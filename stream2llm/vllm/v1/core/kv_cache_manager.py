# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from collections.abc import Iterable
from typing import Dict, List, Optional

from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.core.block_pool import CPUBlockPool, GPUBlockPool
from vllm.v1.core.kv_cache_utils import (BlockHashType, KVCacheBlock,
                                         hash_request_tokens)
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class KVCacheManager:

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        max_model_len: int,
        model_runner,
        sliding_window: Optional[int] = None,
        enable_caching: bool = True,
        num_preallocate_tokens: int = 64,
        log_stats: bool = False,
    ) -> None:
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.max_model_len = max_model_len
        self.max_num_blocks_per_req = cdiv(max_model_len, block_size)
        self.sliding_window = sliding_window
        self.enable_caching = enable_caching
        # FIXME: make prefix cache stats conditional on log_stats
        self.log_stats = log_stats
        # Reference to the model runner for accessing KV cache tensors
        self.model_runner = model_runner
        # NOTE(woosuk): To avoid frequent block allocation, we preallocate some
        # blocks for each request. For example, when a request reaches the end
        # of its block table, we preallocate N blocks in advance. This way, we
        # reduce the overhead of updating free_block_ids and ref_cnts for each
        # request every step (at the cost of some memory waste).
        # NOTE(woosuk): This is different from the "lookahead" slots since this
        # does not guarantee that the request always has N empty blocks. After
        # the request gets N empty blocks, it starts to use the blocks without
        # further allocation. When it uses up all the N empty blocks, it gets
        # N new empty blocks.
        self.num_preallocate_tokens = num_preallocate_tokens
        self.num_preallocate_blocks = cdiv(num_preallocate_tokens, block_size)

        self.gpu_block_pool = GPUBlockPool(num_gpu_blocks, enable_caching)
        # Allocate 2x GPU blocks for CPU to enable overwriting blocks in CPU
        num_cpu_blocks = num_gpu_blocks * 2
        self.cpu_block_pool = CPUBlockPool(num_cpu_blocks, enable_caching)

        # Mapping from request ID to GPU blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_gpu_blocks: defaultdict[
            str, list[KVCacheBlock]] = defaultdict(list)

        # Mapping from request ID to CPU blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request is swapped back to the GPU.
        self.req_to_cpu_blocks: defaultdict[
            str, list[KVCacheBlock]] = defaultdict(list)

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        self.req_to_block_hashes: defaultdict[
            str, list[BlockHashType]] = defaultdict(list)

        # {req_id: The number of cached blocks for this given request}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track the
        # data for reempted ones.
        self.num_cached_block: dict[str, int] = {}
        self.prefix_cache_stats = PrefixCacheStats()

    @property
    def gpu_usage(self) -> float:
        """Get the GPU KV cache usage.

        Returns:
            The GPU KV cache usage (between 0.0 and 1.0).
        """
        return self.gpu_block_pool.get_usage()

    @property
    def cpu_usage(self) -> float:
        """Get the CPU KV cache usage.

        Returns:
            The CPU KV cache usage (between 0.0 and 1.0).
        """
        # TODO: rbachkaniwala3: add cpu usage here
        return 0

    @property
    def total_gpu_blocks(self) -> int:
        return self.num_gpu_blocks

    def make_prefix_cache_stats(self) -> PrefixCacheStats:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats.
        """
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(
            self, request: Request) -> tuple[list[KVCacheBlock], int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        if not self.enable_caching:
            # Prefix caching is disabled.
            if request.status == RequestStatus.PREEMPTED_SWAP:
                return [], request.num_computed_tokens
            return [], 0

        # The block hashes for the request may already be computed
        # if the scheduler has tried to schedule the request before.
        block_hashes = self.req_to_block_hashes[request.request_id]
        if not block_hashes:
            block_hashes = hash_request_tokens(self.block_size, request)
            self.req_to_block_hashes[request.request_id] = block_hashes

        self.prefix_cache_stats.requests += 1
        if request.sampling_params.prompt_logprobs is None:
            # Check for cache hits
            computed_blocks = []
            for block_hash in block_hashes:
                # TODO: rbachkaniwala3: the blocks can be on cpu or gpu, currently only checks if they're on gpu
                # block_hashes is a chain of block hashes. If a block hash
                # is not in the cached_block_hash_to_id, the following
                # block hashes are not computed yet for sure.
                if cached_block := self.gpu_block_pool.get_cached_block(
                        block_hash):
                    computed_blocks.append(cached_block)
                else:
                    break

            self.prefix_cache_stats.queries += len(block_hashes)
            self.prefix_cache_stats.hits += len(computed_blocks)

            # NOTE(woosuk): Since incomplete blocks are not eligible for
            # sharing, `num_computed_tokens` is always a multiple of
            # `block_size`.
            num_computed_tokens = len(computed_blocks) * self.block_size
            return computed_blocks, num_computed_tokens
        else:
            # Skip cache hits for prompt logprobs
            return [], 0

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
        swapped: bool = False,
    ) -> Optional[list[KVCacheBlock]]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_tokens: The number of tokens to allocate. Note that this does
                not include the tokens that have already been computed.
            new_computed_blocks: A list of new computed blocks just hitting the
                prefix caching.

        Blocks layout:
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        if num_tokens == 0:
            raise ValueError("num_tokens must be greater than 0")

        new_computed_blocks = new_computed_blocks or []

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = (request.num_computed_tokens +
                               len(new_computed_blocks) * self.block_size)
        num_required_blocks = cdiv(num_computed_tokens + num_tokens,
                                   self.block_size)
        req_blocks = self.req_to_gpu_blocks[request.request_id]
        if swapped:
            if len(req_blocks) > 0:
                raise ValueError(
                    "If a request is swapped, it should have no blocks on GPU")
        num_new_blocks = (num_required_blocks - len(req_blocks) -
                          len(new_computed_blocks))

        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it cannot be counted as a free block
        # when allocating this request.
        num_evictable_computed_blocks = sum(1 for blk in new_computed_blocks
                                            if blk.ref_cnt == 0)
        if (num_new_blocks > self.gpu_block_pool.get_num_free_blocks() -
                num_evictable_computed_blocks):
            # Cannot allocate new blocks
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.gpu_block_pool.touch(new_computed_blocks)
        else:
            assert not new_computed_blocks, (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        req_blocks.extend(new_computed_blocks)

        # Start to handle new blocks

        if num_new_blocks <= 0:
            # No new block is needed.
            new_blocks = []
        else:
            # Get new blocks from the free block pool considering
            # preallocated blocks.
            num_new_blocks = min(
                num_new_blocks + self.num_preallocate_blocks,
                self.gpu_block_pool.get_num_free_blocks(),
                # Should not exceed the maximum number of blocks per request.
                # This is especially because the block table has the shape
                # [..., max_num_blocks_per_req].
                self.max_num_blocks_per_req - len(req_blocks),
            )
            assert num_new_blocks > 0

            # Concatenate the computed block IDs and the new block IDs.
            new_blocks = self.gpu_block_pool.get_new_blocks(num_new_blocks)
            # For swap in case, the req_blocks should be empty
            assert not swapped or len(req_blocks) == 0
            req_blocks.extend(new_blocks)

        if not self.enable_caching:
            return new_blocks

        # Use `new_computed_blocks` for a new request, and `num_cached_block`
        # for a running request.
        num_cached_blocks = self.num_cached_block.get(request.request_id,
                                                      len(new_computed_blocks))
        # Speculated tokens might be rejected in the future, so we does
        # not cache any speculated tokens. We only cache blocks with
        # generated (accepted) tokens.
        num_full_blocks_after_append = (num_computed_tokens + num_tokens - len(
            request.spec_token_ids)) // self.block_size

        self.gpu_block_pool.cache_full_blocks(
            request=request,
            blocks=req_blocks,
            block_hashes=self.req_to_block_hashes[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks_after_append,
            block_size=self.block_size,
        )

        self.num_cached_block[
            request.request_id] = num_full_blocks_after_append
        return new_blocks

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        When caching is enabled, we free the blocks in reverse order so that
        the tail blocks are evicted first.

        Args:
            request: The request to free the blocks.
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        blocks = self.req_to_gpu_blocks.pop(request.request_id, [])
        ordered_blocks: Iterable[KVCacheBlock] = blocks
        if self.enable_caching:
            # Free blocks in reverse order so that the tail blocks are
            # freed first.
            ordered_blocks = reversed(blocks)

        _ = self.gpu_block_pool.free_blocks(ordered_blocks)
        self.num_cached_block.pop(request.request_id, None)

    def swap_blocks_to_cpu(self, request: Request) -> Dict[int, int]:
        """Swap blocks from GPU to CPU memory for a preempted request. This can only be called once per request during a scheduler's step.
        
        This method updates bookkeeping metadata about the blocks for the request.

        TODO: rbachkanwiwala3 - prefix caching implementation is incomplete. 
        
        Args:
            request: The request whose blocks should be swapped to CPU.
            
        Returns:
            Dict[int, int]: The gpu block ids mapped to cpu block ids
        """
        # Check if we have a model runner reference
        if self.model_runner is None:
            logger.warning(
                "Cannot swap blocks to CPU: No model runner reference")
            return {}

        assert not self.enable_caching, "Prefix caching is not supported when swapping blocks to CPU is enabled."

        # Get the blocks from the req -> gpu_blocks map
        blocks = self.req_to_gpu_blocks.pop(request.request_id, [])

        assert self.req_to_cpu_blocks.get(
            request.request_id
        ) is None, "Cannot swap blocks to CPU: Request already has blocks in CPU. This should never happen."
        assert len(
            blocks
        ) > 0, f"Number of GPU blocks to be swapped for request {request.request_id} is zero. This should never happen."

        # Note: if ref_cnt=0 for the blocks, then they are added to the free queue and returned by free_blocks()
        freed_gpu_blocks = self.gpu_block_pool.free_blocks(blocks)

        # map of blocks to swap out to their cpu block ids
        # Note: If prefix caching is disabled, the blocks_to_swap_out is the same as blocks
        assert len(freed_gpu_blocks) == len(
            blocks
        ), f"Number of blocks to swap out {len(freed_gpu_blocks)} does not match number of blocks for request {request.request_id}. This should never happen."

        # This is done as extra blocks are present due to preallocation of blocks in allocate_slots()
        effective_num_computed_blocks = cdiv(request.num_computed_tokens,
                                             self.block_size)

        # Get the blocks to swap out (limited by the number of computed tokens)
        gpu_blocks_to_swap_out: List[
            KVCacheBlock] = freed_gpu_blocks[:effective_num_computed_blocks]

        # Initialize the list of CPU block IDs for this request
        self.req_to_cpu_blocks[request.request_id] = []

        # Map of GPU block IDs to CPU block IDs
        gpu_to_cpu_block_id_map = {}

        # Loop through blocks and add/update the cpu block pool
        for freed_gpu_block in gpu_blocks_to_swap_out:
            # Use CPUBlockPool's iteration-aware allocation (only uses blocks from previous iterations)
            cpu_block = self.cpu_block_pool.get_new_block(freed_gpu_block)
            # Store the CPU block for this request
            self.req_to_cpu_blocks[request.request_id].append(cpu_block)
            gpu_to_cpu_block_id_map[
                freed_gpu_block.block_id] = cpu_block.block_id

        return gpu_to_cpu_block_id_map

    def restore_blocks_from_cpu(self, request: Request,
                                num_new_tokens: int) -> Dict[int, int]:
        """Restore blocks from CPU memory for a previously preempted request.
            
        Returns:
            Dict[int, int]: A mapping from CPU block IDs to new GPU block IDs.
        """
        # TODO: rbachkanwiwala3 - currently implementation assumes that prefix caching is not enabled.
        assert not self.enable_caching, "Prefix caching is not supported when swapping blocks to CPU is enabled."

        cpu_blocks_to_restore: List[KVCacheBlock] = self.req_to_cpu_blocks.pop(
            request.request_id, [])
        num_cpu_blocks_to_restore = len(cpu_blocks_to_restore)

        assert num_cpu_blocks_to_restore > 0, f"No blocks to restore from CPU for request {request.request_id}"
        assert num_cpu_blocks_to_restore == cdiv(
            request.num_computed_tokens, self.block_size
        ), f"Number of CPU blocks to restore does not match number of computed tokens for request {request.request_id}, num_cpu_blocks={num_cpu_blocks_to_restore}, num_computed_tokens={request.num_computed_tokens}, block_size={self.block_size}"

        # Free up the blocks that are going to be restored
        # CPUBlockPool.free_blocks() will track these as available for next iteration's swap-out
        self.cpu_block_pool.free_blocks(cpu_blocks_to_restore)

        # req_to_gpu_blocks was already updated with new blocks earlier by allocate_slots()
        new_gpu_blocks = self.req_to_gpu_blocks.get(request.request_id, [])

        # Map of CPU block IDs to GPU block IDs
        cpu_to_gpu_block_id_map = {}

        # remove those blocks from the gpu_to_cpu_block_id_map
        for i, cpu_block in enumerate(cpu_blocks_to_restore):
            cpu_to_gpu_block_id_map[
                cpu_block.block_id] = new_gpu_blocks[i].block_id

        return cpu_to_gpu_block_id_map

    def reset_cpu_swap_pools(self):
        """Reset the CPU swap pool tracking at the end of each scheduling cycle."""
        self.cpu_block_pool.reset_iteration_pools()

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if self.gpu_block_pool.reset_prefix_cache():
            self.prefix_cache_stats.reset = True
            return True
        return False

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> int:
        """Calculate the number of common prefix blocks shared by all requests
        in the RUNNING state.

        The function determines this by selecting any request and iterating
        through its blocks.  A block is considered a common prefix block if its
        `ref_cnt` equals the total number of requests in the RUNNING state.

        NOTE(woosuk): The number of requests in the RUNNING state is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because the RUNNING state only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must be in the RUNNING state, the inverse
        is not necessarily true. There may be RUNNING requests that are not
        scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled RUNNING requests that do not
        share the common prefix. Currently, this case cannot be easily detected,
        so the function returns 0 in such cases.

        Args:
            request: Any request in the RUNNING state, used to identify the
                common prefix blocks.
            num_running_requests: The total number of requests in the RUNNING
                state. This can be different from the number of scheduled
                requests in the current step.

        Returns:
            int: The number of common prefix blocks.
        """
        assert request.status == RequestStatus.RUNNING
        blocks = self.req_to_gpu_blocks[request.request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == num_running_requests:
                num_common_blocks += 1
            else:
                break
        return num_common_blocks

    def free_block_hashes(self, request: Request) -> None:
        """Discard the block hashes for the request.

        NOTE: Unlike `free`, this method should be called only when the request
        is finished, not when it is preempted.
        """
        self.req_to_block_hashes.pop(request.request_id, None)

    def invalidate_cache_from_position(self, request: Request,
                                       keep_tokens: int) -> None:
        """Partially invalidate KV cache blocks from a specific position.

        This method is used for prompt update operations where only a portion
        of the prompt changes. It keeps the blocks for the common prefix
        (up to keep_tokens) and frees the blocks for tokens beyond that.

        Handles both GPU blocks (for running requests) and CPU blocks
        (for swapped requests).

        Args:
            request: The request whose KV cache blocks should be partially freed
            keep_tokens: Number of tokens to keep (common prefix length)
        """
        # Number of blocks to keep (rounded up to include partial blocks)
        if keep_tokens == 0:
            num_blocks_to_keep = 0
        else:
            # Use cdiv to round up - partial blocks must be kept
            num_blocks_to_keep = cdiv(keep_tokens, self.block_size)

        blocks_were_freed = False

        # Handle GPU blocks (for running or waiting requests)
        gpu_blocks = self.req_to_gpu_blocks.get(request.request_id, [])
        if gpu_blocks and len(gpu_blocks) > num_blocks_to_keep:
            # Split blocks into those to keep and those to free
            blocks_to_keep = gpu_blocks[:num_blocks_to_keep]
            blocks_to_free = gpu_blocks[num_blocks_to_keep:]

            # Free the blocks beyond the common prefix
            if blocks_to_free:
                self.gpu_block_pool.free_blocks(blocks_to_free)
                blocks_were_freed = True

            # Update the request's block list to only include the kept blocks
            self.req_to_gpu_blocks[request.request_id] = blocks_to_keep

        # Handle CPU blocks (for swapped requests)
        cpu_blocks = self.req_to_cpu_blocks.get(request.request_id, [])
        if cpu_blocks and len(cpu_blocks) > num_blocks_to_keep:
            # Split blocks into those to keep and those to free
            cpu_blocks_to_keep = cpu_blocks[:num_blocks_to_keep]
            cpu_blocks_to_free = cpu_blocks[num_blocks_to_keep:]

            # Free the CPU blocks beyond the common prefix
            if cpu_blocks_to_free:
                self.cpu_block_pool.free_blocks(cpu_blocks_to_free)
                blocks_were_freed = True

            # Update the request's CPU block list
            self.req_to_cpu_blocks[request.request_id] = cpu_blocks_to_keep

        # Set the flag to indicate that blocks were freed.
        # The scheduler will check this flag and set resumed_from_preemption=True
        # to force the worker to reset the block table row instead of appending.
        if blocks_were_freed:
            request.cache_invalidated_this_step = True
