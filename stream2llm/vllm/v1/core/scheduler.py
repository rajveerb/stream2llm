# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import math
import time
from collections import deque
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from vllm.config import (CacheConfig, LoRAConfig, ModelConfig, SchedulerConfig,
                         SpeculativeConfig)
from vllm.logger import init_logger
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.scheduler_output import (CachedRequestData, NewRequestData,
                                           SchedulerOutput)
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        speculative_config: Optional[SpeculativeConfig],
        log_stats: bool,
        structured_output_manager: StructuredOutputManager,
        model_runner: Optional[Any],
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.speculative_config = speculative_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager
        self.model_runner = model_runner

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len

        num_gpu_blocks = cache_config.num_gpu_blocks
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            max_model_len=self.max_model_len,
            model_runner=self.model_runner,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching,
            log_stats=self.log_stats,
        )
        self.block_size = self.cache_config.block_size

        self.sched_policy = None  # TODO: rbachkanwiwala3 add scheduling policy
        # Below function when passed current token budget will return the predicted latency for recomputation
        self.recomputation_latency_cost = None
        # Below function when passed current token budget will return the predicted latency for swap
        self.swap_latency_cost = None

        self.allow_recomp: bool = False
        self.allow_swap: bool = False

        # Scheduler algorithm selection
        self.scheduler_algorithm = os.getenv("SCHEDULER_TYPE", "default_vllm")

        # Validate scheduler algorithm
        valid_algorithms = [
            "default_vllm", "stream_based_v1", "fcfs_lru", "lcas_lifo",
            "mcps_lce", "oeda_pbas", "lcas_cplusp"
        ]

        assert self.scheduler_algorithm in valid_algorithms, "Scheduling algorithm is not supported yet. Pick from the following: " + ", ".join(
            valid_algorithms)

        if self.scheduler_algorithm == "default_vllm":
            self.custom_sched_algo = False
        else:
            self.custom_sched_algo = True

        logger.info(f"Using scheduler algorithm: {self.scheduler_algorithm}")

        # Stream-based scheduler parameters
        self.stream_alpha = float(
            os.getenv("STREAM_ALPHA",
                      "0.5"))  # Weight for time component in priority score
        logger.info(f"Stream-based alpha parameter: {self.stream_alpha}")

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Priority queues for requests.
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        # The requests that have been scheduled and are being executed
        # by the executor.
        self.scheduled_req_ids: set[str] = set()

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # OPTIMIZATION: Cache the CachedRequestData objects to avoid creating
        # them at each scheduling step.
        # Request id -> CachedRequestData
        self._cached_reqs_data: dict[str, CachedRequestData] = {}

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=model_config,
            scheduler_config=scheduler_config,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed). Currently, we assume that the encoder also
        # has the Transformer architecture (e.g., ViT).
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

    def set_recomputation_latency_cost(
            self, recomputation_latency_cost: np.poly1d) -> None:
        self.recomputation_latency_cost = recomputation_latency_cost
        self.allow_recomp = True
        logger.info("Recomputation latency cost function set.")

    def set_swap_latency_cost(self, swap_latency_cost: np.poly1d) -> None:
        self.swap_latency_cost = swap_latency_cost
        self.allow_swap = True
        logger.info("Swap latency cost function set.")

    def set_default_preemption_strategy(self) -> None:
        self.allow_recomp = True
        self.allow_swap = False
        logger.info("Default preemption strategy set to always recompute.")

    def _sort_requests_stream_based_v1(self, ) -> List[Request]:
        # Get all unfinished requests from waiting and running lists
        unfinished_reqs = [
            req for req in (list(self.waiting) + self.running)
            if not req.is_finished()
        ]
        # Removed duplicate line
        all_requests = set(unfinished_reqs)

        ordered_reqs: List[Request] = []

        # sort by streaming prompt requests that have finished streaming or non-streaming requests
        for req in unfinished_reqs:
            if req.check_is_streaming_prompt():
                if req.check_is_streaming_prompt_finished():
                    ordered_reqs.append(req)
                    all_requests.remove(req)
            else:
                ordered_reqs.append(req)
                all_requests.remove(req)
        # sort by arrival time of full prompt requests
        ordered_reqs.sort(key=lambda x: x.arrival_time)
        return ordered_reqs

    def _sort_requests_lcas_cplusp(self, ) -> List[Request]:
        """
        Last Chunk Arrival Scheduling with Complete and then Partial Requests
        Schedules requests in order of the latest observed chunk arrival time (most recent first)
        If there are requests with same arrival time, schedule the complete requests first.
        """
        # Separate full and partial requests
        full_requests = []
        partial_requests = []

        # Get all unfinished requests from waiting and running lists
        unfinished_reqs = [
            req for req in (list(self.waiting) + self.running)
            if not req.is_finished()
        ]

        for req in unfinished_reqs:
            if req.check_is_streaming_prompt():
                if req.check_is_streaming_prompt_finished():
                    full_requests.append(req)
                else:
                    partial_requests.append(req)
            else:
                # Non-streaming requests are considered full
                full_requests.append(req)

        # Sort full requests by arrival time (FCFS)
        full_requests.sort(key=lambda req: req.last_chunk_arrival_time)

        tokens_to_compute_cond = lambda req: req.num_tokens_with_spec - req.num_computed_tokens > 0
        # Sort partial requests by arrival time for opportunistic scheduling, also make sure that there are tokens to schedule
        partial_requests.sort(key=lambda req: req.last_chunk_arrival_time
                              if tokens_to_compute_cond(req) else float('inf'))

        # Full requests get priority, then partial requests
        return full_requests + partial_requests

    def preempt_reqs_fetch_blocks(
        self,
        request: Request,
        num_new_tokens: int,
        preempted_reqs: List[Request],
        scheduled_timestamp: float,
        not_scheduled_reqs: Optional[List[Request]] = None,
    ) -> Tuple[Optional[List[KVCacheBlock]], bool, Optional[Dict[int, int]]]:
        # TODO: rbachkanwiwala3 this is where the eviction logic will be put
        # keep evicting low priority reqs from the back of the running queue
        # until current req can be scheduled
        # Also, fetch block ids for the curr req if possible

        # The block map (swap out) for the preempted reqs that are swapped
        block_map_swap_out_reqs: Dict[int, int] = {}
        while True:
            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens)
            if new_blocks is None:
                # The request cannot be scheduled.
                # Preempt the request with oldest activity (longest time since last chunk/arrival)
                if self.scheduler_algorithm == "stream_based_v1":
                    preempted_req = self._select_request_for_preemption()
                    self.running.remove(preempted_req)
                elif self.scheduler_algorithm == "default_vllm":
                    # Default: preempt the lowest-priority request (LIFO)
                    preempted_req = self.running.pop()
                else:
                    # go from the least priority to the most priority request
                    popped = False
                    preempted_req = None
                    for not_scheduled_req in not_scheduled_reqs[::-1]:
                        if not_scheduled_req in self.running:
                            preempted_req = not_scheduled_req
                            self.running.remove(not_scheduled_req)
                            popped = True
                            break
                    assert popped, "No request to preempt. This should never happen for the scheduling algorithm: '{}' because we already accounted for it's resource requirements.".format(
                        self.scheduler_algorithm)

                # Decide whether to swap blocks to CPU or recompute them
                if self.allow_recomp and self.allow_swap:
                    should_recompute = self.recomputation_latency_cost(
                        num_new_tokens) < self.swap_latency_cost(
                            num_new_tokens)
                elif self.allow_recomp:
                    should_recompute = True
                elif self.allow_swap:
                    should_recompute = False
                else:
                    raise RuntimeError(
                        "Both allow_recomp and allow_swap are False which should never happen."
                    )

                if should_recompute:
                    # Free GPU resources regardless of whether we swapped or not
                    self.kv_cache_manager.free(preempted_req)
                    preempted_req.num_computed_tokens = 0
                    preempted_req.status = RequestStatus.PREEMPTED_RECOMPUTE
                    event_log_status = EngineCoreEventType.PREEMPTED_RECOMPUTE
                else:
                    # Swap blocks to CPU instead of freeing them completely
                    # TODO: rbachkanwiwala3 assumes that swap will always succeed - in reality swap to CPU is bounded by available memory. Account for it in the future.
                    block_map_for_swap_req = self.kv_cache_manager.swap_blocks_to_cpu(
                        preempted_req)
                    # assert that the keys in swap_block_map are not already in block_map_swap_out_reqs
                    assert set(block_map_for_swap_req.keys()).isdisjoint(
                        block_map_swap_out_reqs.keys()
                    ), "Swap block map and block map to swap out have overlapping keys, this should never happen."
                    block_map_swap_out_reqs.update(block_map_for_swap_req)
                    preempted_req.status = RequestStatus.PREEMPTED_SWAP
                    event_log_status = EngineCoreEventType.PREEMPTED_SWAP
                if self.log_stats:
                    preempted_req.record_event(event_log_status,
                                               scheduled_timestamp)

                self.waiting.appendleft(preempted_req)
                preempted_reqs.append(preempted_req)
                if preempted_req == request:
                    # No more request to preempt.
                    can_schedule = False
                    break
            else:
                # The request can be scheduled.
                can_schedule = True
                break
        return new_blocks, can_schedule, block_map_swap_out_reqs


# This file contains the missing method and algorithm additions for the original scheduler

    def _select_request_for_preemption(self) -> Optional[Request]:
        """
        Select request for preemption based on current scheduling algorithm
        """
        if not self.running:
            return None

        if self.scheduler_algorithm == "fcfs_lru":
            return self._evict_lru()
        elif self.scheduler_algorithm == "lcas_lifo":
            return self._evict_lifo()
        elif self.scheduler_algorithm == "mcps_lce":
            return self._evict_lce()
        elif self.scheduler_algorithm == "oeda_pbas":
            return self._evict_oeda()
        elif self.scheduler_algorithm == "stream_based_v1":
            return self._evict_stream_based_v1()
        elif self.scheduler_algorithm == "lcas_cplusp":
            return self._evict_lcas_cplusp()
        else:
            # Default eviction: oldest activity time
            oldest_activity_time = float('inf')
            evict_candidate = None

            for req in self.running:
                if req.check_is_streaming_prompt() and hasattr(
                        req, 'last_chunk_arrival_time'
                ) and req.last_chunk_arrival_time is not None:
                    last_activity_time = req.last_chunk_arrival_time
                else:
                    last_activity_time = req.arrival_time

                if last_activity_time < oldest_activity_time:
                    oldest_activity_time = last_activity_time
                    evict_candidate = req

            return evict_candidate

    # ========== POLICY 1: FCFS-LRU ========== CHECKED
    def _sort_requests_fcfs_lru(self) -> List[Request]:
        """
        First-Come-First-Served with Least-Recently-Used Eviction (FCFS-LRU)
        Schedules full requests in arrival order, with partial requests filled opportunistically
        """
        # Separate full and partial requests
        full_requests = []
        partial_requests = []

        # Get all unfinished requests from waiting and running lists
        unfinished_reqs = [
            req for req in (list(self.waiting) + self.running)
            if not req.is_finished()
        ]

        for req in unfinished_reqs:
            if req.check_is_streaming_prompt():
                if req.check_is_streaming_prompt_finished():
                    full_requests.append(req)
                else:
                    partial_requests.append(req)
            else:
                # Non-streaming requests are considered full
                full_requests.append(req)

        # Sort full requests by arrival time (FCFS)
        full_requests.sort(key=lambda req: req.arrival_time)

        tokens_to_compute_cond = lambda req: req.num_tokens_with_spec - req.num_computed_tokens > 0
        # Sort partial requests by arrival time for opportunistic scheduling, also make sure that there are tokens to schedule
        partial_requests.sort(key=lambda req: req.arrival_time
                              if tokens_to_compute_cond(req) else float('inf'))

        # Full requests get priority, then partial requests
        return full_requests + partial_requests

    # ========== POLICY 2: LCAS-LIFO ========== CHECKED
    def _sort_requests_lcas_lifo(self) -> List[Request]:
        """
        Last Chunk Arrival Scheduling with LIFO Eviction (LCAS-LIFO)
        Schedules requests in order of the latest observed chunk arrival time (most recent first)
        """
        # Get all unfinished requests from waiting and running lists
        all_requests = [
            req for req in (list(self.waiting) + self.running)
            if not req.is_finished()
        ]

        tokens_to_compute_cond = lambda req: req.num_tokens_with_spec - req.num_computed_tokens > 0
        # Sort by most recent chunk arrival time first
        all_requests.sort(key=lambda req: req.last_chunk_arrival_time
                          if tokens_to_compute_cond(req) else float('inf'), )
        return all_requests

    # ========== POLICY 3: MCPS-LCE ========== CHECKED
    def _sort_requests_mcps_lce(self) -> List[Request]:
        """
        Most Chunks Processed Scheduling with Least Chunks Eviction (MCPS-LCE)
        Prioritizes requests by the number of chunks processed (most first)
        """
        # Get all unfinished requests from waiting and running lists
        all_requests = [
            req for req in (list(self.waiting) + self.running)
            if not req.is_finished()
        ]

        def get_sort_key(req):
            # Primary: Number of computed tokens (most first, hence negative)
            # Secondary: For requests with same computed tokens, earliest first chunk arrival
            if req.num_tokens_with_spec - req.num_computed_tokens > 0:
                return (-req.num_computed_tokens, req.arrival_time)
            else:
                return (float('inf'), float('inf'))

        all_requests.sort(key=get_sort_key)
        return all_requests

    # ========== POLICY 4: OEDA-PBAS ==========
    def _sort_requests_oeda_pbas(self) -> List[Request]:
        """
        Overlap-Enhanced Deadline-Aware Priority-Based Adaptive Scheduling (OEDA-PBAS)
        Schedules by urgency and partial readiness
        """
        current_time = time.time()
        # Get all unfinished requests from waiting and running lists
        all_requests = [
            req for req in (list(self.waiting) + self.running)
            if not req.is_finished()
        ]

        def calculate_oeda_priority(req):
            # Urgency calculation
            time_since_first_chunk = current_time - req.arrival_time

            # Assume SLO deadline is 2x expected processing time from arrival
            expected_processing_time = max(
                0.01,
                req.num_computed_tokens / 100.0)  # ~100 tokens/sec, avoid zero
            slo_deadline = req.arrival_time + (2.0 * expected_processing_time)
            time_to_deadline = max(0, slo_deadline - current_time)

            # Higher urgency for requests closer to deadline
            urgency = max(
                0,
                1.0 - (time_to_deadline / max(expected_processing_time, 0.01)))

            # Partial readiness calculation
            if req.num_tokens_with_spec > 0:
                partial_readiness = req.num_computed_tokens / max(
                    req.num_tokens_with_spec, 1)
            else:
                partial_readiness = 0.0

            # Streaming bonus for partial requests
            streaming_bonus = 0.0
            if req.check_is_streaming_prompt(
            ) and not req.check_is_streaming_prompt_finished():
                streaming_bonus = 0.3  # Boost for active streaming requests

            # Combined priority (lower value = higher priority)
            priority = -(urgency + partial_readiness + streaming_bonus)
            return priority

        all_requests.sort(key=calculate_oeda_priority)
        return all_requests

    def _get_sorted_requests_by_algorithm(self) -> List[Request]:
        """
        Get requests sorted by the selected algorithm
        """
        # Get all unfinished requests (waiting + running)
        unfinished_reqs = [
            req for req in (list(self.waiting) + self.running)
            if not req.is_finished()
        ]

        if self.scheduler_algorithm == "fcfs_lru":
            return self._sort_requests_fcfs_lru()
        elif self.scheduler_algorithm == "lcas_lifo":
            return self._sort_requests_lcas_lifo()
        elif self.scheduler_algorithm == "mcps_lce":
            return self._sort_requests_mcps_lce()
        elif self.scheduler_algorithm == "oeda_pbas":
            return self._sort_requests_oeda_pbas()
        elif self.scheduler_algorithm == "stream_based_v1":
            return self._sort_requests_stream_based_v1()
        elif self.scheduler_algorithm == "lcas_cplusp":
            return self._sort_requests_lcas_cplusp()

    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        # NOTE: structured_output_request_ids maps
        # a request's (request that uses structured output)
        # request_id to the running request index.
        # This will helps us determine to slice the grammar bitmask
        # and only applies valid mask for requests that
        # uses structured decoding.
        structured_output_request_ids: dict[str, int] = {}

        req_to_new_block_ids: dict[str, list[int]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        # Need to cache the prompt stream finish status at the time of scheduling, as it may change in the mean time and cause issues in the update_from_output logic
        req_prompt_stream_finished: Dict[str, bool] = {}

        # block map for overall swapped out blocks
        overall_block_map_swap_out_reqs: Dict[int, int] = {}
        overall_block_map_swap_in_reqs: Dict[int, int] = {}

        # For logging.
        scheduled_timestamp = time.time()

        # below variables are for custom scheduling algorithm
        request_to_schedule: List[Request] = []
        request_to_num_new_tokens: Dict[Request, int] = {}
        # request not to be scheduled -> pop requests from ordered_reqs but still preserve the order
        # order is preserved as it's important for the eviction logic for some algorithms
        not_scheduled_reqs: List[Request] = []
        custom_algo_wait_reqs_to_schedule: List[Request] = []

        # new policies: Process requests based on scheduler algorithm
        if self.custom_sched_algo:
            # Use priority-based unified scheduling for enhanced schedulers
            ordered_reqs = self._get_sorted_requests_by_algorithm()

            # construct requests that will be scheduled
            available_token_budget = token_budget
            available_gpu_blocks = self.kv_cache_manager.total_gpu_blocks  # currently no support for prefix caching, hence disabled for streaming feature

            # First, token budget for the following iteration is checked and then gpu blocks are checked
            # if request is not to be scheduled, it is added to `not_scheduled_reqs` but order is still preserved from `ordered_reqs`
            for req in ordered_reqs:
                assert available_token_budget >= 0, f"available_token_budget cannot be negative."
                if available_token_budget == 0:
                    not_scheduled_reqs.append(req)
                    continue

                max_req_new_tokens_to_schedule = req.num_tokens_with_spec - req.num_computed_tokens
                if max_req_new_tokens_to_schedule == 0:
                    not_scheduled_reqs.append(req)
                    continue

                new_tokens_to_schedule = min(max_req_new_tokens_to_schedule,
                                             available_token_budget)

                gpu_blocks_needed = math.ceil(
                    (new_tokens_to_schedule + req.num_computed_tokens) / self.
                    block_size) + self.kv_cache_manager.num_preallocate_blocks
                # not enough gpu blocks to schedule the request
                if (available_gpu_blocks < gpu_blocks_needed
                        or new_tokens_to_schedule == 0):
                    not_scheduled_reqs.append(req)
                    continue

                # request can be scheduled
                request_to_schedule.append(req)
                request_to_num_new_tokens[req] = new_tokens_to_schedule

                # update the available token as well as gpu blocks
                available_token_budget -= new_tokens_to_schedule
                available_gpu_blocks -= gpu_blocks_needed
                if req in self.waiting:
                    custom_algo_wait_reqs_to_schedule.append(req)

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            if request.request_id in self.scheduled_req_ids or (
                    self.custom_sched_algo
                    and request not in request_to_schedule):
                # This request has already been scheduled.
                req_index += 1
                continue
            if self.custom_sched_algo:
                num_new_tokens = request_to_num_new_tokens[request]
            else:
                # default vllm scheduling policy
                num_new_tokens = (request.num_tokens_with_spec -
                                  request.num_computed_tokens)
                num_new_tokens = min(num_new_tokens, token_budget)

            # Schedule encoder inputs.
            encoder_inputs_to_schedule, num_new_tokens, new_encoder_budget = (
                self._try_schedule_encoder_inputs(request,
                                                  request.num_computed_tokens,
                                                  num_new_tokens,
                                                  encoder_budget))
            if num_new_tokens == 0:
                # The request cannot be scheduled because the encoder budget
                # or the encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            new_blocks, can_schedule, block_map_swap_out_reqs = self.preempt_reqs_fetch_blocks(
                request, num_new_tokens, preempted_reqs, scheduled_timestamp,
                not_scheduled_reqs)
            overall_block_map_swap_out_reqs.update(block_map_swap_out_reqs)
            if not can_schedule:
                if self.custom_sched_algo:
                    raise RuntimeError(
                        f"If custom scheduling algorithm is used, the running request should be scheduled."
                    )
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request)
            self.scheduled_req_ids.add(request.request_id)
            if request.use_structured_output:
                # PERF: in case of chunked prefill,
                # request might not include any new tokens.
                # Therefore, we might introduce some additional
                # cycle to fill in the bitmask, which could be a big no-op.
                structured_output_request_ids[request.request_id] = req_index
            req_to_new_block_ids[request.request_id] = [
                b.block_id for b in new_blocks
            ]
            num_scheduled_tokens[request.request_id] = num_new_tokens
            # cache the streaming prompt finish status at the time of scheduling
            if request.check_is_streaming_prompt():
                req_prompt_stream_finished[
                    request.
                    request_id] = request.check_is_streaming_prompt_finished()
            token_budget -= num_new_tokens
            if self.custom_sched_algo and len(preempted_reqs) > 0:
                req_index = 0
            else:
                req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_budget = new_encoder_budget

        # Mark requests that have KV blocks on GPU but were not marked
        # with EngineCoreEventType.KV_ON_GPU and
        # if previously marked with EngineCoreEventType.KV_ON_GPU,
        # mark them as scheduled if request is scheduled.
        if self.log_stats:
            for request in self.running:
                if (request.request_id not in self.scheduled_req_ids
                        and not request.is_kv_on_gpu):
                    # The request has KV blocks on GPU but was not scheduled
                    request.record_event(EngineCoreEventType.KV_ON_GPU,
                                         scheduled_timestamp)
                elif (request.request_id in self.scheduled_req_ids
                      and request.is_kv_on_gpu):
                    # The request has KV blocks on GPU but was scheduled
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)

        # Record the LoRAs in scheduled_running_reqs
        requested_loras: set[int] = set()
        if self.lora_config:
            requested_loras = set(
                req.lora_request.lora_int_id for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0)
            assert len(requested_loras) <= self.lora_config.max_loras

        # check if all the requests supposed to be in scheduled based on the custom scheduling algorithm are scheduled
        if self.custom_sched_algo:
            missed_scheduling_reqs = []
            for req in request_to_schedule:
                if req in self.running and req not in scheduled_running_reqs:
                    missed_scheduling_reqs.append(req)
            if len(missed_scheduling_reqs) > 0:
                for req in missed_scheduling_reqs:
                    print(
                        f"Request {req.request_id}; {req.status.name}; is in request_to_schedule but not in scheduled_running_reqs"
                    )
                print(f"scheduled_req_ids={self.scheduled_req_ids}")
                raise RuntimeError(
                    f"Missed scheduling {len(missed_scheduling_reqs)}/{len(request_to_schedule)} requests"
                )

        # Use a temporary deque to collect requests that need to be skipped
        # and put back at the head of the waiting queue later
        waiting_for_fsm: deque[Request] = deque()

        # Next, schedule the WAITING requests.
        # Note: With 2x CPU blocks and separate CPU pool tracking, we can safely perform
        # simultaneous swap-in and swap-out operations. CPU blocks used for swap-out in the
        # current iteration are tracked separately to prevent conflicts.
        if (not preempted_reqs) or (self.custom_sched_algo
                                    and len(custom_algo_wait_reqs_to_schedule)
                                    > 0):
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break
                if self.custom_sched_algo:
                    if len(custom_algo_wait_reqs_to_schedule) == 0:
                        break
                    request = custom_algo_wait_reqs_to_schedule[0]
                    custom_algo_wait_reqs_to_schedule.pop(0)
                else:
                    request = self.waiting[0]

                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        waiting_structured_output_req = self.waiting.popleft()
                        waiting_for_fsm.appendleft(
                            waiting_structured_output_req)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if self.lora_config and request.lora_request:
                    req_lora_id = request.lora_request.lora_int_id
                    if len(requested_loras) == self.lora_config.max_loras and (
                            req_lora_id not in requested_loras):
                        # Cannot schedule.
                        # TODO (varun): This means all the other requests in
                        # the WAITING queue will be blocked by this request,
                        # even if,
                        # 1. these other requests do not use LoRA, or,
                        # 2. these other requests use the already requested
                        # LoRAs.
                        # This is too conservative and could be optimized.
                        break

                # Get already-cached tokens.
                computed_blocks, num_computed_tokens = \
                    self.kv_cache_manager.get_computed_blocks(request)
                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed requests,
                # which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens
                if num_new_tokens == 0:
                    # This happens when prompt length is divisible by the block
                    # size and all blocks are cached. Now we force to recompute
                    # the last block. Note that we have to re-compute an entire
                    # block because allocate_slots() assumes num_computed_tokens
                    # is always a multiple of the block size. This limitation
                    # can potentially be removed in the future to slightly
                    # improve the performance.
                    num_computed_tokens -= self.block_size
                    num_new_tokens = self.block_size
                    # Currently, when request is in swapped status, the computed blocks are empty.
                    computed_blocks.pop() if len(computed_blocks) > 0 else None
                if self.custom_sched_algo:
                    num_new_tokens = request_to_num_new_tokens[request]
                else:
                    num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                # Schedule encoder inputs.
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_budget) = self._try_schedule_encoder_inputs(
                     request, num_computed_tokens, num_new_tokens,
                     encoder_budget)
                if num_new_tokens == 0:
                    # The request cannot be scheduled.
                    break

                if self.custom_sched_algo:
                    new_blocks, can_schedule, block_map_swap_out_reqs = self.preempt_reqs_fetch_blocks(
                        request, num_new_tokens, preempted_reqs,
                        scheduled_timestamp, not_scheduled_reqs)
                    overall_block_map_swap_out_reqs.update(
                        block_map_swap_out_reqs)
                    assert can_schedule, "If custom scheduling algorithm is used, the waiting request should be scheduled."
                    self.waiting.remove(request)
                else:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request, num_new_tokens, computed_blocks,
                        request.status == RequestStatus.PREEMPTED_SWAP)
                    if new_blocks is None:
                        # The request cannot be scheduled.
                        break
                    self.waiting.popleft()

                if request.use_structured_output:
                    structured_output_request_ids[
                        request.request_id] = req_index
                req_index += 1
                self.running.append(request)
                self.scheduled_req_ids.add(request.request_id)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED_RECOMPUTE:
                    scheduled_resumed_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED_SWAP:
                    # Try to restore blocks from CPU
                    restored_blocks = self.kv_cache_manager.restore_blocks_from_cpu(
                        request, num_new_tokens)
                    overall_block_map_swap_in_reqs.update(restored_blocks)
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    requested_loras.add(request.lora_request.lora_int_id)
                req_to_new_block_ids[request.request_id] = [
                    b.block_id for b in computed_blocks + new_blocks
                ]
                num_scheduled_tokens[request.request_id] = num_new_tokens
                # cache the streaming prompt finish status at the time of scheduling
                if request.check_is_streaming_prompt():
                    req_prompt_stream_finished[
                        request.
                        request_id] = request.check_is_streaming_prompt_finished(
                        )
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens

                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_budget = new_encoder_budget

        # Put back any skipped requests at the head of the waiting queue
        if waiting_for_fsm:
            self.waiting.extendleft(waiting_for_fsm)

        # checks if all the requests supposed to be in scheduled based on the custom scheduling algorithm are scheduled
        if self.custom_sched_algo:
            for req in request_to_schedule:
                if (req not in scheduled_new_reqs
                        and req not in scheduled_resumed_reqs
                        and req not in scheduled_running_reqs):
                    raise RuntimeError(
                        f"Request {req.request_id}; {req.status.name};self.waiting={self.waiting};self.running={self.running};\n\treq in scheduled_new_reqs={req in scheduled_new_reqs};\n\treq in scheduled_resumed_reqs={req in scheduled_resumed_reqs};\n\treq in scheduled_running_reqs={req in scheduled_running_reqs}"
                    )

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) <= len(self.running))

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = 0
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        grammar_bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            len(self.running),
        )
        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        resumed_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=True,
            ) for req in scheduled_resumed_reqs
        ]
        running_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                # If cache was invalidated this step, we need to reset the block
                # table row in the worker (use add_row instead of append_row).
                resumed_from_preemption=req.cache_invalidated_this_step,
            ) for req in scheduled_running_reqs
        ]

        # With 2x CPU blocks and separate CPU pools, we can safely perform simultaneous
        # swap-in and swap-out operations without block conflicts

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=resumed_reqs_data + running_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
            req_prompt_stream_finished=req_prompt_stream_finished,
            gpu_to_cpu_block_map_swap_out=overall_block_map_swap_out_reqs,
            cpu_to_gpu_block_map_swap_in=overall_block_map_swap_in_reqs,
        )

        self.finished_req_ids = set()

        # Clear the cache_invalidated_this_step flag for all scheduled requests
        for req in scheduled_running_reqs:
            req.cache_invalidated_this_step = False

        # Reset CPU swap pool tracking for next iteration
        self.kv_cache_manager.reset_cpu_swap_pools()

        return scheduler_output

    def _make_cached_request_data(
        self,
        request: Request,
        num_scheduled_tokens: int,
        num_scheduled_spec_tokens: int,
        new_block_ids: list[int],
        resumed_from_preemption: bool,
    ) -> CachedRequestData:
        # OPTIMIZATION: Cache the CachedRequestData objects to avoid creating
        # them at each scheduling step.
        num_computed_tokens = request.num_computed_tokens
        num_regular_tokens = num_scheduled_tokens - num_scheduled_spec_tokens
        new_token_ids = request.all_token_ids[
            num_computed_tokens:num_computed_tokens + num_regular_tokens]
        req_data = self._cached_reqs_data.get(request.request_id)
        if req_data is not None:
            req_data.resumed_from_preemption = resumed_from_preemption
            req_data.new_token_ids = new_token_ids
            req_data.new_block_ids = new_block_ids
            req_data.num_computed_tokens = num_computed_tokens
            req_data.is_streaming_prompt = request.check_is_streaming_prompt()
            req_data.is_streaming_prompt_finished = request.check_is_streaming_prompt_finished(
            )
        else:
            req_data = CachedRequestData.from_request(
                request,
                resumed_from_preemption,
                new_token_ids,
                new_block_ids,
                request.check_is_streaming_prompt(),
                request.check_is_streaming_prompt_finished(),
            )
            self._cached_reqs_data[request.request_id] = req_data
        return req_data

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_budget: int,
    ) -> tuple[list[int], int, int]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.
        """
        if not request.has_encoder_inputs():
            return [], num_new_tokens, encoder_budget

        encoder_inputs_to_schedule: list[int] = []
        mm_positions = request.mm_positions
        assert mm_positions is not None
        assert len(mm_positions) > 0
        for i, pos_info in enumerate(mm_positions):
            start_pos = pos_info["offset"]
            num_encoder_tokens = pos_info["length"]

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if start_pos >= num_computed_tokens + num_new_tokens:
                # The encoder input is not needed in this step.
                break
            if start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if self.encoder_cache_manager.has_cache(request, i):
                # The encoder input is already computed and cached.
                continue
            if (not self.encoder_cache_manager.can_allocate(request, i)
                    or num_encoder_tokens > encoder_budget):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - num_computed_tokens
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            encoder_budget -= num_encoder_tokens
            encoder_inputs_to_schedule.append(i)
        return encoder_inputs_to_schedule, num_new_tokens, encoder_budget

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> EngineCoreOutputs:
        sampled_token_ids = model_runner_output.sampled_token_ids
        spec_token_ids = model_runner_output.spec_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        req_prompt_stream_finished = scheduler_output.req_prompt_stream_finished

        new_running: list[Request] = []
        outputs: list[EngineCoreOutput] = []

        # NOTE(woosuk): As len(self.running) can be up to 1K or more, the below
        # loop can be a performance bottleneck. We should do our best to avoid
        # expensive operations inside the loop.
        for request in self.running:
            req_id = request.request_id
            num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
            if num_tokens_scheduled == 0:
                # The request was not scheduled in this step.
                new_running.append(request)
                continue

            if request.check_is_streaming_prompt():
                is_req_prompt_streaming_finished = req_prompt_stream_finished[
                    req_id]

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index]
            if req_id not in scheduler_output.scheduled_spec_decode_tokens:
                # When the request's num_computed_tokens catches up
                # its num_tokens, the request generates output tokens.
                # Otherwise, we ignore the sampler output for the request.
                request.num_computed_tokens += num_tokens_scheduled
                assert request.num_computed_tokens <= request.num_tokens
            else:
                # num_computed_tokens_step represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections.
                # It is calculated as:
                # num_computed_tokens_step = num_scheduled_tokens -
                #                            num_tokens_rejected,
                # where num_tokens_rejected is given by:
                # len(scheduled_spec_token_ids) + 1 - len(generated_token_ids).
                scheduled_spec_token_ids = (
                    scheduler_output.scheduled_spec_decode_tokens[req_id])

                num_computed_tokens_step = num_scheduled_tokens[req_id] - (
                    len(scheduled_spec_token_ids) + 1 -
                    len(generated_token_ids))
                request.num_computed_tokens += num_computed_tokens_step

            cached_encoder_input_ids = (
                self.encoder_cache_manager.get_cached_input_ids(request))
            # OPTIMIZATION: Avoid list(set) if the set is empty.
            if cached_encoder_input_ids:
                for input_id in list(cached_encoder_input_ids):
                    start_pos = request.mm_positions[input_id]["offset"]
                    num_tokens = request.mm_positions[input_id]["length"]
                    if start_pos + num_tokens <= request.num_computed_tokens:
                        # The encoder output is already processed and stored
                        # in the decoder's KV cache.
                        self.encoder_cache_manager.free_encoder_input(
                            request, input_id)

            # Add newly generated spec token ids to the request.
            if spec_token_ids is not None:
                request.spec_token_ids = spec_token_ids[req_index]

            # If the request is streaming prompt, and not all prompt tokens are computed, append req to new_running and move on
            # no output tokens are generated in this case and as a consequence, no EngineCoreOutput is generated
            if request.check_is_streaming_prompt(
            ) and not is_req_prompt_streaming_finished:
                new_running.append(request)
                # This is checked in `def schedule` and if req is present in it it will be skipped so to prevent that we need to remove here.
                self.scheduled_req_ids.remove(request.request_id)
                continue
            stopped = False
            new_logprobs = None
            new_token_ids: list[int] = []

            if request.num_computed_tokens >= request.num_tokens:
                for output_token_id in generated_token_ids:
                    request.append_output_token_ids(output_token_id)
                    new_token_ids.append(output_token_id)

                    # Check for stop and update request state.
                    # This must be called before we make the EngineCoreOutput.
                    stopped = self._check_stop(request)
                    if stopped:
                        if self.log_stats:
                            request.record_event(EngineCoreEventType.FINISHED)
                        self._free_request(request)
                        break

                # Extract sample logprobs if needed.
                if request.sampling_params.logprobs is not None:
                    assert logprobs is not None
                    # NOTE: once we support N tokens per step (spec decode),
                    # the outer lists can be of length > 1.
                    new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and request.use_structured_output:
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # check above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    request.request_id,
                    new_token_ids,
                )

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            # Transmit partial if chunked prefill & prompt logprobs is enabled
            if new_token_ids or prompt_logprobs_tensors is not None:
                # Add EngineCoreOutput for this Request.
                # Include total tokens invalidated if request is finished
                num_tokens_invalidated = None
                if stopped:
                    num_tokens_invalidated = request.total_tokens_invalidated

                outputs.append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        num_tokens_invalidated=num_tokens_invalidated))

            self.scheduled_req_ids.remove(request.request_id)
            if not stopped:
                new_running.append(request)

        self.running = new_running
        return EngineCoreOutputs(
            outputs=outputs,
            scheduler_stats=self.make_stats(),
        )

    def _check_stop(self, request: Request) -> bool:
        if (request.num_tokens >= self.max_model_len
                or request.num_output_tokens >= request.max_tokens):
            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            return True

        sampling_params = request.sampling_params
        last_token_id = request.output_token_ids[-1]
        if (not sampling_params.ignore_eos
                and last_token_id == request.eos_token_id):
            request.status = RequestStatus.FINISHED_STOPPED
            return True

        if last_token_id in (sampling_params.stop_token_ids or ()):
            request.status = RequestStatus.FINISHED_STOPPED
            request.stop_reason = last_token_id
            return True
        return False

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def has_request(self, request_id: str) -> bool:
        return request_id in self.requests

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        else:
            request_ids = set(request_ids)

        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            if request.status == RequestStatus.RUNNING:
                self.running.remove(request)
                self.scheduled_req_ids.discard(request.request_id)
            else:
                self.waiting.remove(request)
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> None:
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        self.kv_cache_manager.free_block_hashes(request)
        self.encoder_cache_manager.free(request)
        self._cached_reqs_data.pop(request.request_id, None)
        del self.requests[request.request_id]
        self.finished_req_ids.add(request.request_id)

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_unfinished_requests(self) -> bool:
        return self.get_num_unfinished_requests() > 0

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def has_requests(self):
        """Returns True if there are unfinished requests, or finished requests
        not yet returned in SchedulerOutputs."""
        return self.has_unfinished_requests() or self.has_finished_requests()

    def get_num_unscheduled_requests(self) -> int:
        """Number of requests that are not being processed by the executor."""
        return self.get_num_unfinished_requests() - len(self.scheduled_req_ids)

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(self) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            gpu_cache_usage=self.kv_cache_manager.gpu_usage,
            cpu_cache_usage=self.kv_cache_manager.cpu_usage,
            prefix_cache_stats=self.kv_cache_manager.make_prefix_cache_stats(),
        )
