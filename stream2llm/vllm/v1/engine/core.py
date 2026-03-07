# SPDX-License-Identifier: Apache-2.0

import json
import os
import queue
import signal
import threading
import time
from concurrent.futures import Future
from inspect import isclass, signature
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional

import msgspec
import numpy as np
import psutil
import zmq
import zmq.asyncio

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from vllm.utils import (get_exception_traceback, resolve_obj_by_qualname,
                        zmq_socket_ctx)
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.core.scheduler import Scheduler as V1Scheduler
from vllm.v1.core.scheduler import SchedulerOutput
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType, UtilityOutput)
from vllm.v1.engine.mm_input_cache import MMInputCacheServer
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5


class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
    ):
        assert vllm_config.model_config.runner_type != "pooling"

        logger.info("Initializing a V1 LLM engine (v%s) with config: %s",
                    VLLM_VERSION, vllm_config)

        self.log_stats = log_stats

        # Setup Model.
        self.model_executor = executor_class(vllm_config)

        # Initialize kv_cache_infos, it will be updated in _initialize_kv_caches
        self.kv_cache_infos: Dict[int, Any] = {}

        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks = self._initialize_kv_caches(
            vllm_config)
        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks

        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # Setup scheduler.
        if isinstance(vllm_config.scheduler_config.scheduler_cls, str):
            Scheduler = resolve_obj_by_qualname(
                vllm_config.scheduler_config.scheduler_cls)
        else:
            Scheduler = vllm_config.scheduler_config.scheduler_cls

        # This warning can be removed once the V1 Scheduler interface is
        # finalized and we can maintain support for scheduler classes that
        # implement it
        if Scheduler is not V1Scheduler:
            logger.warning(
                "Using configured V1 scheduler class %s. "
                "This scheduler interface is not public and "
                "compatibility may not be maintained.",
                vllm_config.scheduler_config.scheduler_cls)

        self.scheduler = Scheduler(
            scheduler_config=vllm_config.scheduler_config,
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            lora_config=vllm_config.lora_config,
            speculative_config=vllm_config.speculative_config,
            log_stats=self.log_stats,
            structured_output_manager=self.structured_output_manager,
            model_runner=self.model_executor,
        )

        # Setup MM Input Mapper.
        self.mm_input_cache_server = MMInputCacheServer(
            vllm_config.model_config)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: Optional[queue.Queue[tuple[Future[ModelRunnerOutput],
                                                     SchedulerOutput]]] = None
        if self.batch_queue_size > 1:
            logger.info("Batch queue is enabled with size %d",
                        self.batch_queue_size)
            self.batch_queue = queue.Queue(self.batch_queue_size)

        self.block_size = vllm_config.cache_config.block_size

        # metadata for recomp and swap latency predictor
        #  it is used to verify the recomputation and swap latency predictor is correct if we use saved data
        #  it is used to save the recomputation and swap latency data to a file if we save this data
        recomp_and_swap_metadata = {
            "model_name":
            vllm_config.model_config.model,
            "gpu_memory_utilization":
            vllm_config.cache_config.gpu_memory_utilization,
            "tensor_parallel_size":
            vllm_config.parallel_config.tensor_parallel_size,
            "max_num_batched_tokens":
            vllm_config.scheduler_config.max_num_batched_tokens,
        }

        # Disabled for large_simple_test  overlap_test  simple_test testing
        # check recomputation predictor env flag is passed as 1, if not default to False
        if os.getenv("USE_RECOMPUTATION_LATENCY_PREDICTOR", "0") == "1":
            recomputation_latency_curve_func = self.initialize_recomputation_latency_predictor(
                vllm_config, recomp_and_swap_metadata)
            # Since, scheduler is already initialized, we need to set the recomputation_latency_curve_func after initialization
            self.scheduler.set_recomputation_latency_cost(
                recomputation_latency_curve_func)

        if os.getenv("USE_SWAP_LATENCY_PREDICTOR", "0") == "1":
            swap_latency_curve_func = self.initialize_swap_latency_predictor(
                vllm_config, recomp_and_swap_metadata)
            # Since, scheduler is already initialized, we need to set the swap_latency_curve_func after initialization
            self.scheduler.set_swap_latency_cost(swap_latency_curve_func)

        if os.getenv("USE_RECOMPUTATION_LATENCY_PREDICTOR",
                     "0") == "0" and os.getenv("USE_SWAP_LATENCY_PREDICTOR",
                                               "0") == "0":
            self.scheduler.set_default_preemption_strategy()

    def initialize_recomputation_latency_predictor(
            self, vllm_config: VllmConfig,
            recomp_and_swap_metadata: Dict[str, Any]) -> np.poly1d:
        """
        Initialize the recomputation latency predictor.

        This function either measures the recomputation latency or loads
        previously saved latency data. If measuring, it can optionally
        save the results to a file for future use.

        Environment variables:
            USE_RECOMPUTATION_LATENCY_PREDICTOR (str): If set to "1", enables the
                recomputation latency predictor.
            USE_SAVED_RECOMPUTATION_DATA_PATH (str): If set, loads
                previously saved latency data from the specified path.
            SAVE_RECOMPUTATION_DATA_PATH (str): If set, saves the
                measured latency data to the specified path.

        Args:
            vllm_config (VllmConfig): Configuration object for vLLM.

        Returns:
            np.poly1d: A polynomial function representing the
                       recomputation latency curve.
        """
        # Check if we should use saved recomputation latency data
        USE_SAVED_RECOMPUTATION_DATA_PATH = os.getenv(
            "USE_SAVED_RECOMPUTATION_DATA_PATH", None)

        # Initialize stats tracker
        stats_tracker = {}

        if USE_SAVED_RECOMPUTATION_DATA_PATH is not None:
            # Skip latency measurement and use saved data
            logger.info(
                f"Reading recomputation latency data from {USE_SAVED_RECOMPUTATION_DATA_PATH}"
            )
            with open(USE_SAVED_RECOMPUTATION_DATA_PATH) as f:
                saved_recomputation_data = json.load(f)

                # Convert # of tokens keys back to integers
                stats_tracker["recomputation_latency"]: Dict[
                    int, List[float]] = {
                        int(num_tokens): latencies
                        for num_tokens, latencies in
                        saved_recomputation_data["recomputation_latencies"].
                        items()
                    }
                sorted_num_tokens = sorted([
                    int(num_tokens) for num_tokens in
                    saved_recomputation_data["recomputation_latencies"].keys()
                ])
                stats_tracker["num_tokens_per_request"]: List[
                    int] = sorted_num_tokens
                logger.info(
                    f"Successfully loaded recomputation latency data from {USE_SAVED_RECOMPUTATION_DATA_PATH}"
                )
                # verify the metadata is correct
                if recomp_and_swap_metadata != saved_recomputation_data[
                        "metadata"]:
                    logger.warning(
                        "Metadata mismatch between saved and current recomputation latency data"
                    )
                    logger.warning(
                        f"Saved metadata: {saved_recomputation_data['metadata']}"
                    )
                    logger.warning(
                        f"Current metadata: {recomp_and_swap_metadata}")
        else:
            # Perform latency measurement
            stats_tracker = self._measure_recomputation_latency(vllm_config)
            # Save current data if SAVE_RECOMPUTATION_DATA_PATH is set
            SAVE_RECOMPUTATION_DATA_PATH = os.getenv(
                "SAVE_RECOMPUTATION_DATA_PATH", None)
            if SAVE_RECOMPUTATION_DATA_PATH is not None:
                with open(SAVE_RECOMPUTATION_DATA_PATH, "w") as f:
                    json.dump(
                        {
                            "recomputation_latencies":
                            stats_tracker["recomputation_latency"],
                            "metadata":
                            recomp_and_swap_metadata,
                        },
                        f,
                        indent=4)
                    logger.info(
                        f"Saved recomputation latency data to {SAVE_RECOMPUTATION_DATA_PATH}"
                    )

        # log stats tracker but in a prettier way and details
        logger.info("Recomputation latency stats:\n"
                    "\tMean latency (ms) per num_tokens:\n" + "\n".join(
                        f"\t\t{num_token:4d} tokens: {np.mean(latencies):8.2f}"
                        for num_token, latencies in sorted(
                            stats_tracker["recomputation_latency"].items())))

        # recomputation latency
        recomputation_data = stats_tracker["recomputation_latency"]
        # Extract keys and values
        x = np.array(list(recomputation_data.keys()))
        y = np.array([np.mean(v) for v in recomputation_data.values()
                      ])  # Use mean values for fitting

        # Fit a quadratic curve (2nd degree polynomial)
        coeffs = np.polyfit(x, y, 2)  # Quadratic fit

        # Create the quadratic function
        recomputation_latency_curve_func = np.poly1d(coeffs)
        y_pred = recomputation_latency_curve_func(x)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum(
            (y - np.mean(y))**2)  # R-squared

        # Check if r2 is close to 1
        if r2 < 0.9:
            logger.warning(
                "Low fidelity for recomputation latency curve predictor")
            logger.info(f"r2: {r2}")

        return recomputation_latency_curve_func

    def _measure_recomputation_latency(self, vllm_config: VllmConfig):
        """Measure recomputation latency by running dummy requests."""
        max_model_len = vllm_config.model_config.max_model_len
        model_vocab_size = vllm_config.model_config.get_vocab_size()

        # get power of 2 values from min to max
        def get_power_of_2_values(min_val, max_val):
            """Get powers of 2 between min_val and max_val."""
            powers = []
            power = 1
            while power < min_val:
                power *= 2
            while power <= max_val:
                powers.append(power)
                power *= 2
            if max_val not in powers:
                powers.append(max_val)
            return powers

        def create_dummy_token_ids(size: int, vocab_size: int) -> List[int]:
            """Create a list of dummy token IDs of specified size.
    
            Uses a simple pattern that cycles through tokens 0 to vocab_size-1 to ensure
            we have valid token IDs within the model's vocabulary range while being
            deterministic.
    
            Args:
                size: Number of token IDs to generate
            vocab_size: Size of the model's vocabulary
    
            Returns:
                List of token IDs within the valid vocabulary range
            """
            return [next(unique_first_token_gen)] + [
                i % vocab_size for i in range(size - 1)
            ]  # Cycle through 0 to vocab_size-1

        def create_dummy_request(token_count: int,
                                 vocab_size: int) -> EngineCoreRequest:
            """Create a dummy EngineCoreRequest with specified number of tokens.
    
            Args:
                token_count: Number of tokens in the request
                vocab_size: Size of the model's vocabulary
    
            Returns:
                EngineCoreRequest with specified number of tokens
            """
            import uuid
            return EngineCoreRequest(
                request_id=str(uuid.uuid4()),
                prompt=None,  # Not needed when added to EngineCoreClient
                prompt_token_ids=create_dummy_token_ids(
                    token_count, vocab_size),
                mm_inputs=None,
                mm_hashes=None,
                mm_placeholders=None,
                sampling_params=SamplingParams(max_tokens=1),
                eos_token_id=None,
                arrival_time=time.time(),
                lora_request=None,
            )

        # generate power of 2 values
        num_tokens_set = get_power_of_2_values(self.block_size, max_model_len)
        # set tracker
        stats_tracker = {}
        stats_tracker["num_tokens_per_request"] = num_tokens_set
        stats_tracker["recomputation_latency"] = {
            num_token: []
            for num_token in num_tokens_set
        }
        # run it 5 times (it is enough since variance is low) - drop first 2 points
        repeat = 7
        discard = 2
        assert repeat * len(
            num_tokens_set
        ) <= model_vocab_size, "If this condition doesn't hold, then prefix caching needs to be disabled first before generating recomputation latencies. And, using saved recomputation number without prefix caching enabled."

        unique_first_token_gen = (i for i in range(model_vocab_size))

        for num_tokens in sorted(num_tokens_set):
            for i in range(repeat):
                self.add_request(
                    create_dummy_request(num_tokens, model_vocab_size))
                start_time = time.time_ns()
                self.step()
                duration = (time.time_ns() - start_time) / 1e6  # in ms
                if i >= discard:
                    # record after `discard` number
                    stats_tracker["recomputation_latency"][num_tokens].append(
                        duration)

        return stats_tracker

    def initialize_swap_latency_predictor(
            self, vllm_config: VllmConfig,
            recomp_and_swap_metadata: Dict[str, Any]) -> np.poly1d:
        """
        Initialize the swap latency predictor.

        This function either measures the swap latency or loads
        previously saved latency data. If measuring, it can optionally
        save the results to a file for future use.

        Environment variables:
            USE_SWAP_LATENCY_PREDICTOR (str): If set to "1", enables the
                swap latency predictor.
            USE_SAVED_SWAP_LATENCY_DATA_PATH (str): If set, loads
                previously saved swap latency data from the specified path.
            SAVE_SWAP_LATENCY_DATA_PATH (str): If set, saves the
                measured swap latency data to the specified path.

        Args:
            vllm_config (VllmConfig): Configuration object for vLLM.

        Returns:
            np.poly1d: A polynomial function representing the
                       swap latency curve.
        """
        # Check if we should use saved swap latency data
        USE_SAVED_SWAP_LATENCY_DATA_PATH = os.getenv(
            "USE_SAVED_SWAP_LATENCY_DATA_PATH", None)

        # Initialize stats tracker
        stats_tracker = {}

        if USE_SAVED_SWAP_LATENCY_DATA_PATH is not None:
            # Skip latency measurement and use saved data
            logger.info(
                f"Reading swap latency data from {USE_SAVED_SWAP_LATENCY_DATA_PATH}"
            )
            with open(USE_SAVED_SWAP_LATENCY_DATA_PATH) as f:
                saved_swap_data = json.load(f)
                # Convert # of tokens keys back to integers
                stats_tracker["swap_in_latency"]: Dict[int, List[float]] = {
                    int(num_tokens): latencies
                    for num_tokens, latencies in
                    saved_swap_data["swap_in_latency"].items()
                }
                stats_tracker["swap_out_latency"]: Dict[int, List[float]] = {
                    int(num_tokens): latencies
                    for num_tokens, latencies in
                    saved_swap_data["swap_out_latency"].items()
                }
                # number of unique tokens
                sorted_num_tokens = sorted([
                    int(num_tokens) for num_tokens in
                    saved_swap_data["swap_in_latency"].keys()
                ])
                stats_tracker["num_tokens_per_request"]: List[
                    int] = sorted_num_tokens
                logger.info(
                    f"Successfully loaded swap latency data from {USE_SAVED_SWAP_LATENCY_DATA_PATH}"
                )
                # verify the metadata is correct
                if recomp_and_swap_metadata != saved_swap_data["metadata"]:
                    logger.warning(
                        "Metadata mismatch between saved and current swap latency data"
                    )
                    logger.warning(
                        f"Saved metadata: {saved_swap_data['metadata']}")
                    logger.warning(
                        f"Current metadata: {recomp_and_swap_metadata}")
        else:
            # Perform latency measurement
            stats_tracker = self._measure_swap_latency(vllm_config)
            # Save current data if SAVE_SWAP_LATENCY_DATA_PATH is set
            SAVE_SWAP_LATENCY_DATA_PATH = os.getenv(
                "SAVE_SWAP_LATENCY_DATA_PATH", None)
            if SAVE_SWAP_LATENCY_DATA_PATH is not None:
                with open(SAVE_SWAP_LATENCY_DATA_PATH, "w") as f:
                    json.dump(
                        {
                            "swap_in_latency":
                            stats_tracker["swap_in_latency"],
                            "swap_out_latency":
                            stats_tracker["swap_out_latency"],
                            "metadata": recomp_and_swap_metadata,
                        },
                        f,
                        indent=4)
                    logger.info(
                        f"Saved swap latency data to {SAVE_SWAP_LATENCY_DATA_PATH}"
                    )

        # Add swap in and out latencies
        swap_data = {}
        for num_tokens in stats_tracker["num_tokens_per_request"]:
            swap_data[num_tokens] = [
                swap_in + swap_out for swap_in, swap_out in zip(
                    stats_tracker["swap_in_latency"][num_tokens],
                    stats_tracker["swap_out_latency"][num_tokens])
            ]
        # log stats tracker but in a prettier way and details
        logger.info("Swap_in/Swap_out/Total mean latency stats:\n" + "\n".join(
            f"{num_token:8d} tokens: {np.mean(stats_tracker['swap_in_latency'][num_token]):.2f}/{np.mean(stats_tracker['swap_out_latency'][num_token]):.2f}/{np.mean(swap_data[num_token]):.2f} ms"
            for num_token in stats_tracker["num_tokens_per_request"]))

        # Extract keys and values
        x = np.array(list(swap_data.keys()))
        y = np.array([np.mean(v) for v in swap_data.values()
                      ])  # Use mean values for fitting

        # Fit a quadratic curve (2nd degree polynomial)
        coeffs = np.polyfit(x, y, 2)  # Quadratic fit

        # Create the quadratic function
        swap_latency_curve_func = np.poly1d(coeffs)
        y_pred = swap_latency_curve_func(x)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum(
            (y - np.mean(y))**2)  # R-squared

        # Check if r2 is close to 1
        if r2 < 0.9:
            logger.warning("Low fidelity for swap latency curve predictor")
            logger.info(f"r2: {r2}")

        return swap_latency_curve_func

    def _measure_swap_latency(self, vllm_config: VllmConfig):
        """Measure swap in and out latency for each number of tokens."""
        max_model_len = vllm_config.model_config.max_model_len

        # Verify all gpu_workers have the same KV cache shape for each layer
        # This implementation is based on the assumption that all gpu_workers have the same KV cache shape for each layer
        gpu_workers = list(self.kv_cache_infos.keys())
        kv_cache_reference_shapes = self.kv_cache_infos[gpu_workers[0]]
        if len(gpu_workers) > 1:
            for gpu_worker in gpu_workers:
                kv_cache_device_shapes: List[int] = self.kv_cache_infos[
                    gpu_worker]
                assert len(kv_cache_reference_shapes) == len(
                    kv_cache_device_shapes
                ), "Different number of layers between devices"
                for i in range(len(kv_cache_reference_shapes)):
                    assert kv_cache_reference_shapes[
                        i] == kv_cache_device_shapes[
                            i], f"Layer {i} has different shapes reference shape {kv_cache_reference_shapes} vs device shape {kv_cache_device_shapes} across devices"

        # define kv_cache_spec which is a dictionary of # of layers, # of devices, and shape
        kv_cache_spec = {
            "num_layers": len(kv_cache_reference_shapes),
            "num_devices": len(gpu_workers),
            "max_gpu_blocks_per_gpu": kv_cache_reference_shapes[0]
            [1]  # Get layer 0's number of blocks Assuming shape is (2, num_blocks, block_size, num_kv_heads, head_size)
        }

        # get power of 2 values from min to max
        def get_power_of_2_values(min_val, max_val):
            """Get powers of 2 between min_val and max_val."""
            powers = []
            power = 1
            while power < min_val:
                power *= 2
            while power <= max_val:
                powers.append(power)
                power *= 2
            if max_val not in powers:
                powers.append(max_val)
            return powers

        # generate power of 2 values
        num_tokens_set = get_power_of_2_values(self.block_size, max_model_len)
        num_blocks_set = [
            num_tokens // self.block_size for num_tokens in num_tokens_set
        ]
        max_num_blocks_per_req = max(num_blocks_set)
        assert max_num_blocks_per_req <= kv_cache_spec[
            "max_gpu_blocks_per_gpu"], f"Cannot model swap latencies for because the max theoretical number of blocks per request ({max_num_blocks_per_req}) exceeds the max number of blocks per GPU ({kv_cache_spec['max_gpu_blocks_per_gpu']})"

        # set tracker
        stats_tracker = {}
        stats_tracker["num_tokens_per_request"] = num_tokens_set
        stats_tracker["swap_in_latency"] = {
            num_tokens: []
            for num_tokens in num_tokens_set
        }
        stats_tracker["swap_out_latency"] = {
            num_tokens: []
            for num_tokens in num_tokens_set
        }
        # run it 5 times (it is enough since variance is low) - drop first 2 points
        repeat = 7
        discard = 2

        # Measure swap latency for each number of blocks in num_blocks_set
        for num_blocks in sorted(num_blocks_set):
            # Call measure_kv_cache_swap_latency on all workers via collective_rpc
            all_swap_metrics = self.model_executor.collective_rpc(
                "measure_kv_cache_swap_latency",
                args=(repeat, discard, num_blocks))

            for worker_metrics in all_swap_metrics:
                num_tokens = num_blocks * self.block_size
                stats_tracker["swap_in_latency"][num_tokens].extend(
                    worker_metrics["swap_in_latency"])
                stats_tracker["swap_out_latency"][num_tokens].extend(
                    worker_metrics["swap_out_latency"])

        return stats_tracker

    def _initialize_kv_caches(self,
                              vllm_config: VllmConfig) -> tuple[int, int]:
        start = time.time()

        # Get all kv cache needed by the model
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        # Profiles the peak memory usage of the model to determine how much
        # memory can be allocated for kv cache.
        available_gpu_memory = self.model_executor.determine_available_memory()

        # Get the kv cache tensor size
        kv_cache_configs = get_kv_cache_configs(vllm_config, kv_cache_specs,
                                                available_gpu_memory)
        num_gpu_blocks_set = set(config.num_blocks
                                 for config in kv_cache_configs)
        assert len(num_gpu_blocks_set) == 1, (
            f"num_gpu_blocks need to be the same across workers, "
            f"but they are different: {num_gpu_blocks_set}")
        num_gpu_blocks = num_gpu_blocks_set.pop()
        num_cpu_blocks = 0

        # Initialize kv cache and warmup the execution
        kv_cache_infos = self.model_executor.initialize_from_config(
            kv_cache_configs)

        # kv_cache_infos is a list of kv_cache_info
        # kv_cache_info is a dict of device_id to [[layer_1 shape], [layer_2 shape], ...]
        for kv_cache_info in kv_cache_infos:
            for device_id in kv_cache_info.keys():
                self.kv_cache_infos[device_id] = kv_cache_info[device_id]

        # assert that kv_cache_infos is not empty
        assert len(self.kv_cache_infos) > 0, "kv_cache_infos is empty"

        elapsed = time.time() - start
        logger.info(("init engine (profile, create kv cache, "
                     "warmup model) took %.2f seconds"), elapsed)
        return num_gpu_blocks, num_cpu_blocks

    def add_request(self, request: EngineCoreRequest):
        """Add request to the scheduler."""

        if request.mm_hashes is not None:
            # Here, if hash exists for a multimodal input, then it will be
            # fetched from the cache, else it will be added to the cache.
            # Note that the cache here is mirrored with the client cache, so
            # anything that has a hash must have a HIT cache entry here
            # as well.
            assert request.mm_inputs is not None
            request.mm_inputs = self.mm_input_cache_server.get_and_update(
                request.mm_inputs, request.mm_hashes)

        # check if the request is a streaming prompt and if it has been added to the scheduler
        # if true, then this request contains other parts of the streaming prompt
        if request.is_streaming_prompt and self.scheduler.has_request(
                request.request_id):

            existing_request: Request = self.scheduler.requests[
                request.request_id]
            assert not existing_request.check_is_streaming_prompt_finished(
            ), "If streaming prompt is finished then this request should not have been sent (out of order request might cause this issue)"

            # Check if this is an update operation or append operation
            if request.is_prompt_update:
                # Update the prompt tokens (for ANNS-style workloads)
                common_prefix_len, tokens_to_invalidate = existing_request.update_prompt_tokens(
                    request)

                # If tokens need to be invalidated, free corresponding KV cache blocks
                if tokens_to_invalidate > 0:
                    self.scheduler.kv_cache_manager.invalidate_cache_from_position(
                        existing_request, common_prefix_len)
            else:
                # Append new tokens (for crawler-style workloads)
                existing_request.append_streaming_prompt_tokens(request)
            return

        req = Request.from_engine_core_request(request)
        if req.use_structured_output:
            # Start grammar compilation asynchronously
            self.structured_output_manager.grammar_init(req)

        self.scheduler.add_request(req)

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids,
                                       RequestStatus.FINISHED_ABORTED)

    def step(self) -> EngineCoreOutputs:
        """Schedule, execute, and make output."""

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        # TODO: rbachkanwiwala3 this can be a problem, it is different than previous 0.7.2's implementation (need to verify)
        if not self.scheduler.has_requests():
            return EngineCoreOutputs(
                outputs=[],
                scheduler_stats=self.scheduler.make_stats(),
            )
        scheduler_output = self.scheduler.schedule()

        if scheduler_output.total_num_scheduled_tokens == 0:
            return EngineCoreOutputs(
                outputs=[],
                scheduler_stats=self.scheduler.make_stats(),
            )

        model_output = self.model_executor.execute_model(scheduler_output)
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output)  # type: ignore

        if len(engine_core_outputs.outputs) == 0:
            # Occurs due to streaming prompt requests
            return EngineCoreOutputs(
                outputs=[],
                scheduler_stats=self.scheduler.make_stats(),
            )
        return engine_core_outputs

    def step_with_batch_queue(self) -> Optional[EngineCoreOutputs]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if there are unscheduled requests
        and the job queue is not full. If a new batch is scheduled, directly
        return an empty engine core output. In other words, we won't check
        and return model outputs before the batch queue is full.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """
        assert self.batch_queue is not None

        engine_core_outputs = None
        scheduler_output = None
        # If there are unscheduled requests and the job queue
        # is not full, schedule a new batch. Note that this is not blocking.
        if (self.scheduler.get_num_unscheduled_requests() > 0
                and not self.batch_queue.full()):
            scheduler_output = self.scheduler.schedule()
            if scheduler_output.total_num_scheduled_tokens > 0:
                future = self.model_executor.execute_model(scheduler_output)
                self.batch_queue.put_nowait(
                    (future, scheduler_output))  # type: ignore

        scheduled_batch = (scheduler_output is not None
                           and scheduler_output.total_num_scheduled_tokens > 0)

        # If no more requests can be scheduled and the job queue is not empty,
        # block until the first batch in the job queue is finished.
        if not scheduled_batch and not self.batch_queue.empty():
            future, scheduler_output = self.batch_queue.get_nowait()
            # Blocking until the first result is available.
            model_output = future.result()
            self.batch_queue.task_done()
            engine_core_outputs = self.scheduler.update_from_output(
                scheduler_output, model_output)

        return engine_core_outputs

    def shutdown(self):
        self.model_executor.shutdown()

    def profile(self, is_start: bool = True):
        self.model_executor.profile(is_start)

    def reset_prefix_cache(self):
        self.scheduler.reset_prefix_cache()

    def sleep(self, level: int = 1):
        self.model_executor.sleep(level)

    def wake_up(self):
        self.model_executor.wake_up()

    def is_sleeping(self) -> bool:
        return self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        self.model_executor.collective_rpc("execute_dummy_batch")

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        ready_pipe: Connection,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
    ):
        super().__init__(vllm_config, executor_class, log_stats)

        # Background Threads and Queues for IO. These enable us to
        # overlap ZMQ socket IO with GPU since they release the GIL,
        # and to overlap some serialization/deserialization with the
        # model forward pass.
        # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
        self.input_queue: queue.Queue[tuple[EngineCoreRequestType,
                                            Any]] = queue.Queue()
        self.output_queue: queue.Queue[EngineCoreOutputs] = queue.Queue()
        threading.Thread(target=self.process_input_socket,
                         args=(input_path, ),
                         daemon=True).start()
        threading.Thread(target=self.process_output_socket,
                         args=(output_path, ),
                         daemon=True).start()

        # Send Readiness signal to EngineClient.
        ready_pipe.send({"status": "READY"})

    @staticmethod
    def run_engine_core(*args, **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        parent_process = psutil.Process().parent()
        engine_core = None
        try:
            engine_core = EngineCoreProc(*args, **kwargs)
            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore interrupted.")

        except Exception:
            traceback = get_exception_traceback()
            logger.error("EngineCore hit an exception: %s", traceback)
            parent_process.send_signal(signal.SIGUSR1)

        finally:
            if engine_core is not None:
                engine_core.shutdown()

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""

        step_fn = (self.step
                   if self.batch_queue is None else self.step_with_batch_queue)

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            while not self.scheduler.has_requests():
                logger.debug("EngineCore busy loop waiting.")
                req = self.input_queue.get()
                self._handle_client_request(*req)

            # 2) Handle any new client requests.
            while not self.input_queue.empty():
                req = self.input_queue.get_nowait()
                self._handle_client_request(*req)

            # 3) Step the engine core.
            outputs = step_fn()

            # 4) Put EngineCoreOutputs into the output queue.
            if outputs.outputs:
                self.output_queue.put_nowait(outputs)

    def _handle_client_request(self, request_type: EngineCoreRequestType,
                               request: Any) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.ADD or request_type == EngineCoreRequestType.STREAM_ADD:
            self.add_request(request)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            call_id, method_name, args = request
            output = UtilityOutput(call_id)
            try:
                method = getattr(self, method_name)
                output.result = method(
                    *self._convert_msgspec_args(method, args))
            except BaseException as e:
                logger.exception("Invocation of %s method failed", method_name)
                output.failure_message = (f"Call to {method_name} method"
                                          f" failed: {str(e)}")
            self.output_queue.put_nowait(
                EngineCoreOutputs(utility_output=output))

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
         arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation) if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation) else v
            for v, p in zip(args, arg_types))

    def process_input_socket(self, input_path: str):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()

        with zmq_socket_ctx(input_path, zmq.constants.PULL) as socket:
            while True:
                # (RequestType, RequestData)
                type_frame, data_frame = socket.recv_multipart(copy=False)
                request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                # Deserialize the request data.
                decoder = add_request_decoder if (
                    request_type == EngineCoreRequestType.ADD or request_type
                    == EngineCoreRequestType.STREAM_ADD) else generic_decoder
                request = decoder.decode(data_frame.buffer)

                # Push to input queue for core busy loop.
                self.input_queue.put_nowait((request_type, request))

    def process_output_socket(self, output_path: str):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Reuse send buffer.
        buffer = bytearray()

        with zmq_socket_ctx(output_path, zmq.constants.PUSH) as socket:
            while True:
                outputs = self.output_queue.get()
                encoder.encode_into(outputs, buffer)
                socket.send_multipart((buffer, ), copy=False)
