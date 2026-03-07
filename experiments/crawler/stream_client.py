# SPDX-License-Identifier: Apache-2.0

import asyncio, os
import time
import uuid
from typing import Dict, List

from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import EngineCoreClient, AsyncMPClient
from vllm.v1.executor.abstract import Executor

if not current_platform.is_cuda():
    print(reason="V1 currently only supported on CUDA.")
    exit()


def make_stream_request(
    params: SamplingParams,
    request_id: str,
    prompt_token_ids: List[int],
    is_streaming_prompt: bool,
    is_streaming_prompt_finished: bool,
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=prompt_token_ids,
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=params,
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
        is_streaming_prompt=is_streaming_prompt,
        is_streaming_prompt_finished=is_streaming_prompt_finished,
    )


def append_stream_request(
    request_id: str,
    prompt_token_ids: List[int],
    is_streaming_prompt: bool,
    is_streaming_prompt_finished: bool,
) -> EngineCoreRequest:
    """Helper function to append data to stream request"""
    return EngineCoreRequest(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=prompt_token_ids,
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=SamplingParams(),
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
        is_streaming_prompt=is_streaming_prompt,
        is_streaming_prompt_finished=is_streaming_prompt_finished,
    )


def update_stream_request(
    request_id: str,
    prompt_token_ids: List[int],
    is_streaming_prompt: bool,
    is_streaming_prompt_finished: bool,
) -> EngineCoreRequest:
    """Helper function to update prompt tokens in a stream request.

    This is used for ANNS-style workloads where the document set changes
    over time, requiring the prompt to be updated rather than appended to.

    Args:
        request_id: The unique ID of the request
        prompt_token_ids: The complete new set of prompt tokens
        is_streaming_prompt: Whether this is a streaming input request
        is_streaming_prompt_finished: Whether prompt streaming is finished

    Returns:
        EngineCoreRequest with is_prompt_update=True
    """
    return EngineCoreRequest(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=prompt_token_ids,
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=SamplingParams(),
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
        is_streaming_prompt=is_streaming_prompt,
        is_streaming_prompt_finished=is_streaming_prompt_finished,
        is_prompt_update=True,  # This flag distinguishes update from append
    )


def create_stream_client(
    model_name: str,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
    enable_prefix_caching: bool = False,
    log_stats: bool = False,
    tensor_parallel_size: int = 2,
    seed: int = 42,
) -> EngineCoreClient:
    os.environ["VLLM_USE_V1"] = "1"
    engine_args = EngineArgs(
        model=model_name,
        enable_chunked_prefill=False,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        enable_prefix_caching=enable_prefix_caching,
        seed=seed,
    )
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.UNKNOWN_CONTEXT)
    executor_class = Executor.get_class(vllm_config)
    client: AsyncMPClient = EngineCoreClient.make_client(
        multiprocess_mode=True,
        asyncio_mode=True,
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=log_stats,
    )
    return client
