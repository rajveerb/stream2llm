import argparse
import asyncio
import pytest
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.core_client import EngineCoreClient, AsyncMPClient
from vllm.v1.executor.abstract import Executor

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.",
                allow_module_level=True)


@pytest.mark.asyncio
async def test_engine_core_client_asyncio(monkeypatch, model_name,
                                          gpu_memory_utilization,
                                          tensor_parallel_size,
                                          max_num_batched_tokens):

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        # Set max_num_batched_tokens to max model length
        engine_args = EngineArgs(
            model=model_name,
            enable_chunked_prefill=False,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=False,
        )
        vllm_config = engine_args.create_engine_config(
            usage_context=UsageContext.UNKNOWN_CONTEXT)
        executor_class = Executor.get_class(vllm_config)
        client: AsyncMPClient = EngineCoreClient.make_client(
            multiprocess_mode=True,
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=True)

        client.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--gpu_memory_utilization", type=float, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, required=True)
    parser.add_argument("--max_num_batched_tokens", type=int, required=True)
    args = parser.parse_args()
    monkeypatch = pytest.MonkeyPatch()
    asyncio.run(
        test_engine_core_client_asyncio(monkeypatch, args.model_name,
                                        args.gpu_memory_utilization,
                                        args.tensor_parallel_size,
                                        args.max_num_batched_tokens))
