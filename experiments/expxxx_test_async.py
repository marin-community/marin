"""Standalone async vLLM smoke test.

This experiment spins up a minimal AsyncvLLMInferenceContext on TPU via Ray
and generates a short sample to validate that initialization succeeds.
"""

import argparse
import logging

import ray
from vllm import SamplingParams

from marin.rl.environments.inference_ctx.async_vllm import AsyncvLLMInferenceContext
from marin.rl.environments.inference_ctx.vllm import InferenceMode, vLLMInferenceContextConfig

logger = logging.getLogger(__name__)


def _build_sampling_params(max_tokens: int) -> SamplingParams:
    """Deterministic params for quick validation runs."""
    return SamplingParams(
        temperature=0.0,
        n=1,
        max_tokens=max_tokens,
        stop=None,
        logprobs=0,
        include_stop_str_in_output=True,
    )


@ray.remote(resources={"TPU": 4})
def init_async_vllm_remote(
    model_name: str,
    *,
    max_model_len: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    prompt: str,
    max_tokens: int,
) -> str:
    """Remote helper that instantiates async vLLM and returns a completion."""
    sampling_params = _build_sampling_params(max_tokens)
    config = vLLMInferenceContextConfig(
        model_name=model_name,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        sampling_params=sampling_params,
        mode=InferenceMode.ASYNC,
    )

    ctx = AsyncvLLMInferenceContext(config)
    completions = ctx.batch_completions(
        prompts=[prompt],
        temperature=0.0,
        n=1,
        max_tokens=max_tokens,
        stop=None,
        system_prompt=None,
    )
    ctx.llm.shutdown()
    return completions[0].choices[0].message.content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test AsyncvLLMInferenceContext on TPU via Ray.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B", help="HF repo to load.")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--ray-address", default=None, help="Ray address (defaults to auto).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if not ray.is_initialized():
        ray.init(address=args.ray_address or "auto", ignore_reinit_error=True)

    logger.info("Launching async vLLM test with model %s", args.model_name)
    output = ray.get(
        init_async_vllm_remote.remote(
            args.model_name,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
        )
    )
    logger.info("Async vLLM response: %s", output)


if __name__ == "__main__":
    main()
