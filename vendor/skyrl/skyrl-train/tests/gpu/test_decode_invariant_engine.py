import asyncio

import pytest
import ray
from transformers import AutoTokenizer

from skyrl_train.config.utils import get_default_config
from skyrl_train.entrypoints.main_base import create_ray_wrapped_inference_engines_from_config
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.utils import initialize_ray
from tests.gpu.grug_exact_checkpoint import write_snowball_checkpoint
from tests.gpu.grug_gpu_gates import require_hoppers

MAX_MODEL_LEN = 4096
# Every id of the checkpoint's 151,665-token tokenizer, rounded up to a multiple of 128. With Snowball's 128,256
# entries, a stock engine returns NaN log-probabilities in steps that decode several greedy requests.
VOCAB_SIZE = 151_680
# Steps of at most 16 tokens prefill prompts in chunks, beside other requests' decode steps; steps of 128 tokens
# compute prefill rows with other expert-kernel tiles and GEMM row counts than decode steps.
SMALL_STEPS, LARGE_STEPS = 16, 128
RESPONSE_TOKENS = 24
LONG_PROMPT = [1 + (index * 7919) % 20000 for index in range(2600)]
# The prompt that runs beside the long one shares its first 256 tokens, which the prefix cache holds; two short ones
# arrive later.
SHARED_PREFIX_PROMPT = LONG_PROMPT[:256] + [7, 11, 13] * 30
LATER_PROMPTS = [list(range(1, 20)), [1, 23]]


def _client(cfg) -> InferenceEngineClient:
    tokenizer = AutoTokenizer.from_pretrained(cfg.trainer.policy.model.path)
    engines = create_ray_wrapped_inference_engines_from_config(cfg, colocate_pg=None, tokenizer=tokenizer)
    return InferenceEngineClient(engines, tokenizer, cfg)


async def _generate(client: InferenceEngineClient, cfg, prompts: list[list[int]], **settings) -> dict:
    sampling_params = get_sampling_params_for_backend("vllm", cfg.generator.sampling_params)
    sampling_params.update(
        {"temperature": 0.0, "ignore_eos": True, "logprobs": 1, "max_tokens": RESPONSE_TOKENS, **settings}
    )
    return await client.generate(InferenceEngineInput(prompt_token_ids=prompts, sampling_params=sampling_params))


async def _long_prompt_beside_others(client: InferenceEngineClient, cfg) -> dict:
    """The long prompt run beside the shared-prefix prompt, with the later prompts sent once a two-token request
    sent at the same time finishes."""

    async def later() -> None:
        await _generate(client, cfg, [[1, 5]], max_tokens=2)
        await _generate(client, cfg, LATER_PROMPTS)

    beside, _ = await asyncio.gather(_generate(client, cfg, [LONG_PROMPT, SHARED_PREFIX_PROMPT]), later())
    return beside


@pytest.mark.vllm
@pytest.mark.parametrize(
    ("decode_invariant", "max_num_batched_tokens"),
    [(False, SMALL_STEPS), (True, SMALL_STEPS), (True, LARGE_STEPS)],
    ids=["stock_small_steps", "exact_small_steps", "exact_large_steps"],
)
def test_decode_invariant_engine_gives_a_response_the_same_log_probabilities_in_any_step(
    tmp_path, decode_invariant, max_num_batched_tokens
):
    require_hoppers(1)
    model_path = tmp_path / "model"
    model_path.mkdir()
    # Snowball's hidden width, attention geometry and 2,048-token window with four layers and eight narrow experts.
    write_snowball_checkpoint(model_path, num_hidden_layers=4, num_local_experts=8, vocab_size=VOCAB_SIZE)
    cfg = get_default_config()
    cfg.trainer.seed = 23
    cfg.trainer.policy.model.path = str(model_path)
    cfg.trainer.numerics.mode = "exact" if decode_invariant else "native"
    cfg.trainer.placement.colocate_all = False
    cfg.generator.backend = "vllm"
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.engine_init_kwargs = {"max_model_len": MAX_MODEL_LEN, "moe_backend": "triton"}
    cfg.generator.model_dtype = "bfloat16"
    cfg.generator.enforce_eager = False
    cfg.generator.vllm_attention_backend = "FLASH_ATTN"
    cfg.generator.vllm_v1_disable_multiproc = True
    cfg.generator.enable_prefix_caching = True
    cfg.generator.max_num_batched_tokens = max_num_batched_tokens
    cfg.generator.max_num_seqs = 4
    cfg.generator.gpu_memory_utilization = 0.3
    initialize_ray(cfg)
    try:
        client = _client(cfg)

        alone = asyncio.run(_generate(client, cfg, [LONG_PROMPT]))
        response, logprobs = alone["response_ids"][0], alone["response_logprobs"][0]
        beside = asyncio.run(_long_prompt_beside_others(client, cfg))
        # The prompt and its response read back as one prompt past the window, in prefill chunks.
        reread = asyncio.run(_generate(client, cfg, [LONG_PROMPT + response], max_tokens=1, prompt_logprobs=0))
        reread_logprobs = [
            reread["prompt_logprobs"][0][len(LONG_PROMPT) + index][token] for index, token in enumerate(response)
        ]
        differing = {
            "beside": sum(left != right for left, right in zip(beside["response_logprobs"][0], logprobs)),
            "reread": sum(left != right for left, right in zip(reread_logprobs, logprobs)),
        }
        print(
            f"decode_invariant={decode_invariant} max_num_batched_tokens={max_num_batched_tokens}: "
            f"log-probabilities differing from the prompt alone {differing}"
        )
        if decode_invariant:
            assert beside["response_ids"][0] == response
            assert differing == {"beside": 0, "reread": 0}
        else:
            assert any(differing.values())
    finally:
        ray.shutdown()
