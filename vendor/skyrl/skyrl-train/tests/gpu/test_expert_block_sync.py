import asyncio

import pytest
import ray
import torch
from omegaconf import OmegaConf
from skyrl_train.entrypoints import main_base
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.utils import initialize_ray
from skyrl_train.weight_sync.expert_block.driver import ExpertBlockSync
from transformers import AutoTokenizer

from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import (
    GRUG_BIAS_NAMES as BIAS_NAMES,
)
from tests.gpu.grug_serving import (
    GRUG_SERVING_EXPERT_INDEX_BY_NAME as SERVING_EXPERT_INDEX_BY_NAME,
)
from tests.gpu.grug_serving import (
    LM_HEAD_NAME,
    MAX_MODEL_LEN,
    ROUTER_NAME,
    STACKED_EXPERT_NAME,
    assert_engine_weights,
    rank0_validation_snapshot,
)
from tests.gpu.grug_tiny_checkpoint import write_tiny_checkpoint
from tests.gpu.test_grug_megatron import (
    ATTN_GATE_NAME,
    GATED_NORM_NAME,
    _config,
    _init_policy,
)
from tests.gpu.test_hero_megatron import write_tiny_hero_checkpoint

GRUG_SYNC_NAMES = [
    *BIAS_NAMES,
    *SERVING_EXPERT_INDEX_BY_NAME,
    LM_HEAD_NAME,
    ROUTER_NAME,
    ATTN_GATE_NAME,
    GATED_NORM_NAME,
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.2.shared_expert.gate_proj.weight",
    "model.layers.2.shared_expert.up_proj.weight",
]
HERO_BIAS_NAMES = [f"model.layers.{layer}.mlp.router.bias" for layer in range(4)]
HERO_EXPERTS = {
    "model.layers.0.mlp.experts.0.gate_proj.weight": 0,
    "model.layers.0.mlp.experts.15.down_proj.weight": 15,
}
HERO_SYNC_NAMES = [
    *HERO_BIAS_NAMES,
    *HERO_EXPERTS,
    LM_HEAD_NAME,
    ROUTER_NAME,
    "model.layers.0.mlp.latent_down_proj.weight",
    "model.layers.0.shared_experts.1.up_proj.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.sconv_mlp.weight",
    "model.layers.0.self_attn.sconv_k.weight",
    "model.layers.0.sconv_attn.weight",
]


def flip_one_installed_byte(worker) -> int:
    """Worker RPC for this test: flip byte 1 of the first expert slot of layer 0."""
    for name, parameter in worker.model_runner.model.named_parameters():
        if name.endswith("layers.0.mlp.experts.routed_experts.w13_weight"):
            with torch.no_grad():
                parameter.data.view(-1).view(torch.uint8)[1] ^= 0xFF
            return 1
    return 0


@pytest.mark.vllm
@pytest.mark.parametrize("hero", [False, True], ids=["grug-dummy", "hero-dummy-shortconv"])
def test_dummy_engines_hold_requests_until_expert_block_sync_installs_verified_weights(tmp_path, hero, monkeypatch):
    engine_dp = 1 if hero else 2
    require_hoppers(2 + engine_dp)
    from skyrl_train.inference_engines.vllm import vllm_engine

    class CorruptibleEngine(vllm_engine.AsyncVLLMInferenceEngine):
        async def flip_installed_byte(self):
            return await self.llm.collective_rpc(flip_one_installed_byte)

    monkeypatch.setattr(vllm_engine, "AsyncVLLMRayActor", ray.remote(CorruptibleEngine))

    model_path = tmp_path / "model"
    model_path.mkdir()
    (write_tiny_hero_checkpoint if hero else write_tiny_checkpoint)(model_path)
    cfg = _config(str(model_path), world_size=2, pp=1, ep=2)
    cfg.generator.inference_engine_data_parallel_size = engine_dp
    cfg.generator.inference_engine_expert_parallel_size = engine_dp
    cfg.generator.weight_sync_transport = "expert_block"
    engine_init_kwargs = {"load_format": "dummy", "max_model_len": MAX_MODEL_LEN}
    if hero:
        engine_init_kwargs["enable_flashinfer_autotune"] = False
    OmegaConf.update(cfg, "generator.engine_init_kwargs", engine_init_kwargs, force_add=True)
    sync_names = HERO_SYNC_NAMES if hero else GRUG_SYNC_NAMES
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    initialize_ray(cfg)
    try:
        experiment = main_base.BasePPOExp.__new__(main_base.BasePPOExp)
        experiment.cfg, experiment.tokenizer, experiment.colocate_pg = cfg, tokenizer, None
        client = experiment.create_inference_engine_client()
        policy = _init_policy(cfg, 2)
        training_names = (
            sync_names
            if hero
            else [*(name for name in sync_names if name not in SERVING_EXPERT_INDEX_BY_NAME), STACKED_EXPERT_NAME]
        )
        expected = rank0_validation_snapshot(policy, training_names)
        sync = ExpertBlockSync(policy_model=policy, inference_engine_client=client, timeout_seconds=180)

        async def sync_while_a_request_waits():
            await sync.prepare()
            # A second client skips the paused client's barrier, so the request reaches the engine schedulers.
            request = {
                "prompts": [[{"role": "user", "content": "What is two plus two?"}]],
                "sampling_params": {"temperature": 0.0, "max_tokens": 1, "logprobs": 0},
            }
            held = asyncio.create_task(InferenceEngineClient(client.engines, tokenizer, cfg).generate(request))
            try:
                done, _pending = await asyncio.wait({held}, timeout=3)
                assert not done, "A request completed before the initial dummy weights were verified"
                await sync.sync(1)
                await sync.verify(1)
                await client.resume_generation(policy_step=1)
                assert len((await asyncio.wait_for(held, timeout=30))["response_ids"]) == 1
            finally:
                held.cancel()
                await asyncio.gather(held, return_exceptions=True)

        asyncio.run(sync_while_a_request_waits())
        bias_names, expert_names = (
            (HERO_BIAS_NAMES, HERO_EXPERTS) if hero else (BIAS_NAMES, SERVING_EXPERT_INDEX_BY_NAME)
        )
        assert_engine_weights(client, sync_names, expected, bias_names, expert_names)

        async def corrupt_and_verify():
            flipped = await client.engines[-1].inference_engine_actor.flip_installed_byte.remote()
            assert sum(flipped) == 1
            await client.pause_generation()
            try:
                with pytest.raises(RuntimeError, match=r"1 of \d+ replayed bytes differ"):
                    await sync.verify(1)
            finally:
                await client.resume_generation(policy_step=1)

        asyncio.run(corrupt_and_verify())
        asyncio.run(sync.close())
    finally:
        ray.shutdown()
