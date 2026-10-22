"""FP32 master export: the rounded masters are the BF16 HF export, byte for byte.

The weight metrics read FP32 master records out of distributed checkpoints. This gate
exports one tiny Grug checkpoint twice through the standard checkpoint-export path —
once with a BF16 model and once with an FP32 model that receives the masters uncast —
and requires the FP32 safetensors to round to the BF16 ones exactly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import ray
import torch
from safetensors.torch import load_file

import skyrl_train.models.grug_megatron_bridge  # noqa: F401  # registers the Grug bridge
from skyrl_train.utils import initialize_ray
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_tiny_checkpoint import write_tiny_checkpoint
from tests.gpu.utils import get_test_actor_config, init_worker_with_type


def _config(model_path: str, *, bf16: bool) -> object:
    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = model_path
    cfg.trainer.critic.model.path = None
    cfg.trainer.strategy = "megatron"
    cfg.trainer.flash_attn = False
    cfg.trainer.bf16 = bf16
    cfg.trainer.gradient_checkpointing = False
    cfg.trainer.use_sample_packing = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
    return cfg


def _export(tmp_path: Path, model_path: Path, ckpt_dir: Path, *, bf16: bool) -> dict[str, torch.Tensor]:
    cfg = _config(str(model_path), bf16=bf16)
    policy = init_worker_with_type("policy", cfg=cfg, num_gpus_per_node=1, num_nodes=1)
    try:
        ray.get(policy.async_run_ray_method("pass_through", "init_model_for_export", str(model_path)))
        ray.get(
            policy.async_run_ray_method(
                "pass_through", "load_checkpoint", ckpt_dir=str(ckpt_dir), load_training_state=False
            )
        )
        export_dir = tmp_path / ("bf16" if bf16 else "fp32")
        ray.get(policy.async_run_ray_method("pass_through", "save_hf_model", str(export_dir), None))
    finally:
        policy.kill_actors()
    shards = sorted((export_dir / "model").glob("*.safetensors"))
    assert shards, f"no safetensors written under {export_dir}"
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(shard))
    return state


@pytest.mark.parametrize("world_size", [1], ids=["pp1"])
def test_fp32_master_export_rounds_to_the_bf16_hf_export(tmp_path, world_size: int):
    require_hoppers(world_size)
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_checkpoint(model_path)

    training_cfg = _config(str(model_path), bf16=True)
    initialize_ray(training_cfg)
    try:
        ckpt_dir = tmp_path / "ckpt" / "policy"
        policy = init_worker_with_type("policy", cfg=training_cfg, num_gpus_per_node=world_size, num_nodes=1)
        try:
            ray.get(policy.async_run_ray_method("pass_through", "init_model", str(model_path)))
            ray.get(
                policy.async_run_ray_method("pass_through", "save_checkpoint", ckpt_dir=str(ckpt_dir), tokenizer=None)
            )
            ray.get(policy.async_run_ray_method("pass_through", "wait_checkpoint_upload"))
        finally:
            policy.kill_actors()

        bf16_state = _export(tmp_path, model_path, ckpt_dir, bf16=True)
        fp32_state = _export(tmp_path, model_path, ckpt_dir, bf16=False)

        assert set(bf16_state) == set(fp32_state)
        for name, bf16_weight in bf16_state.items():
            assert fp32_state[name].dtype == torch.float32, name
            assert torch.equal(bf16_weight, fp32_state[name].to(torch.bfloat16)), name
    finally:
        ray.shutdown()
