# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FP32 master export: the gate can only pass after a real optimizer step.

The weight metrics read FP32 master records out of distributed checkpoints. This
gate takes one optimizer step on a tiny Grug policy, saves the checkpoint WITH
optimizer state, and exports the masters through ``export_fp32_masters``. Because
Adam has moved the masters away from the model weights, an export that silently
upcast BF16 model weights would differ from the DCP masters somewhere and fail —
that is the property the old save-right-after-init gate could not test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import ray
import torch

import skyrl_train.models.grug_megatron_bridge  # noqa: F401  # registers the Grug bridge
from skyrl_train.distributed.megatron.master_export import export_fp32_masters, load_fp32_masters
from skyrl_train.utils import initialize_ray
from skyrl_train.checkpoint_schedule import OptimizerStateMode
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_tiny_checkpoint import write_tiny_checkpoint
from tests.gpu.test_megatron_worker import get_test_training_batch
from tests.gpu.utils import get_test_actor_config, init_worker_with_type


def _training_config(model_path: str, *, world_size: int) -> object:
    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = model_path
    cfg.trainer.critic.model.path = None
    cfg.trainer.strategy = "megatron"
    cfg.trainer.flash_attn = False
    cfg.trainer.bf16 = True
    cfg.trainer.gradient_checkpointing = False
    cfg.trainer.use_sample_packing = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = world_size
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 4
    cfg.generator.n_samples_per_prompt = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    return cfg


@pytest.mark.parametrize("world_size", [1], ids=["pp1"])
def test_fp32_master_export_differs_from_upcast_bf16_after_a_step(tmp_path, world_size: int):
    require_hoppers(world_size)
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_checkpoint(model_path)

    training_cfg = _training_config(str(model_path), world_size=world_size)
    initialize_ray(training_cfg)
    try:
        ckpt_dir = tmp_path / "ckpt" / "policy"
        batch = get_test_training_batch(4)
        batch.metadata["global_step"] = 0
        policy = init_worker_with_type("policy", cfg=training_cfg, num_gpus_per_node=world_size, num_nodes=1)
        try:
            ray.get(policy.async_run_ray_method("pass_through", "init_model", str(model_path)))
            # One real Adam step: the fp32 masters move off the bf16 weights.
            train_outputs = ray.get(policy.async_run_ray_method("pass_through", "ppo_train", batch))
            for output in train_outputs:
                assert output.metadata["train_status"]["policy_update_steps"] == 1
            bf16_dir = tmp_path / "hf-bf16"
            ray.get(policy.async_run_ray_method("pass_through", "save_hf_model", str(bf16_dir), None))
            ray.get(
                policy.async_run_ray_method(
                    "pass_through",
                    "save_checkpoint",
                    ckpt_dir=str(ckpt_dir),
                    tokenizer=None,
                    optimizer_state=OptimizerStateMode.MASTERS_ONLY,
                )
            )
            ray.get(policy.async_run_ray_method("pass_through", "wait_checkpoint_upload"))
        finally:
            policy.kill_actors()

        # The exporter verifies master records from the checkpoint metadata; a
        # bf16-only checkpoint would raise here, not export silently.
        export_dir = tmp_path / "fp32-export"
        export_fp32_masters(str(ckpt_dir), str(export_dir))
        shards = sorted(Path(export_dir).glob("*.safetensors"))
        assert shards, "no safetensors written by the exporter"
        exported: dict[str, torch.Tensor] = {}
        for shard in shards:
            exported.update(load_shard(shard))

        masters = load_fp32_masters(str(ckpt_dir))
        assert set(exported) == set(masters)
        for name, master in masters.items():
            assert torch.equal(exported[name], master), name

        # The step must show: at least one master differs from the upcast bf16
        # model weight. Equal-everywhere means the export read model weights, not
        # masters, and the gate failed.
        bf16_shards = sorted((bf16_dir / "model").glob("*.safetensors"))
        bf16_state: dict[str, torch.Tensor] = {}
        for shard in bf16_shards:
            bf16_state.update(load_shard(shard))
        shared = sorted(set(exported) & set(bf16_state))
        assert shared, "export and bf16 export share no tensor names"
        differing = [name for name in shared if not torch.equal(exported[name], bf16_state[name].to(torch.float32))]
        assert differing, (
            f"all {len(shared)} exported masters equal the upcast bf16 weights; "
            "the export read model weights, not optimizer masters"
        )
    finally:
        ray.shutdown()


def load_shard(shard: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file  # noqa: PLC0415 - optional dep guard

    return load_file(str(shard))
