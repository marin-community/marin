# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FP32 master export: the gate can only pass after a real optimizer step.

Weight analysis reads HF-named FP32 exports of the optimizer masters. This gate
takes one optimizer step on a tiny Grug policy, saves a masters-only checkpoint,
and exports it through ``save_fp32_master_hf_model`` next to an ordinary BF16 HF
export. After Adam moves the masters, the FP32 export must round to the BF16
export exactly yet differ from its upcast somewhere; an export that read the BF16
model weights instead of the masters fails the second check.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
import ray
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

import skyrl_train.models.grug_megatron_bridge  # noqa: F401  # registers the Grug bridge
from skyrl_train.distributed.megatron.master_export import load_fp32_masters
from skyrl_train.utils import initialize_ray
from skyrl_train.checkpoint_schedule import OptimizerStateMode
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_tiny_checkpoint import TOKENIZER, write_tiny_checkpoint
from tests.gpu.test_megatron_worker import get_test_training_batch
from tests.gpu.utils import get_test_actor_config, init_worker_with_type


S3_SCRATCH_ENV = "SKYRL_GPU_TEST_S3_SCRATCH"


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
def test_fp32_master_export_rounds_to_the_bf16_export_after_a_step(tmp_path, world_size: int):
    require_hoppers(world_size)
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_checkpoint(model_path)

    training_cfg = _training_config(str(model_path), world_size=world_size)
    initialize_ray(training_cfg)
    try:
        ckpt_dir = tmp_path / "ckpt" / "policy"
        bf16_dir = tmp_path / "hf-bf16"
        fp32_dir = tmp_path / "hf-fp32"
        batch = get_test_training_batch(4)
        batch.metadata["global_step"] = 0
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        policy = init_worker_with_type("policy", cfg=training_cfg, num_gpus_per_node=world_size, num_nodes=1)
        try:
            ray.get(policy.async_run_ray_method("pass_through", "init_model", str(model_path)))
            # One real Adam step: the fp32 masters move off the bf16 weights.
            train_outputs = ray.get(policy.async_run_ray_method("pass_through", "ppo_train", batch))
            for output in train_outputs:
                assert output.metadata["train_status"]["policy_update_steps"] == 1
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
            ray.get(policy.async_run_ray_method("pass_through", "save_hf_model", str(bf16_dir), tokenizer))
            ray.get(
                policy.async_run_ray_method(
                    "pass_through", "save_fp32_master_hf_model", str(ckpt_dir), str(fp32_dir), tokenizer
                )
            )
        finally:
            policy.kill_actors()

        bf16 = load_dir(bf16_dir)
        fp32 = load_dir(fp32_dir)
        assert set(fp32) == set(bf16) and bf16, "the FP32 export must carry the HF export's tensor names"
        assert {tensor.dtype for tensor in fp32.values()} == {torch.float32}
        for name, tensor in fp32.items():
            assert tensor.shape == bf16[name].shape, name
            assert torch.equal(tensor.to(bf16[name].dtype), bf16[name]), f"{name} does not round to the BF16 export"
        differing = [name for name, tensor in fp32.items() if not torch.equal(tensor, bf16[name].to(torch.float32))]
        assert differing, "every FP32 tensor equals the upcast BF16 export; the export read model weights, not masters"

        # Independent of Bridge: the embedding is a one-to-one mapping, so its HF tensor
        # must be the checkpoint's master record itself.
        masters = load_fp32_masters(str(ckpt_dir))
        embedding = fp32["model.embed_tokens.weight"]
        assert torch.equal(embedding, masters["embedding.word_embeddings.weight"][: embedding.shape[0]])
    finally:
        ray.shutdown()


def load_dir(directory: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(directory.glob("*.safetensors")):
        tensors.update(load_file(str(shard)))
    return tensors


@pytest.mark.parametrize("world_size", [2], ids=["dp2"])
def test_masters_save_without_a_gather_and_survive_a_full_checkpoint_round_trip(tmp_path, world_size: int):
    """Masters-only and dp_reshardable full saves carry the same FP32 masters, bit for bit.

    With two data-parallel ranks each rank holds half of every master. A masters-only
    save assembles them per parameter (no optimizer-state gather); a full save writes
    the optimizer in DP-local shards plus the same master records. Reloading the full
    checkpoint into a fresh policy and saving its masters again must reproduce them.
    """
    require_hoppers(world_size)
    # Megatron restores only from CoreWeave S3, so the round trip needs an S3 scratch prefix.
    s3_prefix = os.environ.get(S3_SCRATCH_ENV)
    if not s3_prefix:
        pytest.skip(f"set {S3_SCRATCH_ENV} to an s3:// scratch prefix for the checkpoint round trip")
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_checkpoint(model_path)
    training_cfg = _training_config(str(model_path), world_size=world_size)
    training_cfg.trainer.policy.megatron_config.optimizer_checkpoint_sharding_type = "dp_reshardable"
    run_root = f"{s3_prefix.rstrip('/')}/{uuid.uuid4().hex}"
    initialize_ray(training_cfg)
    try:
        masters_dir, full_dir, reloaded_dir = (f"{run_root}/{name}/policy" for name in ("masters", "full", "reloaded"))
        batch = get_test_training_batch(4)
        batch.metadata["global_step"] = 0

        def save(policy, ckpt_dir, mode):
            ray.get(
                policy.async_run_ray_method(
                    "pass_through", "save_checkpoint", ckpt_dir=str(ckpt_dir), tokenizer=None, optimizer_state=mode
                )
            )
            ray.get(policy.async_run_ray_method("pass_through", "wait_checkpoint_upload"))

        policy = init_worker_with_type("policy", cfg=training_cfg, num_gpus_per_node=world_size, num_nodes=1)
        try:
            ray.get(policy.async_run_ray_method("pass_through", "init_model", str(model_path)))
            ray.get(policy.async_run_ray_method("pass_through", "ppo_train", batch))
            save(policy, masters_dir, OptimizerStateMode.MASTERS_ONLY)
            save(policy, full_dir, OptimizerStateMode.FULL)
        finally:
            policy.kill_actors()

        fresh = init_worker_with_type("policy", cfg=training_cfg, num_gpus_per_node=world_size, num_nodes=1)
        try:
            ray.get(fresh.async_run_ray_method("pass_through", "init_model", str(model_path)))
            ray.get(fresh.async_run_ray_method("pass_through", "load_checkpoint", str(full_dir)))
            save(fresh, reloaded_dir, OptimizerStateMode.MASTERS_ONLY)
        finally:
            fresh.kill_actors()

        masters = load_fp32_masters(str(masters_dir))
        assert masters and {tensor.dtype for tensor in masters.values()} == {torch.float32}
        for other in (load_fp32_masters(str(full_dir)), load_fp32_masters(str(reloaded_dir))):
            assert set(other) == set(masters)
            for name, tensor in masters.items():
                assert torch.equal(other[name], tensor), name
    finally:
        ray.shutdown()
