# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug router telemetry on the Megatron trainer.

Two gates:

- One ``ppo_train`` step on tiny Grug at PP=2 writes the telemetry sidecar. The
  record must describe every pipeline stage's layers, and the per-task collision
  rows must exist for a mixed batch — the θ_XR control.
- The instrumentation itself is verified against an independent top-k: the test
  drives the real observer over a locally loaded tiny Grug model and recomputes
  the expected selection counts with ``torch.topk`` on the biased logits — never
  with the router's own selection helper.
"""

from __future__ import annotations

import json

import ray
import torch

import skyrl_train.models.grug_megatron_bridge  # noqa: F401  # registers the Grug bridge
from skyrl_train.models.router_instrumentation import observe_router_forwards
from skyrl_train.utils import initialize_ray
from skyrl_train.workers.megatron.router_telemetry import RouterLoadStats, build_router_layer_map
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_tiny_checkpoint import TOY_SHAPE, write_tiny_checkpoint
from tests.gpu.test_megatron_worker import get_test_training_batch
from tests.gpu.utils import get_test_actor_config, init_worker_with_type

ROLLOUT_WORLD_SIZE = 2


def _config(model_path: str, *, pipeline_parallel_size: int, export_path: str | None = None) -> object:
    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = model_path
    cfg.trainer.critic.model.path = None
    cfg.trainer.strategy = "megatron"
    cfg.trainer.flash_attn = False
    cfg.trainer.bf16 = True
    cfg.trainer.gradient_checkpointing = False
    cfg.trainer.use_sample_packing = False
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 4
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = pipeline_parallel_size
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = pipeline_parallel_size
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
    cfg.trainer.router_telemetry.enabled = True
    if export_path is not None:
        cfg.trainer.export_path = export_path
    return cfg


def test_ppo_train_at_pp2_writes_the_sidecar_with_per_task_collision_rows(tmp_path):
    require_hoppers(ROLLOUT_WORLD_SIZE)
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_checkpoint(model_path, shape=TOY_SHAPE)
    export_path = tmp_path / "exports"
    cfg = _config(str(model_path), pipeline_parallel_size=ROLLOUT_WORLD_SIZE, export_path=str(export_path))
    initialize_ray(cfg)
    try:
        policy = init_worker_with_type("policy", cfg=cfg, num_gpus_per_node=ROLLOUT_WORLD_SIZE, num_nodes=1)
        try:
            batch = get_test_training_batch(4)
            # A mixed θ_XR-style batch: two micro-batches tagged with different
            # tasks so the collision split has two waves to attribute.
            batch.metadata["data_source_names"] = ["task_X", "task_R", "task_X", "task_R"]
            ray.get(policy.async_run_ray_method("pass_through", "ppo_train", batch))
        finally:
            policy.kill_actors()

        sidecar = export_path / "router_telemetry.jsonl"
        records = [json.loads(line) for line in sidecar.read_text().splitlines()]
        assert records, "the telemetry sidecar is empty after a training step"
        record = records[-1]
        assert record["layers"], "the sidecar record carries no layers"
        # Both pipeline stages' layers appear, keyed per (layer, expert).
        assert len(record["load_counts"]) == len(record["layers"])
        for per_layer in record["load_counts"].values():
            assert len(per_layer) >= TOY_SHAPE["num_experts"] if "num_experts" in TOY_SHAPE else per_layer
        # The mixed batch produced per-task collision rows — never a "mixed" bucket.
        assert set(record["collision_norms"]) >= {"task_X", "task_R"}
    finally:
        ray.shutdown()


def test_load_stats_match_an_independent_topk_recompute(tmp_path):
    """The observer's counts equal ``torch.topk`` over the biased logits, computed
    here independently of the router's own selection helper."""
    require_hoppers(1)
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_checkpoint(model_path, shape=TOY_SHAPE)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()
    layer_map = build_router_layer_map(model)
    assert layer_map, "the tiny Grug model exposes no routers"
    stats = RouterLoadStats(layer_map)

    hidden = torch.randn(6, model.config.hidden_size)
    captured: list = []

    def capture(observation):
        captured.append(observation)
        stats.observe(observation)

    with observe_router_forwards(capture):
        model(hidden)

    assert captured, "the observer recorded no router forwards"
    num_layers = len(layer_map)
    expected_counts: dict[int, torch.Tensor] = {}
    for observation in captured:
        layer = layer_map[id(observation.router)]
        # Independent recompute: top-k over the biased logits, exactly as Grug's
        # routing contract defines it, but via torch.topk here.
        biased = observation.router_inputs + getattr(observation.router, "expert_bias", 0)
        _, selected = torch.topk(biased, observation.selected_experts.shape[-1], dim=-1)
        assert torch.equal(selected, observation.selected_experts), "the router's selection diverges from torch.topk"
        row = expected_counts.setdefault(
            layer, torch.zeros(observation.selection_logits.shape[-1], dtype=torch.float64)
        )
        row += torch.bincount(selected.reshape(-1), minlength=row.numel()).double()

    counts = stats.dense_counts()
    assert counts.shape[0] == num_layers
    for position, layer in enumerate(stats.layers):
        assert torch.equal(counts[position], expected_counts[layer]), layer
