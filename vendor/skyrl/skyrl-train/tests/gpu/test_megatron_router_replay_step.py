"""1-GPU end-to-end replay tests for Megatron MoE router replay (R3).

Runs the production policy worker at TP1/PP1/EP1/CP1 with
``moe_router_replay`` enabled after config validation: the old-logprob
forward and the PPO training step must consume the rollout's captured
expert choices (observed through log-prob sensitivity to the targets),
missing or malformed routes must fail before any optimizer step, and a
flag-off run with routes present in the batch must be byte-identical to a
run without them — the silent-ignore regression that motivated the
strategy guard. Packing on and off.

Requires 1 GPU; run on an otherwise idle node (not part of the CPU PR gate).
"""

from __future__ import annotations

import math

import pytest
import ray
import torch
from transformers import AutoTokenizer

from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import initialize_ray
from tests.gpu.grug_serving import rank0_validation_snapshot
from tests.gpu.grug_tiny_checkpoint import TOY_SHAPE, write_tiny_checkpoint
from tests.gpu.router_replay_fixtures import random_unique_routes
from tests.gpu.test_grug_megatron import (
    RESPONSE_LENGTH,
    _config,
    _padded_batch,
    _train_step,
)
from tests.gpu.utils import init_worker_with_type

TOPK = 4
NUM_EXPERTS = TOY_SHAPE["num_local_experts"]
NUM_LAYERS = TOY_SHAPE["num_hidden_layers"]
PARAMETER_NAMES = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.3.self_attn.q_proj.weight",
    "model.norm.weight",
    "model.layers.7.mlp.experts.down_proj.weight",
]


def _replay_config(tmp_path, *, packing: bool):
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_checkpoint(model_path, num_experts_per_tok=TOPK)
    cfg = _config(str(model_path), world_size=1, pp=1, ep=1)
    cfg.trainer.use_sample_packing = packing
    cfg.trainer.policy.megatron_config.moe_router_replay = True
    return cfg, model_path


def _routed_batch(pad_token_id: int, *, routes: str = "random", seed: int = 29) -> TrainingInputBatch:
    batch = _padded_batch(pad_token_id)
    generator = torch.Generator().manual_seed(seed)
    shape = (batch["sequences"].shape[0], RESPONSE_LENGTH, NUM_LAYERS, TOPK)

    if routes == "random":
        rollout_routed_experts = random_unique_routes(shape, NUM_EXPERTS, generator=generator)
        rollout_routed_experts[1] = 0  # one sample with fully-lost capture routes natively
    elif routes == "sentinel":
        rollout_routed_experts = torch.zeros(shape, dtype=torch.long)
    elif routes == "wrong_layers":
        rollout_routed_experts = random_unique_routes(
            (*shape[:2], NUM_LAYERS - 1, TOPK), NUM_EXPERTS, generator=generator
        )
    elif routes == "wrong_response":
        # Short local captures are valid; an overlong capture exceeds the global response window.
        shape = (shape[0], RESPONSE_LENGTH + 1, NUM_LAYERS, TOPK)
        rollout_routed_experts = random_unique_routes(shape, NUM_EXPERTS, generator=generator)
    else:
        raise ValueError(routes)
    batch["rollout_routed_experts"] = rollout_routed_experts.to(torch.int32)
    return batch


def _response_logprobs(policy, batch: TrainingInputBatch) -> torch.Tensor:
    outputs = ray.get(policy.async_run_ray_method("mesh", "forward", data=batch))
    return concatenate_outputs_after_mesh_dispatch(policy.actor_infos, outputs)["output"].float()


def _init_policy(cfg):
    return init_worker_with_type(
        "policy", shared_pg=None, colocate_all=False, num_gpus_per_node=1, num_nodes=1, cfg=cfg
    )


@pytest.mark.parametrize("packing", [False, True], ids=["unpacked", "packed"])
def test_old_logprob_forward_replays_captured_routes(tmp_path, packing: bool):
    """Log-probs must be insensitive to an empty capture and sensitive to real targets."""
    cfg, model_path = _replay_config(tmp_path, packing=packing)
    pad_token_id = AutoTokenizer.from_pretrained(model_path).pad_token_id
    initialize_ray(cfg)
    try:
        policy = _init_policy(cfg)
        # An all-sentinel capture replays nothing, so it must reproduce the
        # native log-probs exactly (and repeat exactly).
        native = _response_logprobs(policy, _routed_batch(pad_token_id, routes="sentinel"))
        repeated = _response_logprobs(policy, _routed_batch(pad_token_id, routes="sentinel"))
        replayed = _response_logprobs(policy, _routed_batch(pad_token_id, routes="random"))

        assert torch.equal(native, repeated)
        sample_diff = (replayed - native).abs().amax(dim=1)
        # Samples with a real capture route differently than natively; the
        # all-sentinel sample must be untouched by replay.
        assert (sample_diff[torch.tensor([0, 2, 3])] > 1e-4).all(), sample_diff
        assert sample_diff[1].item() == 0.0, sample_diff
    finally:
        ray.shutdown()


@pytest.mark.parametrize("packing", [False, True], ids=["unpacked", "packed"])
@pytest.mark.parametrize("recompute_modules", [None, ["layernorm"], ["moe"]], ids=["full", "norm", "moe"])
def test_training_step_replays_and_recomputes(tmp_path, packing: bool, recompute_modules):
    cfg, model_path = _replay_config(tmp_path, packing=packing)
    if recompute_modules is not None:
        transformer = cfg.trainer.policy.megatron_config.transformer_config_kwargs
        transformer.recompute_granularity = "selective"
        transformer.recompute_method = None
        transformer.recompute_num_layers = None
        transformer.recompute_modules = recompute_modules
    pad_token_id = AutoTokenizer.from_pretrained(model_path).pad_token_id
    batch = _routed_batch(pad_token_id, routes="random")
    initialize_ray(cfg)
    try:
        policy = _init_policy(cfg)
        status = _train_step(policy, batch)
        assert status["router_replay/hit_fraction"] == 1.0, status
        assert 0.0 <= status["router_replay/sentinel_fraction"] < 1.0, status
        assert math.isfinite(status["policy_loss"])
        assert math.isfinite(status["raw_grad_norm"])
    finally:
        ray.shutdown()


def test_missing_routes_fail_before_any_forward(tmp_path):
    from ray.exceptions import RayActorError

    cfg, model_path = _replay_config(tmp_path, packing=False)
    pad_token_id = AutoTokenizer.from_pretrained(model_path).pad_token_id
    batch = _padded_batch(pad_token_id)  # no rollout_routed_experts key at all
    initialize_ray(cfg)
    try:
        policy = _init_policy(cfg)
        with pytest.raises((ValueError, RayActorError), match="rollout_routed_experts"):
            ray.get(policy.async_run_ray_method("mesh", "ppo_train", batch))[0]
    finally:
        ray.shutdown()


@pytest.mark.parametrize(
    ("routes", "match"),
    [("wrong_layers", "expected_moe_layers"), ("wrong_response", "response_len")],
    ids=["layer_count", "response_len"],
)
def test_malformed_routes_fail_with_named_quantity(tmp_path, routes: str, match: str):
    from ray.exceptions import RayActorError

    cfg, model_path = _replay_config(tmp_path, packing=False)
    pad_token_id = AutoTokenizer.from_pretrained(model_path).pad_token_id
    batch = _routed_batch(pad_token_id, routes=routes)
    initialize_ray(cfg)
    try:
        policy = _init_policy(cfg)
        with pytest.raises((ValueError, RayActorError), match=match):
            ray.get(policy.async_run_ray_method("mesh", "ppo_train", batch))[0]
    finally:
        ray.shutdown()


def test_flag_off_run_with_routes_present_is_byte_identical(tmp_path):
    """Routes in the batch must not change a flag-off training step (#355 regression)."""
    cfg, model_path = _replay_config(tmp_path, packing=False)
    cfg.trainer.policy.megatron_config.moe_router_replay = False
    pad_token_id = AutoTokenizer.from_pretrained(model_path).pad_token_id
    plain = _padded_batch(pad_token_id)
    routed = _routed_batch(pad_token_id, routes="random")
    initialize_ray(cfg)
    try:
        first = _init_policy(cfg)
        status_plain = _train_step(first, plain)
        params_plain = rank0_validation_snapshot(first, PARAMETER_NAMES)

        second = _init_policy(cfg)
        status_routed = _train_step(second, routed)
        params_routed = rank0_validation_snapshot(second, PARAMETER_NAMES)

        for key in ("policy_loss", "policy_entropy", "raw_grad_norm"):
            assert status_plain[key] == status_routed[key], key
        for name in PARAMETER_NAMES:
            assert torch.equal(params_plain[name], params_routed[name]), name
    finally:
        ray.shutdown()
