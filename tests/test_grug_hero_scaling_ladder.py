# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import pytest
from marin.execution.lazy import StepContext

from experiments.grug.moe_hero_ep.launch_diagnostics import build_diagnostic_run
from experiments.grug.moe_hero_ep.launch_scaling_ladder import build_ladder_run
from experiments.grug.moe_hero_ep.optimizer import _gate_router_decay_mask, _scale_by_adam_gate_router_decay

# A miniature stacked-hero parameter tree: the two decay targets plus leaves that must be left alone.
_MOCK_PARAMS = {
    "token_embed": jnp.ones((5, 4)),
    "stacked_blocks": {
        "stacked": {
            "attn": {"attn_gate": jnp.ones((2, 4, 3))},
            "mlp": {"router": jnp.ones((2, 4, 6)), "router_bias": jnp.ones((2, 6))},
            "rms_attn": {"weight": jnp.ones((2, 4))},
        }
    },
}


def test_gate_router_decay_state_matches_plain_adam():
    # The decay must not change the optimizer-state tree, so a checkpoint written without it restores
    # unchanged (moments + step count preserved) when the decay is switched on for a continuation.
    plain = optax.scale_by_adam(0.9, 0.95, 1e-8).init(_MOCK_PARAMS)
    decayed = _scale_by_adam_gate_router_decay(0.9, 0.95, 1e-8, 0.05, 390_251).init(_MOCK_PARAMS)
    assert type(decayed) is type(plain)
    assert jtu.tree_structure(decayed) == jtu.tree_structure(plain)


@pytest.mark.parametrize(("count", "expected_wd"), [(0, 0.05), (54_000, 0.05 * (1 - 54_000 / 390_251)), (390_251, 0.0)])
def test_gate_router_decay_anneals_from_the_step_count(count, expected_wd):
    total_steps = 390_251
    adam = optax.scale_by_adam(0.9, 0.95, 1e-8)
    decay = _scale_by_adam_gate_router_decay(0.9, 0.95, 1e-8, 0.05, total_steps)
    grads = jax.tree.map(lambda x: jnp.full_like(x, 0.01), _MOCK_PARAMS)

    def state_at_count(transform):
        return transform.init(_MOCK_PARAMS)._replace(count=jnp.asarray(count, jnp.int32))

    decayed_updates, _ = decay.update(grads, state_at_count(decay), _MOCK_PARAMS)
    adam_updates, _ = adam.update(grads, state_at_count(adam), _MOCK_PARAMS)
    # Decoupled decay adds wd * param (param == 1) on top of the identical Adam step for the router.
    applied_wd = float(
        (
            decayed_updates["stacked_blocks"]["stacked"]["mlp"]["router"]
            - adam_updates["stacked_blocks"]["stacked"]["mlp"]["router"]
        ).mean()
    )
    assert applied_wd == pytest.approx(expected_wd, abs=1e-6)


def test_gate_router_decay_mask_selects_only_gate_and_router():
    mask = _gate_router_decay_mask(_MOCK_PARAMS)
    stacked = mask["stacked_blocks"]["stacked"]
    assert bool(stacked["attn"]["attn_gate"]) is True
    assert bool(stacked["mlp"]["router"]) is True
    assert bool(stacked["mlp"]["router_bias"]) is False
    assert bool(stacked["rms_attn"]["weight"]) is False
    assert bool(mask["token_embed"]) is False


def test_ladder_defaults_gate_router_weight_decay_on_with_opt_out():
    # Weight decay is on by default for the hero recipe, so a resume can never silently continue
    # without it; passing 0 is the explicit opt-out.
    default_step = build_ladder_run(run_id="test-wd-default", size="d6144", num_steps=1, version="2026.08.18")
    opt_out_step = build_ladder_run(
        run_id="test-wd-off", size="d6144", num_steps=1, gate_router_weight_decay=0.0, version="2026.08.18"
    )

    def optimizer_of(step):
        ctx = StepContext.for_fingerprint(runtime_arg_keys=step.runtime_args, deps=step.deps)
        return step.build_config(ctx).optimizer

    assert optimizer_of(default_step).gate_router_weight_decay == 0.02
    assert optimizer_of(opt_out_step).gate_router_weight_decay == 0.0


def test_diagnostic_run_matches_the_d6144_rack_local_recipe():
    diagnostic = build_diagnostic_run(
        run_id="test-diagnostic",
        dp_racks=1,
        num_steps=1,
        schedule_steps=390_251,
        version="dev",
    )
    ladder = build_ladder_run(run_id="test-ladder", size="d6144", version="dev")
    diagnostic_config = diagnostic.build_config(
        StepContext.for_fingerprint(runtime_arg_keys=diagnostic.runtime_args, deps=diagnostic.deps)
    )
    ladder_config = ladder.build_config(
        StepContext.for_fingerprint(runtime_arg_keys=ladder.runtime_args, deps=ladder.deps)
    )

    assert diagnostic_config.model == ladder_config.model
    assert diagnostic_config.processes_per_task == ladder_config.processes_per_task
    assert diagnostic_config.tensorstore_cache_bytes == ladder_config.tensorstore_cache_bytes
    assert diagnostic_config.trainer == dataclasses.replace(
        ladder_config.trainer,
        trainer=diagnostic_config.trainer.trainer,
        replica_axis_size=1,
        save_checkpoints=False,
    )
    assert diagnostic_config.trainer.trainer == dataclasses.replace(
        ladder_config.trainer.trainer,
        id=diagnostic_config.trainer.trainer.id,
        train_batch_size=diagnostic_config.trainer.trainer.train_batch_size,
        profiler=diagnostic_config.trainer.trainer.profiler,
        tracker=diagnostic_config.trainer.trainer.tracker,
        progress_watchdog=diagnostic_config.trainer.trainer.progress_watchdog,
        checkpointer=diagnostic_config.trainer.trainer.checkpointer,
        load_checkpoint_path=diagnostic_config.trainer.trainer.load_checkpoint_path,
    )
    assert diagnostic_config.data.target_budget is ladder_config.data.target_budget is None
    assert diagnostic_config.data.experiment_budget is ladder_config.data.experiment_budget is None
    assert diagnostic_config.data.train_weights == [
        (step, {name: weight for name, weight in weights.items() if weight > 0})
        for step, weights in ladder_config.data.train_weights
    ]


@pytest.mark.parametrize(
    ("size", "num_steps", "expected_simulated_epoching"),
    [("d2048", None, True), ("d6144", 1, True), ("d6144", None, False)],
)
def test_scaling_ladder_disables_simulated_epoching_above_flop_limit(size, num_steps, expected_simulated_epoching):
    step = build_ladder_run(run_id=f"test-{size}", size=size, num_steps=num_steps, version="2026.08.18")
    ctx = StepContext.for_fingerprint(runtime_arg_keys=step.runtime_args, deps=step.deps)

    data = step.build_config(ctx).data

    assert (data.target_budget is not None) is expected_simulated_epoching
    assert (data.experiment_budget is not None) is expected_simulated_epoching


def test_scaling_ladder_searches_cluster_and_data_local_temp_roots(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin")
    monkeypatch.setenv("MARIN_TEMP_PREFIX", "s3://hero-checkpoints")
    step = build_ladder_run(run_id="test-d6144", size="d6144", num_steps=1, version="2026.08.18")
    output_path = "s3://marin-us-east-02a/marin/grug/test-d6144/v"
    ctx = dataclasses.replace(
        StepContext.for_fingerprint(runtime_arg_keys=step.runtime_args, deps=step.deps),
        output_path=output_path,
    )

    trainer = step.build_config(ctx).trainer.trainer
    assert trainer.checkpoint_search_paths("test-d6144") == [
        f"{output_path}/checkpoints",
        "s3://hero-checkpoints/tmp/ttl=14d/checkpoints-temp/marin-us-east-02a/marin/grug/test-d6144/v/checkpoints",
        "s3://marin-us-east-02a/tmp/ttl=14d/checkpoints-temp/marin-us-east-02a/marin/grug/test-d6144/v/checkpoints",
    ]
