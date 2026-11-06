# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import pytest
from levanter.checkpoint import Checkpointer, CheckpointInterval
from levanter.data.text.datasets import BlockShuffleConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import StepContext

from experiments.grug.moe_hero_ep import launch_scaling_ladder
from experiments.grug.moe_hero_ep.harrier_mix_2026_08_18 import HarrierContextPhase
from experiments.grug.moe_hero_ep.heuristic import MoeHeuristic
from experiments.grug.moe_hero_ep.launch_diagnostics import build_diagnostic_run
from experiments.grug.moe_hero_ep.launch_scaling_ladder import build_ladder_run, hero_permanent_keeps
from experiments.grug.moe_hero_ep.optimizer import (
    ScheduleExtension,
    _gate_router_decay_mask,
    _scale_by_adam_gate_router_decay,
)
from experiments.grug.moe_hero_ep.train import FlopsBaseline

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


@pytest.mark.parametrize(
    ("count", "expected_wd"), [(0, 0.05), (54_000, 0.05 * (1 - 54_000 / 390_251)), (390_251, 0.0), (391_000, 0.0)]
)
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
    default_step = build_ladder_run(run_id="test-wd-default", size="d6144", version="2026.08.18")
    opt_out_step = build_ladder_run(
        run_id="test-wd-off", size="d6144", gate_router_weight_decay=0.0, version="2026.08.18"
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
        long_context_start_step=390_251,
        restore_from="s3://bucket/checkpoints/step-390251",
        version="dev",
        gc_interval=100,
    )
    ladder = build_ladder_run(run_id="test-ladder", size="d6144", version="dev")
    diagnostic_config = diagnostic.build_config(
        StepContext.for_fingerprint(runtime_arg_keys=diagnostic.runtime_args, deps=diagnostic.deps)
    )
    ladder_config = ladder.build_config(
        StepContext.for_fingerprint(runtime_arg_keys=ladder.runtime_args, deps=ladder.deps)
    )

    # The diagnostic is one rack of the hero: the same model, and the hero's per-rack batch at its context.
    assert diagnostic_config.model == ladder_config.model
    assert (
        diagnostic_config.trainer.trainer.train_batch_size
        == ladder_config.trainer.trainer.train_batch_size // ladder_config.trainer.replica_axis_size
    )
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
        load_checkpoint=diagnostic_config.trainer.trainer.load_checkpoint,
    )
    assert diagnostic_config.data.target_budget is ladder_config.data.target_budget is None
    assert diagnostic_config.data.experiment_budget is ladder_config.data.experiment_budget is None
    assert [
        (step, {name: weight for name, weight in weights.items() if weight > 0})
        for step, weights in diagnostic_config.data.train_weights
    ] == [
        (step, {name: weight for name, weight in weights.items() if weight > 0})
        for step, weights in ladder_config.data.train_weights
    ]


@pytest.mark.parametrize(("size", "expected_simulated_epoching"), [("d2048", True), ("d6144", False)])
def test_scaling_ladder_disables_simulated_epoching_above_flop_limit(size, expected_simulated_epoching):
    step = build_ladder_run(run_id=f"test-{size}", size=size, version="2026.08.18")
    ctx = StepContext.for_fingerprint(runtime_arg_keys=step.runtime_args, deps=step.deps)

    data = step.build_config(ctx).data

    assert (data.target_budget is not None) is expected_simulated_epoching
    assert (data.experiment_budget is not None) is expected_simulated_epoching


def test_d6144_continues_the_checked_in_context_history_and_narrow_rungs_start_fresh():
    hero_step = build_ladder_run(run_id="test-d6144-history", size="d6144", version="dev")
    narrow_step = build_ladder_run(run_id="test-d2048-history", size="d2048", version="dev")
    hero = hero_step.build_config(StepContext.for_fingerprint(hero_step.runtime_args, hero_step.deps))
    narrow = narrow_step.build_config(StepContext.for_fingerprint(narrow_step.runtime_args, narrow_step.deps))

    phases = launch_scaling_ladder.HERO_CONTEXT_PHASES
    assert hero.flops_baseline == FlopsBaseline(phases[-1].end_step, launch_scaling_ladder.HERO_FLOPS_BASELINE_TOTAL)
    # The long-context mix starts where the history ends instead of replaying it.
    assert hero.data.prior_context_phases == []
    assert narrow.flops_baseline is None
    assert narrow.data.prior_context_phases == []


def test_d6144_rejects_a_schedule_that_would_rebuild_its_history_differently():
    # The prior phases' data stages derive from the schedule length; another length would move their data offsets.
    with pytest.raises(ValueError, match="keeps its schedule"):
        build_ladder_run(run_id="test-d6144-steps", size="d6144", num_steps=300_000, version="dev")


@pytest.mark.parametrize(("size", "bucket"), [("d6144", "hero-checkpoints"), ("d768", "marin-us-east-08a")])
def test_scaling_ladder_keeps_hero_checkpoints_separate_from_shared_scratch(monkeypatch, size, bucket):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin")
    monkeypatch.setenv("MARIN_TEMP_PREFIX", "s3://marin-us-east-08a")
    run_id = f"test-{size}"
    step = build_ladder_run(run_id=run_id, size=size, version="2026.08.18")
    output_path = f"s3://marin-us-east-02a/marin/grug/{run_id}/v"
    ctx = dataclasses.replace(
        StepContext.for_fingerprint(runtime_arg_keys=step.runtime_args, deps=step.deps),
        output_path=output_path,
    )

    trainer = step.build_config(ctx).trainer.trainer
    temporary_path = f"s3://{bucket}/tmp/ttl=3d/checkpoints-temp/marin-us-east-02a/marin/grug/{run_id}/v/checkpoints"
    search_paths = [
        f"{output_path}/checkpoints",
        temporary_path,
    ]
    if size != "d6144":
        search_paths.extend(
            f"s3://{previous}/tmp/ttl=3d/checkpoints-temp/marin-us-east-02a/marin/grug/{run_id}/v/checkpoints"
            for previous in ("marin-us-east-02a", "hero-checkpoints")
        )
    assert trainer.checkpoint_search_paths(run_id) == search_paths
    assert trainer.checkpointer.temporary_base_path == temporary_path


def _permanent_steps(keep: list[dict[str, int | None]], steps: range) -> list[int]:
    """The steps a Levanter checkpointer built from ``keep`` saves permanently."""
    policies = SimpleNamespace(step_policies=[CheckpointInterval(**k) for k in keep])
    return [step for step in steps if step % Checkpointer._get_current_step_save_interval(policies, step) == 0]


def test_d6144_permanent_checkpoints_pin_55000_and_densify_over_the_final_5000_steps():
    step = build_ladder_run(run_id="test-d6144-ckpt", size="d6144", version="2026.08.18")
    ctx = StepContext.for_fingerprint(runtime_arg_keys=step.runtime_args, deps=step.deps)
    config = step.build_config(ctx)
    keep = config.trainer.trainer.checkpointer.keep
    num_steps = config.trainer.trainer.num_train_steps

    # Building the step already runs CheckpointerConfig's monotonic-interval validation.
    assert num_steps == 390_251
    assert _permanent_steps(keep, range(48_000, 62_001)) == [48_000, 54_000, 55_000, 60_000]
    assert _permanent_steps(keep, range(378_000, num_steps + 1)) == [378_000, 384_000, *range(385_500, 390_001, 500)]


@pytest.mark.parametrize("num_steps", [3_000, 5_000, 30_000, 59_000, 60_000, 61_000])
def test_short_d6144_schedules_keep_valid_increasing_ranges_and_the_dense_tail(monkeypatch, num_steps):
    keep = hero_permanent_keeps(num_steps)
    untils = [k["until"] for k in keep[:-1]]
    assert untils == sorted(set(untils))
    assert keep[-1] == {"until": None, "every": 500}
    tail = range(max(1, num_steps - 5000 + 1), num_steps + 1)
    assert _permanent_steps(keep, tail) == [s for s in tail if s % 500 == 0]
    _set_hero_seq_len(monkeypatch, 4096)  # a fresh run starts at 4K
    step = build_ladder_run(
        run_id="test-short-d6144", size="d6144", num_steps=num_steps, context_phases=(), version="2026.08.18"
    )
    # Building the config runs CheckpointerConfig's strictly increasing `until` validation.
    config = step.build_config(StepContext.for_fingerprint(runtime_arg_keys=step.runtime_args, deps=step.deps))
    assert config.trainer.trainer.checkpointer.keep == keep


def _set_hero_seq_len(monkeypatch, seq_len):
    # A context-switch branch sets HERO_MODEL.max_seq_len, which reaches the launcher as HERO_MODEL_CONFIG.
    monkeypatch.setattr(
        launch_scaling_ladder,
        "HERO_MODEL_CONFIG",
        dataclasses.replace(launch_scaling_ladder.HERO_MODEL_CONFIG, max_seq_len=seq_len),
    )


def test_context_switch_preserves_optimizer(monkeypatch):
    # The child is the checked-in hero with its context history; the parent is the 4K hero it resumes.
    child_step = build_ladder_run(run_id="context-child", size="d6144", version="dev")
    _set_hero_seq_len(monkeypatch, 4096)
    parent_step = build_ladder_run(run_id="context-parent", size="d6144", version="dev", context_phases=())
    parent = parent_step.build_config(StepContext.for_fingerprint(parent_step.runtime_args, parent_step.deps))
    child = child_step.build_config(StepContext.for_fingerprint(child_step.runtime_args, child_step.deps))
    # The child resumes the parent's optimizer state, so its schedule and hyperparameters must not move.
    assert child.trainer.trainer.num_train_steps == parent.trainer.trainer.num_train_steps
    assert child.optimizer == parent.optimizer


def test_hero_context_history_rebuilds_the_prior_phases_it_trained_with(monkeypatch):
    # Each launch re-derives the earlier phases' stages and shuffle from the current code, so a change to
    # them would silently move the data offsets of a run that has already switched context.
    _set_hero_seq_len(monkeypatch, 65536)
    step = build_ladder_run(
        run_id="context-history",
        size="d6144",
        qk_mult=1.6,
        version="dev",
        context_phases=(
            HarrierContextPhase(end_step=231_600, seq_len=4096),
            HarrierContextPhase(end_step=350_400, seq_len=16384),
        ),
        flops_baseline_total=2e24,
    )
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))

    phases = [
        (phase.end_step, phase.seq_len, phase.batch_size, [stage for stage, _ in phase.train_weights], phase.shuffle)
        for phase in config.data.prior_context_phases
    ]
    assert phases == [
        (
            231_600,
            4096,
            11264,
            [0, 108_000, 312_192],
            BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel"),
        ),
        (
            350_400,
            16384,
            2816,
            [0, 108_096, 312_192],
            BlockShuffleConfig(io_block_size=64, window_blocks=512, perm_type="feistel"),
        ),
    ]
    assert config.flops_baseline == FlopsBaseline(completed_steps=350_400, total_flops=2e24)


def test_hero_without_context_parallelism_keeps_its_16k_layout(monkeypatch):
    _set_hero_seq_len(monkeypatch, 16384)
    monkeypatch.setattr(launch_scaling_ladder, "HERO_CONTEXT_AXIS_SIZE", 1)
    monkeypatch.setattr(launch_scaling_ladder, "HERO_BALANCE_ATTENTION_ROWS", False)
    step = build_ladder_run(
        run_id="test-d6144-cp1",
        size="d6144",
        version="dev",
        context_phases=(HarrierContextPhase(end_step=257_553, seq_len=4096),),
        flops_baseline_total=1.8e24,
    )
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))

    assert config.trainer.expert_axis_size == 64
    assert config.trainer.trainer.mesh == MeshConfig()
    assert config.eval.eval_batch_size == 704


def test_hero_context_parallelism_splits_each_rack_into_expert_and_sequence_shards(monkeypatch):
    _set_hero_seq_len(monkeypatch, 65536)
    monkeypatch.setattr(launch_scaling_ladder, "HERO_CONTEXT_AXIS_SIZE", 4)
    monkeypatch.setattr(launch_scaling_ladder, "HERO_BALANCE_ATTENTION_ROWS", True)
    step = build_ladder_run(
        run_id="test-d6144-cp4",
        size="d6144",
        qk_mult=1.66,
        version="dev",
        context_phases=(
            HarrierContextPhase(end_step=257_553, seq_len=4096),
            HarrierContextPhase(end_step=376_900, seq_len=16384),
        ),
        flops_baseline_total=2e24,
    )
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))

    assert config.trainer.expert_axis_size == 16
    mesh_axes, _ = config.trainer.trainer.mesh.axis_shapes(num_devices=704, num_slices=1)
    assert (mesh_axes["data"], mesh_axes["context"]) == (176, 4)
    # One 65,536-token sequence per batch shard: 16,384 tokens per device per eval batch, as at 16K.
    assert config.eval.eval_batch_size == 16 * 11


def test_hero_context_axis_must_divide_the_sequence(monkeypatch):
    _set_hero_seq_len(monkeypatch, 16386)
    monkeypatch.setattr(launch_scaling_ladder, "HERO_CONTEXT_AXIS_SIZE", 4)
    with pytest.raises(ValueError, match="context_axis_size=4 must divide max_seq_len=16386"):
        build_ladder_run(run_id="test-d6144-seq", size="d6144", version="dev")


def test_hero_context_axis_must_split_the_expert_bank(monkeypatch):
    # EP16 x CP4 stores the bank over 64 shards; 32 experts pass a 16-way expert check but fail parameter init.
    monkeypatch.setattr(
        launch_scaling_ladder,
        "HERO_MODEL_CONFIG",
        dataclasses.replace(launch_scaling_ladder.HERO_MODEL_CONFIG, num_experts=32),
    )
    monkeypatch.setattr(launch_scaling_ladder, "HERO_CONTEXT_AXIS_SIZE", 4)
    with pytest.raises(ValueError, match=r"must be divisible by expert \(16\) \* context \(4\) = 64"):
        build_ladder_run(run_id="test-d6144-bank", size="d6144", version="dev")


HERO_SCHEDULE_STEPS = 390_251
_EXTENSION_STEPS = 1084
_LR_EXTENSION_CANDIDATES = [
    pytest.param(ScheduleExtension(steps=_EXTENSION_STEPS, final_lr_ratio=0.05), id="hold-5pct"),
    pytest.param(ScheduleExtension(steps=_EXTENSION_STEPS, final_lr_ratio=0.01), id="decay-5pct-to-1pct"),
]


def _hero_optimizer(extension=None):
    optimizer = MoeHeuristic().build_optimizer_config(
        num_train_steps=HERO_SCHEDULE_STEPS, batch_size=704, hidden_dim=6144, seq_len=65536
    )
    return dataclasses.replace(optimizer, schedule_extension=extension)


def _lr_schedules(optimizer):
    return [
        optimizer.lr_scheduler(HERO_SCHEDULE_STEPS),
        optimizer.lr_scheduler(HERO_SCHEDULE_STEPS, override_lr=optimizer.adam_lr),
    ]


def test_schedule_extension_preserves_rates_through_schedule_end():
    extension = ScheduleExtension(steps=_EXTENSION_STEPS, final_lr_ratio=0.01)
    counts = jnp.arange(HERO_SCHEDULE_STEPS + 1)
    for original, extended in zip(
        _lr_schedules(_hero_optimizer()), _lr_schedules(_hero_optimizer(extension)), strict=True
    ):
        assert jnp.array_equal(jax.vmap(original)(counts), jax.vmap(extended)(counts))


@pytest.mark.parametrize("extension", _LR_EXTENSION_CANDIDATES)
def test_schedule_extension_continues_from_the_floor_to_its_final_ratio(extension):
    optimizer = _hero_optimizer(extension)
    peaks = [optimizer.learning_rate, optimizer.adam_lr]
    counts = jnp.arange(HERO_SCHEDULE_STEPS, HERO_SCHEDULE_STEPS + extension.steps + 2)
    for peak, schedule in zip(peaks, _lr_schedules(optimizer), strict=True):
        rates = np.asarray(jax.vmap(schedule)(counts))
        fractions = np.minimum(np.arange(len(rates)) / extension.steps, 1.0)
        expected = peak * (0.05 + (extension.final_lr_ratio - 0.05) * fractions)
        np.testing.assert_allclose(rates, expected, rtol=1e-5)


def _hero_config():
    step = build_ladder_run(run_id="test-extension", size="d6144", version="dev")
    return step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))


def test_hero_extension_moves_only_the_stop_step_and_the_rate_past_the_schedule(monkeypatch):
    monkeypatch.setattr(launch_scaling_ladder, "HERO_SCHEDULE_EXTENSION", None)
    off = _hero_config()
    extension = ScheduleExtension(steps=_EXTENSION_STEPS, final_lr_ratio=0.01)
    monkeypatch.setattr(launch_scaling_ladder, "HERO_SCHEDULE_EXTENSION", extension)
    on = _hero_config()

    assert off.stop_after_steps == HERO_SCHEDULE_STEPS
    assert on == dataclasses.replace(
        off,
        optimizer=dataclasses.replace(off.optimizer, schedule_extension=extension),
        stop_after_steps=HERO_SCHEDULE_STEPS + _EXTENSION_STEPS,
    )


def test_hero_stop_step_is_not_a_periodic_eval_step():
    # A periodic eval due at the stop step runs before the final save, so an eval failure there would lose
    # the run's last steps. The forced eval after the final save still covers the stop step.
    config = _hero_config()

    assert config.stop_step % config.eval.steps_per_eval != 0
