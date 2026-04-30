# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import datetime
import json
import os
import pathlib
import tempfile
from datetime import timedelta

import equinox
import equinox as eqx
import fsspec
import haliax as hax
import jax
import jax.tree_util as jtu
import numpy as np
import optax
import pytest
from chex import assert_trees_all_close, assert_trees_all_equal
from haliax import Axis
from jax import ShapeDtypeStruct
from jax import numpy as jnp
from test_utils import MLP, arrays_only, assert_trees_not_close, use_test_mesh

from levanter.callbacks import StepInfo
from levanter.checkpoint import (
    CheckpointDebugConfig,
    Checkpointer,
    CheckpointerConfig,
    CheckpointInterval,
    _collect_debug_checkpointer_state,
    _load_metadata,
    discover_latest_checkpoint,
    load_checkpoint,
    load_checkpoint_or_initialize,
    register_debug_checkpointer_state_provider,
    save_checkpoint,
    unregister_debug_checkpointer_state_provider,
)
from levanter.trainer import TrainerConfig
from levanter.trainer_state import TrainerState


def _dummy_step_info(step):
    return StepInfo(
        state=TrainerState(
            # + 1 b/c step here is next step
            step=step + 1,
            model=None,
            optimizer=None,  # type: ignore
            opt_state=None,
            training_key=jax.random.PRNGKey(0),
            is_trainable=True,
            mp=None,
            model_averaging=None,
        ),
        loss=0.0,
        step_duration=0.0,
    )


def _on_step(checkpointer: Checkpointer, step: int, *, force: bool = False):
    info = _dummy_step_info(step)
    checkpointer.on_step(tree=info.state.saveable_state, step=info.step, force=force)


def _get_checkpoint_steps(checkpoint_dir):
    paths = list(pathlib.Path(checkpoint_dir).iterdir())
    return sorted([_load_metadata(f)["step"] for f in paths])


def test_checkpointer_changing_policy():
    with tempfile.TemporaryDirectory(prefix="checkpoints") as tmpdir:
        checkpointer = Checkpointer(
            tmpdir,
            None,
            [
                CheckpointInterval(every=2, until=10),
                CheckpointInterval(every=5, until=20),
                CheckpointInterval(every=10, until=None),
            ],
        )

        for step in range(1, 50):
            _on_step(checkpointer, step)

        checkpointer.wait_until_finished()

        # ensure we saved the right checkpoints
        assert _get_checkpoint_steps(tmpdir) == [2, 4, 6, 8, 10, 15, 20, 30, 40]


def test_checkpointer_temporal_policy():
    fake_now = datetime.datetime(2021, 1, 1, 0, 0, 0)

    tick = 10

    def advance_time(delta_seconds):
        nonlocal fake_now
        fake_now += timedelta(seconds=delta_seconds)

    with tempfile.TemporaryDirectory(prefix="checkpoints") as tmpdir:
        checkpointer = Checkpointer(tmpdir, timedelta(seconds=tick), [], dt_now_injection=lambda: fake_now)

        _on_step(checkpointer, 0)
        advance_time(tick)
        _on_step(checkpointer, 1)
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [1]

        advance_time(tick - 1)
        _on_step(checkpointer, 2)
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [1]
        advance_time(1)
        _on_step(checkpointer, 3)
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [3]


def test_checkpointer_mixed_policy():
    fake_now = datetime.datetime(2021, 1, 1, 0, 0, 0)

    tick = 10

    def advance_time(delta_seconds):
        nonlocal fake_now
        fake_now += timedelta(seconds=delta_seconds)

    with tempfile.TemporaryDirectory(prefix="checkpoints") as tmpdir:
        checkpointer = Checkpointer(
            tmpdir,
            timedelta(seconds=tick),
            [
                CheckpointInterval(every=2, until=10),
                CheckpointInterval(every=5, until=20),
                CheckpointInterval(every=10, until=None),
            ],
            dt_now_injection=lambda: fake_now,
        )

        _on_step(checkpointer, 0)
        advance_time(tick)
        _on_step(checkpointer, 1)
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [1]

        advance_time(tick - 1)
        # time hasn't advanced enough, so we wouldn't save a checkpoint, but we do because of the interval
        _on_step(checkpointer, 2)
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [2]

        advance_time(1)
        # time has advanced enough now from last temporal save, but we don't save a checkpoint because we just saved one
        _on_step(checkpointer, 3)
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [2]

        for step in range(4, 11):
            advance_time(tick)
            _on_step(checkpointer, step)
            # we need this to stop a race condition

        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [2, 4, 6, 8, 10]

        advance_time(tick - 1)
        _on_step(checkpointer, 11)
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [2, 4, 6, 8, 10]

        for step in range(12, 50):
            _on_step(checkpointer, step)
            advance_time(tick)

        # ensure we saved the right checkpoints
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [2, 4, 6, 8, 10, 15, 20, 30, 40, 49]  # 49 is last temporary checkpoint


def _make_state(step, key, depth=3):
    model = MLP(in_size=2, out_size=1, width_size=2, depth=depth, key=key)
    optim = optax.adam(1e-4)
    opt_state = optim.init(arrays_only(model))

    return TrainerState(step, model, optim, opt_state, key, is_trainable=True, mp=None, model_averaging=None)


def test_checkpoint_simple():
    key0 = jax.random.PRNGKey(0)
    key1 = jax.random.PRNGKey(1)

    initial_state = _make_state(10, key0)
    rep_state = _make_state(2, key1)

    assert_trees_not_close(initial_state.model, rep_state.model)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(
            initial_state,
            step=initial_state.step,
            checkpoint_path=tmpdir,
        )
        restored_state = load_checkpoint(
            rep_state,
            checkpoint_path=tmpdir,
        )

        assert_trees_all_equal(
            jax.tree_util.tree_leaves(arrays_only(restored_state.model)),
            jax.tree_util.tree_leaves(arrays_only(initial_state.model)),
        )
        assert all(np.isclose(restored_state.training_key, initial_state.training_key))
        assert restored_state.step == initial_state.step


def test_checkpoint_steps():
    key0 = jax.random.PRNGKey(0)
    key1 = jax.random.PRNGKey(1)

    optim = optax.adam(1e-4)

    initial_state = _make_state(10, key0)
    data = jax.random.uniform(key0, (2, 2))

    @eqx.filter_grad
    def loss_fn(model, data):
        m = jax.vmap(model)
        return jnp.mean(jnp.square(m(data)))

    state = initial_state
    for i in range(3):
        grad = loss_fn(state.model, data)
        updates, new_state = optim.update(grad, state.opt_state)
        model = eqx.apply_updates(state.model, updates)
        state = dataclasses.replace(state, step=state.step + 1, model=model, opt_state=new_state)

    assert_trees_not_close(state, initial_state)

    rep_state = _make_state(42, key1)
    assert_trees_not_close(state, rep_state)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(state, step=3, checkpoint_path=tmpdir)
        restored_state = load_checkpoint(rep_state, checkpoint_path=tmpdir)

        assert_trees_all_equal(
            jax.tree_util.tree_leaves(arrays_only(restored_state)),
            jax.tree_util.tree_leaves(arrays_only(state)),
        )


def test_checkpoint_discovery():
    with tempfile.TemporaryDirectory() as tempdir:
        save_checkpoint(dict(model=1, training_state=2), step=10, checkpoint_path=f"{tempdir}/step-10")
        save_checkpoint(dict(model=3, training_state=4), step=20, checkpoint_path=f"{tempdir}/step-20")
        save_checkpoint(dict(model=5, training_state=6), step=30, checkpoint_path=f"{tempdir}/step-30")

        latest = discover_latest_checkpoint(tempdir)
        assert latest == f"{tempdir}/step-30"

        assert discover_latest_checkpoint("file:///tmp/does-not-exist") is None


def test_checkpoint_discovery_across_multiple_paths():
    with tempfile.TemporaryDirectory() as permanent_dir, tempfile.TemporaryDirectory() as temp_dir:
        save_checkpoint(dict(model=1), step=10, checkpoint_path=f"{permanent_dir}/step-10", is_temporary=False)
        save_checkpoint(dict(model=2), step=15, checkpoint_path=f"{temp_dir}/step-15", is_temporary=True)

        # Without additional paths, only permanent_dir is searched
        latest_single = discover_latest_checkpoint(permanent_dir)
        assert latest_single == f"{permanent_dir}/step-10"

        # With additional paths, the newer checkpoint in temp_dir wins
        latest_both = discover_latest_checkpoint(permanent_dir, temp_dir)
        assert latest_both == f"{temp_dir}/step-15"


def test_checkpointer_temporary_base_path_routes_temp_checkpoints():
    fake_now = datetime.datetime(2021, 1, 1, 0, 0, 0)
    tick = 10

    def advance_time(delta_seconds):
        nonlocal fake_now
        fake_now += timedelta(seconds=delta_seconds)

    with tempfile.TemporaryDirectory() as permanent_dir, tempfile.TemporaryDirectory() as temp_dir:
        checkpointer = Checkpointer(
            permanent_dir,
            timedelta(seconds=tick),
            [CheckpointInterval(every=5, until=None)],
            temporary_base_path=temp_dir,
            dt_now_injection=lambda: fake_now,
        )

        # Step 0 doesn't save
        _on_step(checkpointer, 0)

        # Time-based save goes to temp_dir
        advance_time(tick)
        _on_step(checkpointer, 1)
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(temp_dir) == [1]
        assert _get_checkpoint_steps(permanent_dir) == []

        # Step-based save goes to permanent_dir
        advance_time(tick)
        _on_step(checkpointer, 5)
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(permanent_dir) == [5]
        # Old temp checkpoint should be deleted
        assert _get_checkpoint_steps(temp_dir) == []

        # Another time-based save goes to temp_dir
        advance_time(tick)
        _on_step(checkpointer, 6)
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(temp_dir) == [6]
        assert _get_checkpoint_steps(permanent_dir) == [5]


def test_checkpointer_config_temporary_base_path():
    config = dataclasses.replace(
        CheckpointerConfig(),
        base_path="/tmp/test-perm",
        temporary_base_path="/tmp/test-temp",
        append_run_id_to_base_path=False,
    )
    assert config.expanded_path("run1") == "/tmp/test-perm"
    assert config.expanded_temporary_path("run1") == "/tmp/test-temp"

    config_with_run_id = dataclasses.replace(
        CheckpointerConfig(),
        base_path="/tmp/test-perm",
        temporary_base_path="/tmp/test-temp",
        append_run_id_to_base_path=True,
    )
    assert config_with_run_id.expanded_path("run1") == "/tmp/test-perm/run1"
    assert config_with_run_id.expanded_temporary_path("run1") == "/tmp/test-temp/run1"


def test_checkpointer_config_no_temporary_base_path():
    config = CheckpointerConfig()
    assert config.temporary_base_path is None
    assert config.expanded_temporary_path("run1") is None


def test_trainer_config_checkpoint_search_paths():
    config = dataclasses.replace(
        TrainerConfig(),
        checkpointer=CheckpointerConfig(
            base_path="/tmp/test-perm",
            temporary_base_path="/tmp/test-temp",
            append_run_id_to_base_path=True,
        ),
    )
    assert config.checkpoint_search_paths("run1") == ["/tmp/test-perm/run1", "/tmp/test-temp/run1"]

    pinned_config = dataclasses.replace(config, load_checkpoint_path="/tmp/test-perm/run1/step-100")
    assert pinned_config.checkpoint_search_paths("run1") == ["/tmp/test-perm/run1/step-100"]


def test_checkpointer_config_propagates_debug_settings():
    config = CheckpointerConfig(
        base_path="/tmp/checkpoints",
        delete_previous_temporary_checkpoint_after_save=False,
        debug=CheckpointDebugConfig(
            enabled=True,
            log_interval=12.5,
            dump_stacks_after=45.0,
            tracemalloc_frames=17,
            top_allocations=5,
            force_gc_before_serialize=False,
            flush_logs=False,
        ),
    )

    checkpointer = config.create("run-1")

    assert checkpointer.delete_previous_temporary_checkpoint_after_save is False
    assert checkpointer.debug.enabled is True
    assert checkpointer.debug.log_interval == 12.5
    assert checkpointer.debug.dump_stacks_after == 45.0
    assert checkpointer.debug.tracemalloc_frames == 17
    assert checkpointer.debug.top_allocations == 5
    assert checkpointer.debug.force_gc_before_serialize is False
    assert checkpointer.debug.flush_logs is False


def test_debug_checkpointer_state_providers_register_and_unregister():
    provider_name = "unit-test-provider"
    provider = lambda: {"weight_transfer": {"bytes": 123}}

    try:
        register_debug_checkpointer_state_provider(provider_name, provider)
        assert _collect_debug_checkpointer_state()[provider_name] == {"weight_transfer": {"bytes": 123}}
    finally:
        unregister_debug_checkpointer_state_provider(provider_name)

    assert provider_name not in _collect_debug_checkpointer_state()


def test_checkpointer_config_rejects_invalid_debug_tracemalloc_settings():
    with pytest.raises(AssertionError, match="checkpoint debug tracemalloc_frames must be positive"):
        CheckpointerConfig(debug=CheckpointDebugConfig(tracemalloc_frames=0))


def test_checkpointer_deletes_previous_checkpoints():
    fake_now = datetime.datetime(2021, 1, 1, 0, 0, 0)

    tick = 10

    def advance_time(delta_seconds):
        nonlocal fake_now
        fake_now += timedelta(seconds=delta_seconds)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpointer = Checkpointer(
            tmpdir,
            timedelta(seconds=tick),
            [
                CheckpointInterval(every=5, until=20),
                CheckpointInterval(every=10, until=None),
            ],
            dt_now_injection=lambda: fake_now,
        )

        _on_step(checkpointer, 0)
        advance_time(tick)
        for i in range(1, 6):
            _on_step(checkpointer, i)
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [5]
        advance_time(tick)
        _on_step(checkpointer, 6)
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [5, 6]

        # now make a new one and ensure it deletes the old one
        checkpointer = Checkpointer(
            tmpdir,
            timedelta(seconds=tick),
            [
                CheckpointInterval(every=5, until=20),
                CheckpointInterval(every=10, until=None),
            ],
            dt_now_injection=lambda: fake_now,
        )

        _on_step(checkpointer, 7)
        advance_time(tick)
        _on_step(checkpointer, 8)
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [5, 8]

        # now make sure if we don't enable deleting old checkpoints, it doesn't delete them
        checkpointer = Checkpointer(
            tmpdir,
            timedelta(seconds=tick),
            [
                CheckpointInterval(every=20, until=None),
            ],
            dt_now_injection=lambda: fake_now,
            delete_old_temp_checkpoints=False,
        )

        _on_step(checkpointer, 9)
        advance_time(tick)
        _on_step(checkpointer, 10)
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [5, 8, 10]


def test_checkpointer_deletes_previous_checkpoints_under_relative_base_paths():
    fake_now = datetime.datetime(2021, 1, 1, 0, 0, 0)

    tick = 10

    def advance_time(delta_seconds):
        nonlocal fake_now
        fake_now += timedelta(seconds=delta_seconds)

    with tempfile.TemporaryDirectory(dir=".") as tmpdir:
        # remove ./ if present because tensorstore doesn't like it
        tmpdir = os.path.normpath(tmpdir)
        print(f"tmpdir is {tmpdir}")
        checkpointer = Checkpointer(
            tmpdir,
            timedelta(seconds=tick),
            [],
            dt_now_injection=lambda: fake_now,
        )

        # step 0 doesn't save a checkpoint
        _on_step(checkpointer, 0)

        advance_time(tick)
        _on_step(checkpointer, 1)
        checkpointer.wait_until_finished()
        # step 1 should save a checkpoint
        assert _get_checkpoint_steps(tmpdir) == [1]

        advance_time(tick)
        _on_step(checkpointer, 2)
        checkpointer.wait_until_finished()
        # step 2 should delete step 1 if we're handling relative paths properly
        assert _get_checkpoint_steps(tmpdir) == [2]


def test_checkpointer_can_keep_previous_temporary_checkpoint_after_save():
    fake_now = datetime.datetime(2021, 1, 1, 0, 0, 0)
    tick = 10

    def advance_time(delta_seconds):
        nonlocal fake_now
        fake_now += timedelta(seconds=delta_seconds)

    with tempfile.TemporaryDirectory(prefix="checkpoints") as tmpdir:
        checkpointer = Checkpointer(
            tmpdir,
            timedelta(seconds=tick),
            [],
            dt_now_injection=lambda: fake_now,
            delete_previous_temporary_checkpoint_after_save=False,
        )

        _on_step(checkpointer, 0)

        advance_time(tick)
        _on_step(checkpointer, 1)
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [1]

        advance_time(tick)
        _on_step(checkpointer, 2)
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [1, 2]


def test_checkpointer_force_save_uses_permanent_path_even_when_time_policy_elapsed():
    fake_now = datetime.datetime(2021, 1, 1, 0, 0, 0)
    tick = 10

    def advance_time(delta_seconds):
        nonlocal fake_now
        fake_now += timedelta(seconds=delta_seconds)

    with (
        tempfile.TemporaryDirectory(prefix="checkpoints") as permanent_dir,
        tempfile.TemporaryDirectory(prefix="temp_checkpoints") as temporary_dir,
    ):
        checkpointer = Checkpointer(
            permanent_dir,
            timedelta(seconds=tick),
            [],
            temporary_base_path=temporary_dir,
            dt_now_injection=lambda: fake_now,
        )

        _on_step(checkpointer, 0)

        advance_time(tick)
        _on_step(checkpointer, 1, force=True)
        checkpointer.wait_until_finished()

        assert _get_checkpoint_steps(permanent_dir) == [1]
        assert list(pathlib.Path(temporary_dir).iterdir()) == []


def test_load_from_checkpoint_or_initialize():
    In = Axis("in", 2)
    Out = Axis("out", 1)

    def init_fn(key):
        return hax.nn.MLP.init(In, Out, 2, 1, key=key, use_bias=False, use_final_bias=False)

    with use_test_mesh(), tempfile.TemporaryDirectory() as tmpdir:
        k0 = jax.random.PRNGKey(0)
        k1 = jax.random.PRNGKey(1)
        model0 = eqx.filter_jit(init_fn)(k0)
        model1 = eqx.filter_jit(init_fn)(k1)

        is_checkpointed = hax.tree_util.tree_map(lambda _: False, model0)
        is_checkpointed = eqx.tree_at(lambda t: t.layers[-1], is_checkpointed, replace=True)
        is_checkpointed1 = hax.tree_util.tree_map(lambda _: False, model1)
        is_checkpointed1 = eqx.tree_at(lambda t: t.layers[-1], is_checkpointed1, replace=True)

        filtered = eqx.filter(model0, is_checkpointed)
        save_checkpoint(filtered, step=0, checkpoint_path=tmpdir)

        loaded = load_checkpoint_or_initialize(init_fn, [tmpdir], is_checkpointed=is_checkpointed, donate_args=False)(
            k1
        )
        assert not any(jax.tree_util.tree_leaves(eqx.filter(loaded, lambda x: isinstance(x, ShapeDtypeStruct))))

        latest_checkpoint = discover_latest_checkpoint(tmpdir)
        assert latest_checkpoint is not None
        loaded2 = load_checkpoint(eqx.filter(model1, is_checkpointed), latest_checkpoint)
        loaded2 = eqx.combine(loaded2, model1)

        assert_trees_all_equal(
            jax.tree_util.tree_leaves(arrays_only(loaded)),
            jax.tree_util.tree_leaves(arrays_only(loaded2)),
        )

        assert_trees_all_equal(
            jax.tree_util.tree_leaves(arrays_only(eqx.filter(loaded, is_checkpointed))),
            jax.tree_util.tree_leaves(arrays_only(eqx.filter(model0, is_checkpointed))),
        )

        assert_trees_not_close(
            jax.tree_util.tree_leaves(arrays_only(eqx.filter(loaded, is_checkpointed, inverse=True))),
            jax.tree_util.tree_leaves(arrays_only(eqx.filter(model0, is_checkpointed, inverse=True))),
        )

        assert_trees_not_close(
            jax.tree_util.tree_leaves(arrays_only(eqx.filter(loaded, is_checkpointed))),
            jax.tree_util.tree_leaves(arrays_only(eqx.filter(model1, is_checkpointed))),
        )

        assert_trees_all_equal(
            jax.tree_util.tree_leaves(arrays_only(eqx.filter(loaded, is_checkpointed, inverse=True))),
            jax.tree_util.tree_leaves(arrays_only(eqx.filter(model1, is_checkpointed1, inverse=True))),
        )


def test_load_from_checkpoint_or_initialize_searches_additional_paths():
    In = Axis("in", 2)
    Out = Axis("out", 1)

    def init_fn(key):
        return hax.nn.MLP.init(In, Out, 2, 1, key=key, use_bias=False, use_final_bias=False)

    with use_test_mesh(), tempfile.TemporaryDirectory() as permanent_dir, tempfile.TemporaryDirectory() as temp_dir:
        k0 = jax.random.PRNGKey(0)
        k1 = jax.random.PRNGKey(1)
        model0 = eqx.filter_jit(init_fn)(k0)
        model1 = eqx.filter_jit(init_fn)(k1)

        is_checkpointed = hax.tree_util.tree_map(lambda _: False, model0)
        is_checkpointed = eqx.tree_at(lambda t: t.layers[-1], is_checkpointed, replace=True)

        filtered = eqx.filter(model0, is_checkpointed)
        save_checkpoint(filtered, step=0, checkpoint_path=temp_dir)

        loaded = load_checkpoint_or_initialize(
            init_fn,
            [permanent_dir, temp_dir],
            is_checkpointed=is_checkpointed,
            donate_args=False,
        )(k1)

        assert_trees_all_equal(
            jax.tree_util.tree_leaves(arrays_only(eqx.filter(loaded, is_checkpointed))),
            jax.tree_util.tree_leaves(arrays_only(eqx.filter(model0, is_checkpointed))),
        )
        assert_trees_all_equal(
            jax.tree_util.tree_leaves(arrays_only(eqx.filter(loaded, is_checkpointed, inverse=True))),
            jax.tree_util.tree_leaves(arrays_only(eqx.filter(model1, is_checkpointed, inverse=True))),
        )


def test_load_from_checkpoint_or_initialize_works_if_file_not_found():
    In = Axis("in", 2)
    Out = Axis("out", 1)

    def init_fn(key):
        return hax.nn.MLP.init(In, Out, 2, 3, key=key)

    with use_test_mesh():
        k0 = jax.random.PRNGKey(0)
        k1 = jax.random.PRNGKey(1)
        model0 = init_fn(k0)
        model1 = init_fn(k1)

        is_checkpointed = jtu.tree_map(lambda _: False, model0)
        is_checkpointed = eqx.tree_at(lambda t: t.layers[-1], is_checkpointed, replace=True)

        loaded = load_checkpoint_or_initialize(
            init_fn, ["kanmfklafnmjlkanfjklanfjkh"], is_checkpointed=is_checkpointed
        )(k1)

        assert not any(jax.tree_util.tree_leaves(eqx.filter(loaded, lambda x: isinstance(x, ShapeDtypeStruct))))
        # should be the same as model1
        # on TPU, there's a very slight difference for some reason
        assert_trees_all_close(
            jax.tree_util.tree_leaves(arrays_only(eqx.filter(loaded, is_checkpointed))),
            jax.tree_util.tree_leaves(arrays_only(eqx.filter(model1, is_checkpointed))),
        )


def test_load_from_checkpoint_allows_partial_checkpoints():
    In = Axis("in", 2)
    Out = Axis("out", 1)

    class MyModule(eqx.Module):
        a: hax.NamedArray
        b: hax.NamedArray | None

    def init_fn(key, use_b):
        k_a, k_b = jax.random.split(key)
        return MyModule(a=hax.random.normal(k_a, (In, Out)), b=hax.random.normal(k_b, (In, Out)) if use_b else None)

    is_checkpointed = True

    with use_test_mesh(), tempfile.TemporaryDirectory() as tmpdir:
        k0 = jax.random.PRNGKey(0)
        k1 = jax.random.PRNGKey(1)
        model0 = init_fn(k0, False)
        model1 = init_fn(k1, True)

        save_checkpoint(eqx.filter(model0, is_checkpointed), step=0, checkpoint_path=tmpdir)

        loaded = load_checkpoint_or_initialize(
            init_fn,
            [tmpdir],
            is_checkpointed=is_checkpointed,
            allow_partial=True,
        )(k1, True)

        assert not any(jax.tree_util.tree_leaves(eqx.filter(loaded, lambda x: isinstance(x, ShapeDtypeStruct))))
        assert hax.all(hax.equal(loaded.a, model0.a))
        assert loaded.b is not None
        assert hax.all(hax.equal(loaded.b, model1.b))


def test_ocdbt_merges_files():
    """Test that OCDBT checkpoints create manifest.ocdbt file."""

    for depth in [1, 5, 20]:
        with tempfile.TemporaryDirectory() as tmpdir:
            key0 = jax.random.PRNGKey(0)
            initial_state = _make_state(10, key0, depth=depth)
            save_checkpoint(
                initial_state,
                step=initial_state.step,
                checkpoint_path=tmpdir,
            )

            # Check that manifest.ocdbt exists
            # The manifest should be in one of the checkpoint subdirectories
            checkpoint_dir = pathlib.Path(tmpdir)
            checkpoint_files = [path for path in checkpoint_dir.rglob("*") if path.is_file()]
            assert (
                len(checkpoint_files) <= 25
            ), f"There should be fewer than 25 files in the checkpoint directory: {checkpoint_files}"
            print(depth, len(checkpoint_files), checkpoint_files)

            manifest_files = list(checkpoint_dir.rglob("manifest.ocdbt"))
            assert len(manifest_files) > 0, "OCDBT manifest.ocdbt file should exist in checkpoint"


def test_backward_compatibility_with_ocdbt():
    """Test that we can load old non-OCDBT checkpoints with new OCDBT-enabled code."""
    import jax.experimental.array_serialization.serialization as array_ser

    key0 = jax.random.PRNGKey(0)
    key1 = jax.random.PRNGKey(1)

    initial_state = _make_state(10, key0)
    rep_state = _make_state(2, key1)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save with old format by directly using serialize_with_paths (non-OCDBT)
        manager = array_ser.GlobalAsyncCheckpointManager()
        from levanter.utils import jax_utils

        checkpoint_path = tmpdir

        leaf_key_paths = jax_utils.leaf_key_paths(initial_state, is_leaf=lambda x: x is None)
        paths = []
        for key_path in jax.tree.leaves(leaf_key_paths):
            paths.append(f"{checkpoint_path}/{key_path.replace('.', '/')}")

        arrays = [
            leaf.array if hasattr(leaf, "array") else leaf
            for leaf in jax.tree.leaves(initial_state)
            if hasattr(leaf, "array") or jax.Array in type(leaf).__mro__
        ]

        filtered = [(a, p) for a, p in zip(arrays, paths) if equinox.is_array_like(a)]
        arrays_to_save = [a for a, _ in filtered]
        paths_to_save = [p for _, p in filtered]

        # Save using old non-OCDBT method
        manager.serialize_with_paths(arrays_to_save, paths_to_save)
        manager.wait_until_finished()

        # Save metadata (normally done by save_checkpoint)
        fs, _ = fsspec.core.url_to_fs(checkpoint_path)
        metadata = {"step": 10, "timestamp": datetime.datetime.now().isoformat(), "is_temporary": False}
        with fs.open(f"{checkpoint_path}/metadata.json", "w") as f:
            json.dump(metadata, f)

        # Now try to load it with the new OCDBT-enabled code
        restored_state = load_checkpoint(
            rep_state,
            checkpoint_path=tmpdir,
        )

        # Verify the data was loaded correctly
        assert_trees_all_equal(
            jax.tree_util.tree_leaves(arrays_only(restored_state.model)),
            jax.tree_util.tree_leaves(arrays_only(initial_state.model)),
        )
        assert all(np.isclose(restored_state.training_key, initial_state.training_key))
        assert restored_state.step == initial_state.step
