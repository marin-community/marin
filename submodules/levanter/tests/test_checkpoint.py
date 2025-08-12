import dataclasses
import datetime
import pathlib
import tempfile
from datetime import timedelta

import equinox as eqx
import jax
import jax.tree_util as jtu
import numpy as np
import optax
from chex import assert_trees_all_close, assert_trees_all_equal
from jax import ShapeDtypeStruct
from jax import numpy as jnp

import haliax as hax
from haliax import Axis

from levanter.callbacks import StepInfo
from levanter.checkpoint import (
    Checkpointer,
    CheckpointInterval,
    _load_metadata,
    discover_latest_checkpoint,
    load_checkpoint,
    load_checkpoint_or_initialize,
    save_checkpoint,
)
from levanter.trainer_state import TrainerState
from test_utils import MLP, arrays_only, assert_trees_not_close


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
            checkpointer.on_step(_dummy_step_info(step))

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

        checkpointer.on_step(_dummy_step_info(0))
        advance_time(tick)
        checkpointer.on_step(_dummy_step_info(1))
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [1]

        advance_time(tick - 1)
        checkpointer.on_step(_dummy_step_info(2))
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [1]
        advance_time(1)
        checkpointer.on_step(_dummy_step_info(3))
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

        checkpointer.on_step(_dummy_step_info(0))
        advance_time(tick)
        checkpointer.on_step(_dummy_step_info(1))
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [1]

        advance_time(tick - 1)
        # time hasn't advanced enough, so we wouldn't save a checkpoint, but we do because of the interval
        checkpointer.on_step(_dummy_step_info(2))
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [2]

        advance_time(1)
        # time has advanced enough now from last temporal save, but we don't save a checkpoint because we just saved one
        checkpointer.on_step(_dummy_step_info(3))
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [2]

        for step in range(4, 11):
            advance_time(tick)
            checkpointer.on_step(_dummy_step_info(step))
            # we need this to stop a race condition

        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [2, 4, 6, 8, 10]

        advance_time(tick - 1)
        checkpointer.on_step(_dummy_step_info(11))
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [2, 4, 6, 8, 10]

        for step in range(12, 50):
            checkpointer.on_step(_dummy_step_info(step))
            advance_time(tick)

        # ensure we saved the right checkpoints
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [2, 4, 6, 8, 10, 15, 20, 30, 40, 49]  # 49 is last temporary checkpoint


def _make_state(step, key):
    model = MLP(in_size=2, out_size=1, width_size=2, depth=3, key=key)
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
            discover_latest=False,
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
        restored_state = load_checkpoint(rep_state, checkpoint_path=tmpdir, discover_latest=False)

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

        checkpointer.on_step(_dummy_step_info(0))
        advance_time(tick)
        for i in range(1, 6):
            checkpointer.on_step(_dummy_step_info(i))
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [5]
        advance_time(tick)
        checkpointer.on_step(_dummy_step_info(6))
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

        checkpointer.on_step(_dummy_step_info(7))
        advance_time(tick)
        checkpointer.on_step(_dummy_step_info(8))
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

        checkpointer.on_step(_dummy_step_info(9))
        advance_time(tick)
        checkpointer.on_step(_dummy_step_info(10))
        checkpointer.wait_until_finished()
        assert _get_checkpoint_steps(tmpdir) == [5, 8, 10]


def test_load_from_checkpoint_or_initialize():
    In = Axis("in", 2)
    Out = Axis("out", 1)

    def init_fn(key):
        return hax.nn.MLP.init(In, Out, 2, 1, key=key, use_bias=False, use_final_bias=False)

    k0 = jax.random.PRNGKey(0)
    k1 = jax.random.PRNGKey(1)

    model0 = eqx.filter_jit(init_fn)(k0)
    model1 = eqx.filter_jit(init_fn)(k1)

    is_checkpointed = hax.tree_util.tree_map(lambda _: False, model0)
    is_checkpointed = eqx.tree_at(lambda t: t.layers[-1], is_checkpointed, replace=True)
    is_checkpointed1 = hax.tree_util.tree_map(lambda _: False, model1)
    is_checkpointed1 = eqx.tree_at(lambda t: t.layers[-1], is_checkpointed1, replace=True)

    with jax.sharding.Mesh(jax.devices(), ("devices",)), tempfile.TemporaryDirectory() as tmpdir:
        filtered = eqx.filter(model0, is_checkpointed)
        save_checkpoint(filtered, step=0, checkpoint_path=tmpdir)

        loaded = load_checkpoint_or_initialize(init_fn, tmpdir, is_checkpointed=is_checkpointed, donate_args=False)(k1)
        assert not any(jax.tree_util.tree_leaves(eqx.filter(loaded, lambda x: isinstance(x, ShapeDtypeStruct))))

        loaded2 = load_checkpoint(eqx.filter(model1, is_checkpointed), tmpdir, discover_latest=True)
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


def test_load_from_checkpoint_or_initialize_works_if_file_not_found():
    In = Axis("in", 2)
    Out = Axis("out", 1)

    def init_fn(key):
        return hax.nn.MLP.init(In, Out, 2, 3, key=key)

    k0 = jax.random.PRNGKey(0)
    k1 = jax.random.PRNGKey(1)

    model0 = init_fn(k0)
    model1 = init_fn(k1)

    is_checkpointed = jtu.tree_map(lambda _: False, model0)
    is_checkpointed = eqx.tree_at(lambda t: t.layers[-1], is_checkpointed, replace=True)

    with jax.sharding.Mesh(jax.devices(), ("devices",)):
        loaded = load_checkpoint_or_initialize(init_fn, "kanmfklafnmjlkanfjklanfjkh", is_checkpointed=is_checkpointed)(
            k1
        )

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

    k0 = jax.random.PRNGKey(0)
    k1 = jax.random.PRNGKey(1)

    model0 = init_fn(k0, False)
    model1 = init_fn(k1, True)

    is_checkpointed = True

    with jax.sharding.Mesh(jax.devices(), ("devices",)), tempfile.TemporaryDirectory() as tmpdir:

        save_checkpoint(eqx.filter(model0, is_checkpointed), step=0, checkpoint_path=tmpdir)

        loaded = load_checkpoint_or_initialize(
            init_fn,
            tmpdir,
            is_checkpointed=is_checkpointed,
            allow_partial=True,
        )(k1, True)

        assert not any(jax.tree_util.tree_leaves(eqx.filter(loaded, lambda x: isinstance(x, ShapeDtypeStruct))))
        assert hax.all(hax.equal(loaded.a, model0.a))
        assert loaded.b is not None
        assert hax.all(hax.equal(loaded.b, model1.b))
