"""Small real-checkpoint proofs for explicit 10/05 optimizer-state conversion."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.checkpoint import save_checkpoint
from levanter.checkpoint_manifest import read_manifest

from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.post_training.long_loop.runs.e12 import carried_restore
from experiments.post_training.long_loop.runs.e12.carried_restore import (
    _array_specs,
    _shared,
    restore_1005_carried_state,
)
from experiments.post_training.long_loop.sft_runtime.train import (
    GrugTrainState,
    reset_carried_trainer_step,
    restore_template_from,
)


class Expert(eqx.Module):
    w_gate: jax.Array
    w_up: jax.Array
    w_down: jax.Array


class Stacked(eqx.Module):
    shared: tuple[Expert, ...] | Expert
    matrix: jax.Array


class Blocks(eqx.Module):
    stacked: Stacked


class Parameters(eqx.Module):
    stacked_blocks: Blocks
    token_embed: jax.Array
    output_proj: jax.Array


def make_state():
    def x(shape):
        return jnp.full(shape, 0.125, dtype=jnp.bfloat16)

    params = Parameters(
        stacked_blocks=Blocks(Stacked(shared=(Expert(x((2, 2)), x((2, 2)), x((2, 2))),), matrix=x((2, 2)))),
        token_embed=x((4, 2)),
        output_proj=x((2, 4)),
    )
    optimizer = GrugMoeMuonHConfig(
        learning_rate=5e-6,
        adam_lr=5e-6,
        lr_schedule="constant",
        min_lr_ratio=1.0,
        warmup=0,
        decay=0,
        max_grad_norm=None,
        global_grad_norm=1.0,
        beta1=0.9062,
        beta2=0.95,
        epsilon=3.8339433005718795e-15,
        expert_normalization="all_experts",
        use_syrk=False,  # CPU checkpoint test; GPU SYRK path is not exercised.
    ).build(15)
    return (
        GrugTrainState(
            step=jnp.array(0, dtype=jnp.int32),
            params=params,
            master_params=None,
            opt_state=optimizer.init(params),
            ema_params=None,
            pending_qb_betas=jnp.zeros((2, 2), dtype=jnp.float32),
        ),
        optimizer,
    )


def populated_source(target):
    legacy = _shared(target, legacy=True)

    def fill(path, value):
        if not hasattr(value, "dtype"):
            return value
        name = jax.tree_util.keystr(path)
        if name.endswith(".count") or name.endswith(".step"):
            return jnp.full(value.shape, 161750, dtype=value.dtype)
        if "hyperparams[" in name:
            return jnp.full(value.shape, 5.0947383e-6, dtype=jnp.float32)
        values = jnp.arange(value.size, dtype=jnp.float32).reshape(value.shape)
        return 0.1234567 + values * 0.00314159

    return jax.tree_util.tree_map_with_path(fill, legacy)


@pytest.fixture(autouse=True)
def synthetic_optimizer_path_count(monkeypatch):
    monkeypatch.setattr(carried_restore, "_EXPECTED_MOMENT_LEAVES", 8)


@pytest.fixture
def cpu_mesh():
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape((-1,)), ("data",), axis_types=(jax.sharding.AxisType.Explicit,)
    )
    with jax.set_mesh(mesh):
        yield mesh


def test_real_checkpoint_fills_every_slot_and_rounds_only_fp32_to_bf16(tmp_path, cpu_mesh):
    target, optimizer = make_state()
    saved = populated_source(target)
    path = tmp_path / "checkpoint"
    save_checkpoint(saved, step=161750, checkpoint_path=path, is_temporary=False)
    target = restore_template_from(target)
    restored = restore_1005_carried_state(target, str(path), mesh=cpu_mesh, expected_step=161750)
    after = _shared(restored, legacy=True)
    before_specs, after_specs = _array_specs(saved), _array_specs(after)
    assert before_specs.keys() == after_specs.keys()
    for source, actual in zip(jax.tree.leaves(saved), jax.tree.leaves(after), strict=True):
        expected = source.astype(jnp.bfloat16) if actual.dtype == jnp.bfloat16 else source
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    momentum = restored.opt_state.inner_state.inner_states["muonh"].inner_state[0].momentum_buffer
    assert isinstance(momentum.stacked_blocks.stacked.shared, tuple)
    for leaf in jax.tree.leaves(momentum.stacked_blocks.stacked.shared):
        assert np.all(np.asarray(leaf) != 0)
    assert int(restored.opt_state.count) == 161750
    assert int(restored.opt_state.hyperparams_states["learning_rate"].count) == 161750
    assert int(restored.opt_state.inner_state.inner_states["adam"].inner_state[0].count) == 161750
    assert int(restored.opt_state.inner_state.inner_states["adamh"].inner_state[0].count) == 161750
    assert float(restored.opt_state.hyperparams["learning_rate"]) == float(jnp.asarray(5.0947383e-6, dtype=jnp.bfloat16))
    reset = reset_carried_trainer_step(restored, load_checkpoint=True, allow_partial=False, expected_step=161750)
    assert int(reset.step) == 0 and int(reset.opt_state.count) == 161750
    # Injected learning rates are recomputed from the current constant schedule on the next update.
    grads = jax.tree.map(jnp.ones_like, reset.params)
    _, updated = optimizer.update(grads, reset.opt_state, reset.params)
    assert float(updated.hyperparams["learning_rate"]) == float(jnp.asarray(5e-6, dtype=jnp.bfloat16))
    assert float(updated.hyperparams["adam_lr"]) == float(jnp.asarray(5e-6, dtype=jnp.bfloat16))
    assert int(updated.count) == 161751
    assert int(updated.inner_state.inner_states["adam"].inner_state[0].count) == 161751
    assert int(updated.inner_state.inner_states["adamh"].inner_state[0].count) == 161751


def test_reject_wrong_step_and_non_bf16_template(tmp_path, cpu_mesh):
    target, _ = make_state()
    saved = populated_source(target)
    path = tmp_path / "checkpoint"
    save_checkpoint(saved, step=161750, checkpoint_path=path, is_temporary=False)
    with pytest.raises(ValueError, match="restored step"):
        restore_1005_carried_state(target, str(path), mesh=cpu_mesh, expected_step=161749)
    wrong = dataclasses.replace(target, params=jax.tree.map(lambda x: x.astype(jnp.float32), target.params))
    with pytest.raises(ValueError, match="BF16"):
        restore_1005_carried_state(wrong, str(path), mesh=cpu_mesh, expected_step=161750)


def test_missing_manifest_slot_fails_before_any_restore(tmp_path, cpu_mesh, monkeypatch):
    target, _ = make_state()
    path = tmp_path / "checkpoint"
    save_checkpoint(populated_source(target), step=161750, checkpoint_path=path, is_temporary=False)
    manifest = read_manifest(str(path))
    missing_slot = next(x for x in manifest.arrays if "/momentum_buffer/" in x.path)
    shortened = manifest.model_copy(update={"arrays": tuple(x for x in manifest.arrays if x.path != missing_slot.path)})
    monkeypatch.setattr(carried_restore, "read_manifest", lambda _path: shortened)

    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("a missing manifest leaf must fail before loading")

    monkeypatch.setattr(carried_restore, "load_checkpoint", forbidden_load)
    with pytest.raises(ValueError, match="checkpoint layout mismatch"):
        restore_1005_carried_state(target, str(path), mesh=cpu_mesh, expected_step=161750)


def test_streaming_restore_deletes_each_cast_source_before_loading_next(tmp_path, cpu_mesh, monkeypatch):
    target, _ = make_state()
    path = tmp_path / "checkpoint"
    save_checkpoint(populated_source(target), step=161750, checkpoint_path=path, is_temporary=False)
    target = restore_template_from(target)
    target_specs = _array_specs(_shared(target, legacy=True))
    real_load = carried_restore.load_checkpoint
    prior_cast_source = None
    seen = 0

    def tracked_load(*args, **kwargs):
        nonlocal prior_cast_source, seen
        if prior_cast_source is not None:
            assert prior_cast_source.is_deleted(), "previous FP32 cast source still live before next load"
        out = real_load(*args, **kwargs)
        leaf = next(iter(out.values()))
        subpath = kwargs.get("subpath") or ""
        name = next(iter(out))
        key = "/".join(part for part in (subpath, name) if part)
        prior_cast_source = leaf if leaf.dtype == jnp.float32 and target_specs[key][1] == "bfloat16" else None
        seen += 1
        return out

    monkeypatch.setattr(carried_restore, "load_checkpoint", tracked_load)
    restore_1005_carried_state(target, str(path), mesh=cpu_mesh, expected_step=161750)
    if prior_cast_source is not None:
        assert prior_cast_source.is_deleted()
    assert seen == len(read_manifest(str(path)).arrays)


def test_moment_cast_audit_detects_nonzero_to_zero(cpu_mesh, caplog):
    source = jax.device_put(
        np.asarray([1.25, -0.75, 0.0], dtype=np.float32),
        jax.sharding.NamedSharding(cpu_mesh, jax.sharding.PartitionSpec()),
    )
    good = source.astype(jnp.bfloat16)
    with caplog.at_level("INFO"):
        carried_restore._audit_moment_cast("opt_state/adam/nu/weight", source, good)
    assert "nonzero_to_zero=0" in caplog.text
    assert "scope=global_sharded_tensor" in caplog.text
    wrong = good.at[1].set(0)
    with pytest.raises(ValueError, match="nonzero FP32 moment became BF16 zero"):
        carried_restore._audit_moment_cast("opt_state/adam/nu/weight", source, wrong)


def test_global_moment_audit_with_sharded_and_replicated_cpu_arrays(caplog):
    if len(jax.devices()) < 2:
        pytest.skip("requires two CPU devices")
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("data",), axis_types=(jax.sharding.AxisType.Explicit,))
    vals = np.array([1.1, 2.2, 0.0, -4.4], dtype=np.float32)
    for spec in (jax.sharding.PartitionSpec("data"), jax.sharding.PartitionSpec()):
        source = jax.device_put(vals, jax.sharding.NamedSharding(mesh, spec))
        converted = source.astype(jnp.bfloat16)
        stats = jax.device_get(carried_restore._device_moment_stats(source, converted))
        assert int(np.asarray(stats[3]).sum()) == 3
        assert int(np.asarray(stats[4]).sum()) == 0
        with caplog.at_level("INFO"):
            carried_restore._audit_moment_cast("opt_state/adam/nu/weight", source, converted)
        assert "elements=4 nonzero=3 nonzero_to_zero=0" in caplog.text


def test_moment_audit_rejects_nonfinite_and_bad_path_coverage(cpu_mesh, tmp_path, monkeypatch):
    sharding = jax.sharding.NamedSharding(cpu_mesh, jax.sharding.PartitionSpec())
    source = jax.device_put(np.asarray([1.0, np.nan], dtype=np.float32), sharding)
    with pytest.raises(ValueError, match="nonfinite FP32 moment"):
        carried_restore._audit_moment_cast("opt_state/adam/nu/weight", source, source.astype(jnp.bfloat16))
    target, _ = make_state()
    path = tmp_path / "checkpoint"
    save_checkpoint(populated_source(target), step=161750, checkpoint_path=path, is_temporary=False)
    monkeypatch.setattr(carried_restore, "_EXPECTED_MOMENT_LEAVES", 9)
    with pytest.raises(ValueError, match="moment path coverage mismatch"):
        restore_1005_carried_state(restore_template_from(target), str(path), mesh=cpu_mesh, expected_step=161750)


def test_host_reference_and_device_audit_agree(cpu_mesh, caplog):
    sharding = jax.sharding.NamedSharding(cpu_mesh, jax.sharding.PartitionSpec())
    source = jax.device_put(np.asarray([1.1, -2.2, 0.0, 4.4], dtype=np.float32), sharding)
    converted = source.astype(jnp.bfloat16)
    with caplog.at_level("INFO"):
        carried_restore._audit_moment_cast_reference("opt_state/adam/nu/weight", source, converted, chunk_elements=2)
        carried_restore._audit_moment_cast("opt_state/adam/nu/weight", source, converted)
    assert caplog.text.count("nonzero=3 nonzero_to_zero=0") == len(jax.devices()) + 1
    stats = jax.device_get(carried_restore._device_moment_stats(source, converted))
    actual_min = float(np.asarray(stats[6], dtype=np.uint32).view(np.float32))
    expected_min = float(np.min(np.abs(np.asarray(source))[np.asarray(source) != 0]))
    expected_max = float(
        np.max(
            np.abs(np.asarray(converted, dtype=np.float64) - np.asarray(source, dtype=np.float64))[
                np.asarray(source) != 0
            ]
            / np.abs(np.asarray(source, dtype=np.float64))[np.asarray(source) != 0]
        )
    )
    assert actual_min == expected_min
    np.testing.assert_allclose(float(stats[7]), expected_max, rtol=1e-6)
