# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for threading stateful ragged-dot ops through the EP MoE backends."""

import equinox as eqx
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from haliax.quantization import OverwriteWithGradient
from levanter.grug._moe.common import MoeRaggedDotOps
from levanter.grug._moe.ep_common import _pmax_replicated_cotangent
from levanter.grug.grug_moe import moe_mlp


class _ScaledRaggedDotOp(eqx.Module):
    """Stateful stand-in for a quantized op: scales the activations by a carried array."""

    scale: jnp.ndarray

    def __call__(self, lhs, rhs, group_sizes):
        return jax.lax.ragged_dot(lhs * self.scale, rhs, group_sizes)


@jax.custom_vjp
def _amax_tracking_ragged_dot(amax, lhs, rhs, group_sizes):
    del amax
    return jax.lax.ragged_dot(lhs, rhs, group_sizes)


def _amax_tracking_fwd(amax, lhs, rhs, group_sizes):
    del amax
    return jax.lax.ragged_dot(lhs, rhs, group_sizes), (lhs, rhs, group_sizes)


def _amax_tracking_bwd(res, ct):
    lhs, rhs, group_sizes = res
    del ct
    # The amax "gradient" is the new state (max |lhs| seen this step), mirroring
    # how delayed-scaling ops emit their OverwriteWithGradient state updates.
    d_amax = jnp.max(jnp.abs(lhs))
    zero_group_sizes = np.zeros(group_sizes.shape, dtype=jax.dtypes.float0)
    return d_amax, jnp.zeros_like(lhs), jnp.zeros_like(rhs), zero_group_sizes


_amax_tracking_ragged_dot.defvjp(_amax_tracking_fwd, _amax_tracking_bwd)


class _AmaxRaggedDotOp(OverwriteWithGradient):
    """Overwrite-state stand-in: its state cotangent is the local amax of lhs."""

    amax: jnp.ndarray

    def __call__(self, lhs, rhs, group_sizes):
        return _amax_tracking_ragged_dot(self.amax, lhs, rhs, group_sizes)


def _ep_mesh_or_skip(expert: int = 2) -> Mesh:
    devices = jax.devices()
    if len(devices) < expert or len(devices) % expert != 0:
        pytest.skip(f"needs >= {expert} devices with expert-divisible count")
    mesh_devices = np.array(devices).reshape(len(devices) // expert, expert, 1)
    return Mesh(
        mesh_devices,
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _moe_inputs(mesh, t=64, d=32, i=16, e=16, k=4, seed=0, batch=P(("data", "expert"))):
    key = jax.random.key(seed)
    kx, ks, kw, k1, k2 = jax.random.split(key, 5)
    expert_spec = P("expert", None, None) if "expert" in mesh.shape else P(None, None, None)
    x = jax.device_put(jax.random.normal(kx, (t, d), dtype=jnp.float32), NamedSharding(mesh, batch))
    sel = jax.device_put(
        jnp.argsort(jax.random.uniform(ks, (t, e)), axis=-1)[:, :k].astype(jnp.int32),
        NamedSharding(mesh, batch),
    )
    cw = jax.device_put(jax.nn.softmax(jax.random.normal(kw, (t, k))), NamedSharding(mesh, batch))
    w13 = jax.device_put(jax.random.normal(k1, (e, d, 2 * i)) * 0.05, NamedSharding(mesh, expert_spec))
    w2 = jax.device_put(jax.random.normal(k2, (e, i, d)) * 0.05, NamedSharding(mesh, expert_spec))
    return x, sel, cw, w13, w2


@pytest.mark.parametrize("impl", ["ring", "ragged_all_to_all"])
def test_moe_mlp_with_identity_scaled_op_matches_default(impl):
    # An op with scale=1 must reproduce the default bf16 path: same contraction,
    # same dispatch — only the GEMM callable is swapped, so outputs agree to
    # float32 roundoff.
    if impl == "ragged_all_to_all" and jax.default_backend() != "gpu":
        pytest.skip("ragged_all_to_all has no CPU lowering")
    mesh = _ep_mesh_or_skip()
    x, sel, cw, w13, w2 = _moe_inputs(mesh)
    ops = MoeRaggedDotOps(w13=_ScaledRaggedDotOp(jnp.ones(1)), w2=_ScaledRaggedDotOp(jnp.ones(1)))
    with jax.set_mesh(mesh):
        y_default = moe_mlp(x, sel, cw, w13, w2, implementation=impl, mesh=mesh)
        y_ops = moe_mlp(x, sel, cw, w13, w2, implementation=impl, mesh=mesh, ragged_dot_ops=ops)
    np.testing.assert_allclose(np.asarray(y_ops), np.asarray(y_default), rtol=1e-5, atol=1e-6)


def test_moe_mlp_op_state_receives_cotangents(impl="ring"):
    # An op's ordinary trainable state must receive a cotangent through shard_map
    # (summed across shards by the transpose, like any replicated parameter).
    mesh = _ep_mesh_or_skip()
    x, sel, cw, w13, w2 = _moe_inputs(mesh)
    ops = MoeRaggedDotOps(w13=_ScaledRaggedDotOp(jnp.ones(1)), w2=_ScaledRaggedDotOp(jnp.ones(1)))

    def loss(ops_):
        out = moe_mlp(x, sel, cw, w13, w2, implementation=impl, mesh=mesh, ragged_dot_ops=ops_)
        return jnp.sum(out * out)

    with jax.set_mesh(mesh):
        grads = jax.jit(jax.grad(loss))(ops)
    assert np.isfinite(np.asarray(grads.w13.scale)).all()
    assert float(jnp.abs(grads.w13.scale).sum()) > 0.0
    assert float(jnp.abs(grads.w2.scale).sum()) > 0.0


def test_moe_mlp_overwrite_state_cotangent_is_global_amax(impl="ring"):
    # Delayed-scaling state (OverwriteWithGradient) must combine as the max over
    # shards: each op sees only locally dispatched rows, and the state cotangent
    # reaching the optimizer must be the global amax. Every token row reaches
    # some shard's w13 GEMM and padding rows are zeros, so the expected w13 amax
    # is exactly max |x|. A dropped or double-applied pmax wrapper changes this.
    mesh = _ep_mesh_or_skip()
    x, sel, cw, w13, w2 = _moe_inputs(mesh)
    ops = MoeRaggedDotOps(w13=_AmaxRaggedDotOp(jnp.zeros(())), w2=_AmaxRaggedDotOp(jnp.zeros(())))

    def loss(ops_):
        out = moe_mlp(x, sel, cw, w13, w2, implementation=impl, mesh=mesh, ragged_dot_ops=ops_)
        return jnp.sum(out)

    with jax.set_mesh(mesh):
        grads = jax.jit(jax.grad(loss))(ops)
    np.testing.assert_allclose(float(grads.w13.amax), float(jnp.max(jnp.abs(x))), rtol=1e-6)


class _OwgState(OverwriteWithGradient):
    value: jnp.ndarray


def test_pmax_replicated_cotangent_maxes_overwrite_and_sums_plain():
    # Replicated shard_map inputs get psum'ed cotangents by the transpose.
    # OverwriteWithGradient state must instead combine as the max over shards,
    # while ordinary trainable leaves keep the sum (the true total gradient).
    mesh = _ep_mesh_or_skip()
    n_shards = len(jax.devices())
    per_shard = jnp.arange(1.0, n_shards + 1.0)  # shard i contributes weight i+1
    w = jax.device_put(per_shard, NamedSharding(mesh, P(("data", "expert"))))

    def shard_fn(tree, w_local):
        tree = _pmax_replicated_cotangent(tree)
        return (tree["overwrite"].value + tree["plain"]) * w_local

    f = shard_map(
        shard_fn,
        mesh=mesh,
        in_specs=(P(), P(("data", "expert"))),
        out_specs=P(("data", "expert")),
        check_vma=False,
    )

    def loss(tree):
        return jnp.sum(f(tree, w))

    tree = {"overwrite": _OwgState(jnp.ones(())), "plain": jnp.ones(())}
    with jax.set_mesh(mesh):
        grad = jax.jit(jax.grad(loss))(tree)
    # Each shard's cotangent is w_local: overwrite state contributes pmax/n so
    # the psum reconstructs the max; the plain leaf psums to the sum.
    np.testing.assert_allclose(float(grad["overwrite"].value), float(per_shard.max()), rtol=1e-6)
    np.testing.assert_allclose(float(grad["plain"]), float(per_shard.sum()), rtol=1e-6)


def test_moe_mlp_rejects_ops_without_expert_axis():
    # Contract: ops are only wired into the EP backends; a silent bf16 fallback
    # would defeat the point of requesting quantized GEMMs.
    devices = jax.devices()
    mesh_devices = np.array(devices).reshape(len(devices), 1)
    mesh = Mesh(mesh_devices, axis_names=("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    x, sel, cw, w13, w2 = _moe_inputs(mesh, batch=P("data"))
    ops = MoeRaggedDotOps(w13=_ScaledRaggedDotOp(jnp.ones(1)), w2=_ScaledRaggedDotOp(jnp.ones(1)))
    with jax.set_mesh(mesh), pytest.raises(NotImplementedError, match="ragged_dot_ops"):
        moe_mlp(x, sel, cw, w13, w2, implementation="ring", mesh=mesh, ragged_dot_ops=ops)
