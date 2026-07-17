# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for threading a stateless whole-expert-MLP op through the EP MoE backends."""

from dataclasses import dataclass

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug._moe.common import MoeRaggedDotOps
from levanter.grug.grug_moe import moe_mlp
from levanter.utils.activation import ActivationFunctionEnum


@dataclass(frozen=True)
class _RefExpertMlpOp:
    """Stateless reference op: the default backend body as one call.

    Reproduces exactly what the EP backends compute between dispatch and
    combine (ragged w13, concat gate/up split, silu, ragged w2), so routing an
    op through the backend must be output-identical to the default path.
    """

    def __call__(self, x, w13, w2, group_sizes):
        h = jax.lax.ragged_dot(x, w13, group_sizes)
        gate, up = jnp.split(h, [w2.shape[1]], axis=-1)
        return jax.lax.ragged_dot(jax.nn.silu(gate) * up, w2, group_sizes)


@dataclass(frozen=True)
class _ScaledExpertMlpOp:
    """Marker op that visibly changes the output, to prove the op actually runs."""

    scale: float

    def __call__(self, x, w13, w2, group_sizes):
        return _RefExpertMlpOp()(x * self.scale, w13, w2, group_sizes)


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
def test_moe_mlp_with_reference_expert_op_matches_default(impl):
    # The op replaces the entire GEMM->activation->GEMM body; a faithful
    # reference implementation must therefore reproduce the default path
    # bit-for-bit up to float32 roundoff (same dispatch, same contraction).
    if impl == "ragged_all_to_all" and jax.default_backend() != "gpu":
        pytest.skip("ragged_all_to_all has no CPU lowering")
    mesh = _ep_mesh_or_skip()
    x, sel, cw, w13, w2 = _moe_inputs(mesh)
    with jax.set_mesh(mesh):
        y_default = moe_mlp(x, sel, cw, w13, w2, implementation=impl, mesh=mesh)
        y_op = moe_mlp(x, sel, cw, w13, w2, implementation=impl, mesh=mesh, expert_mlp_op=_RefExpertMlpOp())
    np.testing.assert_allclose(np.asarray(y_op), np.asarray(y_default), rtol=1e-5, atol=1e-6)


def test_moe_mlp_expert_op_actually_runs(impl="ring"):
    # Guards against a silent fall-through to the default path: an op that
    # scales its input must change the output.
    mesh = _ep_mesh_or_skip()
    x, sel, cw, w13, w2 = _moe_inputs(mesh)
    with jax.set_mesh(mesh):
        y_default = moe_mlp(x, sel, cw, w13, w2, implementation=impl, mesh=mesh)
        y_scaled = moe_mlp(x, sel, cw, w13, w2, implementation=impl, mesh=mesh, expert_mlp_op=_ScaledExpertMlpOp(2.0))
    assert not np.allclose(np.asarray(y_scaled), np.asarray(y_default))


def test_moe_mlp_expert_op_grads_flow_to_weights_and_x(impl="ring"):
    # The op's custom differentiation must reach the expert weights and the
    # activations through the shard_map boundary (this is the wiring the MXFP8
    # custom_vjp relies on).
    mesh = _ep_mesh_or_skip()
    x, sel, cw, w13, w2 = _moe_inputs(mesh)

    def loss(x_, w13_, w2_):
        out = moe_mlp(x_, sel, cw, w13_, w2_, implementation=impl, mesh=mesh, expert_mlp_op=_RefExpertMlpOp())
        return jnp.sum(out * out)

    def loss_default(x_, w13_, w2_):
        out = moe_mlp(x_, sel, cw, w13_, w2_, implementation=impl, mesh=mesh)
        return jnp.sum(out * out)

    with jax.set_mesh(mesh):
        gx, gw13, gw2 = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(x, w13, w2)
        rx, rw13, rw2 = jax.jit(jax.grad(loss_default, argnums=(0, 1, 2)))(x, w13, w2)
    np.testing.assert_allclose(np.asarray(gx), np.asarray(rx), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(gw13), np.asarray(rw13), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(gw2), np.asarray(rw2), rtol=1e-4, atol=1e-6)


def test_moe_mlp_rejects_expert_op_without_expert_axis():
    # Contract: the op is only wired into the EP backends; a silent bf16
    # fallback would defeat the point of requesting fused quantized kernels.
    devices = jax.devices()
    mesh_devices = np.array(devices).reshape(len(devices), 1)
    mesh = Mesh(mesh_devices, axis_names=("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    x, sel, cw, w13, w2 = _moe_inputs(mesh, batch=P("data"))
    with jax.set_mesh(mesh), pytest.raises(NotImplementedError, match="expert_mlp_op"):
        moe_mlp(x, sel, cw, w13, w2, implementation="ring", mesh=mesh, expert_mlp_op=_RefExpertMlpOp())


def test_moe_mlp_rejects_expert_op_with_ragged_dot_ops():
    mesh = _ep_mesh_or_skip()
    x, sel, cw, w13, w2 = _moe_inputs(mesh)
    ops = MoeRaggedDotOps(w13=_RefExpertMlpOp(), w2=_RefExpertMlpOp())  # values irrelevant
    with jax.set_mesh(mesh), pytest.raises(ValueError, match="mutually exclusive"):
        moe_mlp(x, sel, cw, w13, w2, implementation="ring", mesh=mesh, ragged_dot_ops=ops, expert_mlp_op=_RefExpertMlpOp())


def test_moe_mlp_rejects_expert_op_with_non_silu_activation():
    mesh = _ep_mesh_or_skip()
    x, sel, cw, w13, w2 = _moe_inputs(mesh)
    with jax.set_mesh(mesh), pytest.raises(ValueError, match="activation"):
        moe_mlp(
            x,
            sel,
            cw,
            w13,
            w2,
            implementation="ring",
            mesh=mesh,
            expert_mlp_op=_RefExpertMlpOp(),
            activation=ActivationFunctionEnum.gelu,
        )
