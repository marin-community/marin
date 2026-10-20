# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate for the fused ``GatedNorm(RMSNorm(x))``.

The op restructures the reference's backward, so every backend is held to the reference itself:
in fp32 to within fp32 reassociation, and in bf16 to an error against an fp32 evaluation no larger
than the bf16 reference's own. The XLA backend runs everywhere; the Triton backward kernels run only
on a GPU, so their tests skip elsewhere. The ``shard_map`` wrapper runs on CPU devices.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.kernels.triton.gated_norm import gated_rms_norm, gated_rms_norm_reference
from levanter.testing.cpu_devices import run_on_cpu_devices

EPS = 1e-5
OUTPUTS = ("y", "dx", "d_rms_weight", "d_w_down", "d_w_up")


def _inputs(tokens, hidden, rank, dtype, seed):
    rng = np.random.default_rng(seed)
    arrays = (
        rng.standard_normal((tokens, hidden)),
        rng.standard_normal((tokens, hidden)),
        # A non-unit RMS weight, and fan-in-scaled bottleneck weights so the gate spans the sigmoid.
        1.0 + 0.1 * rng.standard_normal(hidden),
        rng.standard_normal((hidden, rank)) / np.sqrt(hidden),
        rng.standard_normal((rank, hidden)) / np.sqrt(rank),
    )
    return tuple(jnp.asarray(a, dtype) for a in arrays)


def _value_and_grads(fn, x, dy, rms_weight, w_down, w_up):
    with jax.default_matmul_precision("highest"):
        y, vjp = jax.vjp(jax.jit(fn), x, rms_weight, w_down, w_up)
        return tuple(np.asarray(a, np.float64) for a in (y, *vjp(dy)))


def _reference(x, rms_weight, w_down, w_up):
    return gated_rms_norm_reference(x, rms_weight, w_down, w_up, EPS)


def _fused(implementation):
    def fn(x, rms_weight, w_down, w_up):
        return gated_rms_norm(x, rms_weight, w_down, w_up, eps=EPS, implementation=implementation)

    return fn


def _max_relative_error(got, want) -> float:
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


def _rel_rms(got, want) -> float:
    return float(np.sqrt(np.mean((got - want) ** 2)) / np.sqrt(np.mean(want**2)))


def _requires_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("requires the JAX GPU backend")


def _assert_fp32_parity(implementation, shape):
    inputs = _inputs(*shape, jnp.float32, seed=11)
    got = _value_and_grads(_fused(implementation), *inputs)
    want = _value_and_grads(_reference, *inputs)
    for name, g, w in zip(OUTPUTS, got, want, strict=True):
        assert _max_relative_error(g, w) < 1e-5, f"{name}: max error {_max_relative_error(g, w):.2e} of max |ref|"


def _assert_bf16_no_worse_than_reference(implementation, shape):
    """Against an fp32 evaluation, the bf16 op is at least as accurate as the bf16 reference."""
    inputs = _inputs(*shape, jnp.bfloat16, seed=13)
    exact = _value_and_grads(_reference, *(a.astype(jnp.float32) for a in inputs))
    got = _value_and_grads(_fused(implementation), *inputs)
    reference = _value_and_grads(_reference, *inputs)
    for name, g, r, e in zip(OUTPUTS, got, reference, exact, strict=True):
        assert _rel_rms(g, e) <= _rel_rms(r, e), f"{name}: {_rel_rms(g, e):.2e} vs reference {_rel_rms(r, e):.2e}"


@pytest.mark.parametrize("shape", [(128, 256, 32), (64, 384, 128)], ids=lambda s: "x".join(map(str, s)))
def test_xla_backend_matches_the_reference_in_fp32(shape):
    _assert_fp32_parity("xla", shape)


@pytest.mark.parametrize("shape", [(256, 256, 32), (128, 512, 128)], ids=lambda s: "x".join(map(str, s)))
def test_xla_backend_is_no_less_accurate_than_the_reference_in_bf16(shape):
    _assert_bf16_no_worse_than_reference("xla", shape)


@pytest.mark.parametrize("shape", [(128, 256, 32), (192, 640, 128)], ids=lambda s: "x".join(map(str, s)))
def test_triton_backend_matches_the_reference_in_fp32_on_gpu(shape):
    """Several row and column tiles, so the kernels' partial sums are summed across tiles."""
    _requires_gpu()
    _assert_fp32_parity("triton_gpu", shape)


@pytest.mark.parametrize("shape", [(256, 256, 32), (512, 1024, 128)], ids=lambda s: "x".join(map(str, s)))
def test_triton_backend_is_no_less_accurate_than_the_reference_in_bf16_on_gpu(shape):
    _requires_gpu()
    _assert_bf16_no_worse_than_reference("triton_gpu", shape)


def test_default_implementation_on_gpu_runs_the_triton_kernels():
    _requires_gpu()
    x, dy, rms_weight, w_down, w_up = _inputs(128, 256, 32, jnp.bfloat16, seed=3)

    def backward(x, dy):
        _, vjp = jax.vjp(lambda xx: gated_rms_norm(xx, rms_weight, w_down, w_up, eps=EPS), x)
        return vjp(dy)

    hlo = jax.jit(backward).lower(x, dy).as_text()
    assert hlo.count("triton_kernel_call") == 2, "expected the dg and dx kernels in the backward"


def test_triton_implementation_fails_fast_where_it_cannot_run():
    """Off GPU the backend is missing; on GPU a hidden size that is not a multiple of 128 is unsupported."""
    hidden = 96 if jax.default_backend() == "gpu" else 128
    x, _, rms_weight, w_down, w_up = _inputs(64, hidden, 32, jnp.float32, seed=5)
    with pytest.raises(RuntimeError, match="triton_gpu"):
        gated_rms_norm(x, rms_weight, w_down, w_up, eps=EPS, implementation="triton_gpu")


def test_default_implementation_on_gpu_warns_and_uses_xla_for_unsupported_shapes():
    _requires_gpu()
    x, _, rms_weight, w_down, w_up = _inputs(64, 96, 32, jnp.float32, seed=5)
    with pytest.warns(UserWarning, match="triton_gpu"):
        got = gated_rms_norm(x, rms_weight, w_down, w_up, eps=EPS)
    want = gated_rms_norm(x, rms_weight, w_down, w_up, eps=EPS, implementation="xla")
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


_SHARDED_SCRIPT = """
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.kernels.triton.gated_norm import gated_rms_norm, gated_rms_norm_reference

EPS = 1e-5
rng = np.random.default_rng(7)
x = jnp.asarray(rng.standard_normal((4, 16, 64)), jnp.float32)
dy = jnp.asarray(rng.standard_normal((4, 16, 64)), jnp.float32)
rms_weight = jnp.asarray(1.0 + 0.1 * rng.standard_normal(64), jnp.float32)
w_down = jnp.asarray(rng.standard_normal((64, 16)) / 8.0, jnp.float32)
w_up = jnp.asarray(rng.standard_normal((16, 64)) / 4.0, jnp.float32)
mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2, 2), ("data", "context", "model"), axis_types=(AxisType.Explicit,) * 3)


def value_and_grads(fn, x, dy):
    y, vjp = jax.vjp(jax.jit(fn), x, rms_weight, w_down, w_up)
    return [np.asarray(a, np.float64) for a in (y, *vjp(dy))]


want = value_and_grads(lambda *a: gated_rms_norm_reference(*a, EPS), x, dy)
with jax.set_mesh(mesh):
    # Batch and sequence both sharded, so each weight gradient sums over four shards.
    layout = NamedSharding(mesh, P("data", "context", None))
    fused = lambda *a: gated_rms_norm(*a, eps=EPS, implementation="xla")  # noqa: E731
    got = value_and_grads(fused, jax.device_put(x, layout), jax.device_put(dy, layout))
    for name, g, w in zip(("y", "dx", "d_rms_weight", "d_w_down", "d_w_up"), got, want, strict=True):
        error = np.max(np.abs(g - w)) / np.max(np.abs(w))
        assert error < 1e-5, f"{name}: max error {error:.2e} of max |ref|"

    hidden_sharded = jax.device_put(x, NamedSharding(mesh, P("data", None, "model")))
    try:
        gated_rms_norm(hidden_sharded, rms_weight, w_down, w_up, eps=EPS)
    except ValueError as error:
        assert "hidden axis" in str(error), error
    else:
        raise AssertionError("a sharded hidden axis must be rejected rather than all-gathered")
"""


def test_sharded_op_matches_the_unsharded_reference_and_rejects_a_sharded_hidden_axis():
    run_on_cpu_devices(_SHARDED_SCRIPT, device_count=8)
