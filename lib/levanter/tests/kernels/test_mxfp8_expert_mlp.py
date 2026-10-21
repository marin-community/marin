# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""The MXFP8 expert MLP against the ragged backend's expert-MLP contract and a float32 MLP (SM100).

Each check feeds an expert MLP the receiver buffer the ragged backend lays out for its receiver
alignment: each expert's rows start at a multiple of it, and every row that was not received, in
the gaps between experts or past the last, is unspecified, NaN here. References run in their own
jitted stages. Inside one XLA computation the excess-precision rewrite may drop an f32 -> fp8 or
f32 -> bf16 -> f32 round trip, which would compare against an unrounded reference.

Programs compile with XLA GPU command buffers off, as the hero's do, except the one test that checks
CUDA-graph replays against them.
"""

import functools
import importlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug._moe.ep_ragged_all_to_all import _CuteExpertMlp, _RoutingWeightGradient
from levanter.grug._moe.mxfp8.quantize import dequantize_mxfp8, quantize_mxfp8

_HIGHEST = jax.lax.Precision.HIGHEST
# experiments/grug/moe_hero_ep/train.py: XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG.
_jit = functools.partial(jax.jit, compiler_options={"xla_gpu_enable_command_buffer": ""})
# Captures the quantization fusions and the kernels' custom calls into one CUDA graph.
_COMMAND_BUFFER_JIT = functools.partial(
    jax.jit, compiler_options={"xla_gpu_enable_command_buffer": "FUSION,CUSTOM_CALL"}
)
# MXFP8 keeps three mantissa bits of every GEMM operand, so the whole MLP lands near 7e-2 relative
# Frobenius error against a float32 MLP. The gate is the research one of
# c3cb334f8e:experiments/grug/moe/standalone/test_mxfp8_op_gpu.py.
_MXFP8_TOLERANCE = 0.1
# The row dot against float32 <h, dh> on the MXFP8 operands the kernels consumed: the research gate.
_ROW_DOT_TOLERANCE = 2e-2


class _Problem(NamedTuple):
    hidden: int
    intermediate: int
    capacity: int
    active_group_sizes: tuple[int, ...]

    @property
    def active_rows(self) -> int:
        return sum(self.active_group_sizes)


# Each small problem has one expert without rows, experts whose rows end off a 256-row boundary,
# and unused capacity at the end of the buffer.
_SMALL_PROBLEMS = [
    # A middle expert gets no rows; the dSwiGLU kernel writes two 256-column row-dot partials.
    _Problem(hidden=256, intermediate=512, capacity=1700, active_group_sizes=(300, 0, 1000, 213)),
    # The last expert gets no rows, only the layout's padding; three row-dot partials.
    _Problem(hidden=512, intermediate=768, capacity=1300, active_group_sizes=(700, 512, 0)),
]
# One hero expert chunk: H = I = 3072, three experts, a 302,336-row receiver buffer.
_HERO_PROBLEM = _Problem(
    hidden=3072, intermediate=3072, capacity=302_336, active_group_sizes=(140_000, 100_000, 62_000)
)
_SMALL = [
    pytest.param(problem, id=f"h{problem.hidden}-i{problem.intermediate}-e{len(problem.active_group_sizes)}")
    for problem in _SMALL_PROBLEMS
]
_WITH_HERO = [*_SMALL, pytest.param(_HERO_PROBLEM, id="hero-chunk", marks=pytest.mark.slow)]


def _require_mxfp8_expert_mlp():
    if jax.default_backend() != "gpu":
        pytest.skip("the MXFP8 expert MLP requires an SM100 GPU")
    if int(float(jax.devices("gpu")[0].compute_capability)) != 10:
        pytest.skip("the MXFP8 expert MLP requires SM100")
    pytest.importorskip("cutlass")
    pytest.importorskip("quack")
    expert_mlp = importlib.import_module("levanter.grug._moe.mxfp8.expert_mlp")
    return expert_mlp._Mxfp8ExpertMlp(routing_weight_gradient=_RoutingWeightGradient.EXPERT_SIDE)


class _Inputs(NamedTuple):
    x: jax.Array  # [R, H] bf16
    moe_w13: jax.Array  # [E, H, 2I] bf16, gate columns then up columns
    moe_w2: jax.Array  # [E, I, H] bf16
    cotangent: jax.Array  # [R, H] bf16
    physical_group_sizes: jax.Array
    active_group_sizes: jax.Array
    received: jax.Array  # [active rows] int32: the receiver row of each active row, experts in order


def _round_up(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


def _inputs(problem: _Problem, alignment: int, seed: int, *, unreceived: float = float("nan")) -> _Inputs:
    """Random operands in the receiver layout for ``alignment``, ``unreceived`` in every other row.

    The active rows' values depend only on ``seed``, so expert MLPs that take different receiver
    alignments see the same rows.
    """
    experts = len(problem.active_group_sizes)
    hidden, intermediate = problem.hidden, problem.intermediate
    kx, k13, k2, kct = jax.random.split(jax.random.key(seed), 4)
    # Each expert's span reaches the next multiple of the alignment, and the last takes the rest.
    rows = _round_up(problem.capacity + (alignment - 1) * experts, alignment)
    spans = [_round_up(size, alignment) for size in problem.active_group_sizes]
    spans[-1] += rows - sum(spans)
    starts = np.cumsum(spans) - spans
    received = jnp.asarray(
        np.concatenate(
            [start + np.arange(size) for start, size in zip(starts, problem.active_group_sizes, strict=True)]
        ),
        jnp.int32,
    )

    def receiver_buffer(key):
        active_rows = jax.random.normal(key, (problem.active_rows, hidden), jnp.bfloat16)
        return jnp.full((rows, hidden), unreceived, jnp.bfloat16).at[received].set(active_rows)

    moe_w13 = (jax.random.normal(k13, (experts, hidden, 2 * intermediate), jnp.float32) / hidden**0.5).astype(
        jnp.bfloat16
    )
    moe_w2 = (jax.random.normal(k2, (experts, intermediate, hidden), jnp.float32) / intermediate**0.5).astype(
        jnp.bfloat16
    )
    return _Inputs(
        receiver_buffer(kx),
        moe_w13,
        moe_w2,
        receiver_buffer(kct),
        jnp.asarray(spans, jnp.int32),
        jnp.asarray(problem.active_group_sizes, jnp.int32),
        received,
    )


class _Results(NamedTuple):
    out: jax.Array
    dx: jax.Array
    dw13: jax.Array
    dw2: jax.Array
    output_dot: jax.Array


def _forward_backward(expert_mlp, inputs: _Inputs) -> _Results:
    out, residuals = expert_mlp.forward(
        inputs.x, inputs.moe_w13, inputs.moe_w2, inputs.physical_group_sizes, inputs.active_group_sizes
    )
    return _Results(out, *expert_mlp.backward(residuals, inputs.cotangent))


def _run(expert_mlp, inputs: _Inputs) -> _Results:
    return _jit(functools.partial(_forward_backward, expert_mlp))(inputs)


def _active(inputs: _Inputs, results: _Results) -> _Results:
    """The results the contract specifies: the row outputs' active rows, and both weight gradients."""
    received = inputs.received
    return results._replace(
        out=results.out[received], dx=results.dx[received], output_dot=results.output_dot[received]
    )


def _bits(a: jax.Array) -> jax.Array:
    unsigned = {1: jnp.uint8, 2: jnp.uint16, 4: jnp.uint32}[a.dtype.itemsize]
    return jax.lax.bitcast_convert_type(a, unsigned)


def _relative_error(actual: jax.Array, expected: jax.Array) -> float:
    actual = actual.astype(jnp.float32)
    expected = expected.astype(jnp.float32)
    return float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected))


def _relative_errors(actual: _Results, expected: _Results) -> dict[str, float]:
    return {name: _relative_error(a, e) for name, a, e in zip(_Results._fields, actual, expected, strict=True)}


def _reference_mlp(group_sizes: tuple[int, ...], x, moe_w13, moe_w2):
    """Each expert's float32 SwiGLU MLP over its rows of the packed active rows."""
    intermediate = moe_w2.shape[1]
    outputs, start = [], 0
    for expert, size in enumerate(group_sizes):
        gate_up = jnp.matmul(x[start : start + size], moe_w13[expert], precision=_HIGHEST)
        h = jax.nn.silu(gate_up[:, :intermediate]) * gate_up[:, intermediate:]
        outputs.append(jnp.matmul(h, moe_w2[expert], precision=_HIGHEST))
        start += size
    return jnp.concatenate(outputs)


def _reference_forward_backward(problem: _Problem, inputs: _Inputs) -> _Results:
    """The float32 MLP's output and gradients on the active rows, and each row's ``<y, dy>``."""
    out, pullback = jax.vjp(
        functools.partial(_reference_mlp, problem.active_group_sizes),
        inputs.x[inputs.received].astype(jnp.float32),
        inputs.moe_w13.astype(jnp.float32),
        inputs.moe_w2.astype(jnp.float32),
    )
    cotangent = inputs.cotangent[inputs.received].astype(jnp.float32)
    return _Results(out, *pullback(cotangent), jnp.sum(out * cotangent, axis=-1))


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _SMALL)
def test_mxfp8_expert_mlp_keeps_unreceived_rows_out_of_active_outputs_and_weight_gradients(problem):
    expert_mlp = _require_mxfp8_expert_mlp()
    unspecified = _inputs(problem, expert_mlp.receiver_alignment, seed=5)
    zeroed = _inputs(problem, expert_mlp.receiver_alignment, seed=5, unreceived=0.0)

    differing = [
        name
        for name, a, b in zip(
            _Results._fields,
            _active(unspecified, _run(expert_mlp, unspecified)),
            _active(zeroed, _run(expert_mlp, zeroed)),
            strict=True,
        )
        if not bool(jnp.array_equal(_bits(a), _bits(b)))
    ]
    assert not differing, f"NaN in the gaps between experts and past the last changed {differing}"


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _WITH_HERO)
def test_mxfp8_expert_mlp_matches_float32_mlp_and_quack_within_mxfp8_error(problem):
    expert_mlp = _require_mxfp8_expert_mlp()
    quack = _CuteExpertMlp(routing_weight_gradient=_RoutingWeightGradient.EXPERT_SIDE)
    inputs = _inputs(problem, expert_mlp.receiver_alignment, seed=0)
    packed_inputs = _inputs(problem, quack.receiver_alignment, seed=0)

    mxfp8 = _active(inputs, _run(expert_mlp, inputs))
    bf16 = _active(packed_inputs, _run(quack, packed_inputs))
    reference = _jit(functools.partial(_reference_forward_backward, problem))(inputs)
    errors = {
        "mxfp8 vs f32": _relative_errors(mxfp8, reference),
        "mxfp8 vs quack": _relative_errors(mxfp8, bf16),
        "quack vs f32": _relative_errors(bf16, reference),
    }
    for against, by_output in errors.items():
        print(f"{problem} {against}: " + ", ".join(f"{name}={error:.3e}" for name, error in by_output.items()))

    failures = {
        f"{against} {name}": error
        for against in ("mxfp8 vs f32", "mxfp8 vs quack")
        for name, error in errors[against].items()
        if not error < _MXFP8_TOLERANCE
    }
    assert not failures, f"over the MXFP8 tolerance {_MXFP8_TOLERANCE}: {failures}"


def _row_dot_operands(inputs: _Inputs):
    """The MXFP8 operands of the row dot, as the kernels quantize them: every one along H."""
    return (
        quantize_mxfp8(inputs.x[inputs.received]),
        quantize_mxfp8(jnp.swapaxes(inputs.moe_w13, 1, 2)),
        quantize_mxfp8(inputs.cotangent[inputs.received]),
        quantize_mxfp8(inputs.moe_w2),
    )


def _pre_activation_and_hidden_cotangent(problem: _Problem, operands):
    """The bf16 gate/up pre-activation the forward kernel stores, and the float32 ``dh``."""
    (x, x_scales), (w13, w13_scales), (cotangent, cotangent_scales), (w2, w2_scales) = operands
    x = dequantize_mxfp8(x, x_scales)
    w13 = dequantize_mxfp8(w13, w13_scales)  # [E, 2I, H]
    cotangent = dequantize_mxfp8(cotangent, cotangent_scales)
    w2 = dequantize_mxfp8(w2, w2_scales)  # [E, I, H]
    pre_activation, hidden_cotangent, start = [], [], 0
    for expert, size in enumerate(problem.active_group_sizes):
        pre_activation.append(jnp.matmul(x[start : start + size], w13[expert].T, precision=_HIGHEST))
        hidden_cotangent.append(jnp.matmul(cotangent[start : start + size], w2[expert].T, precision=_HIGHEST))
        start += size
    return jnp.concatenate(pre_activation).astype(jnp.bfloat16), jnp.concatenate(hidden_cotangent)


def _row_dot(intermediate: int, pre_activation, hidden_cotangent):
    gate_up = pre_activation.astype(jnp.float32)
    h = jax.nn.silu(gate_up[:, :intermediate]) * gate_up[:, intermediate:]
    return jnp.sum(h * hidden_cotangent, axis=-1)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _WITH_HERO)
def test_mxfp8_expert_mlp_row_dot_matches_float32_h_dot_dh_on_its_operands(problem):
    expert_mlp = _require_mxfp8_expert_mlp()
    inputs = _inputs(problem, expert_mlp.receiver_alignment, seed=1)

    output_dot = _active(inputs, _run(expert_mlp, inputs)).output_dot
    operands = _jit(_row_dot_operands)(inputs)
    pre_activation, hidden_cotangent = _jit(functools.partial(_pre_activation_and_hidden_cotangent, problem))(operands)
    expected = _jit(functools.partial(_row_dot, problem.intermediate))(pre_activation, hidden_cotangent)
    error = _relative_error(output_dot, expected)
    print(f"{problem}: row dot vs f32 <h, dh> on the MXFP8 operands: {error:.3e}")

    assert error < _ROW_DOT_TOLERANCE


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _WITH_HERO)
def test_mxfp8_expert_mlp_is_bitwise_identical_across_reruns(problem):
    expert_mlp = _require_mxfp8_expert_mlp()
    inputs = _inputs(problem, expert_mlp.receiver_alignment, seed=2)
    run = _jit(functools.partial(_forward_backward, expert_mlp))

    first = run(inputs)
    for rerun in range(3):
        again = run(inputs)
        differing = [
            name
            for name, a, b in zip(_Results._fields, first, again, strict=True)
            if not bool(jnp.array_equal(_bits(a), _bits(b)))
        ]
        assert not differing, f"rerun {rerun + 1} changed {differing}"


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _WITH_HERO)
def test_mxfp8_expert_mlp_apply_returns_forwards_output_bitwise(problem):
    # `apply` runs the forward-only kernels, which must not change a byte of the output that the
    # primal of a recomputed layer passes on.
    expert_mlp = _require_mxfp8_expert_mlp()
    inputs = _inputs(problem, expert_mlp.receiver_alignment, seed=4)
    args = (inputs.x, inputs.moe_w13, inputs.moe_w2, inputs.physical_group_sizes, inputs.active_group_sizes)

    applied, _ready = _jit(expert_mlp.apply)(*args)
    forward, _residuals = _jit(expert_mlp.forward)(*args)

    assert bool(jnp.array_equal(_bits(applied[inputs.received]), _bits(forward[inputs.received])))


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _SMALL)
def test_mxfp8_expert_mlp_cuda_graph_replays_match_the_program_without_command_buffers(problem):
    """At these shapes XLA places both weight-gradient calls' descriptor workspaces at one address.

    Before the wgrad kernel acquired its online scale-factor descriptors for the tensormap proxy,
    every replay after the first execution returned a wrong, partly NaN down-projection gradient.
    """
    expert_mlp = _require_mxfp8_expert_mlp()
    inputs = _inputs(problem, expert_mlp.receiver_alignment, seed=4)
    expected = _run(expert_mlp, inputs)
    replayed = _COMMAND_BUFFER_JIT(functools.partial(_forward_backward, expert_mlp))

    # The first execution runs op by op; the graph replays follow.
    for execution in range(5):
        got = replayed(inputs)
        differing = [
            name
            for name, a, b in zip(_Results._fields, got, expected, strict=True)
            if not bool(jnp.array_equal(_bits(a), _bits(b)))
        ]
        assert not differing, f"execution {execution} changed {differing}"


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _SMALL)
def test_mxfp8_expert_mlp_gives_experts_without_rows_exactly_zero_weight_gradients(problem):
    expert_mlp = _require_mxfp8_expert_mlp()

    results = _run(expert_mlp, _inputs(problem, expert_mlp.receiver_alignment, seed=3))

    for expert, size in enumerate(problem.active_group_sizes):
        if size == 0:
            assert not np.any(np.asarray(results.dw13[expert] != 0)), f"expert {expert} has no rows but a nonzero dw13"
            assert not np.any(np.asarray(results.dw2[expert] != 0)), f"expert {expert} has no rows but a nonzero dw2"
