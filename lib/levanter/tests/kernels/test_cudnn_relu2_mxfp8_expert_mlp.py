# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""The cudnn-frontend MXFP8 ReLU^2 expert MLP against the ragged backend's expert-MLP contract, a float32 MLP, and
the QuACK MXFP8 MLP it can replace (SM100 only).

Each check feeds an expert MLP the receiver buffer the ragged backend lays out for its receiver alignment: each
expert's rows start at a multiple of it, and every row that was not received, in the gaps between experts or past
the last, is unspecified, NaN here. The active rows depend only on the seed, so MLPs that take different receiver
alignments see the same rows. References run in their own jitted stages: inside one XLA computation the
excess-precision rewrite may drop an f32 -> bf16 -> f32 round trip, which would compare against an unrounded
reference.
"""

import functools
import importlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug._moe.ep_ragged_all_to_all import _CuteRelu2Mxfp8WgradExpertMlp

_HIGHEST = jax.lax.Precision.HIGHEST
# Relative L2 error of an MXFP8 expert MLP: test_quack_mxfp8_expert_mlp.py's MXFP8 gate. MXFP8 keeps three
# mantissa bits of every GEMM operand.
_MXFP8_TOLERANCE = 0.1
# The scale of the expert-output cotangent under a mean-reduced loss without loss scaling at the hero's batch.
_TINY_COTANGENT_SCALE = 1e-9


class _Problem(NamedTuple):
    hidden: int
    intermediate: int
    capacity: int
    active_group_sizes: tuple[int, ...]

    @property
    def active_rows(self) -> int:
        return sum(self.active_group_sizes)


_SMALL_PROBLEMS = [
    # test_quack_mxfp8_expert_mlp.py's groups: the middle expert gets no rows, the others end off the 128-row
    # scale tiles, and 133 rows of the packed buffer go unused.
    _Problem(hidden=256, intermediate=512, capacity=97 + 259 + 133, active_group_sizes=(97, 0, 259)),
    # H and I end inside a 256-wide N tile of every GEMM, and the last expert gets no rows.
    _Problem(hidden=384, intermediate=640, capacity=1300, active_group_sizes=(700, 512, 0)),
]
# One expert chunk at the Belay hero's scale: H = 3072, I = 4608, 3 of the 6 local experts.
_HERO_PROBLEM = _Problem(
    hidden=3072, intermediate=4608, capacity=344_064, active_group_sizes=(150_000, 110_000, 84_000)
)
_SMALL = [
    pytest.param(problem, id=f"h{problem.hidden}-i{problem.intermediate}-e{len(problem.active_group_sizes)}")
    for problem in _SMALL_PROBLEMS
]
_WITH_HERO = [*_SMALL, pytest.param(_HERO_PROBLEM, id="hero-chunk", marks=pytest.mark.slow)]


def _require_cudnn_mxfp8_expert_mlp():
    if jax.default_backend() != "gpu":
        pytest.skip("the cudnn-frontend MXFP8 expert MLP requires an SM100 GPU")
    if int(float(jax.devices("gpu")[0].compute_capability)) != 10:
        pytest.skip("the cudnn-frontend MXFP8 expert MLP requires SM100")
    pytest.importorskip("cutlass")
    pytest.importorskip("cudnn")
    pytest.importorskip("quack")
    return importlib.import_module("levanter.grug._moe.mxfp8.relu2_expert_mlp")._CudnnRelu2Mxfp8ExpertMlp()


class _Inputs(NamedTuple):
    x: jax.Array  # [R, H] bf16
    w_up: jax.Array  # [E, H, I] bf16
    w_down: jax.Array  # [E, I, H] bf16
    cotangent: jax.Array  # [R, H] bf16
    physical_group_sizes: jax.Array
    active_group_sizes: jax.Array
    received: jax.Array  # [active rows] int32: the receiver row of each active row, experts in order


def _round_up(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


def _inputs(problem: _Problem, alignment: int, seed: int, *, unreceived: float = float("nan")) -> _Inputs:
    """Random operands in the receiver layout for ``alignment``, ``unreceived`` in every other row."""
    experts = len(problem.active_group_sizes)
    hidden, intermediate = problem.hidden, problem.intermediate
    kx, kup, kdown, kct = jax.random.split(jax.random.key(seed), 4)
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

    w_up = (jax.random.normal(kup, (experts, hidden, intermediate), jnp.float32) / hidden**0.5).astype(jnp.bfloat16)
    w_down = (jax.random.normal(kdown, (experts, intermediate, hidden), jnp.float32) / intermediate**0.5).astype(
        jnp.bfloat16
    )
    return _Inputs(
        receiver_buffer(kx),
        w_up,
        w_down,
        receiver_buffer(kct),
        jnp.asarray(spans, jnp.int32),
        jnp.asarray(problem.active_group_sizes, jnp.int32),
        received,
    )


class _Results(NamedTuple):
    out: jax.Array
    dx: jax.Array
    dw_up: jax.Array
    dw_down: jax.Array
    output_dot: jax.Array


def _forward_backward(expert_mlp, inputs: _Inputs) -> _Results:
    out, residuals = expert_mlp.forward(
        inputs.x, inputs.w_up, inputs.w_down, inputs.physical_group_sizes, inputs.active_group_sizes
    )
    return _Results(out, *expert_mlp.backward(residuals, inputs.cotangent))


def _forward_backward_parts(expert_mlp, inputs: _Inputs) -> _Results:
    out, residuals = expert_mlp.forward(
        inputs.x, inputs.w_up, inputs.w_down, inputs.physical_group_sizes, inputs.active_group_sizes
    )
    dx, output_dot, wgrad_args, wgrad = expert_mlp.backward_parts(residuals, inputs.cotangent)
    return _Results(out, dx, *wgrad(*wgrad_args), output_dot)


def _run(expert_mlp, inputs: _Inputs) -> _Results:
    return jax.jit(functools.partial(_forward_backward, expert_mlp))(inputs)


def _active(inputs: _Inputs, results: _Results) -> _Results:
    """The results the contract specifies: the row outputs' active rows, and both weight gradients."""
    received = inputs.received
    return results._replace(
        out=results.out[received], dx=results.dx[received], output_dot=results.output_dot[received]
    )


def _bits(a: jax.Array) -> jax.Array:
    unsigned = {1: jnp.uint8, 2: jnp.uint16, 4: jnp.uint32}[a.dtype.itemsize]
    return jax.lax.bitcast_convert_type(a, unsigned)


def _differing(a: _Results, b: _Results) -> list[str]:
    return [
        name
        for name, x, y in zip(_Results._fields, a, b, strict=True)
        if not bool(jnp.array_equal(_bits(x), _bits(y)))
    ]


def _relative_error(actual: jax.Array, expected: jax.Array) -> float:
    actual = actual.astype(jnp.float32)
    expected = expected.astype(jnp.float32)
    return float(jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected))


def _relative_errors(actual: _Results, expected: _Results) -> dict[str, float]:
    return {name: _relative_error(a, e) for name, a, e in zip(_Results._fields, actual, expected, strict=True)}


def _reference_mlp(group_sizes: tuple[int, ...], x, w_up, w_down):
    """Each expert's float32 ReLU^2 MLP over its rows of the packed active rows."""
    outputs, start = [], 0
    for expert, size in enumerate(group_sizes):
        pre = jnp.matmul(x[start : start + size], w_up[expert], precision=_HIGHEST)
        outputs.append(jnp.matmul(jnp.square(jax.nn.relu(pre)), w_down[expert], precision=_HIGHEST))
        start += size
    return jnp.concatenate(outputs)


def _reference_forward_backward(problem: _Problem, inputs: _Inputs) -> _Results:
    """The float32 MLP's output and gradients on the active rows, and each row's ``<y, dy>``."""
    out, pullback = jax.vjp(
        functools.partial(_reference_mlp, problem.active_group_sizes),
        inputs.x[inputs.received].astype(jnp.float32),
        inputs.w_up.astype(jnp.float32),
        inputs.w_down.astype(jnp.float32),
    )
    cotangent = inputs.cotangent[inputs.received].astype(jnp.float32)
    return _Results(out, *pullback(cotangent), jnp.sum(out * cotangent, axis=-1))


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _SMALL)
def test_cudnn_relu2_mxfp8_expert_mlp_keeps_unreceived_rows_out_of_active_outputs_and_weight_gradients(problem):
    expert_mlp = _require_cudnn_mxfp8_expert_mlp()
    unspecified = _inputs(problem, expert_mlp.receiver_alignment, seed=5)
    zeroed = _inputs(problem, expert_mlp.receiver_alignment, seed=5, unreceived=0.0)

    differing = _differing(
        _active(unspecified, _run(expert_mlp, unspecified)), _active(zeroed, _run(expert_mlp, zeroed))
    )

    assert not differing, f"NaN in the gaps between experts and past the last changed {differing}"


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _WITH_HERO)
def test_cudnn_relu2_mxfp8_expert_mlp_matches_float32_mlp_and_quack_mxfp8_within_mxfp8_error(problem):
    expert_mlp = _require_cudnn_mxfp8_expert_mlp()
    quack = _CuteRelu2Mxfp8WgradExpertMlp()
    inputs = _inputs(problem, expert_mlp.receiver_alignment, seed=0)
    packed_inputs = _inputs(problem, quack.receiver_alignment, seed=0)

    cudnn = _active(inputs, _run(expert_mlp, inputs))
    quack_mxfp8 = _active(packed_inputs, _run(quack, packed_inputs))
    reference = jax.jit(functools.partial(_reference_forward_backward, problem))(inputs)
    errors = {
        "cudnn vs f32": _relative_errors(cudnn, reference),
        "cudnn vs quack": _relative_errors(cudnn, quack_mxfp8),
        "quack vs f32": _relative_errors(quack_mxfp8, reference),
    }
    for against, by_output in errors.items():
        print(f"{problem} {against}: " + ", ".join(f"{name}={error:.3e}" for name, error in by_output.items()))

    failures = {
        f"{against} {name}": error
        for against in ("cudnn vs f32", "cudnn vs quack")
        for name, error in errors[against].items()
        if not error < _MXFP8_TOLERANCE
    }
    assert not failures, f"over the MXFP8 tolerance {_MXFP8_TOLERANCE}: {failures}"


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _WITH_HERO)
def test_cudnn_relu2_mxfp8_expert_mlp_is_bitwise_identical_across_reruns(problem):
    expert_mlp = _require_cudnn_mxfp8_expert_mlp()
    inputs = _inputs(problem, expert_mlp.receiver_alignment, seed=2)
    run = jax.jit(functools.partial(_forward_backward, expert_mlp))

    first = _active(inputs, run(inputs))
    for rerun in range(3):
        differing = _differing(first, _active(inputs, run(inputs)))
        assert not differing, f"rerun {rerun + 1} changed {differing}"


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _SMALL)
def test_cudnn_relu2_mxfp8_expert_mlp_backward_parts_compose_to_backward_bitwise(problem):
    # The ragged backend's overlap_weight_grads path runs the weight gradients from backward_parts after
    # sending dx; it must train exactly as the plain backward does.
    expert_mlp = _require_cudnn_mxfp8_expert_mlp()
    inputs = _inputs(problem, expert_mlp.receiver_alignment, seed=3)

    whole = _run(expert_mlp, inputs)
    parts = jax.jit(functools.partial(_forward_backward_parts, expert_mlp))(inputs)

    differing = _differing(_active(inputs, whole), _active(inputs, parts))
    assert not differing, f"backward_parts changed {differing}"


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _SMALL)
def test_cudnn_relu2_mxfp8_expert_mlp_gives_experts_without_rows_exactly_zero_weight_gradients(problem):
    expert_mlp = _require_cudnn_mxfp8_expert_mlp()

    results = _run(expert_mlp, _inputs(problem, expert_mlp.receiver_alignment, seed=4))

    for expert, size in enumerate(problem.active_group_sizes):
        if size == 0:
            assert not np.any(np.asarray(results.dw_up[expert])), f"expert {expert} has no rows but nonzero dw_up"
            assert not np.any(np.asarray(results.dw_down[expert])), f"expert {expert} has no rows but nonzero dw_down"


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _SMALL)
def test_cudnn_relu2_mxfp8_expert_mlp_keeps_mxfp8_error_for_cotangents_near_1e9(problem):
    # Every quantization scale must follow a 1e-9 cotangent down. With a floor on the block amax such as
    # TileKernels' 1e-4, dy's blocks would take scale 2^-22 and land below E4M3's normal range, an error of
    # tens of percent in every gradient and in the row dot.
    expert_mlp = _require_cudnn_mxfp8_expert_mlp()
    quack = _CuteRelu2Mxfp8WgradExpertMlp()

    def tiny_cotangent(inputs):
        cotangent = (inputs.cotangent.astype(jnp.float32) * _TINY_COTANGENT_SCALE).astype(jnp.bfloat16)
        return inputs._replace(cotangent=cotangent)

    inputs = tiny_cotangent(_inputs(problem, expert_mlp.receiver_alignment, seed=6))
    packed_inputs = tiny_cotangent(_inputs(problem, quack.receiver_alignment, seed=6))

    cudnn = _active(inputs, _run(expert_mlp, inputs))
    quack_mxfp8 = _active(packed_inputs, _run(quack, packed_inputs))
    reference = jax.jit(functools.partial(_reference_forward_backward, problem))(inputs)
    gradients = ("dx", "dw_up", "dw_down", "output_dot")
    errors = {
        against: {name: error for name, error in _relative_errors(cudnn, other).items() if name in gradients}
        for against, other in (("cudnn vs f32", reference), ("cudnn vs quack", quack_mxfp8))
    }
    errors["quack vs f32"] = {
        name: error for name, error in _relative_errors(quack_mxfp8, reference).items() if name in gradients
    }
    for against, by_output in errors.items():
        print(f"{problem} tiny cotangent {against}: " + ", ".join(f"{n}={e:.3e}" for n, e in by_output.items()))

    failures = {
        f"{against} {name}": error
        for against in ("cudnn vs f32", "cudnn vs quack")
        for name, error in errors[against].items()
        if not error < _MXFP8_TOLERANCE
    }
    assert not failures, f"over the MXFP8 tolerance {_MXFP8_TOLERANCE}: {failures}"
