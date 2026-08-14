# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, reshard
from jax.sharding import PartitionSpec as P
from levanter.grug._moe import rms_gated_norm as rgn
from levanter.grug._moe.rms_gated_norm import (
    exact_gated_norm_up_reverse,
    exact_rms_backward_consumer,
    exact_rms_backward_producer_reference,
    rms_gated_norm,
)

from experiments.grug.moe_hero_fsdp import launch, train
from experiments.grug.moe_hero_fsdp.model import (
    _BATCH_AXES,
    _GATED_NORM_RANK,
    GatedNorm,
    GrugModelConfig,
    GrugMoeHfConfig,
    RMSNorm,
)

_NORM_EPS = 1e-5


class NormInputs(NamedTuple):
    x: jax.Array
    norm_weight: jax.Array
    w_down: jax.Array
    w_up: jax.Array
    cotangent: jax.Array


@pytest.fixture
def cpu_mesh():
    mesh = Mesh(
        np.asarray([jax.devices("cpu")[0]]).reshape(1, 1, 1, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    with jax.set_mesh(mesh):
        yield


def _norm_inputs(dtype: jnp.dtype):
    input_key, norm_key, down_key, up_key, cotangent_key = jax.random.split(jax.random.key(17), 5)
    x = jax.random.normal(input_key, (2, 3, 16), dtype=dtype)
    norm_weight = 1 + 0.1 * jax.random.normal(norm_key, (16,), dtype=jnp.float32)
    w_down = (0.1 * jax.random.normal(down_key, (16, _GATED_NORM_RANK), dtype=dtype)).astype(dtype)
    w_up = (0.1 * jax.random.normal(up_key, (_GATED_NORM_RANK, 16), dtype=dtype)).astype(dtype)
    cotangent = jax.random.normal(cotangent_key, x.shape, dtype=dtype)
    return NormInputs(x, norm_weight, w_down, w_up, cotangent)


def _xla_rms_gated_norm(x, norm_weight, w_down, w_up):
    return GatedNorm(w_down=w_down, w_up=w_up)(RMSNorm(weight=norm_weight, eps=_NORM_EPS)(x))


def _assert_error_below(actual, expected, *, max_threshold: float, mean_threshold: float, label: str) -> None:
    absolute_error = jnp.abs(actual.astype(jnp.float32) - expected.astype(jnp.float32))
    max_error = float(jnp.max(absolute_error))
    mean_error = float(jnp.mean(absolute_error))
    assert max_error <= max_threshold and mean_error <= mean_threshold, (
        f"{label}: max_abs_error={max_error:.8g} (threshold={max_threshold:.8g}), "
        f"mean_abs_error={mean_error:.8g} (threshold={mean_threshold:.8g})"
    )


def test_build_hero_run_uses_run_id_argument(monkeypatch):
    monkeypatch.setenv("RUN_ID", "ignored-environment-run")

    step = launch.build_hero_run(
        run_id="cli-run",
        dp_racks=1,
        num_steps=1,
        version="2026.08.01",
    )

    assert step.name == "grug/cli-run"


def test_run_grug_applies_xla_command_buffer_default_and_keeps_override(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_latency_hiding_scheduler=true")
    config = SimpleNamespace(
        trainer=SimpleNamespace(trainer=SimpleNamespace(id="test-run")),
        resources=object(),
        processes_per_task=1,
        run_mode=train.GrugRunMode.DEFAULT,
    )

    with patch.object(train, "dispatch_grug_training_run"):
        train.run_grug(config)

        assert os.environ["XLA_FLAGS"].split() == [
            "--xla_gpu_enable_latency_hiding_scheduler=true",
            train.XLA_DISABLE_GPU_COMMAND_BUFFER_FLAG,
        ]

        explicit_flags = "--xla_gpu_enable_command_buffer=FUSION"
        monkeypatch.setenv("XLA_FLAGS", explicit_flags)
        train.run_grug(config)

        assert os.environ["XLA_FLAGS"] == explicit_flags


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_fused_reverse_keeps_stock_forward_bit_identical(cpu_mesh, dtype):
    del cpu_mesh
    inputs = _norm_inputs(dtype)

    expected = _xla_rms_gated_norm(inputs.x, inputs.norm_weight, inputs.w_down, inputs.w_up)
    actual, _ = rgn._exact_forward(inputs.x, inputs.norm_weight, inputs.w_down, inputs.w_up, _NORM_EPS)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("dtype", "max_threshold", "mean_threshold"),
    [
        (jnp.float32, 2e-6, 2e-7),
        (jnp.bfloat16, 3e-2, 2e-3),
    ],
)
def test_rms_reverse_algebra_matches_stock_autodiff(cpu_mesh, dtype, max_threshold, mean_threshold):
    del cpu_mesh
    inputs = _norm_inputs(dtype)
    x_flat = inputs.x.reshape((-1, inputs.x.shape[-1]))
    cotangent = inputs.cotangent.reshape(x_flat.shape)
    inverse_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x_flat.astype(jnp.float32)), axis=-1) + _NORM_EPS)

    def rms_norm(x, weight):
        local_inverse_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1) + _NORM_EPS)
        return (x.astype(jnp.float32) * local_inverse_rms[:, None] * weight).astype(dtype)

    _, pullback = jax.vjp(rms_norm, x_flat, inputs.norm_weight)
    expected_x, expected_weight = pullback(cotangent)
    zero_rank_cotangent = jnp.zeros((x_flat.shape[0], _GATED_NORM_RANK), dtype=dtype)
    unweighted_cotangent, row_dot_partial = exact_rms_backward_producer_reference(
        zero_rank_cotangent,
        inputs.w_down,
        cotangent,
        x_flat,
        inputs.norm_weight,
        inverse_rms,
    )
    actual_x = exact_rms_backward_consumer(
        unweighted_cotangent,
        jnp.sum(row_dot_partial, axis=-1),
        x_flat,
        inputs.norm_weight,
        inverse_rms,
    )
    normalized_x = x_flat.astype(jnp.float32) * inverse_rms[:, None]
    actual_weight = jnp.sum(unweighted_cotangent.astype(jnp.float32) * normalized_x, axis=0)

    for name, actual, expected in (
        ("input", actual_x, expected_x),
        ("norm_weight", actual_weight, expected_weight),
    ):
        _assert_error_below(
            actual,
            expected,
            max_threshold=max_threshold,
            mean_threshold=mean_threshold,
            label=f"{dtype} {name} gradient",
        )


def test_gated_norm_up_reverse_matches_autodiff(cpu_mesh):
    del cpu_mesh
    inputs = _norm_inputs(jnp.bfloat16)
    _, residuals = rgn._exact_forward(inputs.x, inputs.norm_weight, inputs.w_down, inputs.w_up, _NORM_EPS)

    def output_gate(normalized, gate_hidden, w_up):
        gate = jax.nn.sigmoid(jnp.einsum("tr,rd->td", gate_hidden, w_up))
        return normalized * gate

    _, pullback = jax.vjp(output_gate, residuals.normalized, residuals.gate_hidden, residuals.w_up)
    expected_direct, expected_gate_hidden, expected_w_up = pullback(inputs.cotangent.reshape(residuals.normalized.shape))
    actual = exact_gated_norm_up_reverse(inputs.cotangent, residuals)
    actual_gate_hidden = jnp.einsum("td,rd->tr", actual.gate_accumulator, residuals.w_up)

    np.testing.assert_array_equal(actual.direct, expected_direct)
    np.testing.assert_array_equal(actual_gate_hidden, expected_gate_hidden)
    np.testing.assert_array_equal(actual.w_up, expected_w_up)


@pytest.mark.parametrize(
    ("dtype", "max_threshold", "mean_threshold"),
    [
        (jnp.float32, 2e-6, 2e-7),
        (jnp.bfloat16, 3e-2, 2e-3),
    ],
)
def test_fused_reverse_matches_stock_autodiff_end_to_end(cpu_mesh, monkeypatch, dtype, max_threshold, mean_threshold):
    """Pin the whole custom VJP -- composition, reductions and psums -- against the XLA path.

    The SM100 kernels are swapped for their pure-JAX references so the composition around them
    is exercised on CPU; the kernels themselves are covered by
    ``lib/levanter/tests/grug/test_quack_rms_cute.py`` on a GPU backend.
    """
    del cpu_mesh

    def gate_silu_reverse(normalized, output_cotangent, w_up, gate_preactivation, gate_hidden):
        gate = jax.nn.sigmoid(jnp.einsum("tr,rd->td", gate_hidden, w_up))
        gate_accumulator = output_cotangent * normalized * (gate * (1 - gate))
        w_up_cotangent = jnp.einsum("tr,td->rd", gate_hidden, gate_accumulator)
        gate_hidden_cotangent = jnp.einsum("td,rd->tr", gate_accumulator, w_up)
        _, silu_pullback = jax.vjp(jax.nn.silu, gate_preactivation)
        return output_cotangent * gate, silu_pullback(gate_hidden_cotangent)[0], w_up_cotangent

    monkeypatch.setattr(
        rgn,
        "_backward_kernels",
        lambda: (
            gate_silu_reverse,
            rgn.exact_rms_backward_partials_reference,
            rgn.exact_rms_backward_recompute_consumer_reference,
        ),
    )
    inputs = _norm_inputs(dtype)
    # The fused path reshards its input to the batch spec, so feed both paths an already-sharded
    # activation and cotangent; otherwise their outputs carry different shardings.
    batch_spec = P(_BATCH_AXES)
    x = reshard(inputs.x, batch_spec)
    cotangent = reshard(inputs.cotangent, batch_spec)

    def fused(x, norm_weight, w_down, w_up):
        return rms_gated_norm(
            x,
            norm_weight=norm_weight,
            w_down=w_down,
            w_up=w_up,
            eps=_NORM_EPS,
            implementation="quack_coda_backward",
        )

    primals = (x, inputs.norm_weight, inputs.w_down, inputs.w_up)
    expected_out, expected_pullback = jax.vjp(_xla_rms_gated_norm, *primals)
    actual_out, actual_pullback = jax.vjp(fused, *primals)
    np.testing.assert_array_equal(actual_out, expected_out)

    names = ("input", "norm_weight", "w_down", "w_up")
    for name, actual, expected in zip(names, actual_pullback(cotangent), expected_pullback(cotangent), strict=True):
        _assert_error_below(
            actual,
            expected,
            max_threshold=max_threshold,
            mean_threshold=mean_threshold,
            label=f"{dtype} {name} gradient",
        )


def test_runtime_rms_implementation_is_not_serialized_in_hf_config():
    config = GrugModelConfig(
        vocab_size=257,
        hidden_dim=16,
        num_heads=2,
        num_kv_heads=1,
        num_experts=4,
        num_experts_per_token=2,
        rms_gated_norm_implementation="quack_coda_backward",
    )

    serialized = config.to_hf_config(config.vocab_size).to_dict()
    restored = GrugModelConfig.from_hf_config(GrugMoeHfConfig.from_dict(serialized))

    assert "rms_gated_norm_implementation" not in serialized
    assert restored.rms_gated_norm_implementation == "xla"


def test_rms_gated_norm_rejects_unknown_implementation():
    inputs = _norm_inputs(jnp.bfloat16)
    rms = RMSNorm(weight=inputs.norm_weight, eps=_NORM_EPS)
    gated = GatedNorm(w_down=inputs.w_down, w_up=inputs.w_up)

    with pytest.raises(ValueError):
        rms_gated_norm(
            inputs.x,
            norm_weight=rms.weight,
            w_down=gated.w_down,
            w_up=gated.w_up,
            eps=_NORM_EPS,
            implementation="unknown",  # type: ignore[arg-type]
        )
