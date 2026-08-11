# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.rms_gated_norm import (
    coda_rms_gated_norm_analytic_backward,
    exact_rms_backward_producer_reference,
)

from experiments.grug.moe_hero_fsdp import launch, train
from experiments.grug.moe_hero_fsdp import model as model_module
from experiments.grug.moe_hero_fsdp.model import (
    GatedNorm,
    GrugModelConfig,
    GrugMoeHfConfig,
    RMSNorm,
    coda_rms_gated_norm_reference,
)

_NORM_EPS = 1e-5
_GATED_NORM_RANK = 128
_GRADIENT_NAMES = ("input", "norm_weight", "w_down", "w_up")


def _norm_inputs(dtype: jnp.dtype):
    input_key, norm_key, down_key, up_key, cotangent_key = jax.random.split(jax.random.key(17), 5)
    x = jax.random.normal(input_key, (2, 3, 16), dtype=dtype)
    norm_weight = 1 + 0.1 * jax.random.normal(norm_key, (16,), dtype=jnp.float32)
    w_down = (0.1 * jax.random.normal(down_key, (16, _GATED_NORM_RANK), dtype=dtype)).astype(dtype)
    w_up = (0.1 * jax.random.normal(up_key, (_GATED_NORM_RANK, 16), dtype=dtype)).astype(dtype)
    cotangent = jax.random.normal(cotangent_key, x.shape, dtype=dtype)
    return x, norm_weight, w_down, w_up, cotangent


def _xla_rms_gated_norm(x, norm_weight, w_down, w_up):
    return GatedNorm(w_down=w_down, w_up=w_up)(RMSNorm(weight=norm_weight, eps=_NORM_EPS)(x))


def _coda_forward_residuals(x, norm_weight, w_down, w_up):
    x_flat = x.reshape((-1, x.shape[-1]))
    inverse_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x_flat.astype(jnp.float32)), axis=-1) + _NORM_EPS)
    scaled_w_down = (norm_weight[:, None] * w_down).astype(x.dtype)
    gate_accumulator = jnp.einsum("td,dr->tr", x_flat, scaled_w_down, preferred_element_type=jnp.float32)
    gate_preactivation = gate_accumulator * inverse_rms[:, None]
    gate_hidden = jax.nn.silu(gate_preactivation).astype(x.dtype)
    gate = jax.nn.sigmoid(jnp.einsum("tr,rd->td", gate_hidden, w_up)).astype(x.dtype)
    return x, norm_weight, w_down, w_up, inverse_rms, gate_preactivation, gate_hidden, gate


def _assert_error_below(actual, expected, *, max_threshold: float, mean_threshold: float, label: str) -> None:
    absolute_error = jnp.abs(actual.astype(jnp.float32) - expected.astype(jnp.float32))
    max_error = float(jnp.max(absolute_error))
    mean_error = float(jnp.mean(absolute_error))
    assert max_error <= max_threshold and mean_error <= mean_threshold, (
        f"{label}: max_abs_error={max_error:.8g} (threshold={max_threshold:.8g}), "
        f"mean_abs_error={mean_error:.8g} (threshold={mean_threshold:.8g})"
    )


def _fake_quack_training_forward(a, b, inverse_rms):
    preactivation = jnp.einsum("td,dr->tr", a, b, preferred_element_type=jnp.float32)
    preactivation = preactivation * inverse_rms[:, None]
    return preactivation, jax.nn.silu(preactivation).astype(a.dtype)


def _fake_quack_silu_backward(output_cotangent, w_up, preactivation):
    hidden_cotangent = jnp.einsum("td,rd->tr", output_cotangent, w_up)
    sigmoid = jax.nn.sigmoid(preactivation)
    sigmoid_derivative = sigmoid * (1.0 - sigmoid)
    preactivation_cotangent = hidden_cotangent * sigmoid + (preactivation * hidden_cotangent) * sigmoid_derivative
    return preactivation_cotangent, preactivation * sigmoid


def _fake_quack_rms_backward_producer(gate_cotangent, w_down, direct_cotangent, x, norm_weight, inverse_rms):
    gate_input_cotangent = jnp.einsum("tr,dr->td", gate_cotangent, w_down)
    unweighted_cotangent = gate_input_cotangent + direct_cotangent
    _, row_dot, _ = exact_rms_backward_producer_reference(
        gate_cotangent,
        w_down,
        direct_cotangent,
        x,
        norm_weight,
        inverse_rms,
    )
    return unweighted_cotangent, row_dot[:, None]


@pytest.fixture
def cpu_mesh():
    cpu_device = jax.devices("cpu")[0]
    mesh = Mesh(
        np.asarray([cpu_device]).reshape(1, 1, 1, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    return cpu_device, mesh


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


@pytest.mark.parametrize(
    ("dtype", "max_threshold", "mean_threshold"),
    [
        (jnp.float32, 5e-7, 1e-7),
        (jnp.bfloat16, 1.6e-2, 1e-3),
    ],
)
def test_coda_rms_gated_norm_forward_matches_xla_path(cpu_mesh, dtype, max_threshold, mean_threshold):
    cpu_device, mesh = cpu_mesh
    with jax.default_device(cpu_device), jax.set_mesh(mesh):
        x, norm_weight, w_down, w_up, _ = _norm_inputs(dtype)
        expected = _xla_rms_gated_norm(x, norm_weight, w_down, w_up)
        actual = coda_rms_gated_norm_reference(x, norm_weight, w_down, w_up, _NORM_EPS)

        _assert_error_below(
            actual,
            expected,
            max_threshold=max_threshold,
            mean_threshold=mean_threshold,
            label=f"{dtype} forward",
        )


@pytest.mark.parametrize(
    ("dtype", "max_threshold", "mean_threshold"),
    [
        (jnp.float32, 1e-7, 1e-8),
        (jnp.bfloat16, 5e-4, 1e-4),
    ],
)
def test_coda_rms_gated_norm_gradients_match_xla_path(cpu_mesh, dtype, max_threshold, mean_threshold):
    cpu_device, mesh = cpu_mesh
    with jax.default_device(cpu_device), jax.set_mesh(mesh):
        x, norm_weight, w_down, w_up, cotangent = _norm_inputs(dtype)

        def xla_loss(x, norm_weight, w_down, w_up):
            output = _xla_rms_gated_norm(x, norm_weight, w_down, w_up)
            return jnp.mean(output.astype(jnp.float32) * cotangent.astype(jnp.float32))

        def coda_loss(x, norm_weight, w_down, w_up):
            output = coda_rms_gated_norm_reference(x, norm_weight, w_down, w_up, _NORM_EPS)
            return jnp.mean(output.astype(jnp.float32) * cotangent.astype(jnp.float32))

        expected_gradients = jax.grad(xla_loss, argnums=(0, 1, 2, 3))(x, norm_weight, w_down, w_up)
        actual_gradients = jax.grad(coda_loss, argnums=(0, 1, 2, 3))(x, norm_weight, w_down, w_up)

        for name, actual, expected in zip(_GRADIENT_NAMES, actual_gradients, expected_gradients, strict=True):
            _assert_error_below(
                actual,
                expected,
                max_threshold=max_threshold,
                mean_threshold=mean_threshold,
                label=f"{dtype} {name} gradient",
            )


@pytest.mark.parametrize(
    ("dtype", "max_threshold", "mean_threshold"),
    [
        (jnp.float32, 2e-6, 2e-7),
        (jnp.bfloat16, 3e-2, 2e-3),
    ],
)
def test_coda_analytic_backward_matches_autodiff(cpu_mesh, dtype, max_threshold, mean_threshold):
    cpu_device, mesh = cpu_mesh
    with jax.default_device(cpu_device), jax.set_mesh(mesh):
        x, norm_weight, w_down, w_up, cotangent = _norm_inputs(dtype)
        _, pullback = jax.vjp(
            lambda x, norm_weight, w_down, w_up: coda_rms_gated_norm_reference(x, norm_weight, w_down, w_up, _NORM_EPS),
            x,
            norm_weight,
            w_down,
            w_up,
        )
        expected_gradients = pullback(cotangent)
        residuals = _coda_forward_residuals(x, norm_weight, w_down, w_up)
        actual_gradients = coda_rms_gated_norm_analytic_backward(cotangent, residuals)

        for name, actual, expected in zip(_GRADIENT_NAMES, actual_gradients, expected_gradients, strict=True):
            _assert_error_below(
                actual,
                expected,
                max_threshold=max_threshold,
                mean_threshold=mean_threshold,
                label=f"{dtype} analytic {name} gradient",
            )


def test_coda_custom_vjp_reduces_replicated_parameter_gradients(cpu_mesh, monkeypatch):
    cpu_device, mesh = cpu_mesh
    fake_quack = ModuleType("levanter.grug._moe.quack_rms_cute")
    fake_quack.quack_rms_scaled_silu_gemm_with_preactivation = _fake_quack_training_forward
    fake_quack.quack_silu_backward_gemm = _fake_quack_silu_backward
    fake_quack.quack_coda_rms_backward_producer = _fake_quack_rms_backward_producer
    monkeypatch.setitem(sys.modules, fake_quack.__name__, fake_quack)

    x, norm_weight, w_down, w_up, cotangent = _norm_inputs(jnp.bfloat16)
    batch_spec = P(("replica_dcn", "data", "expert"), None, None)
    x = jax.device_put(x, NamedSharding(mesh, batch_spec))
    cotangent = jax.device_put(cotangent, NamedSharding(mesh, batch_spec))
    norm_weight = jax.device_put(norm_weight, NamedSharding(mesh, P(None)))
    w_down = jax.device_put(w_down, NamedSharding(mesh, P(None, None)))
    w_up = jax.device_put(w_up, NamedSharding(mesh, P(None, None)))

    mapped = model_module.shard_map(
        lambda x, norm_weight, w_down, w_up: model_module._quack_coda_rms_gated_norm(
            x, norm_weight, w_down, w_up, _NORM_EPS
        ),
        mesh=mesh,
        in_specs=(batch_spec, P(None), P(None, None), P(None, None)),
        out_specs=batch_spec,
    )
    baseline = model_module.shard_map(
        _xla_rms_gated_norm,
        mesh=mesh,
        in_specs=(batch_spec, P(None), P(None, None), P(None, None)),
        out_specs=batch_spec,
    )
    with jax.default_device(cpu_device), jax.set_mesh(mesh):
        _, pullback = jax.vjp(mapped, x, norm_weight, w_down, w_up)
        gradients = pullback(cotangent)
        _, baseline_pullback = jax.vjp(baseline, x, norm_weight, w_down, w_up)
        baseline_gradients = baseline_pullback(cotangent)

    assert [gradient.sharding.spec for gradient in gradients] == [
        batch_spec,
        P(None),
        P(None, None),
        P(None, None),
    ]
    for name, actual, expected in zip(_GRADIENT_NAMES, gradients, baseline_gradients, strict=True):
        _assert_error_below(
            actual,
            expected,
            max_threshold=8e-3,
            mean_threshold=5e-4,
            label=f"distributed {name} gradient",
        )


def test_rms_gated_norm_implementation_survives_hf_config_round_trip():
    config = GrugModelConfig(
        vocab_size=257,
        hidden_dim=16,
        num_heads=2,
        num_kv_heads=1,
        num_experts=4,
        num_experts_per_token=2,
        rms_gated_norm_implementation="quack_coda",
    )

    serialized = config.to_hf_config(config.vocab_size).to_dict()
    restored = GrugModelConfig.from_hf_config(GrugMoeHfConfig.from_dict(serialized))

    assert serialized["rms_gated_norm_implementation"] == "quack_coda"
    assert restored.rms_gated_norm_implementation == "quack_coda"
