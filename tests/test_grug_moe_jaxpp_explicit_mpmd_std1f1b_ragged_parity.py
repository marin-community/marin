# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import optax
import pytest

from experiments.grug.moe import check_jaxpp_explicit_mpmd_std1f1b_ragged_parity as parity
from experiments.grug.moe.check_jaxpp_eager_1f1b_parity import DEFAULT_TOLERANCE
from experiments.grug.moe.check_jaxpp_explicit_mpmd_std1f1b_ragged_parity import (
    build_stage_parity_report,
    captured_gradients,
    gradient_capture_optimizer,
    local_precompile_enabled,
    validate_authoritative_topology,
    validate_device_ragged_flags,
)


def test_gradient_capture_optimizer_preserves_params_and_returns_every_gradient():
    initial = {
        "first": jnp.asarray([1.0, -2.0]),
        "nested": {"second": jnp.asarray([3.0])},
    }
    expected_gradients = {
        "first": jnp.asarray([0.25, -0.5]),
        "nested": {"second": jnp.asarray([1.5])},
    }
    optimizer = gradient_capture_optimizer()
    updates, opt_state = optimizer.update(expected_gradients, optimizer.init(initial), initial)
    updated = optax.apply_updates(initial, updates)

    recovered = captured_gradients(opt_state)

    assert jax.tree.all(jax.tree.map(jnp.array_equal, updated, initial))
    assert recovered.keys() == expected_gradients.keys()
    assert jnp.array_equal(recovered["first"], expected_gradients["first"])
    assert jnp.array_equal(recovered["nested"]["second"], expected_gradients["nested"]["second"])


def test_stage_report_rejects_one_finite_gradient_leaf_above_fixed_tolerance():
    report = build_stage_parity_report(
        stage_index=2,
        explicit_loss=jnp.asarray(2.0),
        direct_loss=jnp.asarray(2.0),
        explicit_gradients={
            "passing": jnp.asarray([1.001]),
            "failing": jnp.asarray([1.003]),
        },
        direct_gradients={
            "passing": jnp.asarray([1.0]),
            "failing": jnp.asarray([1.0]),
        },
    )

    gradients = {gradient.path: gradient for gradient in report.gradients}
    assert report.tolerance == DEFAULT_TOLERANCE == 0.002
    assert not report.passed
    assert gradients["params['stage_2']['passing']"].passed
    assert not gradients["params['stage_2']['failing']"].passed


def test_authoritative_topology_requires_four_pipeline_ranks_with_ep2():
    validate_authoritative_topology(process_count=4, local_device_count=2, device_count=8)

    with pytest.raises(ValueError, match="four JAX processes with two local devices each"):
        validate_authoritative_topology(process_count=4, local_device_count=1, device_count=4)


def test_device_ragged_validation_rejects_host_initiated_mode():
    validate_device_ragged_flags(
        "--xla_gpu_autotune_level=0 " "--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true"
    )

    with pytest.raises(ValueError, match="device-ragged parity requires"):
        validate_device_ragged_flags("--xla_gpu_autotune_level=0")


def test_local_precompile_requires_explicit_binary_opt_in():
    assert not local_precompile_enabled({})
    assert not local_precompile_enabled({"GRUG_JAXPP_PRECOMPILE_LOCAL": "0"})
    assert local_precompile_enabled({"GRUG_JAXPP_PRECOMPILE_LOCAL": "1"})

    with pytest.raises(ValueError, match="must be 0 or 1"):
        local_precompile_enabled({"GRUG_JAXPP_PRECOMPILE_LOCAL": "true"})


def test_direct_reference_can_be_loaded_from_another_pipeline_rank(monkeypatch):
    expected_result = ("loss", ("stage-0", "stage-1", "stage-2", "stage-3"))

    class FakeClient:
        def __init__(self):
            self.values = {
                "grug-ragged-parity-direct-status-1": b"1",
                "grug-ragged-parity-direct-status-2": b"0",
                "grug-ragged-parity-direct-status-3": b"0",
                "grug-ragged-parity-direct-result-1": parity.pickle.dumps(expected_result),
            }

        def key_value_set_bytes(self, key, value):
            self.values[key] = value

        def wait_at_barrier(self, name, timeout):
            assert name == "grug_ragged_parity_direct_results_published"
            assert timeout == 123

        def blocking_key_value_get_bytes(self, key, timeout):
            assert timeout == 123
            return self.values[key]

    fake_client = FakeClient()
    fake_dime2 = type(
        "FakeDime2",
        (),
        {
            "get_distributed_client": staticmethod(lambda: fake_client),
            "env_vars": type(
                "FakeEnvVars",
                (),
                {"jaxpp_client_timeout": type("FakeTimeout", (), {"value": 123})()},
            )(),
        },
    )
    monkeypatch.setattr(parity.grug_train, "jaxpp_dime2", fake_dime2)

    result = parity._share_standalone_direct_result(0, None)

    assert result == expected_result
    assert fake_client.values["grug-ragged-parity-direct-status-0"] == b"0"
