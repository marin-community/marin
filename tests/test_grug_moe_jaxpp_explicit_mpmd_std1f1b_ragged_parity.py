# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import optax
import pytest

from experiments.grug.moe.check_jaxpp_eager_1f1b_parity import DEFAULT_TOLERANCE
from experiments.grug.moe.check_jaxpp_explicit_mpmd_std1f1b_ragged_parity import (
    build_stage_parity_report,
    captured_gradients,
    gradient_capture_optimizer,
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
