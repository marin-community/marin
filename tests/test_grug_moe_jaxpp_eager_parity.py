# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp

from experiments.grug.moe.check_jaxpp_eager_1f1b_parity import build_parity_report


def test_parity_report_accepts_loss_and_every_gradient_within_tolerance():
    report = build_parity_report(
        automatic_loss=jnp.asarray(2.001),
        direct_loss=jnp.asarray(2.0),
        automatic_gradients={
            "first": jnp.asarray([1.001, 2.0]),
            "second": jnp.asarray([3.0]),
        },
        direct_gradients={
            "first": jnp.asarray([1.0, 2.0]),
            "second": jnp.asarray([3.0]),
        },
        tolerance=0.002,
    )

    assert report.passed
    assert report.loss.passed
    assert report.loss.finite
    assert report.loss.max_absolute_error > 0
    assert report.loss.actual_l2 > report.loss.reference_l2
    assert report.loss.norm_ratio > 1.0
    assert report.loss.cosine_similarity == 1.0
    assert [gradient.path for gradient in report.gradients] == ["params['first']", "params['second']"]
    assert all(gradient.passed for gradient in report.gradients)
    assert all(gradient.finite for gradient in report.gradients)


def test_parity_report_rejects_one_gradient_leaf_over_tolerance():
    report = build_parity_report(
        automatic_loss=jnp.asarray(2.0),
        direct_loss=jnp.asarray(2.0),
        automatic_gradients={
            "passing": jnp.asarray([1.001]),
            "failing": jnp.asarray([1.003]),
        },
        direct_gradients={
            "passing": jnp.asarray([1.0]),
            "failing": jnp.asarray([1.0]),
        },
        tolerance=0.002,
    )

    results = {gradient.path: gradient for gradient in report.gradients}
    assert not report.passed
    assert results["params['passing']"].passed
    assert not results["params['failing']"].passed


def test_parity_report_rejects_loss_over_tolerance():
    report = build_parity_report(
        automatic_loss=jnp.asarray(2.005),
        direct_loss=jnp.asarray(2.0),
        automatic_gradients={"weight": jnp.asarray([1.0])},
        direct_gradients={"weight": jnp.asarray([1.0])},
        tolerance=0.002,
    )

    assert not report.passed
    assert not report.loss.passed
    assert report.gradients[0].passed


def test_parity_report_rejects_nonfinite_leaf_and_serializes_required_metrics():
    report = build_parity_report(
        automatic_loss=jnp.asarray(2.0),
        direct_loss=jnp.asarray(2.0),
        automatic_gradients={"weight": jnp.asarray([jnp.nan])},
        direct_gradients={"weight": jnp.asarray([1.0])},
        tolerance=0.002,
        gradient_root="gradients",
    )

    result = report.as_dict()
    gradient = result["gradients"][0]
    assert not report.passed
    assert gradient["path"] == "gradients['weight']"
    assert not gradient["finite"]
    assert gradient["relative_l2"] == float("inf")
    assert {
        "reference_l2",
        "actual_l2",
        "absolute_l2",
        "max_absolute_error",
        "norm_ratio",
        "cosine_similarity",
    } <= gradient.keys()


def test_parity_report_distinguishes_scale_from_direction_error():
    report = build_parity_report(
        automatic_loss=jnp.asarray(1.0),
        direct_loss=jnp.asarray(1.0),
        automatic_gradients={
            "scaled": jnp.asarray([2.0, 0.0]),
            "rotated": jnp.asarray([0.0, 1.0]),
        },
        direct_gradients={
            "scaled": jnp.asarray([1.0, 0.0]),
            "rotated": jnp.asarray([1.0, 0.0]),
        },
        tolerance=0.002,
    )

    results = {gradient.path: gradient for gradient in report.gradients}
    assert results["params['scaled']"].norm_ratio == 2.0
    assert results["params['scaled']"].cosine_similarity == 1.0
    assert results["params['rotated']"].norm_ratio == 1.0
    assert results["params['rotated']"].cosine_similarity == 0.0
