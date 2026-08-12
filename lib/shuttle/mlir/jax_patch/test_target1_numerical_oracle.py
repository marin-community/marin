# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the Target 1 independent numerical reference."""

import copy
import json
from pathlib import Path

import jax
import ml_dtypes
import numpy as np
import pytest
from shuttle_jaxlib_target1_acceptance import boundary_function
from target1_numerical_oracle import (
    ANALYTIC_THRESHOLDS,
    BOUNDARIES,
    OUTPUT_ROLES,
    SHAPES,
    error_metrics,
    fixed_inputs,
    independent_reference,
    load_contract,
    require_accepted,
    validate_contract,
)

CONTRACT = Path(__file__).with_name("target1-rowwise-bf16-numerical-oracle-v1.json")


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_ordinary_jax_boundary_satisfies_independent_reference_contract(shape, boundary) -> None:
    rows, features = shape
    arguments = fixed_inputs(rows, features, boundary)
    actual = tuple(np.asarray(value) for value in jax.tree.leaves(jax.jit(boundary_function(boundary))(*arguments)))
    reference = independent_reference(boundary, arguments)

    assert len(actual) == len(reference) == len(OUTPUT_ROLES[boundary])
    for actual_output, reference_output in zip(actual, reference, strict=True):
        assert actual_output.dtype == reference_output.dtype == ml_dtypes.bfloat16
        metrics = error_metrics(actual_output, reference_output)
        require_accepted(metrics, "source_ordered", identity_roundtrip_bitwise=True)
        require_accepted(metrics, "fast", identity_roundtrip_bitwise=True)


def test_contract_matches_closed_schema_and_generated_input_digests() -> None:
    document = load_contract(CONTRACT)
    assert document["scorecard_effect"]["status_changed"] is False


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    [
        (lambda value: value["shapes"].__setitem__(0, [2048, 2048]), "shapes"),
        (lambda value: value["boundaries"].reverse(), "boundaries"),
        (lambda value: value["outputs"]["7x13/forward"][0].__setitem__("shape", [91]), "outputs"),
        (lambda value: value["outputs"]["7x13/forward"][0].__setitem__("dtype", "float32"), "outputs"),
        (lambda value: value["reference"]["formulas"].__setitem__("dx", "wrong formula"), "formulas"),
        (
            lambda value: value["reference"]["output_digests"]["7x13/forward"].__setitem__(0, "0" * 64),
            "output_digests",
        ),
        (
            lambda value: value["policies"]["fast"]["analytic_thresholds"].__setitem__("max_bfloat16_ulp_error", 9),
            "analytic_thresholds",
        ),
        (lambda value: value["provenance"].__setitem__("xla_revision", "0" * 40), "provenance"),
        (
            lambda value: value["local_observation"]["environment"].__setitem__("host_architecture", "x86_64"),
            "local_observation",
        ),
        (
            lambda value: value["local_observation"]["results"]["7x13/forward"][0].__setitem__(
                "output_digest", "0" * 64
            ),
            "local_observation",
        ),
        (lambda value: value["scorecard_effect"].__setitem__("status_changed", True), "scorecard_effect"),
    ],
)
def test_contract_rejects_semantic_mutation(mutation, diagnostic) -> None:
    document = json.loads(CONTRACT.read_text())
    mutation(document)
    with pytest.raises(ValueError, match=diagnostic):
        validate_contract(document)


def test_contract_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    duplicate = CONTRACT.read_text().replace(
        '  "schema_version": 1,', '  "schema_version": 1,\n  "schema_version": 1,', 1
    )
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_contract(path)


def test_contract_rejects_oversized_payload(tmp_path: Path) -> None:
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (128 * 1024 + 1))
    with pytest.raises(ValueError, match="byte limit"):
        load_contract(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, "0"])
def test_contract_rejects_invalid_observed_metrics(value) -> None:
    document = json.loads(CONTRACT.read_text())
    document["local_observation"]["results"]["7x13/forward"][0]["metrics"]["mean_absolute_error"] = value
    with pytest.raises(ValueError, match="finite nonnegative number"):
        validate_contract(document)


def test_analytic_thresholds_reject_boundary_value_drift() -> None:
    metrics = error_metrics(
        np.asarray([1.0], dtype=ml_dtypes.bfloat16),
        np.asarray([1.0], dtype=ml_dtypes.bfloat16),
    )
    require_accepted(metrics, "source_ordered", identity_roundtrip_bitwise=True)

    for field, limit in ANALYTIC_THRESHOLDS.items():
        mutated = copy.copy(metrics)
        object.__setattr__(mutated, field, limit + (1 if isinstance(limit, int) else np.finfo(np.float64).eps))
        with pytest.raises(AssertionError, match=field):
            require_accepted(mutated, "source_ordered", identity_roundtrip_bitwise=True)


def test_fast_non_identity_rewrite_has_no_invented_acceptance_threshold() -> None:
    metrics = error_metrics(
        np.asarray([1.0], dtype=ml_dtypes.bfloat16),
        np.asarray([1.0], dtype=ml_dtypes.bfloat16),
    )
    with pytest.raises(AssertionError, match="unresolved"):
        require_accepted(metrics, "fast", identity_roundtrip_bitwise=False)


def test_reference_rejects_shape_and_dtype_drift() -> None:
    arguments = fixed_inputs(7, 13, "forward")
    with pytest.raises(ValueError, match="rank-2 bfloat16"):
        independent_reference("forward", (arguments[0].astype(np.float32), arguments[1]))
    with pytest.raises(ValueError, match="closed shape"):
        independent_reference(
            "forward",
            (
                np.ones((7, 12), dtype=ml_dtypes.bfloat16),
                np.ones((12,), dtype=ml_dtypes.bfloat16),
            ),
        )
