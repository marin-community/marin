# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the Target 1 pre-run numerical comparison contract."""

import copy
import json
from pathlib import Path

import pytest
from target1_prerun_comparison import (
    load_contract,
    require_identity_roundtrip,
    require_matched_oracle,
    require_reference_qualified,
    validate_contract,
)

CONTRACT = Path(__file__).with_name("target1-rowwise-bf16-prerun-comparison-v1.json")


def _document() -> dict:
    return json.loads(CONTRACT.read_text())


def _replace(path: tuple[str | int, ...], value: object) -> dict:
    document = copy.deepcopy(_document())
    target = document
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    return document


def _metrics(**updates: float | int) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "max_absolute_error": 0.0,
        "mean_absolute_error": 0.0,
        "relative_linf_error": 0.0,
        "max_bfloat16_ulp_error": 0,
    }
    metrics.update(updates)
    return metrics


def test_contract_predeclares_the_complete_comparison_without_claiming_execution() -> None:
    contract = load_contract(CONTRACT)

    assert contract["run_matrix"]["te_runs_per_hardware"] == 24
    assert contract["run_matrix"]["total_te_runs"] == 48
    assert contract["run_matrix"]["boundaries"] == {
        "forward": {"numerical_reference_boundary": "forward", "outputs": ["y"]},
        "backward_recompute": {
            "numerical_reference_boundary": "backward",
            "outputs": ["dx", "dgamma"],
        },
        "composed": {
            "numerical_reference_boundary": "composed",
            "outputs": ["y", "dx", "dgamma"],
        },
    }
    assert contract["rules"]["fast_non_identity"]["status"] == (
        "blocked_requires_new_reviewed_contract_before_execution_or_timing"
    )
    assert contract["repeatability"]["post_timing_invocations"] == 3
    assert contract["execution_state"]["launch_ready"] is False
    assert contract["execution_state"]["hardware_results"] == []
    assert contract["execution_state"]["scorecard_status_changed"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("thresholds", "dtype_floors", "2048x4096", "y", "max_bfloat16_ulp_error"), 2),
        (
            (
                "thresholds",
                "reference_limits",
                "2048x4096",
                "dx",
                "max_bfloat16_ulp_error",
            ),
            8,
        ),
        (("run_matrix", "boundaries", "forward", "outputs"), ["dx"]),
        (
            (
                "run_matrix",
                "boundaries",
                "backward_recompute",
                "numerical_reference_boundary",
            ),
            "composed",
        ),
        (("run_matrix", "te_backend_pairs", 2, "backward"), "transformer_engine"),
        (("rules", "te_backend_pairing"), "choose_lowest_error_after_execution"),
        (("dependencies", "expert_oracle", "sha256"), "0" * 64),
        (("subjects", "transformer_engine", "required_identity"), ["source_commit"]),
        (("repeatability", "post_timing_invocations"), 2),
        (("rules", "fast_non_identity", "status"), "allowed"),
        (("execution_state", "launch_ready"), True),
        (("execution_state", "hardware_results"), [{"hardware": "h100"}]),
        (("execution_state", "scorecard_status_changed"), True),
    ],
)
def test_contract_rejects_threshold_role_boundary_backend_and_provenance_drift(
    path: tuple[str | int, ...], value: object
) -> None:
    with pytest.raises(ValueError, match="drifted"):
        validate_contract(_replace(path, value))


@pytest.mark.parametrize(
    ("metric", "value", "companion_max"),
    [
        ("max_absolute_error", 0.0078125001, None),
        ("mean_absolute_error", 0.0023046756, 0.007),
        ("relative_linf_error", 0.007751938, None),
        ("max_bfloat16_ulp_error", 2, None),
    ],
)
def test_reference_qualification_rejects_each_just_over_limit_metric(
    metric: str, value: float | int, companion_max: float | None
) -> None:
    contract = load_contract(CONTRACT)
    metrics = _metrics(**{metric: value})
    if companion_max is not None:
        metrics["max_absolute_error"] = companion_max

    with pytest.raises(AssertionError, match="predeclared reference limit"):
        require_reference_qualified(metrics, shape="2048x4096", role="y", contract=contract)


def test_matched_comparison_uses_the_predeclared_dtype_floor_metricwise() -> None:
    contract = load_contract(CONTRACT)
    require_matched_oracle(
        _metrics(max_bfloat16_ulp_error=1),
        _metrics(),
        shape="2048x4096",
        role="dx",
        contract=contract,
    )

    with pytest.raises(AssertionError, match="matched expert-or-dtype-floor"):
        require_matched_oracle(
            _metrics(max_bfloat16_ulp_error=2),
            _metrics(),
            shape="2048x4096",
            role="dx",
            contract=contract,
        )


def test_identity_gate_separates_source_parity_from_non_identity_fast() -> None:
    require_identity_roundtrip(
        policy="source_ordered",
        identity_lowering=False,
        ordinary_digest="same",
        shuttle_digest="same",
    )
    require_identity_roundtrip(
        policy="fast",
        identity_lowering=True,
        ordinary_digest="same",
        shuttle_digest="same",
    )

    with pytest.raises(AssertionError, match="non-identity FAST"):
        require_identity_roundtrip(
            policy="fast",
            identity_lowering=False,
            ordinary_digest="same",
            shuttle_digest="same",
        )
    with pytest.raises(AssertionError, match="bitwise equal"):
        require_identity_roundtrip(
            policy="source_ordered",
            identity_lowering=False,
            ordinary_digest="ordinary",
            shuttle_digest="shuttle",
        )


def test_loader_rejects_duplicate_contract_keys(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 1, "schema_version": 1}')

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_contract(duplicate)
