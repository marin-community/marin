# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from compare_hierarchy_label_versions import label_sensitivity  # noqa: E402


def report(label_version: str, change: float = 0, passed: bool = True) -> dict:
    metrics = {
        "cross_group_neighbor_any_label_fraction": 0.70 + change,
        "cross_group_neighbor_label_jaccard": 0.50 + change,
        "cross_group_nearest_primary_macro_f1": 0.40 + change,
        "cluster_nmi": 0.60 + change,
    }
    return {
        "documents": 10_000,
        "label_version": label_version,
        "variants": {
            "compact": {
                "parent": {
                    "models": {"fast_arctic_3m": metrics},
                    "fast_arctic_3m_all_gates_passed": passed,
                    "fast_arctic_3m_large_group_gates_passed": passed,
                }
            }
        },
    }


def test_label_sensitivity_accepts_small_changes_and_stable_decision() -> None:
    result = label_sensitivity(report("raw_glm"), report("adjudicated", change=0.01))

    assert result["all_global_metric_change_gates_passed"]
    assert result["all_gate_decisions_unchanged"]
    assert result["all_large_group_gate_decisions_unchanged"]
    assert result["variants"]["compact"]["parent"]["maximum_absolute_global_metric_change"] == pytest.approx(0.01)


def test_label_sensitivity_rejects_large_change() -> None:
    result = label_sensitivity(report("raw_glm"), report("adjudicated", change=0.021))

    assert not result["all_global_metric_change_gates_passed"]


def test_label_sensitivity_rejects_changed_gate_decision() -> None:
    result = label_sensitivity(report("raw_glm", passed=True), report("adjudicated", passed=False))

    assert not result["all_gate_decisions_unchanged"]
    assert not result["all_large_group_gate_decisions_unchanged"]


def test_label_sensitivity_requires_correct_versions() -> None:
    with pytest.raises(ValueError, match="raw and adjudicated"):
        label_sensitivity(report("adjudicated"), report("raw_glm"))
