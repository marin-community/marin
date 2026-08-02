# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare raw and adjudicated hierarchy embedding reports."""

import argparse
import json
from pathlib import Path
from typing import Any

from glm_semantic_labels import read_json

MAXIMUM_GLOBAL_METRIC_CHANGE = 0.02
GLOBAL_METRICS = (
    "cross_group_neighbor_any_label_fraction",
    "cross_group_neighbor_label_jaccard",
    "cross_group_nearest_primary_macro_f1",
    "cluster_nmi",
)


def label_sensitivity(raw: dict[str, Any], adjudicated: dict[str, Any]) -> dict[str, Any]:
    """Return fixed metric and gate changes after tail adjudication."""
    if raw.get("label_version") != "raw_glm" or adjudicated.get("label_version") != "adjudicated":
        raise ValueError("The reports do not contain raw and adjudicated label versions")
    if raw.get("documents") != adjudicated.get("documents"):
        raise ValueError("The report document counts differ")
    if set(raw["variants"]) != set(adjudicated["variants"]):
        raise ValueError("The report hierarchy variants differ")
    variants = {}
    for variant_name, raw_levels in raw["variants"].items():
        adjudicated_levels = adjudicated["variants"][variant_name]
        if set(raw_levels) != set(adjudicated_levels):
            raise ValueError("The report hierarchy levels differ")
        levels = {}
        for level_name, raw_level in raw_levels.items():
            adjudicated_level = adjudicated_levels[level_name]
            raw_metrics = raw_level["models"]["fast_arctic_3m"]
            adjudicated_metrics = adjudicated_level["models"]["fast_arctic_3m"]
            metric_changes = {
                metric: float(adjudicated_metrics[metric]) - float(raw_metrics[metric]) for metric in GLOBAL_METRICS
            }
            raw_gate = bool(raw_level["fast_arctic_3m_all_gates_passed"])
            adjudicated_gate = bool(adjudicated_level["fast_arctic_3m_all_gates_passed"])
            raw_group_gate = bool(raw_level["fast_arctic_3m_large_group_gates_passed"])
            adjudicated_group_gate = bool(adjudicated_level["fast_arctic_3m_large_group_gates_passed"])
            levels[level_name] = {
                "global_metric_changes": metric_changes,
                "maximum_absolute_global_metric_change": max(abs(value) for value in metric_changes.values()),
                "global_metric_change_gate_passed": all(
                    abs(value) <= MAXIMUM_GLOBAL_METRIC_CHANGE for value in metric_changes.values()
                ),
                "raw_all_gates_passed": raw_gate,
                "adjudicated_all_gates_passed": adjudicated_gate,
                "all_gate_decision_unchanged": raw_gate == adjudicated_gate,
                "raw_large_group_gates_passed": raw_group_gate,
                "adjudicated_large_group_gates_passed": adjudicated_group_gate,
                "large_group_gate_decision_unchanged": raw_group_gate == adjudicated_group_gate,
            }
        variants[variant_name] = levels
    all_levels = [level for levels in variants.values() for level in levels.values()]
    return {
        "documents": raw["documents"],
        "maximum_allowed_global_metric_change": MAXIMUM_GLOBAL_METRIC_CHANGE,
        "variants": variants,
        "all_global_metric_change_gates_passed": all(level["global_metric_change_gate_passed"] for level in all_levels),
        "all_gate_decisions_unchanged": all(level["all_gate_decision_unchanged"] for level in all_levels),
        "all_large_group_gate_decisions_unchanged": all(
            level["large_group_gate_decision_unchanged"] for level in all_levels
        ),
    }


def main() -> None:
    """Read two reports and write their label-sensitivity result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-report-url", required=True)
    parser.add_argument("--adjudicated-report-url", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    result = label_sensitivity(read_json(args.raw_report_url), read_json(args.adjudicated_report_url))
    args.output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
