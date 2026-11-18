# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Blocking coverage checks for each source's recorded QC and difficulty evidence."""

from collections.abc import Callable
from typing import Any


def browse_invariants(fetch: Callable[[str], Any], sources: list[dict]) -> dict[str, int]:
    inventory = fetch("api/qc/inventory")
    panels = {row["source_id"]: row for row in inventory["panels"]}
    rollouts = {row["measurement_id"]: row for row in inventory["rollouts"]}
    required_measurements = set()
    qc_sources = qc_rows = unmapped_rows = 0
    for source in sources:
        if source.get("data_restriction") == "private_always":
            continue
        if source["quality"] in {"FULL", "SAMPLE"}:
            panel = panels.get(source["id"])
            if not panel or not panel["pipeline_key"] or panel["task_count"] != panel["raw_count"]:
                raise ValueError(f"QC review coverage is missing or incomplete: {source['id']}")
            if (
                panel.get("missing_batches", 0)
                or panel.get("missing_chunks", 0)
                or ("linked_tasks" in panel["receipt"] and panel["linked_count"] != panel["receipt"]["linked_tasks"])
            ):
                raise ValueError(f"Original reviewer batch coverage is incomplete: {source['id']}")
            qc_sources += 1
            qc_rows += panel["task_count"]
            # A not-mapped original is an explicit evidence result, never an invented raw review.
            unmapped_rows += (panel["receipt"].get("mapping_counts") or {}).get("not_mapped", 0)
        for model in [*source["difficulty"]["models"], *source["difficulty"]["historical"]]:
            if model.get("measurement_id") and model.get("attempts_receipt"):
                required_measurements.add(model["measurement_id"])
    for identity in required_measurements:
        staged = rollouts.get(identity)
        if not staged or staged["expected_count"] != staged["attempt_count"]:
            raise ValueError(f"Difficulty attempt coverage is missing or incomplete: {identity}")
    legend = fetch("api/qc/defects")["defects"]
    if len({row["id"] for row in legend}) != len(legend):
        raise ValueError("QC defect legend repeats a typed defect")
    for defect in legend:
        if sum(row["count"] for row in defect["sources"]) != defect["count"]:
            raise ValueError(f"QC defect counts do not reconcile: {defect['id']}")
    return {
        "qc_sources": qc_sources,
        "qc_rows": qc_rows,
        "not_mapped": unmapped_rows,
        "difficulty_measurements": len(required_measurements),
        "defect_types": len(legend),
    }
