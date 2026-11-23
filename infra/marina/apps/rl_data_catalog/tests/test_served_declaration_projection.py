# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The SQL-served declaration rows carry only what the gateway and API consume."""

import json

from infra.marina.apps.rl_data_catalog.load import (
    MAX_BATCH_BYTES,
    batches,
    served_declaration_projection,
)
from infra.marina.apps.rl_data_catalog.server.filtration import filtration


def _declaration(pending_exclusions: list[dict]) -> dict:
    return {
        "id": "OpenAthena:pdbthink-coordinate-tasks",
        "status": "Available",
        "oracle_filtration": {
            "admitted_count": 46805,
            "exclusion_counts": {"resource_calibration:infrastructure": 33651},
            "oracle_filter": {"uri": "s3://bucket/filter.json", "sha256": "a" * 64},
            "validation": {"uri": "s3://bucket/validation.json", "sha256": "b" * 64},
            "pending_exclusions": pending_exclusions,
        },
    }


def test_projection_drops_only_the_per_task_pending_exclusions():
    pending = [{"reason": "resource_calibration:infrastructure", "task_id": f"task-{n}"} for n in range(40_000)]
    declaration = _declaration(pending)
    projected = served_declaration_projection(declaration)
    assert "pending_exclusions" not in projected["oracle_filtration"]
    # The aggregate and every pinned artifact reference stay on the served row.
    assert projected["oracle_filtration"]["exclusion_counts"] == {"resource_calibration:infrastructure": 33651}
    assert projected["oracle_filtration"]["oracle_filter"]["uri"] == "s3://bucket/filter.json"
    assert projected["oracle_filtration"]["validation"]["uri"] == "s3://bucket/validation.json"
    assert projected["oracle_filtration"]["admitted_count"] == 46805
    # The input is untouched: the full evidence remains with the snapshot and receipts.
    assert declaration["oracle_filtration"]["pending_exclusions"] == pending


def test_projection_passes_through_declarations_without_pending_exclusions():
    declaration = _declaration([])
    assert served_declaration_projection(declaration) == declaration
    plain = {"id": "OpenAI:gsm8k", "status": "Excluded"}
    assert served_declaration_projection(plain) is plain


def test_projected_row_fits_the_sql_batch_budget_while_the_full_row_does_not():
    pending = [{"reason": "resource_calibration:infrastructure", "task_id": f"task-{n}"} for n in range(40_000)]
    full = {"source_id": "OpenAthena:pdbthink-coordinate-tasks", "declaration": _declaration(pending)}
    projected = {**full, "declaration": served_declaration_projection(full["declaration"])}

    def row_length(row: dict) -> int:
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        return len(json.dumps(encoded).encode()) + 1

    assert row_length(full) + 2 > MAX_BATCH_BYTES
    assert row_length(projected) + 2 <= MAX_BATCH_BYTES
    # The projected row streams through the real batcher without refusing.
    assert next(iter(batches([projected])))


def _filtration_declaration(pending_exclusions: list[dict]) -> dict:
    """A pdbthink-shaped declaration whose filtration evidence takes the aggregate path."""
    infrastructure = len(pending_exclusions)
    return {
        "id": "OpenAthena:pdbthink-coordinate-tasks",
        "status": "Available",
        "within_source_dedup": {"unique_count": 3},
        "task_count": 3,
        "dataset_id": "open-athena/pdbthink-retained",
        "dataset_revision": "d" * 40,
        "current_release": {
            "repo": "open-athena/pdbthink-qualified",
            "revision": "e" * 40,
            "url": "https://huggingface.co/datasets/open-athena/pdbthink-qualified/tree/" + "e" * 40,
        },
        "oracle_validation": "executed",
        "oracle_filtration": {
            "private": True,
            "original_release": {
                "repo": "open-athena/pdbthink-original",
                "revision": "f" * 40,
                "url": "https://huggingface.co/datasets/open-athena/pdbthink-original/tree/" + "f" * 40,
            },
            "original_cohort": {"uri": "s3://bucket/cohort", "sha256": "a" * 64},
            "original_count": 6,
            "oracle_filter": {"uri": "s3://bucket/filter", "sha256": "b" * 64},
            "oracle_filter_version": 4,
            "filtered_release": {
                "repo": "open-athena/pdbthink-qualified",
                "revision": "e" * 40,
                "url": "https://huggingface.co/datasets/open-athena/pdbthink-qualified/tree/" + "e" * 40,
            },
            "tasks_parquet": {"uri": "s3://bucket/tasks", "sha256": "c" * 64},
            "admitted_count": 3,
            "exclusion_counts": {"resource_calibration:infrastructure": infrastructure},
            "pending_exclusions": pending_exclusions,
            "validation": {"uri": "s3://bucket/validation", "sha256": "d" * 64},
        },
    }


def test_filtration_projection_is_identical_for_the_full_and_the_projected_row():
    pending = [{"reason": "resource_calibration:infrastructure", "task_id": f"pdbthink-task-{n:03d}"} for n in range(3)]
    full = _filtration_declaration(pending)
    projected = served_declaration_projection(full)
    assert "pending_exclusions" not in projected["oracle_filtration"]
    # The projected form no longer trips the typed model's list/count pairing.
    full_output = filtration(full, None)
    projected_output = filtration(projected, None)
    assert projected_output == full_output
    assert projected_output["population_count"] == 3
