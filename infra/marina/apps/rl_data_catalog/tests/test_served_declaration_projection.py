# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The SQL-served declaration rows carry only what the gateway and API consume."""

import json

from infra.marina.apps.rl_data_catalog.load import (
    MAX_BATCH_BYTES,
    batches,
    served_declaration_projection,
)


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
