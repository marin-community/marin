# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Columnar task annotations for curation exports."""

import json
from typing import Any

import pyarrow as pa

from taskcompendium.models import TextMessage
from taskcompendium.pipeline.models import TaskAudit
from taskcompendium.pipeline.verification import grader_readiness

TASK_SCHEMA = pa.schema(
    [
        ("task_id", pa.string()),
        ("source_dataset", pa.string()),
        ("source_revision", pa.string()),
        ("source_row", pa.string()),
        ("intended_use", pa.string()),
        (
            "normalization_changes",
            pa.list_(
                pa.struct(
                    [
                        ("field", pa.string()),
                        ("reason", pa.string()),
                        ("original", pa.string()),
                        ("replacement", pa.string()),
                    ]
                )
            ),
        ),
        ("task_json", pa.string()),
        ("raw_json", pa.string()),
        ("original_task_json", pa.string()),
        ("parent_id", pa.string()),
        ("normalization_reason", pa.string()),
        ("normalization_detail", pa.string()),
        ("filter_status", pa.string()),
        ("filter_reasons", pa.list_(pa.string())),
        ("duplicate_of", pa.string()),
        ("review_status", pa.string()),
        ("review_quality", pa.string()),
        ("review_confidence", pa.string()),
        ("review_reference_status", pa.string()),
        ("review_defects", pa.list_(pa.string())),
        ("review_evidence", pa.string()),
        ("review_detail", pa.string()),
        ("checks", pa.list_(pa.struct([("check", pa.string()), ("status", pa.string()), ("detail", pa.string())]))),
        ("grader_readiness", pa.string()),
        ("cleanup_status", pa.string()),
        ("cleanup_action", pa.string()),
        ("cleanup_reason", pa.string()),
        ("cleanup_edits", pa.list_(pa.struct([("old_text", pa.string()), ("replacement", pa.string())]))),
        ("cleanup_detail", pa.string()),
        ("cleanup_lineage_json", pa.string()),
    ],
    metadata={b"taskcompendium.curation_schema": b"6"},
)


def audit_columns(audit: TaskAudit) -> dict[str, Any]:
    """Expose queryable annotations while keeping arbitrary source/task payloads lossless."""
    review = audit.review
    verdict = review.verdict if review is not None else None
    decision = audit.decision
    cleanup = audit.cleanup
    proposal = cleanup.proposal if cleanup is not None else None
    rejection = audit.normalization_rejection
    data = audit.raw.get("data", {}) if audit.raw is not None else {}
    changes = [change.model_dump(mode="json") for change in audit.normalization_changes]
    if not changes:
        changes = data.get("converted", {}).get("normalization_changes", [])
    if not changes and audit.normalized is not None and isinstance(data.get("instruction"), str):
        events = audit.normalized.context.events
        if len(events) == 1 and isinstance(events[0], TextMessage) and events[0].content != data["instruction"]:
            changes = [
                {
                    "field": "instruction",
                    "reason": "Source normalizer changed instruction delivery",
                    "original": data["instruction"],
                    "replacement": events[0].content,
                }
            ]
    return {
        "task_id": audit.task_id,
        "source_dataset": audit.source.dataset,
        "source_revision": audit.source.revision,
        "source_row": audit.source.row,
        "intended_use": audit.intended_use.value if audit.intended_use is not None else None,
        "normalization_changes": changes,
        "task_json": audit.normalized.model_dump_json() if audit.normalized is not None else None,
        "raw_json": json.dumps(audit.raw, ensure_ascii=False, allow_nan=False) if audit.raw is not None else None,
        "original_task_json": audit.original.model_dump_json() if audit.original is not None else None,
        "parent_id": audit.original.id if audit.original is not None else None,
        "normalization_reason": rejection.reason if rejection is not None else None,
        "normalization_detail": rejection.detail if rejection is not None else None,
        "filter_status": decision.disposition.value if decision is not None else None,
        "filter_reasons": decision.reasons if decision is not None else [],
        "duplicate_of": decision.duplicate_of if decision is not None else None,
        "review_status": review.status.value if review is not None else None,
        "review_quality": verdict.quality.value if verdict is not None else None,
        "review_confidence": verdict.confidence.value if verdict is not None else None,
        "review_reference_status": verdict.reference_status.value if verdict is not None else None,
        "review_defects": [defect.value for defect in verdict.defects] if verdict is not None else [],
        "review_evidence": verdict.evidence if verdict is not None else None,
        "review_detail": review.detail if review is not None else None,
        "checks": [check.model_dump(mode="json") for check in audit.checks],
        "grader_readiness": grader_readiness(audit.checks).value,
        "cleanup_status": cleanup.status.value if cleanup is not None else None,
        "cleanup_action": proposal.action.value if proposal is not None else None,
        "cleanup_reason": proposal.reason if proposal is not None else None,
        "cleanup_edits": [edit.model_dump(mode="json") for edit in proposal.edits] if proposal is not None else [],
        "cleanup_detail": cleanup.detail if cleanup is not None else None,
        "cleanup_lineage_json": audit.lineage.model_dump_json(exclude_none=True) if audit.lineage is not None else None,
    }
