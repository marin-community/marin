# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Columnar task annotations for curation exports."""

import json
from typing import Any

import pyarrow as pa

from taskcompendium.pipeline.conversion import normalization_columns
from taskcompendium.pipeline.models import QualityBasis, ReviewStatus, TaskAudit
from taskcompendium.pipeline.verification import grader_readiness

TASK_SCHEMA = pa.schema(
    [
        ("task_id", pa.string()),
        ("source_dataset", pa.string()),
        ("source_revision", pa.string()),
        ("source_row", pa.string()),
        ("original_path", pa.string()),
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
        ("normalization_kind", pa.string()),
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
        ("quality_basis", pa.string()),
        ("source_quality_report", pa.string()),
        ("checks", pa.list_(pa.struct([("check", pa.string()), ("status", pa.string()), ("detail", pa.string())]))),
        ("grader_readiness", pa.string()),
        ("admission", pa.string()),
    ],
    metadata={b"taskcompendium.curation_schema": b"10"},
)


IDENTITY_FIELDS = [
    ("task_id", pa.string()),
    ("source_locator", pa.string()),
    ("raw_input_sha256", pa.string()),
    ("raw_sha256", pa.string()),
]
RAW_SCHEMA = pa.schema(IDENTITY_FIELDS)
NORMALIZED_COLUMNS = (
    "task_id",
    "source_dataset",
    "source_revision",
    "source_row",
    "original_path",
    "task_json",
    "normalization_kind",
    "normalization_reason",
    "normalization_detail",
    "normalization_changes",
)

NORMALIZED_SCHEMA = pa.schema([*(TASK_SCHEMA.field(column) for column in NORMALIZED_COLUMNS), *IDENTITY_FIELDS[1:]])


def audit_columns(audit: TaskAudit) -> dict[str, Any]:
    """Expose queryable annotations while keeping arbitrary source/task payloads lossless."""
    review = audit.review
    verdict = review.verdict if review is not None else None
    decision = audit.decision
    rejection = audit.normalization_rejection
    quality_basis = audit.quality_basis
    if quality_basis is None and review is not None and review.status == ReviewStatus.REVIEWED:
        quality_basis = QualityBasis.DIRECT_REVIEW
    return {
        **normalization_columns(
            audit.task_id,
            audit.source,
            audit.normalized,
            rejection,
            audit.normalization_changes,
            audit.raw.get("original_path") if audit.raw is not None else None,
        ),
        "intended_use": audit.intended_use.value if audit.intended_use is not None else None,
        "raw_json": json.dumps(audit.raw, ensure_ascii=False, allow_nan=False) if audit.raw is not None else None,
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
        "quality_basis": quality_basis.value if quality_basis is not None else None,
        "source_quality_report": audit.source_quality_report,
        "checks": [check.model_dump(mode="json") for check in audit.checks],
        "grader_readiness": grader_readiness(audit.checks).value,
        "admission": None,
    }
