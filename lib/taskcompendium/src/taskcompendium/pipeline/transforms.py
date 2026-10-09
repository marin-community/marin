# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Domain row transformations used by task curation stages."""

import json
from collections.abc import Iterator
from tempfile import SpooledTemporaryFile
from typing import Any

from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.models import ScriptGrader, TaskSpec, VerifyitGrader
from taskcompendium.pipeline.conversion import ConvertedRow, convert_record
from taskcompendium.pipeline.filtering import task_decision
from taskcompendium.pipeline.fingerprints import deduplication_key, semantic_digest
from taskcompendium.pipeline.models import (
    REJECTING_CHECK_STATUSES,
    RESOURCES_OVER_BUDGET,
    CheckResult,
    Confidence,
    Decision,
    Disposition,
    FilterPolicy,
    ImportFailureKind,
    ImportRejection,
    NormalizedTask,
    QualityBasis,
    ReviewRecord,
    SourceRecipe,
    TaskAudit,
)
from taskcompendium.runtime.resources import resource_bytes

GROUP_MEMORY_BYTES = 1024 * 1024


def task_resource_bytes(task: TaskSpec) -> int:
    """Decoded bytes across all roles and build contexts, counting each occurrence."""
    resources = task.resources
    environments = [task.environment_requirements]
    if isinstance(task.grader, ScriptGrader | VerifyitGrader) and task.grader.environment is not None:
        environments.append(task.grader.environment)
    build_files = tuple(
        resource
        for environment in environments
        if environment.docker_build is not None
        for resource in environment.docker_build.files
    )
    return sum(
        len(resource_bytes(resource))
        for resource in (*resources.all, *resources.worker, *resources.oracle, *resources.verifier, *build_files)
    )


def _within_budget(task: TaskSpec, budget: int) -> TaskSpec | ImportRejection:
    size = task_resource_bytes(task)
    if size <= budget:
        return task
    return ImportRejection(
        kind=ImportFailureKind.UNSUPPORTED,
        reason=RESOURCES_OVER_BUDGET,
        detail=f"The task's resources hold {size} bytes; the source allows {budget}",
    )


def normalize_row(record: dict[str, Any], recipe: SourceRecipe) -> dict[str, Any]:
    return admit_converted_row(convert_record(record, recipe), recipe)


def admit_converted_row(converted: ConvertedRow, recipe: SourceRecipe) -> dict[str, Any]:
    """Add resource admission, audit identity, and deduplication keys after conversion."""
    source, task_id = converted.raw.source, converted.raw.id
    raw = {
        "task_id": task_id,
        "source": source.model_dump(),
        "raw_sha256": canonical_sha256(dict(converted.raw.data)),
        "original_path": converted.original_path,
        "data": converted.raw.data,
    }
    result: TaskSpec | NormalizedTask | ImportRejection = converted.result
    audit = TaskAudit(
        task_id=task_id,
        source=source,
        raw=raw,
        normalized=None,
        normalization_rejection=None,
        checks=[],
        review=None,
        decision=None,
        intended_use=recipe.intended_use,
    )
    public_key, semantic_key = task_id, task_id
    if isinstance(result, NormalizedTask):
        audit = audit.model_copy(update={"normalization_changes": result.changes})
        result = result.task
    if isinstance(result, TaskSpec):
        result = _within_budget(result, recipe.resource_budget_bytes)
    if isinstance(result, ImportRejection):
        audit = audit.model_copy(
            update={
                "normalization_rejection": result,
                "decision": Decision(
                    task_id=task_id,
                    disposition=(
                        Disposition.REJECT if result.kind is ImportFailureKind.SOURCE_DEFECT else Disposition.DEFER
                    ),
                    reasons=[f"normalize:{result.reason}"],
                ),
            }
        )
    else:
        audit = audit.model_copy(update={"normalized": result})
        public_key = deduplication_key(result)
        semantic_key = semantic_digest(result, include_reference=True)
    return {
        "locator": source.row,
        "public_key": public_key,
        "semantic_key": semantic_key,
        "audit": audit.model_dump(mode="json"),
    }


def public_group_key(record: dict[str, Any]) -> str:
    return record["public_key"]


def source_locator_order(record: dict[str, Any]) -> str:
    path, index = record["locator"].rsplit(":", 1)
    return f"{path}:{int(index):020d}"


def deduplicate_group(_: str, records: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Cut every conflicting reference and keep the first identical row in source order."""
    # Spooling prevents a frequently repeated prompt from filling worker memory.
    references = set()
    with SpooledTemporaryFile(max_size=GROUP_MEMORY_BYTES, mode="w+t") as spool:
        for record in records:
            if len(references) < 2:
                references.add(record["semantic_key"])
            spool.write(json.dumps(record) + "\n")
        spool.seek(0)
        first_id = None
        for line in spool:
            record = json.loads(line)
            audit = record["audit"]
            if audit["decision"] is None:
                if len(references) > 1:
                    audit["decision"] = Decision(
                        task_id=audit["task_id"], disposition=Disposition.REJECT, reasons=["conflicting_references"]
                    ).model_dump(mode="json")
                elif first_id is not None:
                    audit["decision"] = Decision(
                        task_id=audit["task_id"],
                        disposition=Disposition.REJECT,
                        reasons=["exact_semantic_duplicate"],
                        duplicate_of=first_id,
                    ).model_dump(mode="json")
                else:
                    first_id = audit["task_id"]
            yield audit


def review_record(row: dict[str, Any]) -> ReviewRecord:
    """Recover the recorded review from curation columns."""
    verdict = None
    if row["review_quality"] is not None:
        verdict = {
            "task_id": row["task_id"],
            "quality": row["review_quality"],
            "confidence": row["review_confidence"],
            "reference_status": row["review_reference_status"],
            "defects": row["review_defects"],
            "evidence": row["review_evidence"],
        }
    return ReviewRecord.model_validate_json(
        json.dumps(
            {
                "task_id": row["task_id"],
                "status": row["review_status"] or "unavailable",
                "verdict": verdict,
                "detail": row["review_detail"] if row["review_status"] else "No quality assessment available",
            }
        )
    )


def filter_row(row: dict[str, Any], policy: FilterPolicy) -> dict[str, Any]:
    if (
        row["normalization_reason"] is not None
        or row["duplicate_of"] is not None
        or "conflicting_references" in row["filter_reasons"]
    ):
        return row
    if row["quality_basis"] in (
        QualityBasis.UNREVIEWED,
        QualityBasis.INFERRED_FROM_SOURCE,
        QualityBasis.SOURCE_REJECTED,
        QualityBasis.SOURCE_INCOMPLETE,
    ):
        # These are source-level decisions, not missing per-task model responses.
        # Cheap failures always retain their own rejection evidence.
        failed = [f"check:{check['check']}" for check in row["checks"] if check["status"] in REJECTING_CHECK_STATUSES]
        if failed:
            return {**row, "filter_status": "reject", "filter_reasons": failed}
        if row["quality_basis"] == QualityBasis.INFERRED_FROM_SOURCE and policy.minimum_confidence == Confidence.HIGH:
            return {**row, "filter_status": "defer", "filter_reasons": ["source_quality:requires_direct_review"]}
        return row
    review = review_record(row)
    decision = task_decision(
        row["task_id"], [CheckResult.model_validate(check) for check in row["checks"]], review, policy
    )
    return {
        **row,
        "filter_status": decision.disposition.value,
        "filter_reasons": decision.reasons,
        "duplicate_of": decision.duplicate_of,
    }


def is_accepted(row: dict[str, Any]) -> bool:
    return row["filter_status"] == Disposition.KEEP.value
