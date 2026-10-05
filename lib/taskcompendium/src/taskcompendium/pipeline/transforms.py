# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Domain row transformations used by task curation stages."""

import json
from collections.abc import Iterator
from tempfile import SpooledTemporaryFile
from typing import Any

from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.filtering import task_decision
from taskcompendium.pipeline.fingerprints import deduplication_key, semantic_digest
from taskcompendium.pipeline.models import (
    CheckResult,
    DatasetRecipe,
    Decision,
    Disposition,
    FilterPolicy,
    ImportRejection,
    NormalizedTask,
    RawRow,
    ReviewRecord,
    TaskAudit,
)

GROUP_MEMORY_BYTES = 1024 * 1024


def canonical_merge_record(row: dict[str, Any]) -> dict[str, Any]:
    task = TaskSpec.model_validate_json(row["task_json"]) if row["task_json"] is not None else None
    return {
        "public_key": deduplication_key(task) if task is not None else row["task_id"],
        "semantic_key": semantic_digest(task, include_reference=True) if task is not None else row["task_id"],
        "row": row,
    }


def canonical_representative_order(record: dict[str, Any]) -> str:
    row = record["row"]
    return json.dumps(
        [
            int(row["intended_use"] != "eval"),
            row["source_dataset"],
            row["source_revision"],
            row["source_row"],
            row["task_id"],
        ]
    )


def canonicalize_group(_: str, records: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Choose a deterministic accepted representative and retain every audit row."""
    references = set()
    representative = None
    has_eval = False
    with SpooledTemporaryFile(max_size=GROUP_MEMORY_BYTES, mode="w+t") as spool:
        for record in records:
            row = record["row"]
            has_eval |= row["intended_use"] == "eval"
            if row["filter_status"] == "keep":
                if len(references) < 2:
                    references.add(record["semantic_key"])
                if representative is None:
                    representative = row["task_id"]
            spool.write(json.dumps(record) + "\n")
        spool.seek(0)
        for line in spool:
            row = json.loads(line)["row"]
            if row["filter_status"] == "keep":
                reason = None
                if len(references) > 1:
                    reason = "cross_source_conflicting_verifier_contracts"
                elif has_eval and row["intended_use"] != "eval":
                    reason = "evaluation_overlap"
                elif row["task_id"] != representative:
                    reason = "cross_source_exact_duplicate"
                    row["duplicate_of"] = representative
                if reason is not None:
                    row["filter_status"] = "reject"
                    row["filter_reasons"] = [*row["filter_reasons"], reason]
            yield row


def selected_view(row: dict[str, Any], view: str) -> bool:
    if row["filter_status"] != "keep":
        return False
    if view == "executable":
        return row["grader_readiness"] == "ready"
    return view == "accepted" or row["intended_use"] == view


def normalize_row(record: dict[str, Any], recipe: DatasetRecipe) -> dict[str, Any]:
    source = Source(
        dataset=recipe.source.dataset,
        revision=recipe.source.revision,
        row=f"{recipe.source.config}:{recipe.source.split}:{record['locator']}",
        importer_revision=recipe.version,
    )
    task_id = f"{recipe.name}-{canonical_sha256(source.model_dump())}"
    raw = {
        "task_id": task_id,
        "source": source.model_dump(),
        "raw_sha256": canonical_sha256(record["data"]),
        "data": record["data"],
    }
    result = recipe.pipeline.normalize(RawRow(task_id, source, record["data"]))
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
    if isinstance(result, ImportRejection):
        audit = audit.model_copy(
            update={
                "normalization_rejection": result,
                "decision": Decision(
                    task_id=task_id,
                    disposition=Disposition.REJECT,
                    reasons=[f"normalize:{result.reason}", result.detail],
                ),
            }
        )
    else:
        if result.id != task_id or result.source != source:
            raise ValueError("A converter must retain its supplied task identity and source provenance")
        audit = audit.model_copy(update={"normalized": result})
        public_key = deduplication_key(result)
        semantic_key = semantic_digest(result, include_reference=True)
    return {
        "locator": record["locator"],
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


def filter_row(row: dict[str, Any], policy: FilterPolicy) -> dict[str, Any]:
    if (
        row["normalization_reason"] is not None
        or row["duplicate_of"] is not None
        or "conflicting_references" in row["filter_reasons"]
    ):
        return row
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
    review = ReviewRecord.model_validate_json(
        json.dumps(
            {
                "task_id": row["task_id"],
                "status": row["review_status"] or "unavailable",
                "verdict": verdict,
                "detail": row["review_detail"] or "No quality assessment available",
            }
        )
    )
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
