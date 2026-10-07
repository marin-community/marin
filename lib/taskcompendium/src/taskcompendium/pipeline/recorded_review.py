# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact-task recorded judgments, with their own provenance and reviewer fallback."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.filesystem.storage_path import StoragePath
from zephyr import counters

from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import ReviewRecord, ReviewRubric, ReviewStatus
from taskcompendium.pipeline.review import Reviewer

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class RecordedReviewEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_sha256: Sha256
    rubric_sha256: Sha256
    review: ReviewRecord
    provenance: dict[str, Any] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_judgment(self) -> Self:
        if self.review.status != ReviewStatus.REVIEWED or self.review.verdict is None:
            raise ValueError("Recorded evidence must contain a reviewed verdict")
        if self.review.task_id != self.review.verdict.task_id:
            raise ValueError("Recorded review and verdict task IDs disagree")
        return self


class RecordedReviewBundle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["recorded-review-v1"]
    provenance: dict[str, Any] = Field(min_length=1)
    records: list[RecordedReviewEvidence]

    @model_validator(mode="after")
    def unique_judgments(self) -> Self:
        keys = [(record.task_sha256, record.rubric_sha256) for record in self.records]
        if len(set(keys)) != len(keys):
            raise ValueError("Recorded evidence contains duplicate task/rubric identities")
        return self


def load_recorded_reviews(path: str, expected_sha256: str) -> RecordedReviewBundle:
    """Read and validate a bundle whose exact bytes match the declared digest."""
    with StoragePath(path).open("rb") as stream:
        payload = stream.read()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Recorded review bundle SHA256 differs from the declared evidence")
    return RecordedReviewBundle.model_validate_json(payload)


@dataclass(frozen=True)
class RecordedReviewer:
    """Reuse exact recorded judgments; delegate unmatched and rewrite tasks."""

    fallback: Reviewer
    evidence_path: str
    evidence_sha256: str

    @property
    def identity(self) -> dict[str, Any]:
        return {
            **self.fallback.identity,
            "recorded_evidence": {
                "path": self.evidence_path,
                "sha256": self.evidence_sha256,
                "schema_version": "recorded-review-v1",
            },
        }

    def review(
        self,
        tasks: Sequence[TaskSpec],
        rubric: ReviewRubric,
        output_path: Path,
        *,
        originals: Mapping[str, TaskSpec] | None = None,
    ) -> list[ReviewRecord]:
        bundle = load_recorded_reviews(self.evidence_path, self.evidence_sha256)
        retained = {(item.task_sha256, item.rubric_sha256): item for item in bundle.records}
        rubric_sha256 = canonical_sha256(asdict(rubric))
        results = {}
        matched = []
        unmatched = []
        if len({task.id for task in tasks}) != len(tasks):
            raise ValueError("Recorded review input has duplicate task IDs")
        for task in tasks:
            # A rewrite request also asks about its original. The recorded
            # source judgment did not inspect that comparison, even if IDs match.
            if originals is not None and task.id in originals:
                unmatched.append(task)
                continue
            task_sha256 = canonical_sha256(task.model_dump(mode="json"))
            evidence = retained.get((task_sha256, rubric_sha256))
            if evidence is None:
                unmatched.append(task)
                continue
            if evidence.review.task_id != task.id:
                raise ValueError("Recorded task hash matches but its review task ID disagrees")
            provenance = {
                "evidence_path": self.evidence_path,
                "evidence_sha256": self.evidence_sha256,
                "task_sha256": task_sha256,
                "rubric_sha256": rubric_sha256,
                "bundle_provenance": bundle.provenance,
                "review_provenance": evidence.provenance,
                "original_detail": evidence.review.detail,
            }
            results[task.id] = evidence.review.model_copy(
                update={"detail": json.dumps({"recorded_review": provenance}, ensure_ascii=False, allow_nan=False)}
            )
            matched.append({"review": results[task.id].model_dump(mode="json"), **provenance})
        metrics = counters.current_stage()
        metrics.update_counter("review/recorded/matched_records", len(matched))
        metrics.update_counter("review/recorded/fallback_records", len(unmatched))
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "recorded-reviews.json").write_text(
            json.dumps({"reviewer": self.identity, "records": matched}, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n"
        )
        if unmatched:
            fallback_records = self.fallback.review(
                unmatched,
                rubric,
                output_path / "fallback",
                originals=(
                    {task.id: originals[task.id] for task in unmatched if task.id in originals}
                    if originals is not None
                    else None
                ),
            )
            expected = {task.id for task in unmatched}
            if len(fallback_records) != len(expected) or {record.task_id for record in fallback_records} != expected:
                raise ValueError("Fallback review records do not match the unmatched tasks")
            results.update((record.task_id, record) for record in fallback_records)
        return [results[task.id] for task in tasks]
