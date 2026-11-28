# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""New SAMPLE campaigns pick the pinned V5 default; V3 and V4 receipts stay valid."""

import pytest
from finestore.rl.curation import Confidence, ReferenceStatus
from taskcompendium.pipeline.models import Quality, ReviewRecord, ReviewStatus, ReviewVerdict
from taskcompendium.pipeline.review import ReviewBatchResult

from experiments.post_training.task_curation.full_qc import (
    QUALIFICATION_TASK,
    SAMPLE_REVIEWER_VERSION,
    ReviewerVersion,
    local_reviewer,
    qualify_reviewer,
)
from experiments.post_training.task_curation.sample_qc_artifacts import (
    PINNED_REVIEWER_SHA256,
    PINNED_REVIEWER_V4_SHA256,
    PINNED_REVIEWER_V5_SHA256,
    SAMPLE_REVIEWER_SHA256_BY_PAYLOAD_REVISION,
    SAMPLE_REVIEWER_SHAS,
)


def test_sample_campaigns_pick_v5():
    assert SAMPLE_REVIEWER_VERSION is ReviewerVersion.V5
    spec = local_reviewer(SAMPLE_REVIEWER_VERSION)
    assert spec.max_tokens == 32768
    assert spec.temperature == 0.7
    assert spec.model == "glm-5.3"
    assert spec.payload_revision == "6"
    assert spec.cache_revision == PINNED_REVIEWER_V5_SHA256
    assert spec.cache_revision == SAMPLE_REVIEWER_SHAS[SAMPLE_REVIEWER_VERSION]


def test_v4_differs_from_v3_only_in_max_tokens():
    v3 = local_reviewer(ReviewerVersion.V3)
    v4 = local_reviewer(ReviewerVersion.V4)
    assert v3.max_tokens == 8192
    assert v4.max_tokens == 16384
    assert {**v3.identity, "max_tokens": 16384} == v4.identity
    assert PINNED_REVIEWER_SHA256 != PINNED_REVIEWER_V4_SHA256


def test_receipt_validation_admits_all_pinned_profiles_for_payload_revision_6():
    v3_sha = local_reviewer(ReviewerVersion.V3).cache_revision
    v4_sha = local_reviewer(ReviewerVersion.V4).cache_revision
    v5_sha = local_reviewer(ReviewerVersion.V5).cache_revision
    assert SAMPLE_REVIEWER_SHA256_BY_PAYLOAD_REVISION["6"] == frozenset({v3_sha, v4_sha, v5_sha})
    assert v3_sha == PINNED_REVIEWER_SHA256
    assert v4_sha == PINNED_REVIEWER_V4_SHA256
    assert v5_sha == PINNED_REVIEWER_V5_SHA256


class StubReviewer:
    def __init__(self, record: ReviewRecord) -> None:
        self.record = record

    def review(self, tasks, rubric, *, originals=None, cached=None):
        return ReviewBatchResult([self.record], [])


def _conclusive(task_id: str) -> ReviewRecord:
    verdict = ReviewVerdict(
        task_id=task_id,
        quality=Quality.GOOD,
        confidence=Confidence.HIGH,
        reference_status=ReferenceStatus.CONSISTENT,
        defects=[],
        evidence="The canary answered as instructed.",
    )
    return ReviewRecord(task_id=task_id, status=ReviewStatus.REVIEWED, verdict=verdict, detail="ok")


def test_qualification_accepts_a_conclusive_canary():
    review = qualify_reviewer(StubReviewer(_conclusive(QUALIFICATION_TASK.id)))
    assert review.status is ReviewStatus.REVIEWED


def test_qualification_refuses_an_unconclusive_canary():
    unconclusive = ReviewRecord(
        task_id=QUALIFICATION_TASK.id, status=ReviewStatus.UNAVAILABLE, verdict=None, detail="judge down"
    )
    with pytest.raises(ValueError, match="did not conclude"):
        qualify_reviewer(StubReviewer(unconclusive))
    low = _conclusive(QUALIFICATION_TASK.id).model_copy(
        update={"verdict": _conclusive(QUALIFICATION_TASK.id).verdict.model_copy(update={"confidence": Confidence.LOW})}
    )
    with pytest.raises(ValueError, match="did not conclude"):
        qualify_reviewer(StubReviewer(low))
