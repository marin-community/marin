# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Review retained content with batch-priority GLM, preserving qualification status."""

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from fray.types import ANY_REGION
from iris.rpc import job_pb2
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.inference.iris import RemoteInferenceSession
from marin.inference.openai_chat import OpenAIChatClient
from pydantic import BaseModel
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import ReviewRecord, ReviewRubric, is_conclusive
from taskcompendium.pipeline.query_cache import CachedRequests
from taskcompendium.pipeline.review import ChatReviewer, ReviewBatchResult, review_tasks
from taskcompendium.pipeline.reviewer_spec import REVIEW_PAYLOAD_REVISION, JudgeTier, ReviewerSpec

from experiments.post_training.task_curation.full_qc_review import load_inputs, review_inputs
from experiments.post_training.task_curation.judges import (
    JUDGE_TIERS,
    LocalJudgeEndpoint,
    LocalJudgeModel,
    PinnedLocalJudgeEndpoint,
)

logger = logging.getLogger(__name__)
MODEL_CONFIG = Path("experiments/evaluation/serve/models/zai-org/GLM-5.3.yaml")
V3_MODEL_CONFIG = Path("experiments/evaluation/serve/models/zai-org/GLM-5.3-full-qc-v3.yaml")
V6_MODEL_CONFIG = Path("experiments/evaluation/serve/models/minimaxai/MiniMax-M3-full-qc-v6.yaml")
WORKER_IMAGE = "vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14"
# Standing policy: data jobs default to a three-day timeout.
FULL_QC_JOB_TIMEOUT = Duration.from_hours(72)
REVIEW_RUBRIC = ReviewRubric(
    id="atlas-retained-full-qc",
    version="2026.11.13.1",
    criteria=(
        "Review the exact retained task, including its public instructions, private grader and references.",
        "Identify supported content, reference, leakage and rubric defects.",
        "Execution qualification is pending and assessed separately; do not infer execution from static evidence.",
    ),
)


class ReviewerVersion(StrEnum):
    V2 = "v2"
    V3 = "v3"
    V4 = "v4"
    V5 = "v5"
    V6 = "v6"


SAMPLE_REVIEWER_VERSION = ReviewerVersion.V6
"""The pinned default SAMPLE reviewer profile for new campaigns (user ruling
2026-11-29: V6 = MiniMax-M3-MXFP8 on 8xH100, off GLM-5.3 — the GLM-5.3 canary
never concludes within its token budget at either effort setting; V2-V5
receipts stay untouched)."""


REVIEWER_SERVING_GPU_TYPE: dict[ReviewerVersion, str] = {
    ReviewerVersion.V2: "GB200",
    ReviewerVersion.V3: "GB200",
    ReviewerVersion.V4: "GB200",
    ReviewerVersion.V5: "GB200",
    ReviewerVersion.V6: "H100",
}
"""The serving accelerator each reviewer profile's tree requests (the
serving-root check validates the root against it at submission time)."""


LONG_REQUEST_TIMEOUT = 1200
REVIEW_REQUEST_TIMEOUTS = {
    ReviewerVersion.V2: 300,
    ReviewerVersion.V3: LONG_REQUEST_TIMEOUT,
    ReviewerVersion.V4: LONG_REQUEST_TIMEOUT,
    ReviewerVersion.V5: LONG_REQUEST_TIMEOUT,
    ReviewerVersion.V6: LONG_REQUEST_TIMEOUT,
}


class ReviewWork(BaseModel):
    inputs_path: str
    output_path: str


@dataclass(frozen=True)
class ReadyChatReviewer:
    """Reconnect to a ready serving task before each uncached review batch."""

    session: RemoteInferenceSession
    spec: ReviewerSpec
    query_cache_root: str
    max_concurrent: int
    request_timeout: float

    @property
    def identity(self) -> dict:
        return self._reviewer(self.session).identity

    def _reviewer(self, session: RemoteInferenceSession) -> ChatReviewer:
        endpoint = session.model.endpoint
        return ChatReviewer(
            OpenAIChatClient(endpoint.base_url, endpoint.api_key or "unused", timeout=self.request_timeout),
            self.spec,
            query_cache_root=self.query_cache_root,
            max_concurrent=self.max_concurrent,
        )

    def read_cache(self, batches: Sequence[Sequence[TaskSpec]], rubric: ReviewRubric) -> list[CachedRequests | None]:
        return self._reviewer(self.session).read_cache(batches, rubric)

    def review(
        self,
        tasks: Sequence[TaskSpec],
        rubric: ReviewRubric,
        *,
        originals: Mapping[str, TaskSpec] | None = None,
        cached: CachedRequests | None = None,
    ) -> ReviewBatchResult:
        current = self.session.ready_direct_session()
        return self._reviewer(current).review(tasks, rubric, originals=originals, cached=cached)


QUALIFICATION_TASK = TaskSpec(
    id="sample-profile-qualification",
    context=ConversationInput(events=(TextMessage(role="user", content="Return the single word: ready."),)),
    environment_requirements=EnvironmentRequirements(),
    answer_type=AnswerType.TEXT,
    answer_format=PlainText(),
    grader=NoGrader(reason="profile qualification fixture"),
    source=Source(dataset="profile-qualification", revision="fixture", row="qualification", importer_revision="1"),
)
"""The fixed canary task a new reviewer profile reviews before it serves a campaign."""


def qualify_reviewer(reviewer) -> ReviewRecord:
    """Qualify a reviewer profile before it serves: one canary review must conclude.

    The standard profile qualification: the pinned identity is already
    validated by the caller's literal-profile check; this proves the serving
    session produces a usable conclusive verdict — the shared ``is_conclusive``
    rule — before any campaign task is attempted.
    """
    result = review_tasks([QUALIFICATION_TASK], REVIEW_RUBRIC, reviewer, cached=None)
    if len(result.reviews) != 1:
        raise ValueError("Profile qualification must observe exactly one canary review")
    review = result.reviews[0]
    if not is_conclusive(review):
        raise ValueError(f"Profile qualification failed: the canary review did not conclude: {review.detail}")
    return review


def _local_endpoint(model_config: Path, context_window_tokens: int) -> LocalJudgeEndpoint:
    # Placement note: only ROOT jobs federate in Iris, so this child serving request's
    # cluster would be ignored - the submitter roots the review job on a cluster that
    # places 4xGB200 (require_review_serving_root enforces it at submission time). The
    # serving holds non-preemptible capacity: long reviewer serving must not be evicted
    # mid-review.
    return LocalJudgeEndpoint(
        tier=JudgeTier.HEAVY,
        model_config=model_config,
        context_window_tokens=context_window_tokens,
        accelerator=AcceleratorChoice(
            Platform.GPU,
            gpu_type="GB200",
            gpu_count=4,
            region=ANY_REGION,
            preemptible=False,
        ),
        priority=job_pb2.PRIORITY_BAND_BATCH,
        env_vars={"MARIN_TEMP_PREFIX": "s3://marin-us-east-08a"},
        worker_image=WORKER_IMAGE,
        worker_setup_scripts=(
            "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends git",
        ),
        worker_timeout=FULL_QC_JOB_TIMEOUT,
    )


def _v6_local_endpoint(model_config: Path, context_window_tokens: int) -> LocalJudgeEndpoint:
    # Placement note: only ROOT jobs federate in Iris, so this child serving request's
    # cluster would be ignored - the submitter roots the review job on a cluster that
    # places 8xH100 (require_review_serving_root enforces it at submission time;
    # cw-rno2a is the reserved H100 fleet). The serving holds non-preemptible
    # capacity: long reviewer serving must not be evicted mid-review.
    return LocalJudgeEndpoint(
        tier=JudgeTier.MEDIUM,
        model_config=model_config,
        context_window_tokens=context_window_tokens,
        accelerator=AcceleratorChoice(
            Platform.GPU,
            gpu_type="H100",
            gpu_count=8,
            region=ANY_REGION,
            preemptible=False,
        ),
        priority=job_pb2.PRIORITY_BAND_BATCH,
        env_vars={"MARIN_TEMP_PREFIX": "s3://marin-rno2a"},
        worker_image=WORKER_IMAGE,
        worker_setup_scripts=(
            "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends git",
        ),
        worker_timeout=FULL_QC_JOB_TIMEOUT,
    )


def local_reviewer(version: ReviewerVersion, *, payload_revision: str = REVIEW_PAYLOAD_REVISION) -> ReviewerSpec:
    if version is ReviewerVersion.V6:
        # User ruling 2026-11-29: V6 moves the SAMPLE reviewer off GLM-5.3 onto
        # MiniMax-M3-MXFP8 (8xH100, cw-rno2a) - the GLM-5.3 canary never emits
        # the review_task tool call within its token budget at either effort
        # setting (probe v6: finish_reason length at 32768 with zero content and
        # zero tool calls, byte-identical with and without the effort kwarg).
        # The context is 262144 tokens; a longer review prompt fails closed as a
        # typed context outcome (ReviewStatus.UNAVAILABLE), never truncated.
        pin = LocalJudgeModel(
            model_identity="MiniMaxAI/MiniMax-M3-MXFP8",
            model_revision="c5454eb03678d8710e54a4e0fc681b9f3b4a3dba",
            api_model="minimax-m3",
            context_window_tokens=262144,
        )
        return ReviewerSpec(
            tier=JudgeTier.MEDIUM,
            model=pin.api_model,
            model_revision=pin.model_revision,
            endpoint=PinnedLocalJudgeEndpoint(_v6_local_endpoint(V6_MODEL_CONFIG, pin.context_window_tokens), pin),
            precision="native-mxfp8",
            temperature=0.7,
            top_p=0.95,
            top_k=0,
            reasoning_effort="low",
            max_tokens=32768,
            retry_max_tokens=32768,
            max_attempts=3,
            prompt_budget=262000,
            retry_prompt_budget=262000,
            base_rubric_version="task-quality-v2",
            base_rubric_sha256="af7ae2a33e3f73d17f27907d930beae35453a22b63ef6f28138c50a8d65131eb",
            payload_revision=payload_revision,
        )
    if version is ReviewerVersion.V5:
        # User ruling 2026-11-28: V5 is V4 with max_tokens 32768 and
        # retry_max_tokens 32768, everything else identical - the V4 canary
        # truncates its structured response on 08a/GB200 (two lanes), and the
        # completion needs double the budget to conclude.
        v4 = local_reviewer(ReviewerVersion.V4, payload_revision=payload_revision)
        return replace(v4, max_tokens=32768, retry_max_tokens=32768)
    if version is ReviewerVersion.V4:
        # User ruling 2026-11-19: V4 is V3 with max_tokens 16384, everything
        # else identical, so long-fixture reviews stop truncating. V4 is the
        # default for new SAMPLE campaigns; V3 receipts stay untouched.
        v4 = local_reviewer(ReviewerVersion.V3, payload_revision=payload_revision)
        return replace(v4, max_tokens=16384)
    if version is ReviewerVersion.V3:
        pin = LocalJudgeModel(
            model_identity="zai-org/GLM-5.3",
            model_revision="aca966e4e02791568aa6a4ced368624b3d897f42",
            api_model="glm-5.3",
            context_window_tokens=524288,
        )
        return ReviewerSpec(
            tier=JudgeTier.HEAVY,
            model=pin.api_model,
            model_revision=pin.model_revision,
            endpoint=PinnedLocalJudgeEndpoint(_local_endpoint(V3_MODEL_CONFIG, pin.context_window_tokens), pin),
            precision="native-fp8-e4m3;kv-cache-fp8-e4m3",
            temperature=0.7,
            top_p=0.95,
            top_k=0,
            reasoning_effort="low",
            max_tokens=8192,
            retry_max_tokens=16384,
            max_attempts=3,
            prompt_budget=512000,
            retry_prompt_budget=512000,
            base_rubric_version="task-quality-v2",
            base_rubric_sha256="af7ae2a33e3f73d17f27907d930beae35453a22b63ef6f28138c50a8d65131eb",
            payload_revision=payload_revision,
        )
    if version is not ReviewerVersion.V2:
        raise ValueError(f"Unknown FULL-QC reviewer version: {version}")
    definition = JUDGE_TIERS[JudgeTier.HEAVY]
    assert definition.qa_reviewer is not None and definition.model_revision is not None
    return replace(
        definition.qa_reviewer,
        payload_revision=payload_revision,
        model_revision=definition.model_revision,
        endpoint=_local_endpoint(MODEL_CONFIG, definition.context_window_tokens),
        precision="native-fp8-e4m3;kv-cache-fp8-e4m3",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, help="JSON list of prepared input and review output paths")
    parser.add_argument("--campaign-output", required=True)
    parser.add_argument("--review-cache", required=True)
    parser.add_argument("--reviewer-version", type=ReviewerVersion, choices=list(ReviewerVersion), required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    work = [ReviewWork.model_validate(row) for row in json.loads(StoragePath(args.work).read_text())]
    if not work or len({row.inputs_path for row in work}) != len(work):
        raise ValueError("A campaign requires distinct prepared populations")
    prepared = []
    for row in work:
        inputs = load_inputs(row.inputs_path)
        prepared.append((row, inputs))
        logger.info("Content review planned for %s: %d retained tasks", inputs.source_id, inputs.retained_count)
    spec = local_reviewer(args.reviewer_version)
    logger.info("Reviewer: %s", json.dumps(spec.identity, sort_keys=True))
    if not args.run:
        return
    campaign_path = StoragePath(args.campaign_output) / "attempts" / uuid4().hex
    endpoint = spec.endpoint
    assert isinstance(endpoint, (LocalJudgeEndpoint, PinnedLocalJudgeEndpoint))
    with endpoint.serve() as session:
        reviewer = ReadyChatReviewer(
            session,
            spec,
            query_cache_root=args.review_cache,
            max_concurrent=32,
            request_timeout=REVIEW_REQUEST_TIMEOUTS[args.reviewer_version],
        )
        for row, inputs in prepared:
            result = review_inputs(inputs, reviewer, REVIEW_RUBRIC, row.output_path)
            evidence = {
                "source_id": inputs.source_id,
                "review_artifact": result.path,
                "qualification_status": "pending",
                "admission_status": "not_assessed",
                "reviewer": spec.identity,
                "serving_jobs": endpoint.job_ids,
            }
            path = campaign_path / f"{canonical_sha256(inputs.source_id)[:20]}.json"
            with path.open("w", auto_mkdir=True) as stream:
                json.dump(evidence, stream, ensure_ascii=False, indent=2)
            logger.info("Source review evidence: %s", path)


if __name__ == "__main__":
    main()
