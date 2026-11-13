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
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import ReviewRubric
from taskcompendium.pipeline.query_cache import CachedRequests
from taskcompendium.pipeline.review import ChatReviewer, ReviewBatchResult
from taskcompendium.pipeline.reviewer_spec import JudgeTier, ReviewerSpec

from experiments.post_training.task_curation.full_qc_review import load_inputs, review_inputs
from experiments.post_training.task_curation.judges import JUDGE_TIERS, LocalJudgeEndpoint

logger = logging.getLogger(__name__)
MODEL_CONFIG = Path("experiments/evaluation/serve/models/zai-org/GLM-5.3.yaml")
V3_MODEL_CONFIG = Path("experiments/evaluation/serve/models/zai-org/GLM-5.3-full-qc-v3.yaml")
WORKER_IMAGE = "vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14"
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


@dataclass(frozen=True)
class ReviewerPolicy:
    model_config: Path
    context_window_tokens: int
    max_tokens: int
    retry_max_tokens: int


REVIEWER_POLICIES: Mapping[ReviewerVersion, ReviewerPolicy] = {
    ReviewerVersion.V2: ReviewerPolicy(MODEL_CONFIG, 262_144, 4_096, 8_192),
    ReviewerVersion.V3: ReviewerPolicy(V3_MODEL_CONFIG, 524_288, 8_192, 16_384),
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

    @property
    def identity(self) -> dict:
        return self._reviewer(self.session).identity

    def _reviewer(self, session: RemoteInferenceSession) -> ChatReviewer:
        endpoint = session.model.endpoint
        return ChatReviewer(
            OpenAIChatClient(endpoint.base_url, endpoint.api_key or "unused", timeout=300),
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


def local_reviewer(version: ReviewerVersion) -> ReviewerSpec:
    definition = JUDGE_TIERS[JudgeTier.HEAVY]
    assert definition.qa_reviewer is not None and definition.model_revision is not None
    policy = REVIEWER_POLICIES[version]
    endpoint = LocalJudgeEndpoint(
        tier=JudgeTier.HEAVY,
        model_config=policy.model_config,
        context_window_tokens=policy.context_window_tokens,
        accelerator=AcceleratorChoice(Platform.GPU, gpu_type="GB200", gpu_count=4, region=ANY_REGION),
        priority=job_pb2.PRIORITY_BAND_BATCH,
        env_vars={"MARIN_TEMP_PREFIX": "s3://marin-us-east-08a"},
        worker_image=WORKER_IMAGE,
        worker_setup_scripts=(
            "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends git",
        ),
        worker_timeout=Duration.from_hours(72),
    )
    return replace(
        definition.qa_reviewer,
        model_revision=definition.model_revision,
        endpoint=endpoint,
        precision="native-fp8-e4m3;kv-cache-fp8-e4m3",
        max_tokens=policy.max_tokens,
        retry_max_tokens=policy.retry_max_tokens,
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
    assert isinstance(endpoint, LocalJudgeEndpoint)
    with endpoint.serve() as session:
        reviewer = ReadyChatReviewer(
            session,
            spec,
            query_cache_root=args.review_cache,
            max_concurrent=32,
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
