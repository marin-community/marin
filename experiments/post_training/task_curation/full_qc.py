# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Review qualified retained populations with batch-priority local GLM inference."""

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pyarrow.parquet as pq
from iris.rpc import job_pb2
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.inference.openai_chat import OpenAIChatClient
from pydantic import BaseModel
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.grading_environment import GradingEnvironmentStatus, assess_grading_environment
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import ReviewRubric
from taskcompendium.pipeline.oracle_filter import load_oracle_filter
from taskcompendium.pipeline.review import ChatReviewer
from taskcompendium.pipeline.reviewer_spec import JudgeTier, ReviewerSpec

from experiments.post_training.task_curation.full_qc_artifacts import FullQCInputs
from experiments.post_training.task_curation.full_qc_review import load_inputs, review_inputs
from experiments.post_training.task_curation.judges import JUDGE_TIERS, LocalJudgeEndpoint
from experiments.post_training.task_curation.published_census import population_documents

logger = logging.getLogger(__name__)
MODEL_CONFIG = Path("experiments/evaluation/serve/models/zai-org/GLM-5.3.yaml")
WORKER_IMAGE = "vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14"
REVIEW_RUBRIC = ReviewRubric(
    id="atlas-retained-full-qc",
    version="2026.11.12.1",
    criteria=(
        "Review the exact retained task, including its public instructions, private grader and references.",
        "Identify supported content, reference, environment, resource, leakage and grading-boundary defects.",
        "Do not treat successful qualification as proof that the task or reference is correct.",
    ),
)


class ReviewWork(BaseModel):
    inputs_path: str
    qualification_receipt: str
    output_path: str


def qualify_inputs(inputs: FullQCInputs, receipt_path: str) -> str:
    """Check complete execution evidence before allocating reviewer workers."""
    receipt = load_oracle_filter(receipt_path)
    index_path = StoragePath(inputs.index_path)
    if file_sha256(index_path) != inputs.index_sha256:
        raise ValueError("Prepared contract index changed")
    with index_path.open("rb") as stream:
        contracts = pq.read_table(stream).to_pylist()
    index = {row["task_id"]: row for row in contracts}
    if len(index) != len(contracts) or len(index) != inputs.retained_count:
        raise ValueError("Prepared retained population is incomplete")
    seen = set()
    retained = set()
    for document, runtime in population_documents(inputs.population):
        task = TaskSpec.model_validate(document)
        if task.id in seen:
            raise ValueError(f"Duplicate task in qualification population: {task.id}")
        seen.add(task.id)
        if task.id not in receipt.tasks:
            raise ValueError(f"Qualification omits population task: {task.id}")
        entry = receipt.tasks[task.id]
        contract = {
            "task_id": task.id,
            "task_sha256": canonical_sha256(task.model_dump(mode="json")),
            "runtime_sha256": canonical_sha256(runtime),
        }
        if (entry.task_sha256, entry.runtime_sha256) != (contract["task_sha256"], contract["runtime_sha256"]):
            raise ValueError(f"Qualification belongs to another task or runtime: {task.id}")
        if task.id not in index:
            continue
        if contract != index[task.id]:
            raise ValueError(f"Task differs from prepared retained contract: {task.id}")
        if not receipt.admits(task, runtime):
            raise ValueError(f"Retained task lacks successful execution qualification: {task.id}")
        if assess_grading_environment(task).status is not GradingEnvironmentStatus.MATCHED:
            raise ValueError(f"Retained task lacks a matched grading boundary: {task.id}")
        retained.add(task.id)
    if seen != set(receipt.tasks) or len(seen) != inputs.retention.raw_count or retained != set(index):
        raise ValueError("Qualification does not cover the complete frozen population")
    return file_sha256(StoragePath(receipt_path))


def local_reviewer() -> ReviewerSpec:
    """Pin the public checkpoint separately from the existing unverified relay."""
    definition = JUDGE_TIERS[JudgeTier.HEAVY]
    assert definition.qa_reviewer is not None and definition.model_revision is not None
    endpoint = LocalJudgeEndpoint(
        tier=JudgeTier.HEAVY,
        model_config=MODEL_CONFIG,
        accelerator=AcceleratorChoice(Platform.GPU, gpu_type="GB200", gpu_count=4, target_cluster="cw-us-east-08a"),
        priority=job_pb2.PRIORITY_BAND_BATCH,
        env_vars={"MARIN_TEMP_PREFIX": "s3://marin-us-east-08a"},
        worker_image=WORKER_IMAGE,
        worker_timeout=Duration.from_hours(72),
    )
    return replace(
        definition.qa_reviewer,
        model_revision=definition.model_revision,
        endpoint=endpoint,
        precision="native-fp8-e4m3;kv-cache-fp8-e4m3",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, help="JSON list of input, qualification and output paths")
    parser.add_argument("--campaign-output", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    work = [ReviewWork.model_validate(row) for row in json.loads(StoragePath(args.work).read_text())]
    if not work or len({row.inputs_path for row in work}) != len(work):
        raise ValueError("A campaign requires distinct prepared populations")
    prepared = []
    for row in work:
        inputs = load_inputs(row.inputs_path)
        digest = qualify_inputs(inputs, row.qualification_receipt)
        prepared.append((row, inputs, digest))
        logger.info("Qualified %s: %d retained tasks", inputs.source_id, inputs.retained_count)
    spec = local_reviewer()
    logger.info("Reviewer: %s", json.dumps(spec.identity, sort_keys=True))
    if not args.run:
        return
    campaign_path = StoragePath(args.campaign_output) / "attempts" / uuid4().hex
    endpoint = spec.endpoint
    assert isinstance(endpoint, LocalJudgeEndpoint)
    with endpoint.connect() as connection:
        reviewer = ChatReviewer(
            OpenAIChatClient(connection.base_url, connection.api_key, timeout=300),
            spec,
            query_cache_root=str(StoragePath(args.campaign_output) / "request-cache"),
            max_concurrent=32,
        )
        for row, inputs, digest in prepared:
            result = review_inputs(inputs, reviewer, REVIEW_RUBRIC, row.output_path)
            evidence = {
                "source_id": inputs.source_id,
                "review_artifact": result.path,
                "qualification_receipt": row.qualification_receipt,
                "qualification_sha256": digest,
                "reviewer": spec.identity,
                "serving_jobs": endpoint.job_ids,
            }
            path = campaign_path / f"{canonical_sha256(inputs.source_id)[:20]}.json"
            with path.open("w", auto_mkdir=True) as stream:
                json.dump(evidence, stream, ensure_ascii=False, indent=2)
            logger.info("Source review evidence: %s", path)


if __name__ == "__main__":
    main()
