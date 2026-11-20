# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete a source's panel coverage with fresh reviews of its top-up tasks.

The carry-forward recovery reports, per source, the exact task IDs whose fresh
review brings the panel's conclusive coverage to ``min(100, retained)``. This
runner freezes those tasks from the current population, reviews them with the
pinned SAMPLE reviewer, and records a typed completion receipt beside the
recovery evidence. The receipts stay informational: qualification and binding
follow the source's own SAMPLE pipeline.
"""

import argparse
import json
import logging
from dataclasses import asdict
from typing import Literal

from marin.execution.artifact import Artifact, ArtifactRecord, result_type_name, write_record
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec

from experiments.post_training.task_curation.full_qc import (
    REVIEW_REQUEST_TIMEOUTS,
    REVIEW_RUBRIC,
    SAMPLE_REVIEWER_VERSION,
    ReadyChatReviewer,
    local_reviewer,
    qualify_reviewer,
)
from experiments.post_training.task_curation.full_qc_inputs import task_contract
from experiments.post_training.task_curation.full_qc_review import (
    ContractBatch,
    review_batch,
    write_observations,
)
from experiments.post_training.task_curation.judges import PinnedLocalJudgeEndpoint
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.published_census import population_documents
from experiments.post_training.task_curation.review_quality import quality_result
from experiments.post_training.task_curation.sample_qc_artifacts import SAMPLE_REVIEWER_SHAS
from infra.marina.apps.rl_data_catalog.server.source_identity import SourceId

logger = logging.getLogger(__name__)


class CompletionTarget(BaseModel):
    """One source's top-up tasks, frozen against its current population."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    population: str
    task_ids: tuple[str, ...]
    topup_evidence: str
    """The recovery record that derived these top-up tasks."""


class CompletionQueue(BaseModel):
    targets: tuple[CompletionTarget, ...]


class CompletionReceipt(Artifact):
    """Typed completion observations for one source's top-up tasks."""

    source_id: SourceId
    task_ids: tuple[str, ...]
    population: str
    reviewer: dict
    rubric: dict
    topup_evidence: str
    qualification_status: Literal["pending"] = "pending"
    attempted_count: int
    result: dict
    records_path: str
    records_sha256: str
    batch_receipts: list[str]


def freeze_topup_tasks(population: SourceTextPopulation, task_ids: tuple[str, ...]) -> tuple[list[TaskSpec], list[dict]]:
    """Freeze the top-up tasks' exact contracts from the current population."""
    wanted = set(task_ids)
    if len(wanted) != len(task_ids):
        raise ValueError("Completion tasks must be distinct")
    tasks: dict[str, TaskSpec] = {}
    runtimes: dict[str, dict] = {}
    for document, runtime in population_documents(population):
        if document["id"] in wanted:
            tasks[document["id"]] = TaskSpec.model_validate_json(json.dumps(document))
            runtimes[document["id"]] = runtime
    missing = sorted(wanted - set(tasks))
    if missing:
        raise ValueError(f"Top-up tasks are absent from the population: {missing[:3]}")
    return [tasks[task_id] for task_id in task_ids], [runtimes[task_id] for task_id in task_ids]


def complete_target(target: CompletionTarget, output_root: str, reviewer) -> CompletionReceipt:
    """Review one source's top-up tasks and record the completion receipt."""
    population = SourceTextPopulation.raw_load(target.population)
    tasks, runtimes = freeze_topup_tasks(population, target.task_ids)
    contracts = [task_contract(task, runtime) for task, runtime in zip(tasks, runtimes, strict=True)]
    output = StoragePath(output_root) / canonical_sha256(target.source_id)[:20]

    def batches():
        reviews, receipt = review_batch(tasks, contracts, REVIEW_RUBRIC, reviewer, output)
        yield ContractBatch(index=0, tasks=tasks, contracts=contracts), reviews, receipt

    observations = write_observations(str(output), batches())
    result = quality_result(observations.reviews)
    receipt = CompletionReceipt(
        path=str(observations.records_path.parent),
        source_id=target.source_id,
        task_ids=tuple(target.task_ids),
        population=population.path,
        reviewer=reviewer.identity,
        rubric=asdict(REVIEW_RUBRIC),
        topup_evidence=target.topup_evidence,
        attempted_count=len(observations.reviews),
        result=result.model_dump(mode="json"),
        records_path=str(observations.records_path),
        records_sha256=file_sha256(observations.records_path),
        batch_receipts=observations.batch_receipts,
    )
    write_record(
        ArtifactRecord(
            output_path=receipt.path, result_type=result_type_name(CompletionReceipt), result=receipt.result_payload()
        )
    )
    logger.info(
        "%s completion: %d reviewed, %d unavailable of %d top-up tasks",
        target.source_id,
        result.reviewed,
        result.unavailable_count,
        len(target.task_ids),
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--queue-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--review-cache", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    queue_path = StoragePath(args.queue)
    if file_sha256(queue_path) != args.queue_sha256:
        raise ValueError("Completion queue changed")
    queue = CompletionQueue.model_validate_json(queue_path.read_text())
    if not args.run:
        print(json.dumps({"targets": len(queue.targets), "tasks": sum(len(t.task_ids) for t in queue.targets)}))
        return
    spec = local_reviewer(SAMPLE_REVIEWER_VERSION)
    if spec.cache_revision != SAMPLE_REVIEWER_SHAS[SAMPLE_REVIEWER_VERSION]:
        raise ValueError("Completion reviews require the pinned default reviewer profile for new campaigns (V4)")
    endpoint = spec.endpoint
    assert isinstance(endpoint, PinnedLocalJudgeEndpoint)
    logger.info("Reviewer: %s", json.dumps(spec.identity, sort_keys=True))
    receipts = {}
    with endpoint.serve() as session:
        reviewer = ReadyChatReviewer(
            session,
            spec,
            args.review_cache,
            max_concurrent=32,
            request_timeout=REVIEW_REQUEST_TIMEOUTS[SAMPLE_REVIEWER_VERSION],
        )
        qualify_reviewer(reviewer)
        for target in queue.targets:
            receipt = complete_target(target, args.output_root, reviewer)
            receipts[receipt.source_id] = receipt.model_dump(mode="json")
    summary = {
        source: {"reviewed": row["result"]["reviewed"], "records": row["records_path"]}
        for source, row in receipts.items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
