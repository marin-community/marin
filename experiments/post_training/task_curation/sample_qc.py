# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Review frozen seed-zero panels without claiming execution qualification."""

import argparse
import json
import logging
from dataclasses import asdict
from uuid import uuid4

import pyarrow.parquet as pq
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import ReviewStatus
from taskcompendium.pipeline.review import Reviewer

from experiments.post_training.task_curation.full_qc import (
    REVIEW_REQUEST_TIMEOUTS,
    REVIEW_RUBRIC,
    ReadyChatReviewer,
    ReviewerVersion,
    local_reviewer,
)
from experiments.post_training.task_curation.full_qc_inputs import task_contract
from experiments.post_training.task_curation.full_qc_review import (
    REVIEW_BATCH_SIZE,
    ContractBatch,
    quality_result,
    review_batch,
    write_observations,
)
from experiments.post_training.task_curation.judges import PinnedLocalJudgeEndpoint
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.sample_qc_artifacts import (
    PINNED_REVIEWER_SHA256,
    SAMPLE_SEED,
    SAMPLE_SIZE,
    SampleInventory,
    SamplePanel,
    SampleQCReview,
)

LOGGER = logging.getLogger(__name__)


def panel_tasks(panel: SamplePanel) -> tuple[list[TaskSpec], list[dict]]:
    """Check source and panel bytes before constructing exact task/runtime contracts."""
    if SampleInventory.model_validate_json(StoragePath(panel.evidence_url).read_text()).qc_panel() != panel:
        raise ValueError("Sample panel differs from its complete-population census")
    population = SourceTextPopulation.raw_load(panel.input_artifact)
    if (
        population.source_id,
        population.dataset_id,
        population.dataset_revision,
        population.source_version,
        population.task_count,
        population.task_specs_sha256,
    ) != (
        panel.source_id,
        panel.dataset_id,
        panel.dataset_revision,
        panel.converter_version,
        panel.retained_tasks,
        panel.task_specs_sha256,
    ):
        raise ValueError(f"Sample panel belongs to another current population: {panel.source_id}")
    if panel.sample_seed != SAMPLE_SEED or panel.sample_size != min(SAMPLE_SIZE, panel.retained_tasks):
        raise ValueError("Sample panel differs from the standard seeded design")
    path = StoragePath(panel.panel_path)
    if file_sha256(path) != panel.panel_sha256:
        raise ValueError(f"Frozen sample panel changed: {panel.source_id}")
    with path.open("rb") as stream:
        rows = pq.read_table(stream).to_pylist()
    tasks = [TaskSpec.model_validate_json(row["task_spec"]) for row in rows]
    if len(tasks) != panel.sample_size or len({task.id for task in tasks}) != len(tasks):
        raise ValueError("Sample panel must contain every distinct selected task exactly once")
    if [task.id for task in tasks] != [row["task_id"] for row in rows]:
        raise ValueError("Sample panel IDs differ from their task contracts")
    return tasks, [task_contract(task, json.loads(row["runtime"])) for task, row in zip(tasks, rows, strict=True)]


def review_panel(panel: SamplePanel, reviewer: Reviewer, output_root: str) -> SampleQCReview:
    """Persist sampled observations and compute Result independently of execution admission."""
    tasks, contracts = panel_tasks(panel)
    output = StoragePath(output_root) / canonical_sha256(panel.source_id)[:20]

    def batches():
        for offset in range(0, len(tasks), REVIEW_BATCH_SIZE):
            batch = ContractBatch(
                offset // REVIEW_BATCH_SIZE,
                tasks[offset : offset + REVIEW_BATCH_SIZE],
                contracts[offset : offset + REVIEW_BATCH_SIZE],
            )
            reviews, receipt = review_batch(batch.tasks, batch.contracts, REVIEW_RUBRIC, reviewer, output)
            yield batch, reviews, receipt

    records_path, all_reviews, receipts = write_observations(str(output), batches())
    result = SampleQCReview(
        path=str(records_path.parent),
        panel=panel,
        reviewer=reviewer.identity,
        rubric=asdict(REVIEW_RUBRIC),
        panel_coverage_complete=all(row.status is ReviewStatus.REVIEWED for row in all_reviews),
        attempted_count=len(all_reviews),
        result=quality_result(all_reviews),
        records_path=str(records_path),
        records_sha256=file_sha256(records_path),
        batch_receipts=receipts,
    )
    write_record(
        ArtifactRecord(
            output_path=result.path, result_type=result_type_name(type(result)), result=result.result_payload()
        )
    )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--queue-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--review-cache", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    queue = StoragePath(args.queue)
    if file_sha256(queue) != args.queue_sha256:
        raise ValueError("Frozen sample queue changed")
    panels = [SampleInventory.model_validate(row).qc_panel() for row in json.loads(queue.read_text())]
    if not panels or len({row.source_id for row in panels}) != len(panels):
        raise ValueError("Sample queue requires distinct source populations")
    spec = local_reviewer(ReviewerVersion.V3)
    if spec.cache_revision != PINNED_REVIEWER_SHA256:
        raise ValueError("SAMPLE QC requires the reviewed literal v3 profile")
    endpoint = spec.endpoint
    assert isinstance(endpoint, PinnedLocalJudgeEndpoint)
    LOGGER.info("Reviewer: %s", json.dumps(spec.identity, sort_keys=True))
    LOGGER.info("Queue: %d sources, %d sampled tasks", len(panels), sum(p.sample_size for p in panels))
    if not args.run:
        return
    campaign = StoragePath(args.output_root) / "campaigns" / uuid4().hex
    with endpoint.serve() as session:
        reviewer = ReadyChatReviewer(
            session,
            spec,
            args.review_cache,
            max_concurrent=32,
            request_timeout=REVIEW_REQUEST_TIMEOUTS[ReviewerVersion.V3],
        )
        for panel in panels:
            result = review_panel(panel, reviewer, args.output_root)
            (campaign / f"{canonical_sha256(panel.source_id)[:20]}.json").write_text(
                json.dumps(
                    {
                        "source_id": panel.source_id,
                        "review_artifact": result.path,
                        "result": result.result.model_dump(mode="json"),
                        "serving_jobs": endpoint.job_ids,
                        "qualification_status": "pending",
                    },
                    indent=2,
                )
                + "\n",
                auto_mkdir=True,
            )
            LOGGER.info("Source review: %s %s", panel.source_id, result.path)


if __name__ == "__main__":
    main()
