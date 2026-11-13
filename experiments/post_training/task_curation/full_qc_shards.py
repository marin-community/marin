# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Review disjoint retained batches; assemble source evidence from their receipts."""

import argparse
import json
import logging
from collections import Counter
from dataclasses import replace
from uuid import uuid4

from iris.client.client import iris_ctx
from iris.cluster.types import JobName
from pydantic import BaseModel
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.full_qc import (
    REVIEW_RUBRIC,
    ReadyChatReviewer,
    ReviewerVersion,
    local_reviewer,
)
from experiments.post_training.task_curation.full_qc_artifacts import FullQCReviewShard
from experiments.post_training.task_curation.full_qc_review import (
    SHARD_REQUEST_CONCURRENCY,
    assemble_shards,
    load_inputs,
    review_shard,
    shard_contract_matches,
)
from experiments.post_training.task_curation.judges import LocalJudgeEndpoint, PinnedLocalJudgeEndpoint

logger = logging.getLogger(__name__)


class ShardWork(BaseModel):
    inputs_path: str
    batch_output_path: str
    shard_output_path: str
    assembly_output_path: str
    batch_offset: int
    prompt_budget: int


def write_pointer(result: FullQCReviewShard, path: StoragePath) -> None:
    """Index an immutable shard result by its record digest."""
    record = StoragePath(result.path) / ".artifact.json"
    with path.open("w", auto_mkdir=True) as stream:
        json.dump({"path": result.path, "sha256": file_sha256(record)}, stream)


def read_shard(path: StoragePath) -> FullQCReviewShard:
    """Resolve a shard pointer and verify its immutable record before loading."""
    pointer = json.loads(path.read_text())
    record = StoragePath(pointer["path"]) / ".artifact.json"
    if file_sha256(record) != pointer["sha256"]:
        raise ValueError(f"Review shard evidence changed: {record}")
    return FullQCReviewShard.raw_load(pointer["path"])


def review_work(work: list[ShardWork], args: argparse.Namespace) -> None:
    spec = local_reviewer(args.reviewer_version)
    endpoint = spec.endpoint
    assert isinstance(endpoint, (LocalJudgeEndpoint, PinnedLocalJudgeEndpoint))
    current_root = iris_ctx().job_id
    if current_root is None:
        raise ValueError("Shard review requires a coordinator job identity")
    completed_prefixes = tuple(f"{root}/inference-" for root in (current_root, *args.resume_roots))
    pending = []
    for row in work:
        inputs = load_inputs(row.inputs_path)
        source_spec = replace(spec, prompt_budget=row.prompt_budget, retry_prompt_budget=row.prompt_budget)
        pointer = StoragePath(row.shard_output_path) / f"{args.shard_index:03}" / "latest.json"
        if pointer.exists():
            previous = read_shard(pointer)
            if previous.worker_jobs and all(job.startswith(completed_prefixes) for job in previous.worker_jobs):
                if (
                    canonical_sha256(previous.reviewer) != canonical_sha256({"mode": "chat", **source_spec.identity})
                    or previous.shard_index != args.shard_index
                    or not shard_contract_matches(
                        previous,
                        inputs,
                        REVIEW_RUBRIC,
                        shard_count=args.shard_count,
                        batch_offset=row.batch_offset,
                    )
                ):
                    raise ValueError("Completed shard differs from the requested source, reviewer or assignment")
                logger.info("Reusing completed primary shard: %s", previous.path)
                continue
        pending.append((row, inputs, source_spec))
    if not pending:
        return
    campaign = StoragePath(args.campaign_output) / "attempts" / uuid4().hex
    with endpoint.serve() as session:
        for row, inputs, source_spec in pending:
            reviewer = ReadyChatReviewer(
                session,
                source_spec,
                query_cache_root=args.review_cache,
                max_concurrent=SHARD_REQUEST_CONCURRENCY // args.batch_workers,
            )
            output = StoragePath(row.shard_output_path) / f"{args.shard_index:03}"
            result = review_shard(
                inputs,
                reviewer,
                REVIEW_RUBRIC,
                row.batch_output_path,
                str(output),
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                batch_offset=row.batch_offset,
                max_passes=args.max_passes,
                batch_workers=args.batch_workers,
                worker_jobs=endpoint.job_ids,
            )
            write_pointer(result, output / "latest.json")
            evidence = {
                "source_id": inputs.source_id,
                "shard_artifact": result.path,
                "serving_jobs": endpoint.job_ids,
                "reviewer": reviewer.identity,
                "qualification_status": "pending",
            }
            with (campaign / f"{canonical_sha256(inputs.source_id)[:20]}.json").open("w", auto_mkdir=True) as stream:
                json.dump(evidence, stream, ensure_ascii=False, indent=2)
            logger.info(
                "%s shard %d/%d: %d valid / %d attempted; %s",
                inputs.source_id,
                args.shard_index,
                args.shard_count,
                result.reviewed_count,
                result.attempted_count,
                result.path,
            )


def assemble_work(work: list[ShardWork], args: argparse.Namespace) -> None:
    campaign = StoragePath(args.campaign_output) / "attempts" / uuid4().hex
    spec = local_reviewer(args.reviewer_version)
    for row in work:
        pointers = [
            StoragePath(row.shard_output_path) / f"{index:03}" / "latest.json" for index in range(args.shard_count)
        ]
        if not all(path.exists() for path in pointers):
            logger.info("Source assembly pending: %s", row.inputs_path)
            continue
        inputs = load_inputs(row.inputs_path)
        source_spec = replace(spec, prompt_budget=row.prompt_budget, retry_prompt_budget=row.prompt_budget)
        shards = [read_shard(path) for path in pointers]
        result = assemble_shards(
            inputs,
            shards,
            {"mode": "chat", **source_spec.identity},
            REVIEW_RUBRIC,
            row.assembly_output_path,
        )
        first_statuses = Counter()
        statuses = Counter()
        for shard in shards:
            first_statuses.update(shard.first_pass_status_counts)
            statuses.update(shard.status_counts)
        evidence = {
            "source_id": inputs.source_id,
            "review_artifact": result.path,
            "coverage_complete": result.coverage_complete,
            "qualification_status": "pending",
            "admission_status": "not_assessed",
            "result": result.result.model_dump(mode="json"),
            "first_shard_pass_status_counts": dict(first_statuses),
            "status_counts": dict(statuses),
            "invalid_rate": statuses["invalid"] / result.attempted_count if result.attempted_count else None,
            "shard_artifacts": [shard.path for shard in shards],
            "review_worker_jobs": sorted({job for shard in shards for job in shard.worker_jobs}),
        }
        with (campaign / f"{canonical_sha256(inputs.source_id)[:20]}.json").open("w", auto_mkdir=True) as stream:
            json.dump(evidence, stream, ensure_ascii=False, indent=2)
        logger.info("Source assembly: %s; complete=%s; %s", inputs.source_id, result.coverage_complete, result.path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("review", "assemble"):
        command = commands.add_parser(name)
        command.add_argument("--work", required=True)
        command.add_argument("--reviewer-version", type=ReviewerVersion, choices=list(ReviewerVersion), required=True)
        command.add_argument("--shard-count", required=True, type=int)
        command.add_argument("--campaign-output", required=True)
        command.add_argument("--run", action="store_true")
        if name == "review":
            command.add_argument("--shard-index", required=True, type=int)
            command.add_argument("--max-passes", required=True, type=int)
            command.add_argument(
                "--batch-workers", required=True, type=int, choices=range(1, SHARD_REQUEST_CONCURRENCY + 1)
            )
            command.add_argument("--review-cache", required=True)
            command.add_argument(
                "--resume-root", dest="resume_roots", type=JobName.from_string, action="append", default=[]
            )
    args = parser.parse_args()
    configure_coreweave_s3()
    work = [ShardWork.model_validate(row) for row in json.loads(StoragePath(args.work).read_text())]
    if not work or len({row.inputs_path for row in work}) != len(work):
        raise ValueError("Shard work requires distinct prepared populations")
    if args.shard_count < 1:
        raise ValueError("Shard count must be positive")
    if args.command == "review" and (not 0 <= args.shard_index < args.shard_count or args.max_passes < 1):
        raise ValueError("Invalid review shard assignment or retry count")
    if not args.run:
        logger.info("Planned %d source populations across %d shards", len(work), args.shard_count)
        return
    if args.command == "review":
        review_work(work, args)
    else:
        assemble_work(work, args)


if __name__ == "__main__":
    main()
