# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repeat a frozen quality-review sample against an existing owned local endpoint."""

import argparse
import hashlib
import json
import logging

from marin.inference.openai_chat import OpenAIChatClient
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import ReviewStatus
from taskcompendium.pipeline.review import ChatReviewer

from experiments.post_training.task_curation.full_qc import REVIEW_RUBRIC, ReviewerVersion, local_reviewer
from experiments.post_training.task_curation.full_qc_review import review_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--sample-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--serving-job", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    configure_coreweave_s3()
    raw = StoragePath(args.sample).read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.sample_sha256:
        raise ValueError("Frozen sample bytes changed")
    rows = json.loads(raw)
    tasks = [TaskSpec.model_validate(row["task"]) for row in rows]
    if len(tasks) != 100 or len({task.id for task in tasks}) != 100:
        raise ValueError("The paired audit requires exactly 100 distinct frozen tasks")
    for row, task in zip(rows, tasks, strict=True):
        if canonical_sha256(task.model_dump(mode="json")) != row["contract"]["task_sha256"]:
            raise ValueError("Frozen task contract changed")
    output = StoragePath(args.output)
    reviewer = ChatReviewer(
        OpenAIChatClient(require_secret("QA_ENDPOINT_URL"), "unused", timeout=600),
        local_reviewer(ReviewerVersion.V2),
        query_cache_root=str(output / "request-cache"),
        max_concurrent=8,
    )
    evidence = {"sample_sha256": args.sample_sha256, "serving_job": args.serving_job, "reviewer": reviewer.identity}
    with (output / "identity.json").open("w", auto_mkdir=True) as stream:
        json.dump(evidence, stream, indent=2)
    records, paths = {}, []
    for attempt in range(3):
        for start in range(0, len(tasks), 20):
            batch = tasks[start : start + 20]
            contracts = [row["contract"] for row in rows[start : start + 20]]
            reviewed, path = review_batch(batch, contracts, REVIEW_RUBRIC, reviewer, output)
            records.update((record.task_id, record) for record in reviewed)
            paths.append(path)
            print(
                json.dumps(
                    {
                        "pass": attempt,
                        "batch": start // 20,
                        "valid": sum(record.status is ReviewStatus.REVIEWED for record in records.values()),
                        "accounted": len(records),
                    }
                ),
                flush=True,
            )
        if all(record.status is ReviewStatus.REVIEWED for record in records.values()):
            break
    result = {
        **evidence,
        "reviews": [records[task.id].model_dump(mode="json") for task in tasks],
        "attempt_paths": paths,
    }
    with (output / "result.json").open("w", auto_mkdir=True) as stream:
        json.dump(result, stream)
    print(
        json.dumps(
            {
                "result": str(output / "result.json"),
                "valid": sum(record.status is ReviewStatus.REVIEWED for record in records.values()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
