# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run screened-accept notebook and repository ideas through combined Taskforge/v1."""

import argparse
import asyncio
import hashlib
import json
import tempfile
from collections.abc import Iterable, Iterator
from functools import partial
from pathlib import Path

import fsspec

from experiments.post_training.capability_driven_envs.biotasks_library.pilot_first3 import BioTaskIdea
from experiments.post_training.capability_driven_envs.biotasks_library.taskforge_flow import run_idea
from experiments.post_training.taskforge_zephyr import TaskforgeZephyrConfig, run_taskforge_zephyr

ROOT = "s3://marin-us-east-02a/marin4sc/taskforge/task-library/campaigns/biotasks"
DEFAULT_NOTEBOOKS = f"{ROOT}/notebooks/v1/snapshots/complete-1/screening/accept.jsonl"
DEFAULT_REPOSITORIES = f"{ROOT}/repositories/v1/snapshots/complete-2/screening/accept.jsonl"
DEFAULT_OUTPUT = f"{ROOT}/combined/v1/taskforge-results.jsonl"


def _read_screened(uri: str, source_campaign: str) -> list[dict[str, object]]:
    with fsspec.open(uri, "rt") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    result: list[dict[str, object]] = []
    for row in rows:
        idea = BioTaskIdea.model_validate(row["idea"])
        if row["glm_review"]["recommended_action"] != "accept":
            raise ValueError(f"non-accept idea in screened input: {idea.idea_id}")
        if idea.null_reason is not None:
            raise ValueError(f"null idea in screened input: {idea.idea_id}")
        result.append(
            {
                "idea_json": idea.model_dump_json(),
                "source_campaign": source_campaign,
                "screening_record_sha256": hashlib.sha256(
                    json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    return result


async def _run_one(row: dict[str, object]) -> dict[str, object]:
    idea = BioTaskIdea.model_validate_json(str(row["idea_json"]))
    with tempfile.TemporaryDirectory(prefix="biotasks-combined-v1-") as temporary:
        result = await run_idea(Path(temporary), idea)
    return {
        **result,
        "source_campaign": row["source_campaign"],
        "screening_record_sha256": row["screening_record_sha256"],
    }


def run_shard(rows: Iterable[dict[str, object]], shard_info: object) -> Iterator[dict[str, object]]:
    del shard_info
    for row in rows:
        yield asyncio.run(_run_one(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks", default=DEFAULT_NOTEBOOKS)
    parser.add_argument("--repositories", default=DEFAULT_REPOSITORIES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=32)
    args = parser.parse_args()

    rows = [
        *_read_screened(args.notebooks, "biotasks/notebooks/v1/snapshots/complete-1"),
        *_read_screened(args.repositories, "biotasks/repositories/v1/snapshots/complete-2"),
    ]
    idea_ids = [BioTaskIdea.model_validate_json(str(row["idea_json"])).idea_id for row in rows]
    if len(set(idea_ids)) != len(idea_ids):
        raise ValueError("combined screened inputs contain duplicate idea identities")
    run_taskforge_zephyr(
        rows,
        run_shard=partial(run_shard),
        output=args.output,
        config=TaskforgeZephyrConfig(
            name="biotasks-combined-v1-taskforge",
            max_workers=args.max_workers,
            logical_shards=len(rows),
            region=None,
            preemptible=False,
        ),
    )


if __name__ == "__main__":
    main()
