# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pin selected TMax Harbor tasks individually to avoid a full registry scan."""

import argparse
import asyncio
import json
from pathlib import Path

import yaml
from harbor.auth.client import create_authenticated_client
from harbor.db.client import RegistryDB
from harbor_config.models.job.config import JobConfig

DATASET = "tmax/tmax-13k-v1.1"
RELEASE = "sha256:119bd1ba7bb6b948f9313a6379e3182d30cfe1afd182167162d8a6240ab164e4"
TASK_TAG = "v1.1"
MAX_CONCURRENT_LOOKUPS = 4
MEMBERSHIP_BATCH_SIZE = 100


async def pinned_task(name: str, limiter: asyncio.Semaphore) -> tuple[dict[str, str], str]:
    org, task_name = name.split("/", 1)
    async with limiter:
        version = await RegistryDB().resolve_task_version(org, task_name, TASK_TAG)
    return {"name": name, "ref": f"sha256:{version.content_hash}", "source": DATASET}, version.id


async def pin_names(names: list[str]) -> list[dict[str, str]]:
    limiter = asyncio.Semaphore(MAX_CONCURRENT_LOOKUPS)
    resolved = await asyncio.gather(*(pinned_task(name, limiter) for name in names))
    _, dataset_version = await RegistryDB().resolve_dataset_version("tmax", "tmax-13k-v1.1", RELEASE)
    client = await create_authenticated_client()
    task_version_ids = [version_id for _, version_id in resolved]
    members = set()
    for start in range(0, len(task_version_ids), MEMBERSHIP_BATCH_SIZE):
        membership = await (
            client.table("dataset_version_task")
            .select("task_version_id")
            .eq("dataset_version_id", dataset_version["id"])
            .in_("task_version_id", task_version_ids[start : start + MEMBERSHIP_BATCH_SIZE])
            .execute()
        )
        members.update(row["task_version_id"] for row in membership.data or [])
    if members != set(task_version_ids):
        raise ValueError("selected task versions are not all in the pinned dataset release")
    return [task for task, _ in resolved]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    datasets = config.pop("datasets")
    if len(datasets) != 1 or datasets[0]["name"] != DATASET or datasets[0]["ref"] != RELEASE:
        raise ValueError("expected the pinned TMax v1.1 dataset release")
    names = datasets[0]["task_names"]
    if len(names) != len(set(names)) or any(not name.startswith("tmax/") for name in names):
        raise ValueError("task names must be unique TMax package names")
    tasks = asyncio.run(pin_names(names))
    config["tasks"] = tasks
    JobConfig.model_validate(config)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {
        "dataset": DATASET,
        "release": RELEASE,
        "task_tag": TASK_TAG,
        "release_membership_verified": True,
        "tasks": tasks,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Pinned {len(tasks)} task package digests: {args.output}")


if __name__ == "__main__":
    main()
