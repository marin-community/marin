# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve the published Mini840 Harbor image keys against existing Daytona snapshots."""

import asyncio
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as pq
from daytona import AsyncDaytona
from harbor.utils.container_cache import environment_dir_hash_truncated
from huggingface_hub import hf_hub_download
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from shellbox.image import DockerfileSource, RegistryImage, image_source_key
from taskcompendium.models import TaskSpec

from experiments.evaluation.mini_policy_runtime.profile import DATASET, DESTINATION, REVISION


def publish(name, data):
    content = json.dumps(data, indent=2).encode()
    (Path(os.environ["IRIS_OUTPUT_DIR"]) / name).write_bytes(content)
    fs, key = filesystem_for(str(StoragePath(DESTINATION) / name))
    fs.pipe_file(key, content)


@dataclass(frozen=True)
class EnvironmentRecord:
    task_id: str
    benchmark: str
    image_source_key: str
    environment_hash: str
    docker_image: str | None
    package_sha256: str
    context: str
    cpus: int | None
    memory_mb: int | None
    storage_mb: int | None


def restored_environments(temporary: Path) -> list[EnvironmentRecord]:
    root = Path(__file__).resolve().parents[3]
    importer_spec = importlib.util.spec_from_file_location(
        "mini_runtime_harbor_import", root / "vendor/skyrl/skyrl-train/skyrl_train/dataset/harbor_import.py"
    )
    importer = importlib.util.module_from_spec(importer_spec)
    sys.modules[importer_spec.name] = importer
    importer_spec.loader.exec_module(importer)
    parquet = hf_hub_download(DATASET, "data/validation.parquet", repo_type="dataset", revision=REVISION)
    rows = pq.read_table(parquet).to_pylist()
    inventory = []
    for row in rows:
        if row["agent"] != "pi":
            continue
        task = TaskSpec.model_validate_json(row["task_spec"])
        directory = Path(temporary) / str(row["selection_order"])
        importer.restore_harbor_trial_task(task, directory)
        config = importer.TaskConfig.model_validate_toml((directory / "task.toml").read_text())
        context = directory / "environment"
        source = (
            RegistryImage(config.environment.docker_image)
            if config.environment.docker_image
            else DockerfileSource(context, context / "Dockerfile")
        )
        inventory.append(
            EnvironmentRecord(
                task_id=row["task_id"],
                benchmark=row["benchmark"],
                image_source_key=image_source_key(source),
                environment_hash=environment_dir_hash_truncated(context),
                docker_image=config.environment.docker_image,
                package_sha256=hashlib.sha256(row["task_spec"].encode()).hexdigest(),
                context=str(context),
                cpus=config.environment.cpus,
                memory_mb=config.environment.memory_mb,
                storage_mb=config.environment.storage_mb,
            )
        )
    return inventory


async def main():
    with tempfile.TemporaryDirectory() as temporary:
        inventory = [asdict(row) for row in restored_environments(Path(temporary))]
    snapshots = []
    async with AsyncDaytona() as client:
        page = 1
        while True:
            result = await client.snapshot.list(page=page, limit=100)
            for item in result.items:
                snapshots.append({"name": item.name, "state": str(item.state), "image_name": item.image_name})
            if page >= result.total_pages:
                break
            page += 1
    active = {row["name"]: row for row in snapshots if row["state"].lower() in {"active", "snapshotstate.active"}}
    mapping = {}
    for task in inventory:
        prefix = "harbor__" + task["environment_hash"] + "__"
        candidates = sorted(name for name in active if name.startswith(prefix) and name.endswith("snapshot"))
        task["snapshot_candidates"] = candidates
        if candidates:
            mapping[task["image_source_key"]] = candidates[0]
    missing = [row for row in inventory if row["image_source_key"] not in mapping]
    report = {
        "dataset": DATASET,
        "revision": REVISION,
        "tasks": inventory,
        "snapshots": snapshots,
        "approved_environment_snapshots": mapping,
        "missing": missing,
        "summary": {
            "agentic_tasks": len(inventory),
            "benchmarks": dict(Counter(r["benchmark"] for r in inventory)),
            "unique_source_keys": len({r["image_source_key"] for r in inventory}),
            "mapped_source_keys": len(mapping),
            "missing_tasks": len(missing),
            "listed_snapshots": len(snapshots),
        },
    }
    publish("environment-inventory.json", report)
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
