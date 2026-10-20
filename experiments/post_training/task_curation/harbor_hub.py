# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Acquire pinned Harbor Hub packages using the isolated config/external/harbor runtime."""

import argparse
import asyncio
import hashlib
import json
import logging
import re
import shutil
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

from harbor.db.client import RegistryDB
from harbor.models.task.id import PackageTaskId
from harbor.publisher.packager import Packager
from harbor.registry.client.package import PackageDatasetClient
from harbor.storage.supabase import SupabaseStorage
from rigging.filesystem.storage_path import StoragePath

LOGGER = logging.getLogger(__name__)


async def acquire_release(release: str, output: StoragePath, selected: list[str]) -> None:
    """Cache original task archives after verifying their native content hashes."""
    _, digest = release.rsplit("@", 1)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise ValueError("Select a release by content digest, not a tag")
    metadata = await PackageDatasetClient().get_dataset_metadata(release)
    if metadata.version != digest:
        raise ValueError("Hub resolved a different release")
    tasks = metadata.task_ids
    refs: dict[str, PackageTaskId] = {}
    for task in tasks:
        if not isinstance(task, PackageTaskId):
            raise ValueError("Expected native Hub task references")
        refs[f"{task.org}/{task.name}@{task.ref}"] = task
    if len(refs) != len(tasks):
        raise ValueError("Duplicate task identities in release")
    if metadata.files:
        raise ValueError("Dataset-level scoring files require an explicit adapter")
    unknown = set(selected) - refs.keys()
    if unknown:
        raise ValueError(f"Tasks absent from pinned release: {sorted(unknown)}")
    selected_refs = sorted(set(selected) if selected else refs)
    registry = RegistryDB()
    storage = SupabaseStorage()
    task_output = output / "tasks"
    for ref in selected_refs:
        # Filenames preserve task identity even when multiple names share package contents.
        task_identity = hashlib.sha256(ref.encode()).hexdigest()
        destination = task_output / f"{task_identity}.tar.gz"
        if destination.exists():
            continue
        task = refs[ref]
        resolved = await registry.resolve_task_version(task.org, task.name, task.ref)
        if f"sha256:{resolved.content_hash}" != task.ref:
            raise ValueError(f"Hub resolved a different task: {ref}")
        with TemporaryDirectory(prefix="harbor-hub-task-") as directory:
            archive_path = Path(directory) / "task.tar.gz"
            await storage.download_file(resolved.archive_path, archive_path)
            root = Path(directory) / "task"
            root.mkdir()
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(root, filter="data")
            digest, _ = Packager.compute_content_hash(root)
            if f"sha256:{digest}" != task.ref:
                raise ValueError(f"Downloaded task differs from the pinned package: {ref}")
            pending = task_output / f"{task_identity}.partial"
            with archive_path.open("rb") as source, pending.open("wb", auto_mkdir=True) as target:
                shutil.copyfileobj(source, target)
            pending.rename(destination)
            LOGGER.info("Acquired %s", ref)

    (output / "release.json").write_text(
        json.dumps(
            {
                "release": release,
                "population_count": len(refs),
                "selected": selected_refs,
                "task_refs": sorted(refs),
            },
            indent=2,
        )
        + "\n",
        auto_mkdir=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--output", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task", action="append", default=[], help="Immutable org/task@sha256:... reference")
    selection.add_argument("--all", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(acquire_release(args.release, StoragePath(args.output), args.task))


if __name__ == "__main__":
    main()
