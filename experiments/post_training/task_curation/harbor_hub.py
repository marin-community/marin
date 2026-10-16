# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export pinned Harbor Hub packages using the isolated config/external/harbor runtime."""

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import re
import stat
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NotRequired, TypedDict

from harbor.models.task.id import PackageTaskId
from harbor.publisher.packager import Packager
from harbor.registry.client.package import PackageDatasetClient
from harbor.tasks.client import TaskClient
from rigging.filesystem.storage_path import StoragePath

MAX_TASK_BYTES = 8 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


class ExportedTask(TypedDict):
    task_ref: str
    release: str
    files: NotRequired[dict[str, str]]
    modes: NotRequired[dict[str, str]]
    unsupported: NotRequired[str]
    bytes: NotRequired[int]


def export_task(root: Path, task_ref: str, release: str) -> ExportedTask:
    """Check the native package hash before encoding only its hash-covered files."""
    paths = Packager.collect_files(root)
    if any(path.is_symlink() or not path.is_file() for path in paths):
        return {"task_ref": task_ref, "release": release, "unsupported": "unsupported_package_links"}
    digest, paths = Packager.compute_content_hash(root)
    if task_ref.rsplit("@", 1)[1] != f"sha256:{digest}":
        raise ValueError(f"Downloaded task differs from the pinned package: {task_ref}")
    size = sum(path.stat().st_size for path in paths)
    if size > MAX_TASK_BYTES:
        return {"task_ref": task_ref, "release": release, "unsupported": "export_size_ceiling", "bytes": size}
    return {
        "task_ref": task_ref,
        "release": release,
        "files": {path.relative_to(root).as_posix(): base64.b64encode(path.read_bytes()).decode() for path in paths},
        "modes": {path.relative_to(root).as_posix(): format(stat.S_IMODE(path.stat().st_mode), "o") for path in paths},
    }


async def export_release(release: str, output: StoragePath, selected: list[str]) -> None:
    """Export hash-verified task rows and the pinned release manifest."""
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
    client = TaskClient()
    task_output = output / "tasks"
    for ref in selected_refs:
        # Filenames preserve task identity even when multiple names share package contents.
        task_identity = hashlib.sha256(ref.encode()).hexdigest()
        destination = task_output / f"{task_identity}.jsonl"
        if destination.exists():
            existing = json.loads(destination.read_text())
            if existing["task_ref"] != ref or existing["release"] != release:
                raise ValueError("Export directory contains a different task or release")
            continue
        with TemporaryDirectory(prefix="harbor-hub-task-") as directory:
            downloaded = await client.download_tasks([refs[ref]], output_dir=Path(directory), export=True)
            root = downloaded.results[0].path
            row = export_task(root, ref, release)
            pending = task_output / f"{task_identity}.partial"
            pending.write_text(json.dumps(row) + "\n", auto_mkdir=True)
            pending.rename(destination)
            LOGGER.info("Exported %s", ref)

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
    asyncio.run(export_release(args.release, StoragePath(args.output), args.task))


if __name__ == "__main__":
    main()
