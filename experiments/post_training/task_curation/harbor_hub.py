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
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

from harbor.models.task.id import PackageTaskId
from harbor.publisher.packager import Packager
from harbor.registry.client.package import PackageDatasetClient
from harbor.tasks.client import TaskClient
from rigging.filesystem.storage_path import StoragePath

# Full chunks must contain complete base64 groups; only the final chunk is padded.
ENCODING_CHUNK_BYTES = 3 * 64 * 1024
LOGGER = logging.getLogger(__name__)


def task_header(task_ref: str, release: str) -> bytes:
    return (json.dumps({"task_ref": task_ref, "release": release})[:-1] + ", ").encode()


def export_task(root: Path, task_ref: str, release: str) -> Iterator[bytes]:
    """Verify native content and stream a JSONL row without buffering package contents."""
    paths = Packager.collect_files(root)
    if any(path.is_symlink() or not path.is_file() for path in paths):
        yield task_header(task_ref, release) + b'"unsupported": "unsupported_package_links"}\n'
        return
    digest, paths = Packager.compute_content_hash(root)
    if task_ref.rsplit("@", 1)[1] != f"sha256:{digest}":
        raise ValueError(f"Downloaded task differs from the pinned package: {task_ref}")
    yield task_header(task_ref, release) + b'"files": {'
    for index, path in enumerate(paths):
        if index:
            yield b", "
        yield json.dumps(path.relative_to(root).as_posix()).encode() + b': "'
        with path.open("rb") as stream:
            while chunk := stream.read(ENCODING_CHUNK_BYTES):
                yield base64.b64encode(chunk)
        yield b'"'
    yield b'}, "modes": {'
    for index, path in enumerate(paths):
        if index:
            yield b", "
        yield json.dumps(path.relative_to(root).as_posix()).encode() + b": "
        yield json.dumps(format(stat.S_IMODE(path.stat().st_mode), "o")).encode()
    yield b"}}\n"


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
            # Completed rows are renamed only after writing; inspect identity without loading file payloads.
            header = task_header(ref, release)
            with destination.open("rb") as stream:
                if stream.read(len(header)) != header:
                    raise ValueError("Export directory contains a different task or release")
            continue
        with TemporaryDirectory(prefix="harbor-hub-task-") as directory:
            downloaded = await client.download_tasks([refs[ref]], output_dir=Path(directory), export=True)
            root = downloaded.results[0].path
            pending = task_output / f"{task_identity}.partial"
            with pending.open("wb", auto_mkdir=True) as stream:
                for chunk in export_task(root, ref, release):
                    stream.write(chunk)
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
