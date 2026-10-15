# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run with config/external/harbor: export digest-verified native tasks for conversion."""

import argparse
import asyncio
import base64
import json
import logging
import re
import stat
from pathlib import Path
from tempfile import TemporaryDirectory

from harbor.models.dataset.manifest import DatasetManifest
from harbor.models.task.id import PackageTaskId
from harbor.publisher.packager import Packager
from harbor.registry.client.package import PackageDatasetClient
from harbor.tasks.client import TaskClient

MAX_TASK_BYTES = 8 * 1024 * 1024
# The published Hub release ID is not the current SDK's task-membership hash.
# Pin the independently retained 7,496-task membership as well as resolving the release ID.
MEMBERSHIP_SHA256 = "1292dbfd97e6bde83142366afc4f730e61388cc4f66c7e13b65402eb2caabe41"
LOGGER = logging.getLogger(__name__)


def export_task(root: Path, task_ref: str) -> dict:
    """Check the native package hash before encoding only its hash-covered files."""
    paths = Packager.collect_files(root)
    if any(path.is_symlink() or not path.is_file() for path in paths):
        return {"task_ref": task_ref, "unsupported": "unsupported_package_links"}
    digest, paths = Packager.compute_content_hash(root)
    if task_ref.rsplit("@", 1)[1] != f"sha256:{digest}":
        raise ValueError(f"Downloaded task differs from the pinned package: {task_ref}")
    size = sum(path.stat().st_size for path in paths)
    if size > MAX_TASK_BYTES:
        return {"task_ref": task_ref, "unsupported": "resources_over_budget", "bytes": size}
    return {
        "task_ref": task_ref,
        "files": {path.relative_to(root).as_posix(): base64.b64encode(path.read_bytes()).decode() for path in paths},
        "modes": {path.relative_to(root).as_posix(): format(stat.S_IMODE(path.stat().st_mode), "o") for path in paths},
    }


async def prepare(release: str, output: Path, selected: list[str]) -> None:
    name, digest = release.rsplit("@", 1)
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
    manifest = DatasetManifest.model_validate(
        {
            "dataset": {"name": name},
            "tasks": [{"name": ref.rsplit("@", 1)[0], "digest": ref.rsplit("@", 1)[1]} for ref in refs],
            "files": [{"path": file.path, "digest": f"sha256:{file.content_hash}"} for file in metadata.files],
        }
    )
    if len(refs) != 7496 or manifest.compute_content_hash() != MEMBERSHIP_SHA256:
        raise ValueError("Hub release membership differs from the pinned Skill2Env manifest")
    if metadata.files:
        raise ValueError("Dataset-level scoring files require an explicit adapter")
    unknown = set(selected) - refs.keys()
    if unknown:
        raise ValueError(f"Tasks absent from pinned release: {sorted(unknown)}")
    selected_refs = sorted(set(selected) if selected else refs)
    output.mkdir(parents=True, exist_ok=False)
    (output / "release.json").write_text(
        json.dumps(
            {
                "release": release,
                "population_count": len(refs),
                "membership_sha256": MEMBERSHIP_SHA256,
                "selected": selected_refs,
                "task_refs": sorted(refs),
            },
            indent=2,
        )
        + "\n"
    )
    client = TaskClient()
    task_output = output / "tasks"
    task_output.mkdir()
    for ref in selected_refs:
        with TemporaryDirectory(prefix="skill2env-task-") as directory:
            downloaded = await client.download_tasks([refs[ref]], output_dir=Path(directory), export=True)
            root = downloaded.results[0].path
            row = export_task(root, ref)
            row["release"] = release
            # The shared pipeline uses filename and row index as the task identity.
            task_digest = ref.rsplit("@sha256:", 1)[1]
            (task_output / f"{task_digest}.jsonl").write_text(json.dumps(row) + "\n")
            LOGGER.info("Exported %s", ref)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task", action="append", default=[], help="Immutable org/task@sha256:... reference")
    selection.add_argument("--all", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(prepare(args.release, args.output, args.task))


if __name__ == "__main__":
    main()
