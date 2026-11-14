# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Move shared-image pilot tasks onto the TMax v1.1 verifiers (``TMaxxx/tmax-13k-v1.1-harbor``).

v1.1 is upstream's hardened release of the same task IDs. Its instructions and setup recipes are byte-identical to
the original release, so a shared-image pilot task keeps its environment, setup and snapshot and takes only the
v1.1 ``tests/``. Tasks whose v1.1 image was rebuilt (an answer leak removed from the image) cannot keep the shared
image, so they are skipped, as are tasks v1.1 dropped and tasks on the given exclusion lists. Any task whose
instruction or setup recipe differs from v1.1 is an error.
"""

import argparse
import hashlib
import json
import shutil
import stat
import tomllib
import zipfile
from pathlib import Path

from experiments.post_training.tmax_reserve.shared_image import task_digest

V11_REPOSITORY = "TMaxxx/tmax-13k-v1.1-harbor"
# The lines v1.1's setup.sh adds around the unchanged post_install.sh body.
SETUP_HEADER = frozenset({"#!/bin/bash", "set -e", "export DEBIAN_FRONTEND=noninteractive"})
SKIP_NOT_IN_V11 = "not_in_v11"
SKIP_IMAGE_REBUILT = "image_rebuilt"
SKIP_EXCLUDED = "excluded"


def recipe_lines(script: str) -> list[str]:
    """A setup script's commands with indentation, blank lines and v1.1's header removed."""
    return [line.strip() for line in script.splitlines() if line.strip() and line.strip() not in SETUP_HEADER]


def excluded_tasks(lists: list[Path]) -> dict[str, str]:
    """``{task: reason}`` from exclusion lists shaped like ``{"tasks": [{"task", "reason"}, ...]}``."""
    return {row["task"]: row["reason"] for path in lists for row in json.loads(path.read_text())["tasks"]}


def extract_tests(archive: zipfile.ZipFile, task: str, dest: Path) -> None:
    """Write the v1.1 ``tests/`` of ``task`` under ``dest``, keeping each file's executable bit."""
    prefix = f"{task}/tests/"
    members = [info for info in archive.infolist() if info.filename.startswith(prefix) and not info.is_dir()]
    if not members:
        raise ValueError(f"{task} has no tests/ in the v1.1 archive")
    for info in members:
        out = dest / info.filename[len(prefix) :]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(archive.read(info))
        executable = (info.external_attr >> 16) & stat.S_IXUSR
        out.chmod(0o755 if executable else 0o644)


def convert_task(shared_task: Path, archive: zipfile.ZipFile, task: str, dest: Path) -> None:
    """Copy one shared-image task to ``dest`` with v1.1's tests and a ``-v1.1`` Harbor task name."""
    if archive.read(f"{task}/instruction.md").decode().strip() != (shared_task / "instruction.md").read_text().strip():
        raise ValueError(f"{task}: the v1.1 instruction differs from the pilot's")
    v11_setup = archive.read(f"{task}/setup.sh").decode()
    if recipe_lines(v11_setup) != recipe_lines((shared_task / "task_setup" / "post_install.sh").read_text()):
        raise ValueError(f"{task}: the v1.1 setup recipe differs from the pilot's post_install.sh")
    shutil.copytree(shared_task, dest, ignore=shutil.ignore_patterns("tests"))
    extract_tests(archive, task, dest / "tests")
    toml = dest / "task.toml"
    old_name = f'name = "tmax/{task}"'
    text = toml.read_text()
    if text.count(old_name) != 1:
        raise ValueError(f"{task}: expected one {old_name!r} line in task.toml")
    toml.write_text(text.replace(old_name, f'name = "tmax/{task}-v1.1"'))


def package_v11(shared: Path, v11_zip: Path, v11_revision: str, exclusion_lists: list[Path], dest: Path) -> dict:
    """Convert a shared-image pilot (verified against its manifest) to v1.1 verifiers under ``dest``."""
    shared_manifest = json.loads((shared.parent / f"{shared.name}.manifest.json").read_text())
    for task, row in shared_manifest["tasks"].items():
        if task_digest(shared / task) != row["sha256"]:
            raise ValueError(f"{task} does not match {shared.name}.manifest.json")
    excluded = excluded_tasks(exclusion_lists)
    archive = zipfile.ZipFile(v11_zip)
    in_v11 = {name.split("/", 1)[0] for name in archive.namelist()}
    dest.mkdir(parents=True)
    tasks, skipped = {}, {}
    for task, row in sorted(shared_manifest["tasks"].items()):
        if task in excluded:
            skipped[task] = f"{SKIP_EXCLUDED}:{excluded[task]}"
            continue
        if task not in in_v11:
            skipped[task] = SKIP_NOT_IN_V11
            continue
        if tomllib.loads(archive.read(f"{task}/task.toml").decode())["metadata"]["image_rebuilt"]:
            skipped[task] = SKIP_IMAGE_REBUILT
            continue
        convert_task(shared / task, archive, task, dest / task)
        tasks[task] = {"sha256": task_digest(dest / task), "template": row["template"]}
    manifest = {
        "source": shared.name,
        "source_sha256": shared_manifest["sha256"],
        "v11_repository": V11_REPOSITORY,
        "v11_revision": v11_revision,
        "v11_tasks_zip_sha256": hashlib.sha256(v11_zip.read_bytes()).hexdigest(),
        "exclusion_lists": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in exclusion_lists},
        # The environments are copied unchanged, so each template keeps the shared pilot's snapshot.
        "snapshot_keys": shared_manifest["snapshot_keys"],
        "skipped": skipped,
        "tasks": tasks,
        "sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
    }
    (dest.parent / f"{dest.name}.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shared", type=Path, required=True, help="Shared-image pilot directory (with manifest).")
    parser.add_argument("--v11-zip", type=Path, required=True, help=f"tasks.zip from {V11_REPOSITORY}.")
    parser.add_argument("--v11-revision", required=True, help=f"The {V11_REPOSITORY} commit the zip came from.")
    parser.add_argument("--exclude", type=Path, action="append", default=[], help="Exclusion list JSON (repeatable).")
    parser.add_argument("--dest", type=Path, required=True, help="New directory for the v1.1 tasks.")
    args = parser.parse_args()
    manifest = package_v11(args.shared, args.v11_zip, args.v11_revision, args.exclude, args.dest)
    reasons: dict[str, int] = {}
    for reason in manifest["skipped"].values():
        reasons[reason] = reasons.get(reason, 0) + 1
    print(json.dumps({"tasks": len(manifest["tasks"]), "skipped": reasons, "sha256": manifest["sha256"]}))


if __name__ == "__main__":
    main()
