# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import stat
import zipfile

import pytest

from experiments.post_training.tmax_reserve.shared_image import task_digest
from experiments.post_training.tmax_reserve.v11 import package_v11

INSTRUCTION = "Write /home/user/out.txt.\n"
RECIPE = "    apt-get update && apt-get install -y python3\n    mkdir -p /home/user\n"


def _shared_task(root, task):
    (root / task / "environment").mkdir(parents=True)
    (root / task / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (root / task / "task_setup").mkdir()
    (root / task / "task_setup" / "post_install.sh").write_text(RECIPE)
    (root / task / "tests").mkdir()
    (root / task / "tests" / "test.sh").write_text("old verifier\n")
    (root / task / "tests" / "test_final_state.py").write_text("old pytest\n")
    (root / task / "instruction.md").write_text(INSTRUCTION)
    (root / task / "task.toml").write_text(f'[task]\nname = "tmax/{task}"\n')


def _v11_task(archive, task, *, image_rebuilt=False, recipe=RECIPE):
    archive.writestr(f"{task}/instruction.md", INSTRUCTION)
    archive.writestr(f"{task}/setup.sh", f"#!/bin/bash\nset -e\nexport DEBIAN_FRONTEND=noninteractive\n{recipe}")
    archive.writestr(f"{task}/task.toml", f"[metadata]\nimage_rebuilt = {str(image_rebuilt).lower()}\n")
    test = zipfile.ZipInfo(f"{task}/tests/test.sh")
    test.external_attr = (stat.S_IFREG | 0o750) << 16
    archive.writestr(test, "new verifier\n")


def _pilot(tmp_path, tasks):
    shared = tmp_path / "pilot-shared"
    for task in tasks:
        _shared_task(shared, task)
    rows = {task: {"sha256": task_digest(shared / task), "template": "A"} for task in tasks}
    manifest = {"sha256": "s", "snapshot_keys": {"A": "shellbox-a", "B": "shellbox-b"}, "tasks": rows}
    (tmp_path / "pilot-shared.manifest.json").write_text(json.dumps(manifest))
    return shared


def test_package_v11_swaps_verifiers_and_skips_tasks_it_cannot_carry(tmp_path):
    shared = _pilot(tmp_path, ["task_kept", "task_rebuilt", "task_dropped", "task_excluded"])
    v11_zip = tmp_path / "tasks.zip"
    with zipfile.ZipFile(v11_zip, "w") as archive:
        _v11_task(archive, "task_kept")
        _v11_task(archive, "task_rebuilt", image_rebuilt=True)
        _v11_task(archive, "task_excluded")
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(json.dumps({"tasks": [{"task": "task_excluded", "reason": "verifier_or_task_defect"}]}))

    manifest = package_v11(shared, v11_zip, "rev", [exclusions], tmp_path / "pilot-v11")

    kept = tmp_path / "pilot-v11" / "task_kept"
    assert sorted(manifest["tasks"]) == ["task_kept"]
    assert manifest["skipped"] == {
        "task_dropped": "not_in_v11",
        "task_excluded": "excluded:verifier_or_task_defect",
        "task_rebuilt": "image_rebuilt",
    }
    assert sorted(path.name for path in (kept / "tests").iterdir()) == ["test.sh"]
    assert (kept / "tests" / "test.sh").read_text() == "new verifier\n"
    assert os.access(kept / "tests" / "test.sh", os.X_OK)
    assert 'name = "tmax/task_kept-v1.1"' in (kept / "task.toml").read_text()
    assert (kept / "task_setup" / "post_install.sh").read_text() == RECIPE
    assert manifest["snapshot_keys"] == {"A": "shellbox-a", "B": "shellbox-b"}


def test_package_v11_refuses_a_task_whose_setup_recipe_changed(tmp_path):
    shared = _pilot(tmp_path, ["task_changed"])
    v11_zip = tmp_path / "tasks.zip"
    with zipfile.ZipFile(v11_zip, "w") as archive:
        _v11_task(archive, "task_changed", recipe="    mkdir -p /home/user/other\n")
    with pytest.raises(ValueError, match="setup recipe differs"):
        package_v11(shared, v11_zip, "rev", [], tmp_path / "pilot-v11")
