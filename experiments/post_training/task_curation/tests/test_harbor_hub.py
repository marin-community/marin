# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor", reason="Requires the isolated config/external/harbor runtime")

from experiments.post_training.task_curation import harbor_hub


def test_export_preserves_distinct_references_with_identical_contents(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    (package / "instruction.md").write_text("Solve the task.\n")
    digest, _ = harbor_hub.Packager.compute_content_hash(package)
    task_ids = [harbor_hub.PackageTaskId(org="fixture", name=name, ref=f"sha256:{digest}") for name in ("one", "two")]
    refs = [f"{task.org}/{task.name}@{task.ref}" for task in task_ids]
    release_digest = "sha256:" + "a" * 64
    release = f"fixture/tasks@{release_digest}"
    manifest = harbor_hub.DatasetManifest.model_validate(
        {
            "dataset": {"name": "fixture/tasks"},
            "tasks": [{"name": f"fixture/{name}", "digest": f"sha256:{digest}"} for name in ("one", "two")],
            "files": [],
        }
    )

    async def metadata(self, requested):
        assert requested == release
        return SimpleNamespace(version=release_digest, task_ids=task_ids, files=[])

    async def download(self, tasks, *, output_dir, export):
        assert tasks[0] in task_ids
        return SimpleNamespace(results=[SimpleNamespace(path=package)])

    monkeypatch.setattr(harbor_hub.PackageDatasetClient, "get_dataset_metadata", metadata)
    monkeypatch.setattr(harbor_hub.TaskClient, "download_tasks", download)
    identities = []
    for name, selection in (("full", []), ("subset", refs[:1])):
        output = tmp_path / name
        asyncio.run(
            harbor_hub.export_release(
                release, manifest.compute_content_hash(), harbor_hub.StoragePath(str(output)), selection
            )
        )
        rows = {json.loads(path.read_text())["task_ref"]: path.name for path in (output / "tasks").glob("*.jsonl")}
        assert set(rows) == set(selection or refs)
        assert json.loads((output / "release.json").read_text())["selected"] == sorted(selection or refs)
        identities.append(rows)
    assert identities[0][refs[0]] == identities[1][refs[0]]
