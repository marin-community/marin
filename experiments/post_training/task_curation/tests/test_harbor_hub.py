# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor", reason="Requires the isolated config/external/harbor runtime")

from experiments.post_training.task_curation import harbor_hub


@pytest.mark.parametrize("interrupt", [False, True])
def test_export_preserves_distinct_references_and_resumes_downloads(tmp_path, monkeypatch, interrupt):
    package = tmp_path / "package"
    package.mkdir()
    (package / "instruction.md").write_text("Solve the task.\n")
    digest, _ = harbor_hub.Packager.compute_content_hash(package)
    task_ids = [harbor_hub.PackageTaskId(org="fixture", name=name, ref=f"sha256:{digest}") for name in ("one", "two")]
    refs = [f"{task.org}/{task.name}@{task.ref}" for task in task_ids]
    release_digest = "sha256:" + "a" * 64
    release = f"fixture/tasks@{release_digest}"

    async def metadata(self, requested):
        assert requested == release
        return SimpleNamespace(version=release_digest, task_ids=task_ids, files=[])

    attempts = {task.name: 0 for task in task_ids}

    async def download(self, tasks, *, output_dir, export):
        assert tasks[0] in task_ids
        attempts[tasks[0].name] += 1
        if interrupt and tasks[0].name == "two" and attempts["two"] == 1:
            raise ConnectionError("Download interrupted")
        return SimpleNamespace(results=[SimpleNamespace(path=package)])

    monkeypatch.setattr(harbor_hub.PackageDatasetClient, "get_dataset_metadata", metadata)
    monkeypatch.setattr(harbor_hub.TaskClient, "download_tasks", download)
    identities = []
    for name, selection in (("full", []), ("subset", refs[:1])):
        output = tmp_path / name
        if interrupt and name == "full":
            with pytest.raises(ConnectionError, match="Download interrupted"):
                asyncio.run(harbor_hub.export_release(release, harbor_hub.StoragePath(str(output)), selection))
            assert not (output / "release.json").exists()
        asyncio.run(harbor_hub.export_release(release, harbor_hub.StoragePath(str(output)), selection))
        rows = {json.loads(path.read_text())["task_ref"]: path.name for path in (output / "tasks").glob("*.jsonl")}
        assert set(rows) == set(selection or refs)
        assert json.loads((output / "release.json").read_text())["selected"] == sorted(selection or refs)
        identities.append(rows)
    assert identities[0][refs[0]] == identities[1][refs[0]]
    assert attempts == {"one": 2, "two": 2 if interrupt else 1}
