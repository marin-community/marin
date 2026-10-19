# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
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


def test_export_round_trips_large_binary_files_and_chunk_boundaries(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "tests").mkdir()
    sizes = [0, 1, 2, 3, 196607, 196608, 196609, 9 * 1024 * 1024 + 1]
    for index, size in enumerate(sizes):
        path = package / f'tests/file-{index}-"-é.bin'
        path.write_bytes((bytes(range(256)) * (size // 256 + 1))[:size])
        path.chmod(0o755 if index == 0 else 0o644)
    digest, paths = harbor_hub.Packager.compute_content_hash(package)
    task_ref = f"fixture/large@sha256:{digest}"
    output = tmp_path / "task.jsonl"
    with output.open("wb") as stream:
        for chunk in harbor_hub.export_task(package, task_ref, "fixture/release"):
            stream.write(chunk)
    row = json.loads(output.read_text())
    assert row["task_ref"] == task_ref
    assert row["release"] == "fixture/release"
    assert set(row["files"]) == {path.relative_to(package).as_posix() for path in paths}
    for index in range(len(sizes)):
        name = f'tests/file-{index}-"-é.bin'
        assert base64.b64decode(row["files"][name], validate=True) == (package / name).read_bytes()
        assert row["modes"][name] == ("755" if index == 0 else "644")
