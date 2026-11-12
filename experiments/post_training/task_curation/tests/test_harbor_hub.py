# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import shutil
import tarfile
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor", reason="Requires the isolated vendor/harbor runtime")

from experiments.post_training.task_curation import harbor_hub


@pytest.mark.parametrize("interrupt", [False, True])
def test_acquisition_preserves_native_archives_and_resumes_downloads(tmp_path, monkeypatch, interrupt):
    package = tmp_path / "package"
    package.mkdir()
    (package / "instruction.md").write_text("Solve the task.\n")
    (package / "tests").mkdir()
    verifier = package / "tests/test.sh"
    verifier.write_text("exit 0\n")
    verifier.chmod(0o755)
    digest, _ = harbor_hub.Packager.compute_content_hash(package)
    archive_path = tmp_path / "native.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(package, arcname=".")
    task_ids = [harbor_hub.PackageTaskId(org="fixture", name=name, ref=f"sha256:{digest}") for name in ("one", "two")]
    refs = [f"{task.org}/{task.name}@{task.ref}" for task in task_ids]
    release_digest = "sha256:" + "a" * 64
    release = f"fixture/tasks@{release_digest}"

    async def metadata(self, requested):
        assert requested == release
        return SimpleNamespace(version=release_digest, task_ids=task_ids, files=[])

    async def resolve(self, org, name, ref):
        return SimpleNamespace(content_hash=digest, archive_path=name)

    attempts = {task.name: 0 for task in task_ids}

    async def download(self, remote_path, file_path):
        attempts[remote_path] += 1
        if interrupt and remote_path == "two" and attempts["two"] == 1:
            raise ConnectionError("Download interrupted")
        shutil.copyfile(archive_path, file_path)

    monkeypatch.setattr(harbor_hub.PackageDatasetClient, "get_dataset_metadata", metadata)
    monkeypatch.setattr(harbor_hub.RegistryDB, "resolve_task_version", resolve)
    monkeypatch.setattr(harbor_hub.SupabaseStorage, "download_file", download)
    for name, selection in (("full", []), ("subset", refs[:1])):
        output = tmp_path / name
        if interrupt and name == "full":
            with pytest.raises(ConnectionError, match="Download interrupted"):
                asyncio.run(harbor_hub.acquire_release(release, harbor_hub.StoragePath(str(output)), selection))
            assert not (output / "release.json").exists()
        asyncio.run(harbor_hub.acquire_release(release, harbor_hub.StoragePath(str(output)), selection))
        expected = {hashlib.sha256(ref.encode()).hexdigest() + ".tar.gz" for ref in selection or refs}
        assert {path.name for path in (output / "tasks").glob("*.tar.gz")} == expected
        for path in (output / "tasks").glob("*.tar.gz"):
            assert path.read_bytes() == archive_path.read_bytes()
        assert json.loads((output / "release.json").read_text())["selected"] == sorted(selection or refs)
    assert attempts == {"one": 2, "two": 2 if interrupt else 1}


def test_acquisition_rejects_modified_package_without_publishing_archive(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    (package / "instruction.md").write_text("Original instruction")
    digest, _ = harbor_hub.Packager.compute_content_hash(package)
    task = harbor_hub.PackageTaskId(org="fixture", name="task", ref=f"sha256:{digest}")
    release_digest = "sha256:" + "a" * 64

    async def metadata(self, requested):
        return SimpleNamespace(version=release_digest, task_ids=[task], files=[])

    async def resolve(self, org, name, ref):
        return SimpleNamespace(content_hash=digest, archive_path="task")

    async def download(self, remote_path, file_path):
        (package / "instruction.md").write_text("Modified instruction")
        with tarfile.open(file_path, "w:gz") as archive:
            archive.add(package, arcname=".")

    monkeypatch.setattr(harbor_hub.PackageDatasetClient, "get_dataset_metadata", metadata)
    monkeypatch.setattr(harbor_hub.RegistryDB, "resolve_task_version", resolve)
    monkeypatch.setattr(harbor_hub.SupabaseStorage, "download_file", download)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Downloaded task differs"):
        asyncio.run(
            harbor_hub.acquire_release(f"fixture/tasks@{release_digest}", harbor_hub.StoragePath(str(output)), [])
        )
    assert not list(output.rglob("*.tar.gz"))
    assert not (output / "release.json").exists()
