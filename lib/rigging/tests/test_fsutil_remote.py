# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI behavior for Iris transfers and private Hugging Face destinations."""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
import rigging.fsutil.remote as remote_module
from click.testing import CliRunner
from huggingface_hub import HfApi, HfFileSystem
from huggingface_hub.utils import _http as hf_http
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import TransferError, copy_plan
from rigging.fsutil.cli import cli
from rigging.fsutil.listing import list_entries

HF_STORAGE_ROOTS = (
    "hf://buckets/owner/files",
    "hf://owner/files",
    "hf://models/owner/files",
    "hf://datasets/owner/files",
)


@pytest.mark.parametrize("operation", ["cp", "mv"])
@pytest.mark.parametrize("root", HF_STORAGE_ROOTS)
def test_remote_transfer_forwards_saved_hf_token_without_putting_it_in_command(tmp_path, monkeypatch, operation, root):
    capture = tmp_path / "submission.json"
    iris = tmp_path / "iris"
    iris.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['FSUTIL_CAPTURE_PATH'], 'w') as output:\n"
        "    json.dump({'args': sys.argv[1:], 'token': os.environ.get('HF_TOKEN')}, output)\n"
    )
    iris.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("FSUTIL_CAPTURE_PATH", str(capture))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(remote_module, "get_token", lambda: "saved-token")

    result = CliRunner().invoke(
        cli,
        [operation, "--remote", "s3://marin-us-east-02a/example", f"{root}/example"],
    )

    assert result.exit_code == 0, result.output
    submission = json.loads(capture.read_text())
    assert submission["token"] == "saved-token"
    assert all("saved-token" not in argument for argument in submission["args"])


@dataclass
class HubStorage:
    exists: bool = False
    private: bool = False
    files: dict[str, bytes] = field(default_factory=dict)

    def request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and (path == "/api/repos/create" or path == "/api/buckets/owner/files"):
            self.exists = True
            payload = json.loads(request.content)
            self.private = payload["visibility"] == "private" if path == "/api/repos/create" else payload["private"]
            return httpx.Response(200, json={"url": "https://huggingface.co/owner/files"})
        if not self.exists:
            return httpx.Response(404, headers={"X-Error-Code": "RepoNotFound"}, json={"error": "Not found"})
        if "/tree" in path:
            return httpx.Response(
                200,
                json=[
                    {"type": "file", "path": name, "size": len(data), "oid": "blob", "xetHash": "xet"}
                    for name, data in self.files.items()
                ],
            )
        if "/paths-info/" in path:
            return httpx.Response(200, json=[])
        if path.startswith("/api/") and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "id": "owner/files",
                    "private": self.private,
                    "createdAt": "2026-01-01T00:00:00Z",
                    "size": sum(map(len, self.files.values())),
                    "totalFiles": len(self.files),
                },
            )
        if "/resolve/" in path:
            return httpx.Response(200, content=self.files[path.rsplit("/", 1)[-1]])
        raise AssertionError(f"Unexpected Hub request: {request.method} {request.url}")


@pytest.fixture
def hub(monkeypatch):
    storage = HubStorage()

    def upload_file(_api, *, path_or_fileobj, path_in_repo, **kwargs):
        storage.files[path_in_repo] = Path(path_or_fileobj).read_bytes()

    def batch_bucket_files(_api, _bucket_id, *, add):
        for source, destination in add:
            storage.files[destination] = Path(source).read_bytes()

    HfFileSystem.clear_instance_cache()
    monkeypatch.setattr(HfApi, "upload_file", upload_file)
    monkeypatch.setattr(HfApi, "batch_bucket_files", batch_bucket_files)
    with httpx.Client(transport=httpx.MockTransport(storage.request)) as client:
        monkeypatch.setattr(hf_http, "_GLOBAL_CLIENT", client)
        yield storage
    HfFileSystem.clear_instance_cache()


@pytest.mark.parametrize("operation", ["cp", "mv"])
@pytest.mark.parametrize("root", HF_STORAGE_ROOTS)
@pytest.mark.parametrize("existing_public", [False, True])
def test_transfer_uploads_into_new_private_or_existing_public_storage(tmp_path, hub, operation, root, existing_public):
    hub.exists = existing_public
    source = tmp_path / "payload.txt"
    source.write_bytes(b"transfer payload")
    destination = f"{root}/payload.txt"

    result = CliRunner().invoke(cli, [operation, str(source), destination], catch_exceptions=False)

    assert result.exit_code == 0, (result.output, result.exception)
    assert hub.files == {"payload.txt": b"transfer payload"}
    assert hub.private is not existing_public
    assert source.exists() is (operation == "cp")
    assert StoragePath(destination).read_bytes() == b"transfer payload"


@pytest.mark.parametrize("root", HF_STORAGE_ROOTS)
def test_hf_listing_and_glob_paths_can_be_reopened_and_copied(tmp_path, hub, root):
    hub.exists = True
    hub.files = {"first.txt": b"first", "second.txt": b"second"}

    entries = list_entries(root)
    assert {entry.url for entry in entries} == {f"{root}/first.txt", f"{root}/second.txt"}
    assert {StoragePath(entry.url).read_bytes() for entry in entries} == {b"first", b"second"}
    assert {path.read_bytes() for path in StoragePath(f"{root}/*.txt").glob()} == {b"first", b"second"}

    result = CliRunner().invoke(cli, ["cp", f"{root}/*.txt", str(tmp_path)])

    assert result.exit_code == 0, (result.output, result.exception)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == {
        "first.txt": b"first",
        "second.txt": b"second",
    }


@pytest.mark.parametrize(
    ("source_url", "destination_url"),
    [
        ("hf://owner/files/payload.txt", "hf://models/owner/files/payload.txt"),
        ("hf://owner/files/payload.txt", "hf://models/owner/files@main/payload.txt"),
        ("hf://datasets/owner/files/payload.txt", "hf://datasets/owner/files@main/payload.txt"),
    ],
)
def test_copy_rejects_same_hf_file_through_equivalent_urls(hub, source_url, destination_url):
    hub.exists = True
    hub.files = {"payload.txt": b"keep this data"}

    with pytest.raises(TransferError):
        copy_plan((source_url,), destination_url, recursive=False, no_clobber=False)
    assert hub.files == {"payload.txt": b"keep this data"}
