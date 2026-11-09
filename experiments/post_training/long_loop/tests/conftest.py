# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.post_training.long_loop import hf_archive


class Hub:
    """Content snapshots at the external Hub API boundary."""

    def __init__(self, directory):
        self.directory = directory
        self.files = {}
        self.private = True
        self.token = "test-token"
        self.revision = "a" * 40

    def create_repo(self, *args, **kwargs):
        if not self.files:
            self.private = kwargs["private"]

    def list_repo_commits(self, *args, **kwargs):
        return [SimpleNamespace(commit_id=self.revision)]

    def parse_safetensors_file_metadata(self, repo, name, **kwargs):
        raw = self.files[name]
        length = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8 : 8 + length])
        return SimpleNamespace(metadata=header.pop("__metadata__", {}), tensors=header)

    def model_info(self, repo, revision=None, files_metadata=False):
        siblings = [
            SimpleNamespace(
                rfilename=name,
                blob_id=hashlib.sha256(data).hexdigest(),
                size=len(data),
                lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if name.endswith(".safetensors") else None,
            )
            for name, data in self.files.items()
        ]
        return SimpleNamespace(private=self.private, sha=self.revision, siblings=siblings)

    def list_repo_files(self, *args, **kwargs):
        return list(self.files)

    def upload_file(self, *, path_in_repo, path_or_fileobj, **kwargs):
        self.files[path_in_repo] = path_or_fileobj

    def upload_large_folder(self, *, folder_path, allow_patterns, **kwargs):
        self.files.update({name: (Path(folder_path) / name).read_bytes() for name in allow_patterns})

    def download(self, repo, filename, **kwargs):
        path = self.directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[filename])
        return str(path)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    result = Hub(tmp_path / "hub")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(hf_archive, "HfApi", lambda **kwargs: result)
    monkeypatch.setattr(hf_archive, "hf_hub_download", result.download)
    return result
