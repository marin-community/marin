# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned payload acquisition stays regional and refuses same-size corrupt objects."""

import hashlib
import subprocess
from types import SimpleNamespace

import fsspec
import pytest
from fsspec.implementations.memory import MemoryFileSystem
from huggingface_hub import HfApi
from huggingface_hub.hf_api import BlobLfsInfo, RepoSibling
from iris.cluster.client.job_info import JobInfo, get_job_info, set_job_info
from iris.cluster.types import JobName


class MemoryS3(MemoryFileSystem):
    """S3 protocol transport with actual fsspec operations against memory."""

    protocol = "s3"

    @classmethod
    def _strip_protocol(cls, path):
        return MemoryFileSystem._strip_protocol(str(path).removeprefix("s3://"))

    def glob(self, path, detail=False, **kwargs):
        rows = super().glob(path, detail=detail, **kwargs)
        if detail:
            return {name.lstrip("/"): {**info, "name": name.lstrip("/")} for name, info in rows.items()}
        return [name.lstrip("/") for name in rows]


class MemoryHf(MemoryFileSystem):
    """HF file listing returns revision-qualified paths like HfFileSystem."""

    protocol = "hf"

    @classmethod
    def _strip_protocol(cls, path):
        return MemoryFileSystem._strip_protocol(str(path).removeprefix("hf://"))

    def glob(self, path, **kwargs):
        parts = self._strip_protocol(path).strip("/").split("/")
        parts[2] += "@" + kwargs.pop("revision")
        rows = super().glob("/" + "/".join(parts), **kwargs)
        return {name.lstrip("/"): {**info, "name": name.lstrip("/")} for name, info in rows.items()}


@pytest.fixture
def remote(monkeypatch):
    original = {name: fsspec.get_filesystem_class(name) for name in ("s3", "hf")}
    before = set(MemoryFileSystem.store)
    records = {}
    fsspec.register_implementation("s3", MemoryS3, clobber=True)
    fsspec.register_implementation("hf", MemoryHf, clobber=True)

    def info(_client, repository, *, revision, files_metadata):
        assert files_metadata
        return SimpleNamespace(sha=revision, siblings=records[repository, revision], private=False)

    monkeypatch.setattr(HfApi, "dataset_info", info)
    prior = get_job_info()
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    set_job_info(JobInfo(task_id=JobName.from_wire("/worker/atlas-ingest-test").task(0), worker_region="US-EAST-02A"))
    yield records
    set_job_info(prior)
    for name, implementation in original.items():
        fsspec.register_implementation(name, implementation, clobber=True)
    for name in set(MemoryFileSystem.store) - before:
        MemoryFileSystem.store.pop(name)


def populate_source(records, policy, files):
    siblings = []
    filesystem = MemoryHf()
    for index, (name, content) in enumerate(files.items()):
        path = f"datasets/{policy.upstream.repository}@{policy.upstream.revision}/{name}"
        filesystem.pipe_file(path, content)
        sha256 = hashlib.sha256(content).hexdigest()
        # An LFS sibling's Git blob identifies its pointer, not its payload.
        blob_content = f"oid sha256:{sha256}\nsize {len(content)}\n".encode() if index == 0 else content
        blob = subprocess.check_output(["git", "hash-object", "--stdin"], input=blob_content).decode().strip()
        siblings.append(
            RepoSibling(
                rfilename=name,
                size=len(content),
                blob_id=blob,
                lfs=BlobLfsInfo(size=len(content), sha256=sha256, pointer_size=127) if index == 0 else None,
            )
        )
    records[policy.upstream.repository, policy.upstream.revision] = siblings
