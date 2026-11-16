# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Commit references to private Hugging Face datasets instead of committed data files.

A reference index pins one dataset revision and lists every file it serves with its
SHA-256. Readers download through :func:`hf_hub_download` and verify bytes before
parsing; writers commit new or updated files as a new immutable revision parented at
the current head, then rewrite the committed index for the next landing.
"""

import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from pydantic import BaseModel, ConfigDict, Field


class DataReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DataReferenceIndex(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    references: list[DataReference]


def read_reference_index(path: Path) -> DataReferenceIndex:
    return DataReferenceIndex.model_validate_json(path.read_bytes())


def read_referenced_documents(index_path: Path) -> dict[str, bytes]:
    """Download every referenced document at its pinned revision, verifying its bytes."""
    index = read_reference_index(index_path)
    documents: dict[str, bytes] = {}
    for reference in index.references:
        downloaded = hf_hub_download(
            repo_id=reference.repo,
            repo_type="dataset",
            revision=reference.revision,
            filename=reference.path,
        )
        data = Path(downloaded).read_bytes()
        if hashlib.sha256(data).hexdigest() != reference.sha256:
            raise ValueError(f"Referenced dataset bytes changed: {reference.repo}@{reference.revision}:{reference.path}")
        documents[reference.path] = data
    return documents


def load_referenced_json(index_path: Path) -> dict[str, Any]:
    return {path: json.loads(data) for path, data in read_referenced_documents(index_path).items()}


def _private_dataset(api: HfApi, repo: str) -> None:
    if not api.repo_info(repo_id=repo, repo_type="dataset").private:
        raise ValueError(f"Reference dataset must be private: {repo}")


def commit_reference_documents(
    index_path: Path, documents: Mapping[str, bytes], *, message: str, max_attempts: int = 5
) -> str:
    """Upload new or updated documents to the referenced private dataset and rewrite the index.

    Each write creates a new immutable revision parented at the dataset's current head;
    an existing file is replaced in that revision, so an upsert rewrites one shard while
    every other pinned file keeps its bytes. A concurrent writer moves the head between
    the listing and the commit, so the commit retries against the fresh tree. The
    rewritten index is staged on disk for the caller's next git commit; this function
    never touches git.
    """
    index = read_reference_index(index_path)
    api = HfApi()
    _private_dataset(api, index.repo)
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        head = api.repo_info(repo_id=index.repo, repo_type="dataset").sha
        operations = [
            CommitOperationAdd(path_in_repo=path, path_or_fileobj=data) for path, data in sorted(documents.items())
        ]
        try:
            commit = api.create_commit(
                repo_id=index.repo,
                repo_type="dataset",
                parent_commit=head,
                commit_message=message,
                operations=operations,
            )
        except Exception as error:
            last_error = error
            time.sleep(2**attempt)
            continue
        expected = {reference.path: reference.sha256 for reference in index.references}
        expected.update({path: hashlib.sha256(data).hexdigest() for path, data in documents.items()})
        for path, sha in sorted(expected.items()):
            downloaded = hf_hub_download(repo_id=index.repo, repo_type="dataset", revision=commit.oid, filename=path)
            if hashlib.sha256(Path(downloaded).read_bytes()).hexdigest() != sha:
                raise ValueError(f"Referenced dataset bytes changed: {index.repo}@{commit.oid}:{path}")
        retained = [reference for reference in index.references if reference.path not in documents]
        updated_references = [
            DataReference(repo=index.repo, revision=commit.oid, path=path, sha256=hashlib.sha256(data).hexdigest())
            for path, data in sorted(documents.items())
        ]
        updated = DataReferenceIndex(
            repo=index.repo,
            revision=commit.oid,
            references=[*retained, *updated_references],
        )
        index_path.write_text(updated.model_dump_json(indent=2) + "\n")
        return commit.oid
    raise RuntimeError(f"Could not commit to {index.repo} after {max_attempts} attempts") from last_error


def create_reference_index(index_path: Path, repo: str, documents: Mapping[str, bytes], *, message: str) -> str:
    """Seed a fresh reference index: create the private dataset and commit its files."""
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    _private_dataset(api, repo)
    commit = api.create_commit(
        repo_id=repo,
        repo_type="dataset",
        commit_message=message,
        operations=[
            CommitOperationAdd(path_in_repo=path, path_or_fileobj=data) for path, data in sorted(documents.items())
        ],
    )
    for path, data in documents.items():
        downloaded = hf_hub_download(repo_id=repo, repo_type="dataset", revision=commit.oid, filename=path)
        if hashlib.sha256(Path(downloaded).read_bytes()).hexdigest() != hashlib.sha256(data).hexdigest():
            raise ValueError(f"Referenced dataset bytes changed: {repo}@{commit.oid}:{path}")
    index = DataReferenceIndex(
        repo=repo,
        revision=commit.oid,
        references=[
            DataReference(repo=repo, revision=commit.oid, path=path, sha256=hashlib.sha256(data).hexdigest())
            for path, data in sorted(documents.items())
        ],
    )
    index_path.write_text(index.model_dump_json(indent=2) + "\n")
    return commit.oid
