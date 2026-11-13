# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read immutable Atlas registry documents from their pinned private archive."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from pydantic import BaseModel, ConfigDict, Field


class RegistryReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def read_registry_json(path: Path) -> dict[str, Any]:
    """Download an exact registry revision and verify its bytes before decoding."""
    reference = RegistryReference.model_validate_json(path.read_bytes())
    downloaded = hf_hub_download(
        repo_id=reference.repo,
        repo_type="dataset",
        revision=reference.revision,
        filename=reference.path,
    )
    data = Path(downloaded).read_bytes()
    if hashlib.sha256(data).hexdigest() != reference.sha256:
        raise ValueError(f"Registry bytes changed: {reference.repo}@{reference.revision}:{reference.path}")
    document = json.loads(data)
    if not isinstance(document, dict):
        raise ValueError(f"Registry document must contain an object: {reference.path}")
    return document


def archive_registry_documents(
    staged_root: Path, references_root: Path, *, repo: str, parent_revision: str, commit_message: str
) -> str:
    """Append staged documents, verify readback, then update their Git references.

    Staged files use their repository-relative paths. Existing archive revisions
    and files are preserved; this operation never deletes remote or local files.
    """
    documents = {
        path.relative_to(staged_root).as_posix(): path.read_bytes() for path in sorted(staged_root.rglob("*.json"))
    }
    if not documents:
        raise ValueError(f"No registry documents staged in {staged_root}")
    shard_directories = {Path(name).parent for name in documents if Path(name).match("part-*.json")}
    for directory in shard_directories:
        for previous in (references_root / directory).glob("part-*.json"):
            name = previous.relative_to(references_root).as_posix()
            # Preserve the old artifact in its prior immutable revision while
            # excluding inactive tail shards from the refreshed collection.
            documents.setdefault(name, b"{}\n")
    for name, data in documents.items():
        if not isinstance(json.loads(data), dict):
            raise ValueError(f"Registry document must contain an object: {name}")
    api = HfApi()
    if not api.repo_info(repo_id=repo, repo_type="dataset", revision=parent_revision).private:
        raise ValueError(f"Registry repository must be private: {repo}")
    commit = api.create_commit(
        repo_id=repo,
        repo_type="dataset",
        parent_commit=parent_revision,
        commit_message=commit_message,
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=data) for name, data in documents.items()],
    )
    references = {
        name: RegistryReference(repo=repo, revision=commit.oid, path=name, sha256=hashlib.sha256(data).hexdigest())
        for name, data in documents.items()
    }
    with TemporaryDirectory(prefix="atlas-registry-readback-") as temporary:
        for name, reference in references.items():
            pinned = Path(temporary) / name
            pinned.parent.mkdir(parents=True, exist_ok=True)
            pinned.write_text(reference.model_dump_json(indent=2) + "\n")
            read_registry_json(pinned)
    for name, reference in references.items():
        destination = references_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(reference.model_dump_json(indent=2) + "\n")
    return commit.oid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--references-root", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--parent-revision", required=True)
    parser.add_argument("--message", required=True)
    args = parser.parse_args()
    revision = archive_registry_documents(
        args.staged_root,
        args.references_root,
        repo=args.repo,
        parent_revision=args.parent_revision,
        commit_message=args.message,
    )
    print(json.dumps({"repo": args.repo, "revision": revision}))


if __name__ == "__main__":
    main()
