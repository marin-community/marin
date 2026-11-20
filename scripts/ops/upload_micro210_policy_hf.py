# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the Micro210 evaluation policy inputs to Hugging Face and verify pinned readback.

The policy JSON is public content and goes to the public ``open-athena/micro210-policy``
repository, one git tag per policy version. The prepared-task table re-hosts benchmark
items that must not be public, so it goes to the private
``open-athena/micro210-prepared-tasks`` repository. Both uploads run from an in-region
CPU Iris task: the sources are the regional CoreWeave S3 copies, the task has no GCP
credentials, and the script fails loudly before uploading anything whose bytes do not
match the pins that ``experiments.evaluation.micro210_policy`` holds.

After uploading, every pinned ``hf://`` reference is read back through
:class:`~rigging.filesystem.storage_path.StoragePath` and verified (byte size, SHA-256,
parquet row count), the public policy is re-read anonymously over the resolve endpoint,
and the private repository is checked to reject anonymous reads. The script prints the
pinned commit revisions for the code constants and never prints token material.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import requests
from huggingface_hub import CommitOperationAdd, HfApi, create_commit, create_repo, create_tag
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import get_secret

from experiments.evaluation.micro210_policy import (
    LEGACY_POLICY_REFERENCE,
    POLICY_REFERENCE,
    POLICY_REPO,
    PREPARED_REPO,
    Micro210PolicyReference,
)

# Migration source: the regional CoreWeave copies written when the inputs left GCS.
# Code references never point here; they point at the HF repositories pinned above.
S3_ROOT = "s3://marin-us-east-08a/marin/users/benfeuer/evaluation-policies/micro210"
PREPARED_ROWS = 440

POLICY_CARD = """\
---
license: apache-2.0
language:
- en
tags:
- evaluation
- reinforcement-learning
---

# Micro210 evaluation policy

Content-pinned protocol definitions for the Micro210 evaluation
([marin-community/marin#10804](https://github.com/marin-community/marin/issues/10804)):
one git tag per policy version, one `policy.json` per tag. The evaluation code pins the
repository, commit revision, byte size and SHA-256, and verifies the bytes on every read.

Tags:

- `2026.11.18` — the first published policy revision, SHA-256
  `e0faa32c68994e8bf01d0ab7177893aa9c29f8c4309b9183de5c3dfcb6fa67aa`, 122243 bytes.
- `2026.11.20-together` — the current policy, SHA-256
  `8f7ef8170fbef2880214cbf03f70d80cffa3114276e024389617777bf07f53e5`, 121292 bytes.

A policy is a human-readable JSON protocol: dataset selection, judge configuration, and
runtime settings for the frozen evaluation. It contains no PII: no names, emails, or
usernames.

Prepared task inputs derived from these policies are hosted separately in the private
[`open-athena/micro210-prepared-tasks`](https://huggingface.co/datasets/open-athena/micro210-prepared-tasks)
repository, because they re-host benchmark items whose upstream sources ask that they
not be reposted publicly.
"""

PREPARED_CARD = """\
---
license: other
language:
- en
tags:
- evaluation
- reinforcement-learning
---

# Micro210 prepared task inputs

The prepared-task table for the Micro210 evaluation protocol
([marin-community/marin#10804](https://github.com/marin-community/marin/issues/10804)):
`rollouts.parquet`, 440 rows, SHA-256
`ba099cbc869af0dd1c14f14586557e38a812d6b956442e8b1f406b36cc42c7a0`, 17152253 bytes,
derived from policy `2026.11.20-together` in the public
[`open-athena/micro210-policy`](https://huggingface.co/datasets/open-athena/micro210-policy)
repository (tag `2026.11.20-together`).

This repository is private on purpose: the table re-hosts GPQA-Diamond, AIME,
OlympiadBench and SWE-bench items, and GPQA asks that examples not be posted publicly.
Evaluation tasks read it with the standard HF_TOKEN. The evaluation code pins the
repository, commit revision, byte size and SHA-256, and verifies the bytes on every read.

The table contains no PII: no names, emails, or usernames.
"""


@dataclass(frozen=True)
class PolicyRevisions:
    """The pinned commit revisions a publication left on the policy repository."""

    legacy: str
    current: str


@dataclass(frozen=True)
class PreparedSource:
    """A verified local copy of the prepared table, with its protocol identity."""

    path: Path
    filename: str
    digest: str


def prepare_hf_token() -> None:
    """Expose the task's forwarded HF_TOKEN to huggingface_hub; fail fast when absent."""
    token = os.environ.get("HF_TOKEN") or get_secret("HF_TOKEN")
    if not token:
        raise SystemExit("No HF_TOKEN available on this task; refusing to publish anonymously")
    os.environ["HF_TOKEN"] = token


def read_and_verify(uri: str, digest: str, byte_size: int | None = None) -> bytes:
    """Read bytes through StoragePath and enforce the pinned SHA-256 (and size, when given)."""
    raw = StoragePath(uri).read_bytes()
    if (byte_size is not None and len(raw) != byte_size) or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Pinned Micro210 input bytes mismatch at {uri}")
    return raw


def s3_policy_source(reference: Micro210PolicyReference) -> str:
    """The regional S3 copy of one policy version, keyed by its own digest."""
    return f"{S3_ROOT}/{reference.revision}/{reference.sha256}/policy.json"


def read_prepared_source(protocol: dict) -> PreparedSource:
    """Fetch the prepared table from the regional copy and verify its digest and rows."""
    digest = protocol["prepared_rollouts_sha256"]
    filename = protocol["evaluation"]["data_relative_path"]
    directory = StoragePath(f"{S3_ROOT}/{POLICY_REFERENCE.revision}/{digest}")
    try:
        raw = read_and_verify(str(directory / filename), digest)
    except Exception as error:
        raise ValueError(
            f"Prepared table not verifiable at {directory / filename}: "
            f"directory holds {[str(entry) for entry in directory.ls()]}"
        ) from error
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as handle:
        handle.write(raw)
        path = Path(handle.name)
    rows = pq.read_metadata(path).num_rows
    if rows != PREPARED_ROWS:
        raise ValueError(f"Prepared table has {rows} rows, expected {PREPARED_ROWS}")
    print(f"prepared source verified: {len(raw)} bytes, sha256 {digest}, {rows} rows")
    return PreparedSource(path=path, filename=filename, digest=digest)


def commit_revision(api: HfApi, repo: str, revision: str) -> str:
    """Return the 40-character commit revision a repository revision resolves to."""
    sha = api.dataset_info(repo, revision=revision).sha
    if len(sha) != 40:
        raise ValueError(f"Repository {repo} revision {revision} resolved to a non-commit: {sha}")
    return sha


def tag_revision(api: HfApi, repo: str, tag: str) -> str | None:
    """The commit revision a tag points at, or None when the tag does not exist."""
    refs = api.list_repo_refs(repo_id=repo, repo_type="dataset")
    return next((ref.target_commit for ref in refs.tags if ref.name == tag), None)


def tag_verifies(repo: str, tag: str, path: str, digest: str, byte_size: int | None) -> bool:
    """Whether an existing tag already holds the pinned bytes (read failures propagate)."""
    try:
        read_and_verify(f"hf://datasets/{repo}@{tag}/{path}", digest, byte_size)
    except ValueError:
        return False
    return True


def publish_version(
    api: HfApi,
    repo: str,
    tag: str,
    commit_message: str,
    card: str,
    content: bytes | str,
    path: str,
    digest: str,
    byte_size: int | None,
) -> str:
    """Commit one version unless its tag already carries verifying bytes; return the pinned revision."""
    existing = tag_revision(api, repo, tag)
    if existing is not None:
        if not tag_verifies(repo, tag, path, digest, byte_size):
            raise ValueError(f"Tag {tag!r} on {repo} exists with bytes that do not match the pin")
        return existing
    operations = [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode()),
        CommitOperationAdd(path_in_repo=path, path_or_fileobj=content),
    ]
    create_commit(repo_id=repo, repo_type="dataset", operations=operations, commit_message=commit_message)
    sha = commit_revision(api, repo, "main")
    create_tag(repo_id=repo, repo_type="dataset", tag=tag, revision=sha)
    return sha


def publish_policy(api: HfApi, legacy: bytes, current: bytes) -> PolicyRevisions:
    """Publish one policy commit per version to the public repo; return pinned revisions."""
    create_repo(POLICY_REPO, repo_type="dataset", private=False, exist_ok=True)
    legacy_sha = publish_version(
        api,
        POLICY_REPO,
        LEGACY_POLICY_REFERENCE.revision,
        f"Publish Micro210 policy {LEGACY_POLICY_REFERENCE.revision}",
        POLICY_CARD,
        legacy,
        "policy.json",
        LEGACY_POLICY_REFERENCE.sha256,
        LEGACY_POLICY_REFERENCE.byte_size,
    )
    current_sha = publish_version(
        api,
        POLICY_REPO,
        POLICY_REFERENCE.revision,
        f"Publish Micro210 policy {POLICY_REFERENCE.revision}",
        POLICY_CARD,
        current,
        "policy.json",
        POLICY_REFERENCE.sha256,
        POLICY_REFERENCE.byte_size,
    )
    return PolicyRevisions(legacy=legacy_sha, current=current_sha)


def publish_prepared(api: HfApi, source: PreparedSource) -> str:
    """Publish the prepared table to the private repo; return the pinned revision."""
    create_repo(PREPARED_REPO, repo_type="dataset", private=True, exist_ok=True)
    return publish_version(
        api,
        PREPARED_REPO,
        POLICY_REFERENCE.revision,
        "Publish Micro210 prepared tasks",
        PREPARED_CARD,
        str(source.path),
        source.filename,
        source.digest,
        None,
    )


def verify_readback(revisions: PolicyRevisions, source: PreparedSource, prepared_sha: str) -> None:
    """Read every pinned reference back and verify bytes, and the public/private split."""
    current = read_and_verify(
        f"hf://datasets/{POLICY_REPO}@{revisions.current}/policy.json",
        POLICY_REFERENCE.sha256,
        POLICY_REFERENCE.byte_size,
    )
    read_and_verify(
        f"hf://datasets/{POLICY_REPO}@{revisions.legacy}/policy.json",
        LEGACY_POLICY_REFERENCE.sha256,
        LEGACY_POLICY_REFERENCE.byte_size,
    )
    read_and_verify(f"hf://datasets/{PREPARED_REPO}@{prepared_sha}/{source.filename}", source.digest)
    public_url = f"https://huggingface.co/datasets/{POLICY_REPO}/resolve/{revisions.current}/policy.json"
    anonymous = requests.get(public_url, timeout=120)
    if anonymous.status_code != 200 or anonymous.content != current:
        raise ValueError(f"Public policy is not anonymously readable: HTTP {anonymous.status_code}")
    private_url = f"https://huggingface.co/datasets/{PREPARED_REPO}/resolve/{prepared_sha}/{source.filename}"
    rejected = requests.get(private_url, timeout=120, allow_redirects=False)
    if rejected.status_code in (200, 301, 302, 307, 308):
        raise ValueError("Prepared table repository is anonymously readable; it must stay private")
    print(f"readback verified public policy anonymous read: HTTP 200, sha256 {POLICY_REFERENCE.sha256}")
    print(
        f"readback verified legacy policy revision {LEGACY_POLICY_REFERENCE.revision}: "
        f"sha256 {LEGACY_POLICY_REFERENCE.sha256}"
    )
    print(f"readback verified prepared table anonymously rejected: HTTP {rejected.status_code}")


def report_gcp_absence() -> None:
    """Print the no-GCP-credential state the readback was verified under."""
    default_credentials = Path("~/.config/gcloud/application_default_credentials.json").expanduser()
    print(
        f"gcp_credentials: GOOGLE_APPLICATION_CREDENTIALS={bool(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))}, "
        f"GOOGLE_CLOUD_PROJECT={bool(os.environ.get('GOOGLE_CLOUD_PROJECT'))}, "
        f"adc_file={default_credentials.exists()}"
    )


def main() -> None:
    configure_coreweave_s3()
    prepare_hf_token()
    api = HfApi()
    identity = api.whoami()
    in_org = any(org.get("name") == "open-athena" for org in identity.get("orgs", []))
    print(f"hf_auth: ok, open-athena member={in_org}")
    if not in_org:
        raise SystemExit("HF identity is not an open-athena member; refusing to publish")

    legacy = read_and_verify(
        s3_policy_source(LEGACY_POLICY_REFERENCE),
        LEGACY_POLICY_REFERENCE.sha256,
        LEGACY_POLICY_REFERENCE.byte_size,
    )
    current = read_and_verify(s3_policy_source(POLICY_REFERENCE), POLICY_REFERENCE.sha256, POLICY_REFERENCE.byte_size)
    print(
        f"policy sources verified: legacy {LEGACY_POLICY_REFERENCE.byte_size} bytes, "
        f"current {POLICY_REFERENCE.byte_size} bytes"
    )
    source = read_prepared_source(json.loads(current))
    report_gcp_absence()

    revisions = publish_policy(api, legacy, current)
    prepared_sha = publish_prepared(api, source)
    print("pinned revisions:")
    print(f"  POLICY_COMMIT = {revisions.current}")
    print(f"  LEGACY_POLICY_COMMIT = {revisions.legacy}")
    print(f"  PREPARED_COMMIT = {prepared_sha}")
    verify_readback(revisions, source, prepared_sha)
    print("PASS")


if __name__ == "__main__":
    main()
