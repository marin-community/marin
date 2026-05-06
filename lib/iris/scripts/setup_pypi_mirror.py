#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provision Artifact Registry remote-Python repos for the Iris PyPI mirror.

Creates four AR remote-Python repositories (us / europe x pypi-mirror /
pytorch-cpu-mirror) under ``hai-gcp-models``, applies the same 30-day
cleanup policy used by ``ghcr-mirror``, and grants the iris worker service
account ``roles/artifactregistry.reader`` on each repo. Re-running is safe:
existing repos are left in place but their cleanup policy and IAM bindings
are re-applied so drift is corrected.

Usage:
    uv run python lib/iris/scripts/setup_pypi_mirror.py
    uv run python lib/iris/scripts/setup_pypi_mirror.py --dry-run

Reference: ``.agents/projects/iris_pypi_mirror/spec.md`` §"Infra contract".
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import click

logger = logging.getLogger("setup-pypi-mirror")

PROJECT: str = "hai-gcp-models"
REPOSITORY_FORMAT: str = "python"
WORKER_SA_ID: str = "iris-worker"
WORKER_READER_ROLE: str = "roles/artifactregistry.reader"

PYPI_MIRROR_REPO: str = "pypi-mirror"
PYTORCH_CPU_MIRROR_REPO: str = "pytorch-cpu-mirror"
PYTORCH_CPU_UPSTREAM: str = "https://download.pytorch.org/whl/cpu"

# Copied verbatim from `ghcr-mirror` (see lib/iris/docs/image-push.md).
# Keep recent 16 versions; delete anything older than 30 days.
CLEANUP_POLICY: list[dict] = [
    {
        "name": "delete-older-than-30d",
        "action": {"type": "Delete"},
        "condition": {
            "tagState": "any",
            "olderThan": "2592000s",
        },
    },
    {
        "name": "keep-latest",
        "action": {"type": "Keep"},
        "mostRecentVersions": {
            "keepCount": 16,
        },
    },
]


@dataclass(frozen=True)
class RepoSpec:
    """A single AR remote-Python repo to provision."""

    location: str
    name: str
    # If set, points at a custom upstream (e.g. download.pytorch.org).
    # If None, mirrors PyPI directly via --remote-python-repo=PYPI.
    custom_upstream: str | None
    description: str


REPOS: tuple[RepoSpec, ...] = (
    RepoSpec(
        location="us",
        name=PYPI_MIRROR_REPO,
        custom_upstream=None,
        description="Remote proxy for pypi.org (US multi-region)",
    ),
    RepoSpec(
        location="us",
        name=PYTORCH_CPU_MIRROR_REPO,
        custom_upstream=PYTORCH_CPU_UPSTREAM,
        description="Remote proxy for download.pytorch.org/whl/cpu (US multi-region)",
    ),
    RepoSpec(
        location="europe",
        name=PYPI_MIRROR_REPO,
        custom_upstream=None,
        description="Remote proxy for pypi.org (Europe multi-region)",
    ),
    RepoSpec(
        location="europe",
        name=PYTORCH_CPU_MIRROR_REPO,
        custom_upstream=PYTORCH_CPU_UPSTREAM,
        description="Remote proxy for download.pytorch.org/whl/cpu (Europe multi-region)",
    ),
)


def _run(
    cmd: list[str],
    *,
    dry_run: bool = False,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    logger.info("$ %s", " ".join(cmd))
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    return subprocess.run(cmd, check=True, text=True, capture_output=capture_output)


def _worker_sa_email(project: str) -> str:
    return f"{WORKER_SA_ID}@{project}.iam.gserviceaccount.com"


def _repo_exists(spec: RepoSpec, project: str) -> bool:
    """Return True iff the repo already exists in AR."""
    result = subprocess.run(
        [
            "gcloud",
            "artifacts",
            "repositories",
            "describe",
            spec.name,
            f"--project={project}",
            f"--location={spec.location}",
            "--format=value(name)",
        ],
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def _create_repo(spec: RepoSpec, project: str, *, dry_run: bool) -> None:
    cmd = [
        "gcloud",
        "artifacts",
        "repositories",
        "create",
        spec.name,
        f"--project={project}",
        f"--location={spec.location}",
        f"--repository-format={REPOSITORY_FORMAT}",
        "--mode=remote-repository",
        f"--description={spec.description}",
    ]
    # gcloud uses a single flag for both PYPI and custom upstreams. The
    # value is either the literal `PYPI` or a full http(s) URI.
    if spec.custom_upstream is None:
        cmd.append("--remote-python-repo=PYPI")
    else:
        cmd.append(f"--remote-python-repo={spec.custom_upstream}")
    _run(cmd, dry_run=dry_run)


def _apply_cleanup_policy(spec: RepoSpec, project: str, *, dry_run: bool) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(CLEANUP_POLICY, f)
        policy_path = f.name
    try:
        _run(
            [
                "gcloud",
                "artifacts",
                "repositories",
                "set-cleanup-policies",
                spec.name,
                f"--project={project}",
                f"--location={spec.location}",
                f"--policy={policy_path}",
                "--no-dry-run",
            ],
            dry_run=dry_run,
        )
    finally:
        os.unlink(policy_path)


def _grant_reader(spec: RepoSpec, project: str, member: str, *, dry_run: bool) -> None:
    """Grant ``roles/artifactregistry.reader`` to *member* on this repo.

    `add-iam-policy-binding` is idempotent on the GCP side: if the binding
    already exists, the call is a no-op (still returns 0). We always call it
    so drifted bindings get re-applied.
    """
    _run(
        [
            "gcloud",
            "artifacts",
            "repositories",
            "add-iam-policy-binding",
            spec.name,
            f"--project={project}",
            f"--location={spec.location}",
            f"--member={member}",
            f"--role={WORKER_READER_ROLE}",
            "--condition=None",
        ],
        dry_run=dry_run,
        capture_output=True,
    )


def _provision(spec: RepoSpec, project: str, worker_member: str, *, dry_run: bool) -> str:
    """Provision one repo end-to-end. Returns 'created' or 'exists'."""
    if _repo_exists(spec, project):
        logger.info("Repo already exists: %s/%s", spec.location, spec.name)
        status = "exists"
    else:
        _create_repo(spec, project, dry_run=dry_run)
        status = "created"

    _apply_cleanup_policy(spec, project, dry_run=dry_run)
    _grant_reader(spec, project, worker_member, dry_run=dry_run)
    return status


@click.command(help=__doc__)
@click.option("--project", default=PROJECT, show_default=True, help="GCP project id")
@click.option(
    "--worker-sa-id",
    default=WORKER_SA_ID,
    show_default=True,
    help="Iris worker service account id (without @project suffix)",
)
@click.option("--dry-run", is_flag=True, help="Print planned gcloud commands without changing state")
@click.option("-v", "--verbose", is_flag=True, default=True, help="Verbose logging (default on)")
def main(project: str, worker_sa_id: str, dry_run: bool, verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

    worker_email = f"{worker_sa_id}@{project}.iam.gserviceaccount.com"
    worker_member = f"serviceAccount:{worker_email}"

    logger.info("Project:       %s", project)
    logger.info("Worker SA:     %s", worker_email)
    logger.info("Dry run:       %s", dry_run)
    logger.info("Repos to provision: %d", len(REPOS))

    summary: list[dict[str, str]] = []
    for spec in REPOS:
        logger.info("--- %s/%s ---", spec.location, spec.name)
        status = _provision(spec, project, worker_member, dry_run=dry_run)
        summary.append(
            {
                "location": spec.location,
                "name": spec.name,
                "status": status,
                "upstream": spec.custom_upstream or "PYPI",
            }
        )

    print(json.dumps({"project": project, "worker_member": worker_member, "repos": summary}, indent=2))


if __name__ == "__main__":
    main()
