#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deploy the probes daemon to a single Container-Optimized OS GCP VM.

Subcommands:
    build    docker build, tag with git sha + "latest", push to Artifact Registry.
    apply    roll the prod VM to the "latest" image.
    status   show VM state + tail container logs.
    create   one-time: service account, IAM bindings, and the COS VM itself.

Run with ``uv run deploy/deploy.py <command>`` (click resolves from the project
venv), or ``python deploy/deploy.py <command>`` if click is on the path.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import click
from rigging.filesystem import REGION_TO_DATA_BUCKET

logger = logging.getLogger("deploy")

IMAGE_NAME = "infra-probes"
# The probes daemon writes its JSONL roll-ups under this bucket+prefix (see
# infra_probes.py). Rolling a day up overwrites a deterministic per-day object
# when a stranded local file is re-uploaded after a restart, so the SA needs
# create+get+delete — granted via objectUser, scoped by IAM condition to the
# prefix so the canary can't touch the rest of this shared data bucket.
RESULTS_BUCKET = REGION_TO_DATA_BUCKET["us-central1"]
RESULTS_GCS_PREFIX = "infra/probes"
RESULTS_HOST_PATH = "/var/lib/probes"
# Build context / git repo root for `build`: this script lives in deploy/.
PROBES_DIR = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    logger.info("$ %s", " ".join(cmd))
    return subprocess.run(cmd, check=True, text=True, capture_output=capture_output)


def _artifact_registry(region: str, project: str, repo: str) -> str:
    return f"{region}-docker.pkg.dev/{project}/{repo}/{IMAGE_NAME}"


def _service_account(project: str) -> str:
    return f"{IMAGE_NAME}@{project}.iam.gserviceaccount.com"


@click.group()
@click.option("--project", envvar="MARIN_PROBES_PROJECT", default="hai-gcp-models", show_default=True)
@click.option("--region", envvar="MARIN_PROBES_REGION", default="us-central1", show_default=True)
@click.option("--zone", envvar="MARIN_PROBES_ZONE", default="us-central1-b", show_default=True)
@click.option("--vm-name", envvar="MARIN_PROBES_VM", default="infra-probes", show_default=True)
@click.option("--repo", envvar="MARIN_PROBES_REPO", default="marin", show_default=True)
@click.pass_context
def cli(ctx: click.Context, project: str, region: str, zone: str, vm_name: str, repo: str) -> None:
    ctx.obj = {
        "project": project,
        "region": region,
        "zone": zone,
        "vm_name": vm_name,
        "registry": _artifact_registry(region, project, repo),
    }


@cli.command()
@click.pass_obj
def build(cfg: dict[str, str]) -> None:
    """Build the image, tag with git sha and 'latest', push to Artifact Registry."""
    sha = _run(
        ["git", "-C", str(PROBES_DIR), "rev-parse", "--short", "HEAD"],
        capture_output=True,
    ).stdout.strip()
    image_sha = f"{cfg['registry']}:{sha}"
    image_latest = f"{cfg['registry']}:latest"

    logger.info("Building %s", image_sha)
    _run(
        [
            "docker",
            "build",
            "--platform=linux/amd64",
            "-f",
            str(PROBES_DIR / "deploy" / "Dockerfile"),
            "-t",
            image_sha,
            "-t",
            image_latest,
            str(PROBES_DIR),
        ]
    )
    logger.info("Pushing %s and :latest", image_sha)
    _run(["docker", "push", image_sha])
    _run(["docker", "push", image_latest])


@cli.command()
@click.pass_obj
def apply(cfg: dict[str, str]) -> None:
    """Roll the prod VM to the 'latest' image."""
    image_latest = f"{cfg['registry']}:latest"
    logger.info("Rolling VM %s (%s) to %s", cfg["vm_name"], cfg["zone"], image_latest)
    _run(
        [
            "gcloud",
            "compute",
            "instances",
            "update-container",
            cfg["vm_name"],
            f"--project={cfg['project']}",
            f"--zone={cfg['zone']}",
            f"--container-image={image_latest}",
        ]
    )


@cli.command()
@click.pass_obj
def status(cfg: dict[str, str]) -> None:
    """Print VM state and the last 50 lines of container logs."""
    logger.info("VM %s (%s)", cfg["vm_name"], cfg["zone"])
    _run(
        [
            "gcloud",
            "compute",
            "instances",
            "describe",
            cfg["vm_name"],
            f"--project={cfg['project']}",
            f"--zone={cfg['zone']}",
            "--format=value(status,labels,metadata.items.filter(key=gce-container-declaration).extract(value))",
        ]
    )
    logger.info("Last 50 lines of container logs")
    _run(
        [
            "gcloud",
            "compute",
            "ssh",
            cfg["vm_name"],
            f"--project={cfg['project']}",
            f"--zone={cfg['zone']}",
            "--command=" f"docker ps --filter ancestor={cfg['registry']} -q | head -1 | xargs -r docker logs --tail 50",
        ]
    )


@cli.command()
@click.option(
    "--iris-endpoint",
    default="http://iris-controller-marin.c.hai-gcp-models.internal:10000",
    show_default=True,
    help="controller RPC the daemon canaries against.",
)
@click.option("--machine-type", default="e2-small", show_default=True)
@click.pass_obj
def create(cfg: dict[str, str], iris_endpoint: str, machine_type: str) -> None:
    """One-time: create the service account, IAM bindings, and the COS VM.

    Idempotent enough to re-run: gcloud add-iam-policy-binding is a no-op when the
    binding exists, but the service-account and instance creates fail if they
    already exist — delete them first to recreate.
    """
    project = cfg["project"]
    region = cfg["region"]
    sa = _service_account(project)
    member = f"serviceAccount:{sa}"

    logger.info("Creating service account %s", sa)
    _run(["gcloud", "iam", "service-accounts", "create", IMAGE_NAME, f"--project={project}"])

    # SA needs: pull image, ship stdout to Cloud Logging, manage GCS roll-ups.
    logger.info("Granting IAM roles to %s", sa)
    _run(
        [
            "gcloud",
            "artifacts",
            "repositories",
            "add-iam-policy-binding",
            "marin",
            f"--project={project}",
            f"--location={region}",
            f"--member={member}",
            "--role=roles/artifactregistry.reader",
        ]
    )
    _run(
        [
            "gcloud",
            "projects",
            "add-iam-policy-binding",
            project,
            f"--member={member}",
            "--role=roles/logging.logWriter",
            "--condition=None",
        ]
    )
    # objectUser (create/get/delete) restricted to the roll-up prefix. The
    # bucket-scoped objects.list it implies is intentionally not covered by the
    # object-name condition; gcsfs only uses list to sniff bucket type and falls
    # back gracefully, so the upload still succeeds.
    prefix_condition = (
        f'expression=resource.name.startsWith("projects/_/buckets/{RESULTS_BUCKET}'
        f'/objects/{RESULTS_GCS_PREFIX}/"),title=infra-probes-prefix,'
        "description=Limit infra-probes SA object access to its rollup prefix"
    )
    _run(
        [
            "gcloud",
            "storage",
            "buckets",
            "add-iam-policy-binding",
            f"gs://{RESULTS_BUCKET}",
            f"--member={member}",
            "--role=roles/storage.objectUser",
            f"--condition={prefix_condition}",
        ]
    )

    # The host mount persists the JSONL across container restarts; the
    # startup-script makes it writable by the uid-1000 container.
    startup_script = f"#!/bin/bash\nmkdir -p {RESULTS_HOST_PATH} && chown 1000:1000 {RESULTS_HOST_PATH}"
    image_latest = f"{cfg['registry']}:latest"
    logger.info("Creating VM %s (%s) on %s", cfg["vm_name"], cfg["zone"], image_latest)
    _run(
        [
            "gcloud",
            "compute",
            "instances",
            "create-with-container",
            cfg["vm_name"],
            f"--project={project}",
            f"--zone={cfg['zone']}",
            f"--machine-type={machine_type}",
            f"--service-account={sa}",
            "--scopes=cloud-platform",
            f"--container-image={image_latest}",
            "--container-restart-policy=always",
            f"--container-arg=--iris-endpoint={iris_endpoint}",
            f"--container-mount-host-path=mount-path={RESULTS_HOST_PATH},host-path={RESULTS_HOST_PATH},mode=rw",
            f"--metadata=startup-script={startup_script}",
            "--tags=infra-probes",
        ]
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cli()
