#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deploy the delete-o-tron dashboard to a GCP VM.

Usage:
    uv run scripts/storage/deploy.py deploy [--zone ZONE] [--machine-type TYPE]
    uv run scripts/storage/deploy.py sync-data [--region REGION]
"""

import logging
import os
import subprocess
from pathlib import Path

import click

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PURGE_DIR = REPO_ROOT / "scripts" / "storage" / "purge"
DEFAULT_ZONE = "us-central2-b"
DEFAULT_MACHINE_TYPE = "e2-highmem-4"
VM_NAME = "delete-o-tron"
IMAGE_NAME = "delete-o-tron"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, printing it first."""
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def _get_project() -> str:
    project = os.environ.get("GCP_PROJECT")
    if project:
        return project
    result = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        capture_output=True,
        text=True,
        check=False,
    )
    project = result.stdout.strip()
    if result.returncode == 0 and project and project != "(unset)":
        return project
    raise click.ClickException("Could not determine GCP project. Set GCP_PROJECT or gcloud config.")


@click.group()
def cli():
    """Deploy and manage the delete-o-tron dashboard."""
    pass


@cli.command()
@click.option("--zone", default=DEFAULT_ZONE, show_default=True, help="GCP zone for the VM.")
@click.option("--machine-type", default=DEFAULT_MACHINE_TYPE, show_default=True)
@click.option("--skip-build", is_flag=True, help="Skip Docker build+push, just update the VM.")
@click.option("--skip-data", is_flag=True, help="Skip syncing data to GCS.")
def deploy(zone: str, machine_type: str, skip_build: bool, skip_data: bool):
    """Build, push, and deploy the dashboard to a GCP VM."""
    project = _get_project()
    region = zone.rsplit("-", 1)[0]
    temp_bucket = f"marin-tmp-{region}"
    gcs_prefix = f"gs://{temp_bucket}/ttl=30d/delete-o-tron"
    registry = f"gcr.io/{project}"
    image_tag = f"{registry}/{IMAGE_NAME}:latest"

    # Build and push
    if not skip_build:
        click.echo("\n==> Building Docker image...")
        _run(
            [
                "docker",
                "build",
                "-t",
                f"{IMAGE_NAME}:latest",
                "-f",
                str(REPO_ROOT / "scripts" / "storage" / "Dockerfile"),
                str(REPO_ROOT),
            ]
        )
        _run(["docker", "tag", f"{IMAGE_NAME}:latest", image_tag])
        click.echo(f"\n==> Pushing to {image_tag}...")
        _run(["docker", "push", image_tag])

    # Sync data
    if not skip_data:
        _sync_data(gcs_prefix)

    # Create or update VM
    click.echo(f"\n==> Deploying VM {VM_NAME} in {zone}...")
    exists = (
        subprocess.run(
            ["gcloud", "compute", "instances", "describe", VM_NAME, f"--zone={zone}", f"--project={project}"],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )

    if exists:
        _run(
            [
                "gcloud",
                "compute",
                "instances",
                "update-container",
                VM_NAME,
                f"--zone={zone}",
                f"--project={project}",
                f"--container-image={image_tag}",
                f"--container-env=GCS_DATA_PREFIX={gcs_prefix}",
            ]
        )
    else:
        _run(
            [
                "gcloud",
                "compute",
                "instances",
                "create-with-container",
                VM_NAME,
                f"--zone={zone}",
                f"--project={project}",
                f"--machine-type={machine_type}",
                f"--container-image={image_tag}",
                f"--container-env=GCS_DATA_PREFIX={gcs_prefix}",
                "--scopes=storage-full,compute-ro",
                "--tags=http-server",
                "--boot-disk-size=100GB",
            ]
        )
        # Create firewall rule; ignore failure if it already exists.
        subprocess.run(
            [
                "gcloud",
                "compute",
                "firewall-rules",
                "create",
                f"allow-{VM_NAME}",
                "--allow=tcp:8000",
                "--target-tags=http-server",
                f"--project={project}",
                "--quiet",
            ],
            check=False,
            capture_output=True,
        )

    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "instances",
            "describe",
            VM_NAME,
            f"--zone={zone}",
            f"--project={project}",
            "--format=get(networkInterfaces[0].accessConfigs[0].natIP)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    ip = result.stdout.strip() or "pending"
    click.echo(f"\n==> Dashboard: http://{ip}:8000")


@cli.command("sync-data")
@click.option("--region", default="us-central2", show_default=True)
def sync_data_cmd(region: str):
    """Sync local parquet data and DB to GCS."""
    temp_bucket = f"marin-tmp-{region}"
    gcs_prefix = f"gs://{temp_bucket}/ttl=30d/delete-o-tron"
    _sync_data(gcs_prefix)


def _sync_data(gcs_prefix: str):
    """Sync parquet dirs and DB to GCS."""
    click.echo("\n==> Syncing data to GCS...")
    for name in ["objects_parquet", "dir_summary_parquet"]:
        local = PURGE_DIR / name
        if local.is_dir():
            _run(["gsutil", "-m", "rsync", "-r", str(local) + "/", f"{gcs_prefix}/{name}/"])
    for rule_file in ["protect_rules.json", "delete_rules.json"]:
        rule_path = PURGE_DIR / rule_file
        if rule_path.exists():
            _run(["gsutil", "cp", str(rule_path), f"{gcs_prefix}/{rule_file}"])


if __name__ == "__main__":
    cli()
