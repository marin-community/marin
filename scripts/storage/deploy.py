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
import time
from pathlib import Path

import click

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PURGE_DIR = REPO_ROOT / "scripts" / "storage" / "purge"
DEFAULT_ZONE = "us-central2-b"
DEFAULT_MACHINE_TYPE = "n2-highmem-4"
VM_NAME = "delete-o-tron"
IMAGE_NAME = "delete-o-tron"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, printing it first."""
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def _get_dashboard_password() -> str:
    password = os.environ.get("DASHBOARD_PASSWORD")
    if not password:
        raise click.ClickException("DASHBOARD_PASSWORD env var must be set for deployment")
    return password


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
@click.option("--skip-build", is_flag=True, help="Skip Docker build and restart; only sync data.")
@click.option("--skip-data", is_flag=True, help="Skip syncing data to GCS.")
@click.option("--force-rules", is_flag=True, help="Overwrite rule JSON files on GCS from local copies.")
def deploy(zone: str, machine_type: str, skip_build: bool, skip_data: bool, force_rules: bool):
    """Build and deploy the dashboard to a GCP VM.

    Syncs source to the VM, builds the Docker image there, and restarts the container.
    Creates the VM first if it does not exist.
    """
    project = _get_project()
    region = zone.rsplit("-", 1)[0]
    temp_bucket = f"marin-tmp-{region}"
    gcs_prefix = f"gs://{temp_bucket}/ttl=30d/delete-o-tron"

    # Sync data
    if not skip_data:
        _sync_data(gcs_prefix, force_rules=force_rules)

    # Create VM if it doesn't exist
    click.echo(f"\n==> Checking VM {VM_NAME} in {zone}...")
    exists = (
        subprocess.run(
            ["gcloud", "compute", "instances", "describe", VM_NAME, f"--zone={zone}", f"--project={project}"],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )

    if not exists:
        click.echo(f"\n==> Creating VM {VM_NAME}...")
        _run(
            [
                "gcloud",
                "compute",
                "instances",
                "create",
                VM_NAME,
                f"--zone={zone}",
                f"--project={project}",
                f"--machine-type={machine_type}",
                "--image-family=cos-stable",
                "--image-project=cos-cloud",
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

    if not skip_build:
        _build_on_vm(zone, project, gcs_prefix)

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
@click.option("--force-rules", is_flag=True, help="Overwrite rule JSON files on GCS from local copies.")
def sync_data_cmd(region: str, force_rules: bool):
    """Sync local parquet data to GCS. Rule JSON files are skipped unless --force-rules."""
    temp_bucket = f"marin-tmp-{region}"
    gcs_prefix = f"gs://{temp_bucket}/ttl=30d/delete-o-tron"
    _sync_data(gcs_prefix, force_rules=force_rules)


def _build_on_vm(zone: str, project: str, gcs_prefix: str):
    """Rsync source to the VM, rebuild the image there, and restart the container."""
    remote_dir = "/tmp/delete-o-tron-build"
    ssh_base = ["gcloud", "compute", "ssh", VM_NAME, f"--zone={zone}", f"--project={project}"]

    click.echo("\n==> Waiting for SSH...")
    for attempt in range(20):
        result = subprocess.run([*ssh_base, "--command", "true"], capture_output=True)
        if result.returncode == 0:
            break
        click.echo(f"    SSH not ready, retrying in 10s (attempt {attempt + 1}/20)...")
        time.sleep(10)
    else:
        raise click.ClickException("SSH did not become available after 200s")

    click.echo("\n==> Syncing source to VM...")
    # Pipe a tar archive over SSH — avoids needing rsync or scp with excludes.
    tar = subprocess.Popen(
        [
            "tar",
            "-czf",
            "-",
            "--no-xattrs",
            "--exclude=scripts/storage/purge/objects_parquet",
            "--exclude=scripts/storage/purge/dir_summary_parquet",
            "--exclude=scripts/storage/__pycache__",
            "--exclude=scripts/storage/*/__pycache__",
            "scripts/storage",
        ],
        stdout=subprocess.PIPE,
        cwd=str(REPO_ROOT),
    )
    # Extract and promote the standalone pyproject.toml to the build root.
    remote_setup = (
        f"rm -rf {remote_dir} && mkdir -p {remote_dir} && tar -xzf - -C {remote_dir} && "
        f"cp {remote_dir}/scripts/storage/pyproject.toml {remote_dir}/pyproject.toml"
    )
    extract = subprocess.Popen(
        [*ssh_base, "--", remote_setup],
        stdin=tar.stdout,
    )
    tar.stdout.close()
    extract.wait()
    tar.wait()
    if tar.returncode != 0 or extract.returncode != 0:
        raise click.ClickException("Failed to sync source to VM")

    click.echo("\n==> Building image and restarting container on VM...")
    build_and_restart = " && ".join(
        [
            f"cd {remote_dir}",
            f"docker build -t {IMAGE_NAME}:latest -f scripts/storage/Dockerfile .",
            f"docker rm -f {VM_NAME} 2>/dev/null || true",
            f"docker run -d --name {VM_NAME} -p 8000:8000"
            f" -e GCS_DATA_PREFIX={gcs_prefix}"
            f" -e DASHBOARD_PASSWORD={_get_dashboard_password()}"
            f" -v /mnt/stateful_partition/delete-o-tron-data:/app/scripts/storage/purge"
            f" {IMAGE_NAME}:latest",
        ]
    )
    _run([*ssh_base, "--command", build_and_restart])


@cli.command()
@click.option("--zone", default=DEFAULT_ZONE, show_default=True)
def ssh(zone: str):
    """Open an interactive SSH session to the VM."""
    project = _get_project()
    os.execvp(
        "gcloud",
        ["gcloud", "compute", "ssh", VM_NAME, f"--zone={zone}", f"--project={project}"],
    )


@cli.command()
@click.option("--zone", default=DEFAULT_ZONE, show_default=True)
@click.option("--follow", "-f", is_flag=True, help="Follow log output.")
@click.option("--tail", default=100, show_default=True, help="Number of lines to show from the end.")
def logs(zone: str, follow: bool, tail: int):
    """Fetch Docker logs from the running container on the VM."""
    project = _get_project()
    ssh_base = ["gcloud", "compute", "ssh", VM_NAME, f"--zone={zone}", f"--project={project}"]
    docker_cmd = f"docker logs --tail={tail}"
    if follow:
        docker_cmd += " -f"
    docker_cmd += f" {VM_NAME}"
    _run([*ssh_base, "--command", docker_cmd])


@cli.command()
@click.option("--zone", default=DEFAULT_ZONE, show_default=True)
def status(zone: str):
    """Show VM and container status."""
    project = _get_project()
    click.echo(f"==> VM {VM_NAME} in {zone}:")
    subprocess.run(
        [
            "gcloud",
            "compute",
            "instances",
            "describe",
            VM_NAME,
            f"--zone={zone}",
            f"--project={project}",
            "--format=table(name,status,networkInterfaces[0].accessConfigs[0].natIP:label=EXTERNAL_IP)",
        ],
        check=False,
    )
    ssh_base = ["gcloud", "compute", "ssh", VM_NAME, f"--zone={zone}", f"--project={project}"]
    click.echo(f"\n==> Container {VM_NAME}:")
    subprocess.run(
        [
            *ssh_base,
            "--command",
            f"docker ps --filter name={VM_NAME} --format 'table {{{{.Names}}}}\\t{{{{.Status}}}}\\t{{{{.Ports}}}}'",
        ],
        check=False,
    )


def _sync_data(gcs_prefix: str, force_rules: bool = False):
    """Sync parquet dirs to GCS. Rule JSON files are server-canonical and skipped unless force_rules=True."""
    click.echo("\n==> Syncing data to GCS...")
    for name in ["objects_parquet", "dir_summary_parquet"]:
        local = PURGE_DIR / name
        if local.is_dir():
            _run(["gcloud", "storage", "rsync", "--recursive", str(local) + "/", f"{gcs_prefix}/{name}/"])
    if force_rules:
        click.echo("\n==> Force-syncing rule files to GCS...")
        for rule_file in ["protect_rules.json", "delete_rules.json"]:
            rule_path = PURGE_DIR / rule_file
            if rule_path.exists():
                _run(["gcloud", "storage", "cp", str(rule_path), f"{gcs_prefix}/{rule_file}"])


if __name__ == "__main__":
    cli()
