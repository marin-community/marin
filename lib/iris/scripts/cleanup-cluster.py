#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Clean iris state from a GCP zone.

This script finds and deletes all iris-managed resources in a zone:
- Controller VMs (name matches 'iris-controller*')
- TPU slices (name matches 'iris-*')

By default runs in dry-run mode to show what would be deleted.
Use --no-dry-run to actually perform deletions.

Examples:
    # Show what would be deleted (safe, no changes)
    uv run python scripts/cleanup-cluster.py --zone europe-west4-b

    # Actually delete resources
    uv run python scripts/cleanup-cluster.py --zone europe-west4-b --no-dry-run
"""

import json
import subprocess
import sys

import click


def run_gcloud(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a gcloud command and return the result."""
    cmd = ["gcloud", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def list_controller_vms(zone: str, project: str) -> list[str]:
    """Find iris controller VMs in the zone."""
    result = run_gcloud(
        [
            "compute",
            "instances",
            "list",
            f"--project={project}",
            f"--zones={zone}",
            "--filter=name~^iris-controller",
            "--format=value(name)",
        ],
        check=False,
    )
    if result.returncode != 0:
        click.echo(f"Warning: Failed to list VMs: {result.stderr.strip()}", err=True)
        return []

    names = result.stdout.strip().split("\n")
    return [n for n in names if n]


def list_tpu_slices(zone: str, project: str) -> list[str]:
    """Find iris-managed TPU slices in the zone."""
    result = run_gcloud(
        [
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            f"--project={project}",
            f"--zone={zone}",
            "--filter=name~^iris-",
            "--format=json",
        ],
        check=False,
    )
    if result.returncode != 0:
        click.echo(f"Warning: Failed to list TPUs: {result.stderr.strip()}", err=True)
        return []

    if not result.stdout.strip():
        return []

    try:
        tpus = json.loads(result.stdout)
    except json.JSONDecodeError:
        click.echo(f"Warning: Failed to parse TPU list: {result.stdout[:200]}", err=True)
        return []

    names = []
    for tpu in tpus:
        name = tpu.get("name", "")
        # GCP returns full resource path like 'projects/proj/locations/zone/nodes/my-tpu'
        if "/" in name:
            name = name.split("/")[-1]
        if name:
            names.append(name)
    return names


def delete_vm(name: str, zone: str, project: str) -> bool:
    """Delete a GCE VM. Returns True on success."""
    result = run_gcloud(
        [
            "compute",
            "instances",
            "delete",
            name,
            f"--project={project}",
            f"--zone={zone}",
            "--quiet",
        ],
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.strip()
        if "not found" in error.lower():
            click.echo(f"  VM {name} already deleted")
            return True
        click.echo(f"  Failed to delete VM {name}: {error}", err=True)
        return False
    return True


def delete_tpu(name: str, zone: str, project: str) -> bool:
    """Delete a TPU slice. Returns True on success."""
    result = run_gcloud(
        [
            "compute",
            "tpus",
            "tpu-vm",
            "delete",
            name,
            f"--project={project}",
            f"--zone={zone}",
            "--quiet",
        ],
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.strip()
        if "not found" in error.lower():
            click.echo(f"  TPU {name} already deleted")
            return True
        click.echo(f"  Failed to delete TPU {name}: {error}", err=True)
        return False
    return True


@click.command()
@click.option("--zone", default="europe-west4-b", help="GCP zone to clean")
@click.option("--project", default="hai-gcp-models", help="GCP project ID")
@click.option(
    "--dry-run/--no-dry-run",
    default=True,
    help="Dry-run mode (default: True). Use --no-dry-run to actually delete.",
)
def cleanup(zone: str, project: str, dry_run: bool) -> None:
    """Clean all iris VMs and TPUs from a GCP zone.

    By default runs in dry-run mode to show what would be deleted.
    Use --no-dry-run to actually perform deletions.
    """
    click.echo(f"Scanning zone {zone} in project {project}...")
    if dry_run:
        click.echo("(DRY-RUN mode - no changes will be made)")
    click.echo()

    # Find controller VMs
    controller_vms = list_controller_vms(zone, project)
    if controller_vms:
        click.echo(f"Found {len(controller_vms)} controller VM(s):")
        for name in controller_vms:
            click.echo(f"  - {name}")
    else:
        click.echo("No controller VMs found.")
    click.echo()

    # Find TPU slices
    tpu_slices = list_tpu_slices(zone, project)
    if tpu_slices:
        click.echo(f"Found {len(tpu_slices)} TPU slice(s):")
        for name in tpu_slices:
            click.echo(f"  - {name}")
    else:
        click.echo("No TPU slices found.")
    click.echo()

    total_resources = len(controller_vms) + len(tpu_slices)
    if total_resources == 0:
        click.echo("Nothing to clean up.")
        return

    if dry_run:
        click.echo(f"Would delete {total_resources} resource(s). Use --no-dry-run to delete.")
        return

    # Perform deletions
    click.echo("Deleting resources...")
    failed = 0

    for name in controller_vms:
        click.echo(f"Deleting VM: {name}")
        if not delete_vm(name, zone, project):
            failed += 1

    for name in tpu_slices:
        click.echo(f"Deleting TPU: {name}")
        if not delete_tpu(name, zone, project):
            failed += 1

    click.echo()
    deleted = total_resources - failed
    click.echo(f"Deleted {deleted}/{total_resources} resource(s).")

    if failed > 0:
        click.echo(f"Failed to delete {failed} resource(s).", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cleanup()
