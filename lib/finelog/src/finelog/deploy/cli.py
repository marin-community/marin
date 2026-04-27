# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""finelog deploy CLI — manage a single finelog GCE VM via gcloud subprocess.

Mirrors iris's pattern: shell out to `gcloud compute instances ...` rather
than depending on the google-cloud-compute SDK. The bootstrap script is the
same one used at VM-create time and re-applied over SSH on restart, so the
bootstrap path is the only path that ever starts the container.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

import click

from finelog.deploy.bootstrap import CONTAINER_NAME, render_bootstrap

DEFAULT_ZONE = "us-central1-a"
DEFAULT_MACHINE_TYPE = "n2-standard-4"
DEFAULT_BOOT_DISK_SIZE = 200
DEFAULT_IMAGE = "ghcr.io/marin-community/finelog:latest"
DEFAULT_PORT = 10001

LABEL_KEY = "finelog-name"
LABEL_MARKER = "finelog"


def _project_default() -> str | None:
    return os.environ.get("GOOGLE_CLOUD_PROJECT")


def _require_project(project: str | None) -> str:
    if not project:
        raise click.ClickException(
            "--project is required (or set GOOGLE_CLOUD_PROJECT)",
        )
    return project


def _resolve_image_digest(image: str) -> str:
    """Pin a tag to its content digest via `docker manifest inspect`.

    Returns `ghcr.io/...@sha256:...` on success, or the original tag with a
    warning on any failure (no docker CLI, no network, private registry, etc.).
    """
    if "@sha256:" in image:
        return image
    if ":" not in image.rsplit("/", 1)[-1]:
        # No tag at all — leave it alone; gcloud bootstrap will resolve.
        return image
    repo, _, _ = image.rpartition(":")
    try:
        result = subprocess.run(
            ["docker", "manifest", "inspect", "-v", image],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        click.echo(f"warning: could not resolve digest for {image} ({e}); using tag", err=True)
        return image
    if result.returncode != 0:
        stderr_msg = result.stderr.strip()[:200]
        click.echo(
            f"warning: `docker manifest inspect` failed for {image}: {stderr_msg}; using tag",
            err=True,
        )
        return image
    digest = _extract_digest(result.stdout)
    if not digest:
        click.echo(f"warning: could not parse digest from manifest of {image}; using tag", err=True)
        return image
    return f"{repo}@{digest}"


def _extract_digest(manifest_json: str) -> str | None:
    """Pull a top-level `Descriptor.digest` out of `docker manifest inspect -v` output."""
    try:
        parsed = json.loads(manifest_json)
    except json.JSONDecodeError:
        return None
    # Multi-arch: list of entries, each with a Descriptor. Pick linux/amd64.
    if isinstance(parsed, list):
        for entry in parsed:
            desc = entry.get("Descriptor", {})
            platform = desc.get("platform", {})
            if platform.get("os") == "linux" and platform.get("architecture") == "amd64":
                digest = desc.get("digest")
                if digest:
                    return digest
        # Fallback: first entry.
        if parsed:
            return parsed[0].get("Descriptor", {}).get("digest")
        return None
    # Single-arch: object with Descriptor.
    return parsed.get("Descriptor", {}).get("digest")


def _gcloud(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ["gcloud", *args]
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def _instance_describe(name: str, project: str, zone: str) -> dict | None:
    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "instances",
            "describe",
            name,
            f"--project={project}",
            f"--zone={zone}",
            "--format=json",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


@click.group()
def cli() -> None:
    """Manage a finelog GCE VM."""


@cli.command("create")
@click.option("--name", required=True, help="GCE instance name")
@click.option("--project", default=_project_default(), help="GCP project (or $GOOGLE_CLOUD_PROJECT)")
@click.option("--zone", default=DEFAULT_ZONE, show_default=True)
@click.option("--machine-type", default=DEFAULT_MACHINE_TYPE, show_default=True)
@click.option("--boot-disk-size", type=int, default=DEFAULT_BOOT_DISK_SIZE, show_default=True, help="GB")
@click.option("--image", default=DEFAULT_IMAGE, show_default=True, help="Finelog Docker image (tag or digest)")
@click.option("--port", type=int, default=DEFAULT_PORT, show_default=True)
@click.option("--remote-log-dir", required=True, help="GCS path for FINELOG_REMOTE_DIR (e.g. gs://my-bucket/finelog)")
@click.option("--service-account", default=None, help="VM service account email")
@click.option("--network-tag", "network_tags", multiple=True, help="Network tag (repeatable)")
@click.option("--wait/--no-wait", default=True, help="Wait for /health after create")
def create_cmd(
    name: str,
    project: str | None,
    zone: str,
    machine_type: str,
    boot_disk_size: int,
    image: str,
    port: int,
    remote_log_dir: str,
    service_account: str | None,
    network_tags: tuple[str, ...],
    wait: bool,
) -> None:
    """Create a finelog VM."""
    project = _require_project(project)

    pinned = _resolve_image_digest(image)
    if pinned != image:
        click.echo(f"Pinned image: {image} -> {pinned}")
    else:
        click.echo(f"Using image: {pinned}")

    bootstrap = render_bootstrap(image=pinned, port=port, remote_log_dir=remote_log_dir)

    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(bootstrap)
        startup_path = f.name

    args = [
        "compute",
        "instances",
        "create",
        name,
        f"--project={project}",
        f"--zone={zone}",
        f"--machine-type={machine_type}",
        f"--boot-disk-size={boot_disk_size}GB",
        "--image-family=debian-12",
        "--image-project=debian-cloud",
        f"--metadata-from-file=startup-script={startup_path}",
        f"--labels={LABEL_KEY}={name},{LABEL_MARKER}=true",
    ]
    if service_account:
        args += [f"--service-account={service_account}", "--scopes=cloud-platform"]
    if network_tags:
        args.append(f"--tags={','.join(network_tags)}")

    click.echo(f"Creating GCE instance {name} in {zone}...")
    _gcloud(*args)
    click.echo("Instance created. Startup script will install Docker and launch finelog.")

    if wait:
        click.echo("Waiting for finelog /health (up to ~3 minutes)...")
        ok = _wait_health(name, project, zone, port)
        if not ok:
            raise click.ClickException("finelog did not become healthy; inspect via `finelog logs`")
        click.echo("finelog is healthy.")


@cli.command("restart")
@click.option("--name", required=True)
@click.option("--project", default=_project_default())
@click.option("--zone", default=DEFAULT_ZONE, show_default=True)
@click.option("--image", default=DEFAULT_IMAGE, show_default=True)
@click.option("--port", type=int, default=DEFAULT_PORT, show_default=True)
@click.option("--remote-log-dir", required=True)
def restart_cmd(
    name: str,
    project: str | None,
    zone: str,
    image: str,
    port: int,
    remote_log_dir: str,
) -> None:
    """Restart finelog in-place by re-running the bootstrap over SSH.

    Does NOT recreate the VM — only the container.
    """
    project = _require_project(project)

    pinned = _resolve_image_digest(image)
    if pinned != image:
        click.echo(f"Pinned image: {image} -> {pinned}")
    else:
        click.echo(f"Using image: {pinned}")

    bootstrap = render_bootstrap(image=pinned, port=port, remote_log_dir=remote_log_dir)

    click.echo(f"Re-running bootstrap on {name} via SSH...")
    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "ssh",
            name,
            f"--project={project}",
            f"--zone={zone}",
            "--command=bash -s",
        ],
        input=bootstrap,
        text=True,
    )
    if result.returncode != 0:
        raise click.ClickException("Bootstrap re-run failed; see SSH output above")
    click.echo("Bootstrap re-applied. Verifying health...")
    if not _wait_health(name, project, zone, port):
        raise click.ClickException("finelog did not become healthy after restart")
    click.echo("finelog is healthy.")


@cli.command("delete")
@click.option("--name", required=True)
@click.option("--project", default=_project_default())
@click.option("--zone", default=DEFAULT_ZONE, show_default=True)
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation")
def delete_cmd(name: str, project: str | None, zone: str, yes: bool) -> None:
    """Delete the finelog VM."""
    project = _require_project(project)
    if not yes:
        click.confirm(f"Delete instance {name} in {zone} (project {project})?", abort=True)
    _gcloud(
        "compute",
        "instances",
        "delete",
        name,
        f"--project={project}",
        f"--zone={zone}",
        "--quiet",
    )
    click.echo(f"Deleted {name}.")


@cli.command("status")
@click.option("--name", required=True)
@click.option("--project", default=_project_default())
@click.option("--zone", default=DEFAULT_ZONE, show_default=True)
def status_cmd(name: str, project: str | None, zone: str) -> None:
    """Show VM + container status."""
    project = _require_project(project)
    info = _instance_describe(name, project, zone)
    if info is None:
        click.echo(f"Instance {name} not found in {zone}")
        sys.exit(1)
    click.echo(f"Instance: {info.get('name')}")
    click.echo(f"  status:  {info.get('status')}")
    interfaces = info.get("networkInterfaces", [])
    if interfaces:
        click.echo(f"  internalIP: {interfaces[0].get('networkIP')}")
        access_configs = interfaces[0].get("accessConfigs") or []
        if access_configs:
            click.echo(f"  externalIP: {access_configs[0].get('natIP')}")
    labels = info.get("labels") or {}
    if labels:
        click.echo(f"  labels: {labels}")

    # Best-effort container probe over SSH.
    fmt = "{{.State.Status}}"
    probe_cmd = f"sudo docker inspect --format='{fmt}' {CONTAINER_NAME} 2>/dev/null || echo not_found"
    probe = subprocess.run(
        [
            "gcloud",
            "compute",
            "ssh",
            name,
            f"--project={project}",
            f"--zone={zone}",
            f"--command={probe_cmd}",
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        click.echo(f"  container: {probe.stdout.strip()}")
    else:
        click.echo("  container: <ssh failed>")


@cli.command("logs")
@click.option("--name", required=True)
@click.option("--project", default=_project_default())
@click.option("--zone", default=DEFAULT_ZONE, show_default=True)
@click.option("--tail", type=int, default=200, show_default=True)
@click.option("-f", "--follow", is_flag=True, help="Stream logs")
def logs_cmd(name: str, project: str | None, zone: str, tail: int, follow: bool) -> None:
    """Tail finelog container logs over SSH."""
    project = _require_project(project)
    follow_flag = "-f" if follow else ""
    cmd = f"sudo docker logs {CONTAINER_NAME} --tail {tail} {follow_flag}".strip()
    args = [
        "gcloud",
        "compute",
        "ssh",
        name,
        f"--project={project}",
        f"--zone={zone}",
        f"--command={cmd}",
    ]
    if follow:
        # Stream output in real time.
        proc = subprocess.Popen(args)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
    else:
        subprocess.run(args)


def _wait_health(name: str, project: str, zone: str, port: int, max_attempts: int = 60) -> bool:
    """Poll the VM's /health endpoint over SSH+curl. Returns True on success."""
    for _ in range(max_attempts):
        result = subprocess.run(
            [
                "gcloud",
                "compute",
                "ssh",
                name,
                f"--project={project}",
                f"--zone={zone}",
                f"--command=curl -sf http://localhost:{port}/health > /dev/null",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True
        # SSH connection itself might still be coming up; keep retrying.
        time.sleep(3)
    return False


if __name__ == "__main__":
    cli()
