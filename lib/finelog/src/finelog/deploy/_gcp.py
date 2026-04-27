# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCE deployment backend for finelog.

Lifted verbatim from the original `cli.py`, reshaped to take a
`FinelogConfig` instead of click args. All subprocess-to-gcloud plumbing
(image-digest pinning, instance create, SSH-based bootstrap re-run,
/health poll, status, logs) lives here.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time

import click

from finelog.deploy.bootstrap import CONTAINER_NAME, render_bootstrap
from finelog.deploy.config import FinelogConfig

LABEL_KEY = "finelog-name"
LABEL_MARKER = "finelog"


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
    if isinstance(parsed, list):
        for entry in parsed:
            desc = entry.get("Descriptor", {})
            platform = desc.get("platform", {})
            if platform.get("os") == "linux" and platform.get("architecture") == "amd64":
                digest = desc.get("digest")
                if digest:
                    return digest
        if parsed:
            return parsed[0].get("Descriptor", {}).get("digest")
        return None
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
        time.sleep(3)
    return False


def gcp_up(cfg: FinelogConfig) -> None:
    """Create a finelog GCE VM. Idempotent: errors out cleanly if the VM exists."""
    assert cfg.deployment.gcp is not None
    gcp = cfg.deployment.gcp

    existing = _instance_describe(cfg.name, gcp.project, gcp.zone)
    if existing is not None:
        click.echo(f"Instance {cfg.name} already exists in {gcp.zone}; skipping create.")
        click.echo("Run `finelog deploy restart <name>` to refresh the container in place.")
        return

    pinned = _resolve_image_digest(cfg.image)
    if pinned != cfg.image:
        click.echo(f"Pinned image: {cfg.image} -> {pinned}")
    else:
        click.echo(f"Using image: {pinned}")

    bootstrap = render_bootstrap(image=pinned, port=cfg.port, remote_log_dir=cfg.remote_log_dir)

    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(bootstrap)
        startup_path = f.name

    args = [
        "compute",
        "instances",
        "create",
        cfg.name,
        f"--project={gcp.project}",
        f"--zone={gcp.zone}",
        f"--machine-type={gcp.machine_type}",
        f"--boot-disk-size={gcp.boot_disk_size_gb}GB",
        "--boot-disk-type=pd-ssd",
        "--image-family=debian-12",
        "--image-project=debian-cloud",
        f"--metadata-from-file=startup-script={startup_path}",
        f"--labels={LABEL_KEY}={cfg.name},{LABEL_MARKER}=true",
    ]
    if gcp.service_account:
        args += [f"--service-account={gcp.service_account}", "--scopes=cloud-platform"]
    if gcp.network_tags:
        args.append(f"--tags={','.join(gcp.network_tags)}")

    click.echo(f"Creating GCE instance {cfg.name} in {gcp.zone}...")
    _gcloud(*args)
    click.echo("Instance created. Startup script will install Docker and launch finelog.")

    click.echo("Waiting for finelog /health (up to ~3 minutes)...")
    if not _wait_health(cfg.name, gcp.project, gcp.zone, cfg.port):
        raise click.ClickException("finelog did not become healthy; inspect via `finelog deploy logs`")
    click.echo("finelog is healthy.")


def gcp_down(cfg: FinelogConfig, *, yes: bool) -> None:
    """Delete the finelog VM."""
    assert cfg.deployment.gcp is not None
    gcp = cfg.deployment.gcp
    if not yes:
        click.confirm(
            f"Delete instance {cfg.name} in {gcp.zone} (project {gcp.project})?",
            abort=True,
        )
    _gcloud(
        "compute",
        "instances",
        "delete",
        cfg.name,
        f"--project={gcp.project}",
        f"--zone={gcp.zone}",
        "--quiet",
    )
    click.echo(f"Deleted {cfg.name}.")


def gcp_restart(cfg: FinelogConfig) -> None:
    """Restart finelog in-place by re-running the bootstrap over SSH."""
    assert cfg.deployment.gcp is not None
    gcp = cfg.deployment.gcp

    pinned = _resolve_image_digest(cfg.image)
    if pinned != cfg.image:
        click.echo(f"Pinned image: {cfg.image} -> {pinned}")
    else:
        click.echo(f"Using image: {pinned}")

    bootstrap = render_bootstrap(image=pinned, port=cfg.port, remote_log_dir=cfg.remote_log_dir)

    click.echo(f"Re-running bootstrap on {cfg.name} via SSH...")
    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "ssh",
            cfg.name,
            f"--project={gcp.project}",
            f"--zone={gcp.zone}",
            "--command=bash -s",
        ],
        input=bootstrap,
        text=True,
    )
    if result.returncode != 0:
        raise click.ClickException("Bootstrap re-run failed; see SSH output above")
    click.echo("Bootstrap re-applied. Verifying health...")
    if not _wait_health(cfg.name, gcp.project, gcp.zone, cfg.port):
        raise click.ClickException("finelog did not become healthy after restart")
    click.echo("finelog is healthy.")


def gcp_status(cfg: FinelogConfig) -> None:
    """Show VM + container status."""
    assert cfg.deployment.gcp is not None
    gcp = cfg.deployment.gcp
    info = _instance_describe(cfg.name, gcp.project, gcp.zone)
    if info is None:
        click.echo(f"Instance {cfg.name} not found in {gcp.zone}")
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

    fmt = "{{.State.Status}}"
    probe_cmd = f"sudo docker inspect --format='{fmt}' {CONTAINER_NAME} 2>/dev/null || echo not_found"
    probe = subprocess.run(
        [
            "gcloud",
            "compute",
            "ssh",
            cfg.name,
            f"--project={gcp.project}",
            f"--zone={gcp.zone}",
            f"--command={probe_cmd}",
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        click.echo(f"  container: {probe.stdout.strip()}")
    else:
        click.echo("  container: <ssh failed>")


def gcp_logs(cfg: FinelogConfig, *, tail: int, follow: bool) -> None:
    """Tail finelog container logs over SSH."""
    assert cfg.deployment.gcp is not None
    gcp = cfg.deployment.gcp
    follow_flag = "-f" if follow else ""
    cmd = f"sudo docker logs {CONTAINER_NAME} --tail {tail} {follow_flag}".strip()
    args = [
        "gcloud",
        "compute",
        "ssh",
        cfg.name,
        f"--project={gcp.project}",
        f"--zone={gcp.zone}",
        f"--command={cmd}",
    ]
    if follow:
        proc = subprocess.Popen(args)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
    else:
        subprocess.run(args)
