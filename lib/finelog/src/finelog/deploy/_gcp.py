# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCE deployment backend for finelog.

Takes a `FinelogConfig` and drives all subprocess-to-gcloud plumbing:
image-digest pinning, instance create, SSH-based bootstrap re-run, /health
poll, status, and logs.
"""

import json
import subprocess
import sys
import tempfile
import time

import click

from finelog.deploy.bootstrap import CONTAINER_NAME, render_bootstrap
from finelog.deploy.config import FinelogConfig, auth_policy_json
from finelog.deploy.image import resolve_image_digest

LABEL_KEY = "finelog-name"
LABEL_MARKER = "finelog"


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


def _ssh_args(cfg: FinelogConfig, command: str) -> list[str]:
    """Build a `gcloud compute ssh` argv that respects the config's service account.

    When the deployment specifies a service account, the gcloud client
    impersonates it for SSH-key push (mirroring iris's pattern in
    `iris.cluster.platforms.remote_exec`). Without impersonation, an
    operator without OS Login privileges on the VM cannot connect even
    if they have the role on its service account.
    """
    assert cfg.deployment.gcp is not None
    gcp = cfg.deployment.gcp
    args = [
        "gcloud",
        "compute",
        "ssh",
        cfg.name,
        f"--project={gcp.project}",
        f"--zone={gcp.zone}",
        f"--command={command}",
    ]
    if gcp.service_account:
        args.append(f"--impersonate-service-account={gcp.service_account}")
    return args


def _wait_health_via_ssh(cfg: FinelogConfig, port: int, max_attempts: int = 90) -> bool:
    """Poll ``/health`` from inside the VM over SSH.

    Used by both ``gcp_up`` (where the first attempts may fail while
    OS Login propagates the SSH key on the fresh VM) and ``gcp_restart``.
    A direct probe is unambiguous: serial-console marker scraping was
    fragile because ``gcp_restart`` re-runs the bootstrap over SSH and
    that path never reaches the console.
    """
    probe = f"curl -sf -m 5 http://localhost:{port}/health"
    for _ in range(max_attempts):
        result = subprocess.run(
            _ssh_args(cfg, probe),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True
        time.sleep(2)
    return False


def render_bootstrap_for(cfg: FinelogConfig, image: str) -> str:
    """Render the VM bootstrap script running ``image`` under ``cfg``'s port,
    remote archive, and auth policy.

    ``image`` is separate from ``cfg.image`` so callers can supply an
    already-pinned digest.

    Raises if the config forwards: the rendered script becomes the instance's
    startup-script metadata, readable by anyone with `compute.instances.get` and by
    every process on the VM through the metadata server, so this backend has nowhere
    to put a private signing key. A forwarding finelog belongs on the k8s backend,
    which projects the key through a Secret.
    """
    if cfg.forwarding is not None:
        raise click.ClickException(
            f"finelog config {cfg.name!r}: forwarding is not supported on the gcp backend — "
            "its signing key would land in world-readable instance metadata"
        )
    return render_bootstrap(
        image=image,
        port=cfg.port,
        remote_log_dir=cfg.remote_log_dir,
        # The rendered script is world-readable to anyone with instance-metadata
        # access, and every layer is inline-safe: a jwt layer carries only Ed25519
        # public keys, a cidr layer only network prefixes.
        auth_policy=auth_policy_json(cfg.auth) if cfg.auth else "",
    )


def _pinned_bootstrap(cfg: FinelogConfig) -> str:
    """Pin ``cfg.image`` to a content digest and render the bootstrap for it."""
    pinned = resolve_image_digest(cfg.image)
    if pinned != cfg.image:
        click.echo(f"Pinned image: {cfg.image} -> {pinned}")
    else:
        click.echo(f"Using image: {pinned}")
    return render_bootstrap_for(cfg, pinned)


def gcp_up(cfg: FinelogConfig) -> None:
    """Create a finelog GCE VM. Idempotent: errors out cleanly if the VM exists."""
    assert cfg.deployment.gcp is not None
    gcp = cfg.deployment.gcp

    existing = _instance_describe(cfg.name, gcp.project, gcp.zone)
    if existing is not None:
        click.echo(f"Instance {cfg.name} already exists in {gcp.zone}; skipping create.")
        click.echo("Run `finelog deploy restart <name>` to refresh the container in place.")
        return

    bootstrap = _pinned_bootstrap(cfg)

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
    if not _wait_health_via_ssh(cfg, cfg.port):
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


def _set_startup_script(cfg: FinelogConfig, bootstrap: str) -> None:
    """Persist ``bootstrap`` as the instance's startup-script metadata, which a VM
    reboot (host maintenance, manual reset) re-runs."""
    assert cfg.deployment.gcp is not None
    gcp = cfg.deployment.gcp
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(bootstrap)
        startup_path = f.name
    click.echo("Updating instance startup-script metadata...")
    _gcloud(
        "compute",
        "instances",
        "add-metadata",
        cfg.name,
        f"--project={gcp.project}",
        f"--zone={gcp.zone}",
        f"--metadata-from-file=startup-script={startup_path}",
    )


def apply_bootstrap(cfg: FinelogConfig, bootstrap: str) -> bool:
    """Run ``bootstrap`` on the VM over SSH, then persist it as startup-script
    metadata. Returns whether the container came up.

    The script polls ``/health`` itself and exits non-zero on a crash-loop or
    timeout, so a ``False`` return means the image never became healthy. Metadata
    is written only on success, so a reboot re-runs the last bootstrap known to
    boot rather than one that just failed.
    """
    result = subprocess.run(_ssh_args(cfg, "bash -s"), input=bootstrap, text=True)
    if result.returncode != 0:
        return False
    _set_startup_script(cfg, bootstrap)
    return True


def gcp_restart(cfg: FinelogConfig) -> None:
    """Restart finelog in-place by re-running the bootstrap over SSH."""
    bootstrap = _pinned_bootstrap(cfg)
    click.echo(f"Re-running bootstrap on {cfg.name} via SSH...")
    if not apply_bootstrap(cfg, bootstrap):
        raise click.ClickException("Bootstrap re-run failed; see SSH output above")
    click.echo("Bootstrap re-applied. Verifying health...")
    if not _wait_health_via_ssh(cfg, cfg.port):
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
        _ssh_args(cfg, probe_cmd),
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
    follow_flag = "-f" if follow else ""
    cmd = f"sudo docker logs {CONTAINER_NAME} --tail {tail} {follow_flag}".strip()
    args = _ssh_args(cfg, cmd)
    if follow:
        proc = subprocess.Popen(args)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
    else:
        subprocess.run(args)
