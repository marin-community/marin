# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes deployment backend for finelog.

Templates `lib/finelog/deploy/k8s/*.yaml` against a `FinelogConfig` and
shells out to `kubectl`. No kubernetes-client Python dep — the manifest
list is small enough that subprocess is the right tool.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import click

from finelog.deploy.bootstrap import render_template
from finelog.deploy.config import FinelogConfig

# Manifests live at `lib/finelog/deploy/k8s/*.yaml` in the repo. We resolve
# this once at import time; the directory is part of the source tree, not
# the wheel, but k8s deployments are operator-driven and run from a checkout.
_K8S_MANIFEST_DIR = Path(__file__).resolve().parents[3] / "deploy" / "k8s"

_MANIFESTS = ("01-pvc.yaml.tmpl", "02-deployment.yaml.tmpl", "03-service.yaml.tmpl")


def _render_manifest(template_path: Path, cfg: FinelogConfig) -> str:
    """Render a single k8s manifest template against `cfg`."""
    assert cfg.deployment.k8s is not None
    k8s = cfg.deployment.k8s
    storage_class_block = (
        f"storageClassName: {k8s.storage_class}" if k8s.storage_class else "# storageClassName: <cluster default>"
    )
    template = template_path.read_text()
    return render_template(
        template,
        name=cfg.name,
        namespace=k8s.namespace,
        image=cfg.image,
        port=cfg.port,
        remote_log_dir=cfg.remote_log_dir,
        storage_class_block=storage_class_block,
        storage_gb=k8s.storage_gb,
    )


def _kubectl(*args: str, stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["kubectl", *args], input=stdin, text=True, check=check)


def _kubectl_apply(manifest: str) -> None:
    _kubectl("apply", "-f", "-", stdin=manifest)


def k8s_up(cfg: FinelogConfig) -> None:
    """Render manifests and apply them; wait for the deployment to roll out."""
    assert cfg.deployment.k8s is not None
    k8s = cfg.deployment.k8s
    for manifest_name in _MANIFESTS:
        rendered = _render_manifest(_K8S_MANIFEST_DIR / manifest_name, cfg)
        click.echo(f"Applying {manifest_name}...")
        _kubectl_apply(rendered)
    click.echo(f"Waiting for deployment/{cfg.name} to become Ready...")
    _kubectl("rollout", "status", f"deployment/{cfg.name}", "-n", k8s.namespace)
    click.echo("finelog is healthy.")


def k8s_down(cfg: FinelogConfig, *, yes: bool) -> None:
    """Delete deployment + service. Delete the PVC only when `yes=True`."""
    assert cfg.deployment.k8s is not None
    k8s = cfg.deployment.k8s
    _kubectl(
        "delete",
        f"deployment/{cfg.name}",
        f"service/{cfg.name}",
        "-n",
        k8s.namespace,
        "--ignore-not-found",
    )
    if yes:
        _kubectl(
            "delete",
            f"pvc/{cfg.name}-cache",
            "-n",
            k8s.namespace,
            "--ignore-not-found",
        )
        click.echo(f"Deleted {cfg.name} (deployment, service, pvc).")
    else:
        click.echo(
            f"Deleted {cfg.name} (deployment, service). "
            f"PVC {cfg.name}-cache retained — pass -y to delete it as well."
        )


def k8s_restart(cfg: FinelogConfig) -> None:
    """Roll the deployment by re-setting its image, then wait for rollout."""
    assert cfg.deployment.k8s is not None
    k8s = cfg.deployment.k8s
    _kubectl(
        "set",
        "image",
        f"deployment/{cfg.name}",
        f"finelog={cfg.image}",
        "-n",
        k8s.namespace,
    )
    _kubectl("rollout", "status", f"deployment/{cfg.name}", "-n", k8s.namespace)
    click.echo("finelog is healthy.")


def k8s_status(cfg: FinelogConfig) -> None:
    """Show deployment, service, and PVC status."""
    assert cfg.deployment.k8s is not None
    k8s = cfg.deployment.k8s
    _kubectl(
        "get",
        f"deployment/{cfg.name}",
        f"service/{cfg.name}",
        f"pvc/{cfg.name}-cache",
        "-n",
        k8s.namespace,
    )


def k8s_logs(cfg: FinelogConfig, *, tail: int, follow: bool) -> None:
    """Tail logs from the deployment's pod."""
    assert cfg.deployment.k8s is not None
    k8s = cfg.deployment.k8s
    args = [
        "logs",
        f"deployment/{cfg.name}",
        "-n",
        k8s.namespace,
        f"--tail={tail}",
    ]
    if follow:
        args.append("-f")
    _kubectl(*args)
