# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manage the environment Secret and observe Pulumi-managed Finelog servers."""

import base64
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import click
from rigging.secrets import resolve_secret_spec

from finelog.deploy.config import FinelogConfig, k8s_env_secret_name

# S3-compatible endpoints that accept only virtual-hosted-style requests
# (bucket as a host subdomain).
_VIRTUAL_HOST_ONLY_S3_DOMAINS = ("cwobject.com", "cwlota.com")


def _s3_env(cfg: FinelogConfig) -> dict[str, str]:
    """The ``AWS_*`` environment for an ``s3://`` archive, or ``{}`` when none.

    Carries the operator's R2 credentials (from ``R2_KEY_ID`` /
    ``R2_KEY_SECRET`` in the deploy shell) plus the configured endpoint and
    ``region=auto``, under the names ``AmazonS3Builder::from_env`` reads in the server.
    ``gs://`` and local archives need nothing (GCS uses workload identity).

    Raises if an ``s3://`` archive is configured without an endpoint or creds —
    deploying then would silently start a server that cannot reach its archive.
    """
    assert cfg.deployment.k8s is not None
    if not cfg.remote_log_dir.startswith("s3://"):
        return {}
    k8s = cfg.deployment.k8s
    if not k8s.object_storage_endpoint:
        raise click.ClickException(
            f"finelog config {cfg.name!r}: remote_log_dir is s3:// but "
            "deployment.k8s.object_storage_endpoint is unset"
        )
    key_id = os.environ.get("R2_KEY_ID")
    key_secret = os.environ.get("R2_KEY_SECRET")
    if not key_id or not key_secret:
        raise click.ClickException(
            "R2_KEY_ID and R2_KEY_SECRET must be set in the deploy "
            f"environment to deploy {cfg.name!r} with an s3:// archive"
        )
    endpoint = k8s.object_storage_endpoint
    env = {
        "AWS_ACCESS_KEY_ID": key_id,
        "AWS_SECRET_ACCESS_KEY": key_secret,
        # Non-AWS S3 endpoints (R2, CoreWeave) reject a real region in the v4
        # signature; "auto" skips region validation.
        "AWS_REGION": "auto",
        "AWS_DEFAULT_REGION": "auto",
    }
    # The server's Rust object_store S3 client takes the addressing style and
    # the plain-http opt-in from env. CoreWeave Object Storage endpoints
    # (cwobject.com; cwlota.com, the in-cluster LOTA cache, plain http) accept
    # only virtual-hosted-style requests, and object_store uses the endpoint
    # verbatim as the base URL in that mode — so the archive bucket must be
    # baked into the endpoint host (http://<bucket>.cwlota.com).
    parsed = urlparse(endpoint)
    hostname = parsed.hostname or ""
    if any(hostname == d or hostname.endswith("." + d) for d in _VIRTUAL_HOST_ONLY_S3_DOMAINS):
        env["AWS_VIRTUAL_HOSTED_STYLE_REQUEST"] = "true"
        bucket = cfg.remote_log_dir.removeprefix("s3://").split("/", 1)[0]
        if not hostname.startswith(f"{bucket}."):
            endpoint = f"{parsed.scheme}://{bucket}.{parsed.netloc}"
    env["AWS_ENDPOINT_URL"] = endpoint
    if endpoint.startswith("http://"):
        env["AWS_ALLOW_HTTP"] = "true"
    return env


def _forwarding_env(cfg: FinelogConfig) -> dict[str, str]:
    """The forwarding private key, resolved from its secret reference, or ``{}``.

    Raises `SecretResolutionError` if no configured source yields the key: a server
    that cannot authenticate to its hub forwards nothing, and looks exactly like a
    quiet cluster.
    """
    if cfg.forwarding is None:
        return {}
    resolved = resolve_secret_spec(cfg.forwarding.signing_key)
    click.echo(f"Resolved forwarding signing key from {resolved.source}")
    return {"FINELOG_SIGNING_KEY": resolved.value}


def _build_env_secret_manifest(cfg: FinelogConfig) -> str | None:
    """Build the pod's secret-environment Secret manifest, or ``None`` when empty."""
    assert cfg.deployment.k8s is not None
    env = _s3_env(cfg) | _forwarding_env(cfg)
    if not env:
        return None
    manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": k8s_env_secret_name(cfg), "namespace": cfg.deployment.k8s.namespace},
        "type": "Opaque",
        "data": {k: base64.b64encode(v.encode()).decode() for k, v in env.items()},
    }
    return json.dumps(manifest)


def _kube_flags(cfg: FinelogConfig) -> list[str]:
    """Global kubectl flags binding this deploy to its configured kubeconfig/context.

    Empty when the config sets neither — kubectl then falls back to its own
    resolution (KUBECONFIG env var or ~/.kube/config, current-context).
    """
    assert cfg.deployment.k8s is not None
    k8s = cfg.deployment.k8s
    flags: list[str] = []
    if k8s.kubeconfig:
        flags.extend(["--kubeconfig", str(Path(k8s.kubeconfig).expanduser())])
    if k8s.kube_context:
        flags.extend(["--context", k8s.kube_context])
    return flags


def _kubectl(
    cfg: FinelogConfig,
    *args: str,
    stdin: str | None = None,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kubectl", *_kube_flags(cfg), *args],
        input=stdin,
        text=True,
        check=check,
        capture_output=capture_output,
    )


def _kubectl_apply(cfg: FinelogConfig, manifest: str) -> None:
    _kubectl(cfg, "apply", "-f", "-", stdin=manifest)


def k8s_sync_secret(cfg: FinelogConfig) -> None:
    """Create or update the out-of-band environment Secret referenced by Pulumi."""
    manifest = _build_env_secret_manifest(cfg)
    if manifest is None:
        raise click.ClickException(f"finelog config {cfg.name!r} does not require an environment Secret")
    click.echo(f"Applying Secret {k8s_env_secret_name(cfg)}...")
    _kubectl_apply(cfg, manifest)


def k8s_status(cfg: FinelogConfig) -> None:
    """Show deployment, service, and PVC status."""
    assert cfg.deployment.k8s is not None
    k8s = cfg.deployment.k8s
    _kubectl(
        cfg,
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
    _kubectl(cfg, *args)
