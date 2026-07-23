# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes deployment backend for finelog.

Templates `lib/finelog/deploy/k8s/*.yaml` against a `FinelogConfig` and
shells out to `kubectl`. No kubernetes-client Python dep — the manifest
list is small enough that subprocess is the right tool.
"""

import base64
import json
import os
import re
import subprocess
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

import click
from rigging.secrets import resolve_secret_spec

from finelog.deploy.bootstrap import render_template
from finelog.deploy.config import FinelogConfig, auth_policy_json
from finelog.deploy.image import resolve_image_digest

_TEMPLATE_VAR_RE = re.compile(r"\{\{ (\w+) \}\}")

# Suffix for the finelog-owned Secret that carries the pod's secret environment:
# S3 credentials and the forwarding signing key. Distinct from iris's own task-env
# Secret so finelog manages its own lifecycle.
_ENV_SECRET_SUFFIX = "-env"

# S3-compatible endpoints that accept only virtual-hosted-style requests
# (bucket as a host subdomain).
_VIRTUAL_HOST_ONLY_S3_DOMAINS = ("cwobject.com", "cwlota.com")

# Manifests live at `lib/finelog/deploy/k8s/*.yaml` in the repo. We resolve
# this once at import time; the directory is part of the source tree, not
# the wheel, but k8s deployments are operator-driven and run from a checkout.
_K8S_MANIFEST_DIR = Path(__file__).resolve().parents[3] / "deploy" / "k8s"

_MANIFESTS = ("01-pvc.yaml.tmpl", "02-deployment.yaml.tmpl", "03-service.yaml.tmpl")


def _env_entry(name: str, value: str) -> str:
    """Render one single-quoted container-env entry at the template's indentation."""
    if "'" in value:
        raise ValueError(f"{name} must not contain a single quote")
    return f"            - name: {name}\n              value: '{value}'"


def _inline_env_block(cfg: FinelogConfig) -> str:
    """Render the non-secret container-env entries the templates splice in, or "".

    `FINELOG_AUTH_POLICY` is inline-safe (a cidr layer carries network prefixes, a jwt
    layer Ed25519 public keys), as is `FINELOG_FORWARDING` (a url and a cluster name).
    The forwarding *private* key travels in the `<name>-env` Secret instead.
    """
    entries = []
    if cfg.auth:
        entries.append(_env_entry("FINELOG_AUTH_POLICY", auth_policy_json(cfg.auth)))
    if cfg.forwarding:
        entries.append(_env_entry("FINELOG_FORWARDING", cfg.forwarding.to_env_json()))
    return "\n".join(entries)


def _priority_class_block(cfg: FinelogConfig) -> str:
    """Render the pod-spec `priorityClassName` line, or "" when none is configured."""
    assert cfg.deployment.k8s is not None
    name = cfg.deployment.k8s.priority_class_name
    if not name:
        return ""
    return f"      priorityClassName: {name}"


def _render_manifest(template_path: Path, cfg: FinelogConfig) -> str:
    """Render a single k8s manifest template against `cfg`.

    `render_template` raises on unused variables, and the three manifests use
    disjoint subsets of the available config fields (PVC needs storage_*;
    Deployment needs image/port/remote_log_dir; Service needs port). We pass
    only the variables the template actually references.
    """
    assert cfg.deployment.k8s is not None
    k8s = cfg.deployment.k8s
    storage_class_block = (
        f"storageClassName: {k8s.storage_class}" if k8s.storage_class else "# storageClassName: <cluster default>"
    )
    template = template_path.read_text()
    all_vars: dict[str, str | int] = {
        "name": cfg.name,
        "namespace": k8s.namespace,
        "image": cfg.image,
        "port": cfg.port,
        "remote_log_dir": cfg.remote_log_dir,
        "storage_class_block": storage_class_block,
        "storage_gb": k8s.storage_gb,
        "cpu_request": k8s.cpu_request,
        "cpu_limit": k8s.cpu_limit,
        "memory_request": k8s.memory_request,
        "memory_limit": k8s.memory_limit,
        "inline_env_block": _inline_env_block(cfg),
        "priority_class_block": _priority_class_block(cfg),
    }
    referenced = set(_TEMPLATE_VAR_RE.findall(template))
    return render_template(template, **{k: v for k, v in all_vars.items() if k in referenced})


def _env_secret_name(cfg: FinelogConfig) -> str:
    return f"{cfg.name}{_ENV_SECRET_SUFFIX}"


def _s3_env(cfg: FinelogConfig) -> dict[str, str]:
    """The ``AWS_*`` environment for an ``s3://`` archive, or ``{}`` when none.

    Carries the operator's R2 credentials (from ``R2_ACCESS_KEY_ID`` /
    ``R2_SECRET_ACCESS_KEY`` in the deploy shell) plus the configured endpoint and
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
    key_id = os.environ.get("R2_ACCESS_KEY_ID")
    key_secret = os.environ.get("R2_SECRET_ACCESS_KEY")
    if not key_id or not key_secret:
        raise click.ClickException(
            "R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY must be set in the deploy "
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
        "metadata": {"name": _env_secret_name(cfg), "namespace": cfg.deployment.k8s.namespace},
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
    cfg: FinelogConfig, *args: str, stdin: str | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    return subprocess.run(["kubectl", *_kube_flags(cfg), *args], input=stdin, text=True, check=check)


def _kubectl_apply(cfg: FinelogConfig, manifest: str) -> None:
    _kubectl(cfg, "apply", "-f", "-", stdin=manifest)


def _ensure_priority_class(cfg: FinelogConfig) -> None:
    """Create the configured PriorityClass (idempotently) before the Deployment.

    A pod referencing a missing PriorityClass is rejected at admission, and on a
    fresh cluster finelog is brought up before Iris creates the iris-* bands. So
    finelog provisions its own scheduling dependency rather than depending on
    ordering. `kubectl apply` is a no-op when the class already exists with the
    same immutable value/preemptionPolicy (e.g. Iris created it first), and fails
    loudly on a real mismatch. PreemptLowerPriority matches the iris-system band:
    the control plane may evict a lower-priority pod to stay scheduled.
    """
    assert cfg.deployment.k8s is not None
    k8s = cfg.deployment.k8s
    if k8s.priority_class_name is None:
        return
    manifest = {
        "apiVersion": "scheduling.k8s.io/v1",
        "kind": "PriorityClass",
        "metadata": {"name": k8s.priority_class_name},
        "value": k8s.priority_class_value,
        "preemptionPolicy": "PreemptLowerPriority",
        "globalDefault": False,
    }
    click.echo(f"Ensuring PriorityClass {k8s.priority_class_name} (value {k8s.priority_class_value})...")
    _kubectl_apply(cfg, json.dumps(manifest))


def k8s_up(cfg: FinelogConfig) -> None:
    """Render manifests and apply them; wait for the deployment to roll out.

    ``cfg.image`` is pinned to its content digest before rendering, so the
    Deployment references an immutable image and a redeploy lands exactly what
    the tag points to now (with ``imagePullPolicy: IfNotPresent``, cache-safe).
    """
    assert cfg.deployment.k8s is not None
    cfg = replace(cfg, image=resolve_image_digest(cfg.image))
    k8s = cfg.deployment.k8s
    _ensure_priority_class(cfg)
    secret_manifest = _build_env_secret_manifest(cfg)
    if secret_manifest is not None:
        click.echo(f"Applying Secret {_env_secret_name(cfg)}...")
        _kubectl_apply(cfg, secret_manifest)
    for manifest_name in _MANIFESTS:
        rendered = _render_manifest(_K8S_MANIFEST_DIR / manifest_name, cfg)
        click.echo(f"Applying {manifest_name}...")
        _kubectl_apply(cfg, rendered)
    click.echo(f"Waiting for deployment/{cfg.name} to become Ready...")
    _kubectl(cfg, "rollout", "status", f"deployment/{cfg.name}", "-n", k8s.namespace)
    click.echo("finelog is healthy.")


def k8s_down(cfg: FinelogConfig, *, yes: bool) -> None:
    """Delete deployment, service, and the env Secret. Delete the PVC only when `yes=True`.

    The Secret goes with them: it holds the archive credentials and the forwarding
    signing key, and a torn-down deployment has no use for either. `deploy up` mints it
    again from the operator's environment and the config's secret references.
    """
    assert cfg.deployment.k8s is not None
    k8s = cfg.deployment.k8s
    _kubectl(
        cfg,
        "delete",
        f"deployment/{cfg.name}",
        f"service/{cfg.name}",
        f"secret/{_env_secret_name(cfg)}",
        "-n",
        k8s.namespace,
        "--ignore-not-found",
    )
    if yes:
        _kubectl(
            cfg,
            "delete",
            f"pvc/{cfg.name}-cache",
            "-n",
            k8s.namespace,
            "--ignore-not-found",
        )
        click.echo(f"Deleted {cfg.name} (deployment, service, secret, pvc).")
    else:
        click.echo(
            f"Deleted {cfg.name} (deployment, service, secret). "
            f"PVC {cfg.name}-cache retained — pass -y to delete it as well."
        )


def k8s_restart(cfg: FinelogConfig) -> None:
    """Roll the deployment by re-setting its image, then wait for rollout."""
    assert cfg.deployment.k8s is not None
    k8s = cfg.deployment.k8s
    _kubectl(
        cfg,
        "set",
        "image",
        f"deployment/{cfg.name}",
        f"finelog={resolve_image_digest(cfg.image)}",
        "-n",
        k8s.namespace,
    )
    _kubectl(cfg, "rollout", "status", f"deployment/{cfg.name}", "-n", k8s.namespace)
    click.echo("finelog is healthy.")


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
