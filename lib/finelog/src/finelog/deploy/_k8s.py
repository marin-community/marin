# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deploy, roll back, and operate Pulumi-managed Kubernetes Finelog servers."""

import base64
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import click
from rigging.secrets import resolve_secret_spec

from finelog.deploy.bootstrap import HEALTH_OK, REGISTRATION_FAILED, health_probe_command
from finelog.deploy.config import (
    K8S_APP_LABEL,
    K8S_CONTAINER_NAME,
    SOURCE_REVISION_ANNOTATION,
    FinelogConfig,
    k8s_env_secret_name,
)

# S3-compatible endpoints that accept only virtual-hosted-style requests
# (bucket as a host subdomain).
_VIRTUAL_HOST_ONLY_S3_DOMAINS = ("cwobject.com", "cwlota.com")
_DEPLOYMENT_REVISION_ANNOTATION = "deployment.kubernetes.io/revision"
_PULUMI_PROJECT_DIR = Path(__file__).resolve().parents[5] / "infra" / "finelog"
_ROLLOUT_TIMEOUT = "10m"
_REVISION_DISCOVERY_ATTEMPTS = 60


@dataclass(frozen=True)
class K8sRevision:
    """A retained Finelog Deployment revision."""

    replica_set: str
    revision: int
    created_at: datetime
    image: str
    source_revision: str | None


@dataclass(frozen=True)
class K8sRevisionHistory:
    """The live Deployment and its retained ReplicaSet history."""

    deployment_uid: str
    current: K8sRevision
    revisions: tuple[K8sRevision, ...]


class K8sRevisionConflict(click.ClickException):
    """The active Deployment changed after rollback planning."""


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
    secret_name = k8s_env_secret_name(cfg)
    assert secret_name is not None
    manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": secret_name, "namespace": cfg.deployment.k8s.namespace},
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


def _deployment_json(cfg: FinelogConfig) -> dict | None:
    """Read the Deployment, or return ``None`` when it has not been created."""
    assert cfg.deployment.k8s is not None
    result = _kubectl(
        cfg,
        "get",
        f"deployment/{cfg.name}",
        "-n",
        cfg.deployment.k8s.namespace,
        "--ignore-not-found",
        "-o",
        "json",
        capture_output=True,
    )
    if not result.stdout.strip():
        return None
    return json.loads(result.stdout)


def _revision_number(metadata: dict, resource: str) -> int:
    annotations = metadata.get("annotations") or {}
    value = annotations.get(_DEPLOYMENT_REVISION_ANNOTATION)
    if value is None:
        raise click.ClickException(f"{resource} has no Kubernetes Deployment revision annotation")
    try:
        return int(value)
    except ValueError as exc:
        raise click.ClickException(f"{resource} has invalid Kubernetes Deployment revision {value!r}") from exc


def _replica_set_revision(item: dict) -> K8sRevision:
    metadata = item["metadata"]
    replica_set = metadata["name"]
    created_at = datetime.fromisoformat(metadata["creationTimestamp"].replace("Z", "+00:00"))
    template = item["spec"]["template"]
    containers = template["spec"]["containers"]
    container = next((container for container in containers if container.get("name") == K8S_CONTAINER_NAME), None)
    if container is None:
        raise click.ClickException(f"ReplicaSet {replica_set} has no finelog container")
    annotations = template.get("metadata", {}).get("annotations") or {}
    return K8sRevision(
        replica_set=replica_set,
        revision=_revision_number(metadata, f"ReplicaSet {replica_set}"),
        created_at=created_at,
        image=container["image"],
        source_revision=annotations.get(SOURCE_REVISION_ANNOTATION),
    )


def _deployment_owns(owner: dict, deployment_uid: str) -> bool:
    return owner.get("uid") == deployment_uid and owner.get("kind") == "Deployment" and owner.get("controller") is True


def _revision_history(cfg: FinelogConfig, deployment: dict) -> K8sRevisionHistory:
    assert cfg.deployment.k8s is not None
    metadata = deployment["metadata"]
    deployment_uid = metadata["uid"]
    current_revision = _revision_number(metadata, f"Deployment {cfg.name}")
    result = _kubectl(
        cfg,
        "get",
        "replicasets",
        "-n",
        cfg.deployment.k8s.namespace,
        "-l",
        f"{K8S_APP_LABEL}={cfg.name}",
        "-o",
        "json",
        capture_output=True,
    )
    revisions = tuple(
        _replica_set_revision(item)
        for item in json.loads(result.stdout)["items"]
        if any(_deployment_owns(owner, deployment_uid) for owner in item["metadata"].get("ownerReferences", ()))
    )
    current = next((revision for revision in revisions if revision.revision == current_revision), None)
    if current is None:
        raise click.ClickException(f"Deployment {cfg.name} revision {current_revision} has no retained ReplicaSet")
    return K8sRevisionHistory(
        deployment_uid=deployment_uid,
        current=current,
        revisions=revisions,
    )


def k8s_revision_history(cfg: FinelogConfig) -> K8sRevisionHistory:
    """Read the active Finelog revision and retained rollback targets."""
    deployment = _deployment_json(cfg)
    if deployment is None:
        raise click.ClickException(f"Kubernetes Deployment {cfg.name!r} does not exist")
    return _revision_history(cfg, deployment)


def select_rollback_revision(history: K8sRevisionHistory, *, to_revision: int | None = None) -> K8sRevision:
    """Select an explicit revision or the next older ReplicaSet release."""
    if to_revision is not None:
        target = next((revision for revision in history.revisions if revision.revision == to_revision), None)
        if target is None:
            raise click.ClickException(f"Kubernetes revision {to_revision} is not retained")
        if target.replica_set == history.current.replica_set:
            raise click.ClickException(f"Kubernetes revision {to_revision} is already active")
        return target

    ordered = sorted(
        history.revisions,
        key=lambda revision: (revision.created_at, revision.revision),
        reverse=True,
    )
    current_index = next(
        index for index, revision in enumerate(ordered) if revision.replica_set == history.current.replica_set
    )
    if current_index + 1 == len(ordered):
        raise click.ClickException(f"no older retained revision exists before {history.current.replica_set}")
    return ordered[current_index + 1]


def _revision_summary(revision: K8sRevision) -> str:
    source = f", source {revision.source_revision}" if revision.source_revision else ""
    return f"revision {revision.revision} {revision.replica_set} ({revision.image}{source})"


def _pulumi(stack: str, command: str, *args: str) -> None:
    if not (_PULUMI_PROJECT_DIR / "Pulumi.yaml").is_file():
        raise click.ClickException("Finelog Pulumi deploys must run from a Marin repository checkout")
    subprocess.run(
        ["pulumi", command, "--stack", stack, *args],
        cwd=_PULUMI_PROJECT_DIR,
        check=True,
    )


def _wait_for_active_replica_set(
    cfg: FinelogConfig,
    *,
    replica_set: str,
    after_revision: int,
) -> K8sRevisionHistory:
    for _ in range(_REVISION_DISCOVERY_ATTEMPTS):
        history = k8s_revision_history(cfg)
        if history.current.replica_set == replica_set and history.current.revision > after_revision:
            return history
        time.sleep(1)
    raise click.ClickException(f"Deployment {cfg.name} did not activate ReplicaSet {replica_set}")


def _activate_revision(
    cfg: FinelogConfig,
    *,
    expected_history: K8sRevisionHistory,
    target: K8sRevision,
) -> K8sRevision:
    assert cfg.deployment.k8s is not None
    history = k8s_revision_history(cfg)
    if (
        history.deployment_uid != expected_history.deployment_uid
        or history.current.replica_set != expected_history.current.replica_set
        or history.current.revision != expected_history.current.revision
    ):
        raise K8sRevisionConflict(
            f"Deployment {cfg.name} changed from revision {expected_history.current.revision} "
            f"to {history.current.revision}; plan the rollback again"
        )
    retained_target = next(
        (revision for revision in history.revisions if revision.replica_set == target.replica_set),
        None,
    )
    if retained_target is None:
        raise click.ClickException(f"ReplicaSet {target.replica_set} is no longer retained")

    _kubectl(
        cfg,
        "rollout",
        "undo",
        f"deployment/{cfg.name}",
        "-n",
        cfg.deployment.k8s.namespace,
        f"--to-revision={retained_target.revision}",
    )
    activated = _wait_for_active_replica_set(
        cfg,
        replica_set=retained_target.replica_set,
        after_revision=history.current.revision,
    )
    _kubectl(
        cfg,
        "rollout",
        "status",
        f"deployment/{cfg.name}",
        "-n",
        cfg.deployment.k8s.namespace,
        f"--revision={activated.current.revision}",
        f"--timeout={_ROLLOUT_TIMEOUT}",
    )
    k8s_verify_ingest_ready(cfg)
    return activated.current


def _refresh_after_rollback(stack: str) -> None:
    try:
        _pulumi(stack, "refresh", "--yes")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise click.ClickException(
            f"Finelog is serving the restored revision, but `pulumi refresh` failed for stack {stack!r}"
        ) from exc


def _restore_source_revision(
    cfg: FinelogConfig,
    *,
    stack: str,
    source: K8sRevision,
    failure: BaseException,
    operation: str,
) -> K8sRevision:
    live = k8s_revision_history(cfg)
    if live.current.replica_set == source.replica_set:
        raise click.ClickException(
            f"{operation} failed before changing the active ReplicaSet; revision {source.revision} is still serving"
        ) from failure
    try:
        restored = _activate_revision(cfg, expected_history=live, target=source)
    except (OSError, subprocess.CalledProcessError, click.ClickException) as rollback_failure:
        raise click.ClickException(
            f"{operation} failed and automatic recovery to {source.replica_set} also failed: {rollback_failure}"
        ) from failure
    _refresh_after_rollback(stack)
    return restored


def k8s_pulumi_rollout(cfg: FinelogConfig, *, stack: str, yes: bool) -> None:
    """Run the Finelog Pulumi update and restore the captured revision on failure."""
    deployment = _deployment_json(cfg)
    source = _revision_history(cfg, deployment).current if deployment is not None else None
    if source is not None:
        click.echo(f"Captured Kubernetes {_revision_summary(source)}")
    try:
        _pulumi(stack, "up", *(["--yes"] if yes else []))
    except (OSError, subprocess.CalledProcessError) as failure:
        if source is None:
            raise click.ClickException(
                "Pulumi update failed; no previous Kubernetes revision was available"
            ) from failure
        restored = _restore_source_revision(
            cfg,
            stack=stack,
            source=source,
            failure=failure,
            operation="Pulumi update",
        )
        raise click.ClickException(
            f"Pulumi update failed; restored Kubernetes revision {source.revision} "
            f"as revision {restored.revision} ({source.replica_set})"
        ) from failure


def k8s_rollback(
    cfg: FinelogConfig,
    *,
    stack: str,
    to_revision: int | None,
    yes: bool = False,
) -> None:
    """Move the Deployment to a retained revision and restore the source on failure."""
    history = k8s_revision_history(cfg)
    target = select_rollback_revision(history, to_revision=to_revision)
    click.echo(f"Current: {_revision_summary(history.current)}")
    click.echo(f"Target:  {_revision_summary(target)}")
    if not yes and not click.confirm("Roll back this Finelog Deployment?"):
        click.echo("Aborted.")
        return

    try:
        activated = _activate_revision(cfg, expected_history=history, target=target)
    except K8sRevisionConflict:
        raise
    except (OSError, subprocess.CalledProcessError, click.ClickException) as failure:
        restored = _restore_source_revision(
            cfg,
            stack=stack,
            source=history.current,
            failure=failure,
            operation=f"Rollback to {target.replica_set}",
        )
        raise click.ClickException(
            f"Rollback to {target.replica_set} failed; restored {history.current.replica_set} "
            f"as revision {restored.revision}"
        ) from failure
    _refresh_after_rollback(stack)
    click.echo(f"Finelog is healthy on revision {activated.revision} ({activated.replica_set}).")


def k8s_sync_secret(cfg: FinelogConfig) -> None:
    """Create or update the out-of-band environment Secret referenced by Pulumi."""
    manifest = _build_env_secret_manifest(cfg)
    if manifest is None:
        raise click.ClickException(f"finelog config {cfg.name!r} does not require an environment Secret")
    secret_name = k8s_env_secret_name(cfg)
    assert secret_name is not None
    click.echo(f"Applying Secret {secret_name}...")
    _kubectl_apply(cfg, manifest)


def k8s_verify_ingest_ready(cfg: FinelogConfig, max_attempts: int = 60) -> None:
    """Fail when the deployed server is listening but not accepting writes."""
    assert cfg.deployment.k8s is not None
    probe = health_probe_command(cfg.port)
    body = "unreachable"
    for _ in range(max_attempts):
        result = _kubectl(
            cfg,
            "exec",
            f"deployment/{cfg.name}",
            "-n",
            cfg.deployment.k8s.namespace,
            "--",
            "sh",
            "-c",
            probe,
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise click.ClickException(f"could not read /health from deployment/{cfg.name}: {result.stderr.strip()}")
        body = result.stdout.strip()
        if body == HEALTH_OK:
            return
        if REGISTRATION_FAILED in body:
            break
        time.sleep(2)
    raise click.ClickException(f"finelog is serving but not ingesting: {body}")


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
