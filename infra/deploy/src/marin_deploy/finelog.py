# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog deployment commands."""

import json
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import click
from finelog.deploy import _k8s
from finelog.deploy._gcp import _wait_health_via_ssh, render_bootstrap_for
from finelog.deploy.bootstrap import CONTAINER_NAME, HEALTH_OK
from finelog.deploy.build import build_image as build_finelog_image
from finelog.deploy.config import FinelogConfig, load_finelog_config
from finelog.deploy.image import resolve_image_digest

from marin_deploy.gce import GceVmTarget, StartupScriptPersistence, activate_startup_script, remote_stdout

STATE_DIR = Path.home() / ".cache" / "finelog" / "deploy-state"
GCE_ACTIVATION_TIMEOUT = 900
GCE_INSPECT_TIMEOUT = 60
STATE_FIELDS = (
    "current_digest",
    "previous_digest",
    "attempted_digest",
    "rollout_started_at",
    "rollout_succeeded_at",
    "rolled_back_from",
    "rolled_back_at",
)


@dataclass(frozen=True)
class _FinelogRolloutState:
    current_digest: str | None = None
    previous_digest: str | None = None
    attempted_digest: str | None = None
    rollout_started_at: str | None = None
    rollout_succeeded_at: str | None = None
    rolled_back_from: str | None = None
    rolled_back_at: str | None = None


def _state_path(config: FinelogConfig) -> Path:
    return STATE_DIR / f"{config.name}.json"


def _read_state(config: FinelogConfig) -> _FinelogRolloutState:
    path = _state_path(config)
    if not path.is_file():
        return _FinelogRolloutState()
    return _FinelogRolloutState(**json.loads(path.read_text()))


def _write_state(config: FinelogConfig, state: _FinelogRolloutState) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _state_path(config)
    payload = {key: value for key, value in asdict(state).items() if value is not None}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _gce_target(config: FinelogConfig) -> GceVmTarget:
    deployment = config.deployment.gcp
    assert deployment is not None
    return GceVmTarget(
        project=deployment.project,
        zone=deployment.zone,
        instance=config.name,
        impersonate_service_account=deployment.service_account,
    )


def _running_repo_digest(config: FinelogConfig) -> str | None:
    digests_template = "{{range .RepoDigests}}{{.}}|{{end}}"
    command = (
        f"set -e; "
        f"image=$(sudo docker inspect --format='{{{{.Image}}}}' {CONTAINER_NAME} 2>/dev/null) || exit 0; "
        f"sudo docker image inspect --format='{digests_template}' \"$image\" 2>/dev/null || true"
    )
    try:
        output = remote_stdout(
            _gce_target(config),
            command,
            timeout=GCE_INSPECT_TIMEOUT,
            attempts=3,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise click.ClickException(f"Could not inspect the running Finelog image on {config.name}: {error}") from error
    for chunk in output.replace("\n", "|").split("|"):
        digest = chunk.strip().strip("'")
        if "@sha256:" in digest:
            return digest
    return None


def _bootstrap_with_image(config: FinelogConfig, image: str) -> bool:
    try:
        activate_startup_script(
            _gce_target(config),
            render_bootstrap_for(config, image),
            persistence=StartupScriptPersistence.AFTER_SUCCESS,
            timeout=GCE_ACTIVATION_TIMEOUT,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True


def _verify_health(config: FinelogConfig) -> bool:
    health = _wait_health_via_ssh(config, config.port)
    if health == HEALTH_OK:
        return True
    click.echo(f"Finelog is not ingesting: {health}", err=True)
    return False


def _pinned_image(image: str) -> str:
    pinned = resolve_image_digest(image)
    if "@sha256:" not in pinned:
        raise click.ClickException(f"Could not pin {image} to a content digest; refusing to deploy a mutable tag.")
    return pinned


def _gce_rollout(
    config: FinelogConfig,
    *,
    auto_rollback: bool,
    force: bool,
    build: bool,
    fast: bool,
) -> None:
    click.echo(f"== rollout: {config.name} ==")
    if build:
        cargo_profile = "fast" if fast else "release"
        click.echo(f"building & pushing {config.image} (cargo profile: {cargo_profile})...")
        build_finelog_image(image=config.image, cargo_profile=cargo_profile)

    previous_digest = _running_repo_digest(config)
    if previous_digest is None:
        click.echo("warning: no running container or digest unavailable; auto-rollback disabled.", err=True)
        auto_rollback = False
    else:
        click.echo(f"captured running digest: {previous_digest}")

    candidate_digest = _pinned_image(config.image)
    click.echo(f"new pinned digest:       {candidate_digest}")
    if previous_digest == candidate_digest and not force:
        click.echo("new digest matches running digest; nothing to do (pass --force to redeploy).")
        return
    if previous_digest == candidate_digest:
        click.echo("--force: redeploying the same digest.")

    state = replace(
        _read_state(config),
        previous_digest=previous_digest,
        attempted_digest=candidate_digest,
        rollout_started_at=_now(),
    )
    click.echo(f"state recorded at {_write_state(config, state)}")
    click.echo("re-running bootstrap with new image...")
    healthy = _bootstrap_with_image(config, candidate_digest)
    if healthy:
        click.echo("waiting for /health...")
        healthy = _verify_health(config)

    if healthy:
        _write_state(
            config,
            replace(state, current_digest=candidate_digest, rollout_succeeded_at=_now()),
        )
        click.echo(f"OK — {config.name} healthy on {candidate_digest}")
        if previous_digest:
            click.echo(f"rollback target preserved: {previous_digest}")
        return

    click.echo("FAIL — Finelog did not become healthy on the new image.", err=True)
    if not auto_rollback or previous_digest is None:
        raise click.ClickException(
            "Health check failed. Run `marin-deploy finelog rollback <name>` with optional `--to` to recover."
        )

    click.echo(f"auto-rolling back to {previous_digest}...", err=True)
    if not _bootstrap_with_image(config, previous_digest) or not _verify_health(config):
        raise click.ClickException(f"Rollback to {previous_digest} also failed; manual intervention is required.")
    _write_state(
        config,
        replace(
            state,
            current_digest=previous_digest,
            rolled_back_from=candidate_digest,
            rolled_back_at=_now(),
        ),
    )
    raise click.ClickException(
        f"Rolled back to {previous_digest}. Investigate the failed image {candidate_digest} before retrying."
    )


def _gce_rollback(config: FinelogConfig, to_digest: str | None) -> None:
    click.echo(f"== rollback: {config.name} ==")
    state = _read_state(config)
    if to_digest is None:
        to_digest = state.previous_digest
        if not to_digest:
            raise click.ClickException(f"No previous_digest recorded for {config.name}; pass --to <image> explicitly.")
        click.echo(f"using previous_digest from state: {to_digest}")
    else:
        to_digest = _pinned_image(to_digest)
        click.echo(f"using explicit target: {to_digest}")

    running_digest = _running_repo_digest(config)
    if running_digest:
        click.echo(f"currently running:       {running_digest}")
        if running_digest == to_digest:
            click.echo("rollback target matches running digest; nothing to do.")
            return

    if not _bootstrap_with_image(config, to_digest) or not _verify_health(config):
        raise click.ClickException(f"Rollback to {to_digest} did not become healthy.")
    _write_state(
        config,
        replace(
            state,
            current_digest=to_digest,
            rolled_back_at=_now(),
            rolled_back_from=running_digest,
        ),
    )
    click.echo(f"OK — {config.name} healthy on {to_digest}")


def _gce_status(config: FinelogConfig) -> None:
    state = _read_state(config)
    click.echo(f"== status: {config.name} ==")
    click.echo(f"state file: {_state_path(config)}")
    values = asdict(state)
    if any(values.values()):
        for key in STATE_FIELDS:
            if values[key] is not None:
                click.echo(f"  {key}: {values[key]}")
    else:
        click.echo("  (no recorded state)")
    click.echo(f"running digest: {_running_repo_digest(config) or '<unavailable>'}")


def _unsupported_k8s_rollout_options(*, auto_rollback: bool, force: bool, build: bool, fast: bool) -> list[str]:
    options: list[str] = []
    if not auto_rollback:
        options.append("--no-auto-rollback")
    if force:
        options.append("--force")
    if not build:
        options.append("--no-build")
    if fast:
        options.append("--fast")
    return options


@click.group()
def finelog() -> None:
    """Deploy Finelog servers."""


@finelog.command("rollout")
@click.argument("name")
@click.option("-y", "--yes", is_flag=True, help="Skip Kubernetes Pulumi confirmation.")
@click.option("--auto-rollback/--no-auto-rollback", default=True, help="Restore the prior GCE digest on failure.")
@click.option("--force", is_flag=True, help="Redeploy a matching GCE digest.")
@click.option("--build/--no-build", default=True, show_default=True, help="Build the GCE image before rollout.")
@click.option("--fast", is_flag=True, help="Use the Finelog fast Cargo profile for a GCE build.")
def rollout_cmd(name: str, yes: bool, auto_rollback: bool, force: bool, build: bool, fast: bool) -> None:
    """Deploy and verify one Finelog server."""
    config = load_finelog_config(name)
    if config.deployment.k8s is not None:
        unsupported = _unsupported_k8s_rollout_options(
            auto_rollback=auto_rollback,
            force=force,
            build=build,
            fast=fast,
        )
        if unsupported:
            raise click.ClickException(f"Kubernetes Finelog rollout does not support {', '.join(unsupported)}")
        _k8s.k8s_pulumi_rollout(config, stack=name, yes=yes)
        return
    if yes:
        raise click.ClickException("--yes applies only to Kubernetes Finelog rollouts")
    _gce_rollout(config, auto_rollback=auto_rollback, force=force, build=build, fast=fast)


@finelog.command("rollback")
@click.argument("name")
@click.option("--to-revision", type=int, help="Restore an exact retained Kubernetes Deployment revision.")
@click.option("--to", "to_digest", help="Restore a GCE image tag or digest; tags are pinned before activation.")
@click.option("-y", "--yes", is_flag=True, help="Skip Kubernetes confirmation.")
def rollback_cmd(name: str, to_revision: int | None, to_digest: str | None, yes: bool) -> None:
    """Restore the previous healthy Finelog deployment."""
    config = load_finelog_config(name)
    if config.deployment.k8s is not None:
        if to_digest is not None:
            raise click.ClickException("--to applies only to GCE Finelog rollbacks")
        _k8s.k8s_rollback(config, stack=name, to_revision=to_revision, yes=yes)
        return
    if to_revision is not None:
        raise click.ClickException("--to-revision applies only to Kubernetes Finelog rollbacks")
    if yes:
        raise click.ClickException("--yes applies only to Kubernetes Finelog rollbacks")
    _gce_rollback(config, to_digest)


@finelog.command("status")
@click.argument("name")
def status_cmd(name: str) -> None:
    """Show deployment status for one Finelog server."""
    config = load_finelog_config(name)
    if config.deployment.k8s is not None:
        _k8s.k8s_status(config)
        return
    _gce_status(config)
