#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Safe finelog deploy with rollout / rollback.

Wraps the GCE bootstrap path used by ``finelog deploy restart`` with:
  - capture of the currently-running container's pinned image digest *before*
    the restart,
  - persistence of that digest under ``~/.cache/finelog/deploy-state/<name>.json``,
  - on health failure, an automatic re-bootstrap with the captured digest,
  - a separate ``rollback`` subcommand to restore the last good digest later.

Usage:
    uv run python lib/finelog/scripts/safe_deploy.py rollout marin-dev
    uv run python lib/finelog/scripts/safe_deploy.py rollback marin-dev
    uv run python lib/finelog/scripts/safe_deploy.py rollback marin-dev --to ghcr.io/...@sha256:...
    uv run python lib/finelog/scripts/safe_deploy.py status marin-dev
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import click
from finelog.deploy._gcp import _resolve_image_digest, _ssh_args, _wait_health_via_ssh
from finelog.deploy.bootstrap import CONTAINER_NAME, render_bootstrap
from finelog.deploy.build import build_image as build_finelog_image
from finelog.deploy.config import FinelogConfig, load_finelog_config

STATE_DIR = Path.home() / ".cache" / "finelog" / "deploy-state"


def _state_path(cfg: FinelogConfig) -> Path:
    return STATE_DIR / f"{cfg.name}.json"


def _read_state(cfg: FinelogConfig) -> dict:
    path = _state_path(cfg)
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _write_state(cfg: FinelogConfig, **updates: str | None) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = _read_state(cfg)
    state.update({k: v for k, v in updates.items() if v is not None})
    path = _state_path(cfg)
    path.write_text(json.dumps(state, indent=2, sort_keys=True))
    return path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _running_repo_digest(cfg: FinelogConfig) -> str | None:
    """Return the currently-running container's pinned ``ghcr.io/...@sha256:...``
    image, or ``None`` if no container is running or the digest is unavailable.

    Two-step inspect: container --> image config sha (``.Image``), then image
    --> ``RepoDigests``. Container-level inspect doesn't expose RepoDigests.
    Locally-built images with no published digest yield an empty list and
    we return ``None``.
    """
    # Bash heredoc keeps Go template braces literal; pipes the image id from
    # `docker inspect <container>` into `docker image inspect`.
    digests_tpl = "{{range .RepoDigests}}{{.}}|{{end}}"
    cmd = (
        f"set -e; "
        f"img=$(sudo docker inspect --format='{{{{.Image}}}}' {CONTAINER_NAME} 2>/dev/null) || exit 0; "
        f"sudo docker image inspect --format='{digests_tpl}' \"$img\" 2>/dev/null || true"
    )
    result = subprocess.run(_ssh_args(cfg, cmd), capture_output=True, text=True)
    for chunk in result.stdout.replace("\n", "|").split("|"):
        chunk = chunk.strip().strip("'")
        if "@sha256:" in chunk:
            return chunk
    return None


def _require_gcp(cfg: FinelogConfig) -> None:
    if cfg.deployment.gcp is None:
        raise click.ClickException("safe_deploy only supports GCP deployments.")


def _bootstrap_with_image(cfg: FinelogConfig, image: str) -> None:
    """Re-render and re-run the bootstrap with an explicit image (no pinning)."""
    bootstrap = render_bootstrap(image=image, port=cfg.port, remote_log_dir=cfg.remote_log_dir)
    result = subprocess.run(_ssh_args(cfg, "bash -s"), input=bootstrap, text=True)
    if result.returncode != 0:
        raise click.ClickException(f"Bootstrap failed for image {image}; see SSH output above")


def _verify_health(cfg: FinelogConfig) -> bool:
    assert cfg.deployment.gcp is not None
    return _wait_health_via_ssh(cfg, cfg.port)


@click.group()
def cli() -> None:
    """Safe finelog deploy: rollout with auto-rollback, plus explicit rollback."""


@cli.command("rollout")
@click.argument("name")
@click.option(
    "--auto-rollback/--no-auto-rollback",
    default=True,
    help="On health failure, re-bootstrap with the captured previous digest.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-bootstrap even when the new pinned digest matches the running one.",
)
@click.option(
    "--build/--no-build",
    default=True,
    help="Build and push cfg.image before resolving its digest. Default: build.",
)
def rollout_cmd(name: str, auto_rollback: bool, force: bool, build: bool) -> None:
    """Roll forward to the digest pinned from cfg.image; capture the previous digest."""
    cfg = load_finelog_config(name)
    _require_gcp(cfg)

    click.echo(f"== rollout: {cfg.name} ==")

    if build:
        click.echo(f"building & pushing {cfg.image}...")
        build_finelog_image(image=cfg.image)

    old_digest = _running_repo_digest(cfg)
    if old_digest is None:
        click.echo(
            "warning: no running container or digest unavailable; auto-rollback disabled.",
            err=True,
        )
        auto_rollback = False
    else:
        click.echo(f"captured running digest: {old_digest}")

    new_digest = _resolve_image_digest(cfg.image)
    if "@sha256:" not in new_digest:
        raise click.ClickException(f"Could not pin {cfg.image} to a content digest; refusing to deploy a mutable tag.")
    click.echo(f"new pinned digest:       {new_digest}")

    if old_digest == new_digest and not force:
        click.echo("new digest matches running digest; nothing to do (pass --force to redeploy).")
        return
    if old_digest == new_digest:
        click.echo("--force: redeploying the same digest.")

    state_path = _write_state(
        cfg,
        previous_digest=old_digest,
        attempted_digest=new_digest,
        rollout_started_at=_now(),
    )
    click.echo(f"state recorded at {state_path}")

    click.echo("re-running bootstrap with new image...")
    _bootstrap_with_image(cfg, new_digest)

    click.echo("waiting for /health...")
    if _verify_health(cfg):
        _write_state(cfg, current_digest=new_digest, rollout_succeeded_at=_now())
        click.echo(f"OK — {cfg.name} healthy on {new_digest}")
        if old_digest:
            click.echo(f"rollback target preserved: {old_digest}")
        return

    click.echo("FAIL — finelog did not become healthy on the new image.", err=True)
    if not auto_rollback or old_digest is None:
        raise click.ClickException(
            "Health check failed. Run `safe_deploy rollback <name>` (optionally with --to) to recover."
        )

    click.echo(f"auto-rolling back to {old_digest}...", err=True)
    _bootstrap_with_image(cfg, old_digest)
    if not _verify_health(cfg):
        raise click.ClickException(f"Rollback to {old_digest} ALSO failed — manual intervention required.")
    _write_state(
        cfg,
        current_digest=old_digest,
        rolled_back_from=new_digest,
        rolled_back_at=_now(),
    )
    raise click.ClickException(
        f"Rolled back to {old_digest}. Investigate the failed image {new_digest} before retrying."
    )


@cli.command("rollback")
@click.argument("name")
@click.option(
    "--to",
    "to_digest",
    default=None,
    help="Image (tag or digest) to restore. Defaults to the previous_digest captured on the last rollout.",
)
def rollback_cmd(name: str, to_digest: str | None) -> None:
    """Restore a previously-captured (or explicitly given) image digest."""
    cfg = load_finelog_config(name)
    _require_gcp(cfg)

    click.echo(f"== rollback: {cfg.name} ==")

    if to_digest is None:
        state = _read_state(cfg)
        to_digest = state.get("previous_digest")
        if not to_digest:
            raise click.ClickException(f"No previous_digest recorded for {cfg.name}; pass --to <image> explicitly.")
        click.echo(f"using previous_digest from state: {to_digest}")
    else:
        click.echo(f"using explicit target: {to_digest}")

    running = _running_repo_digest(cfg)
    if running:
        click.echo(f"currently running:       {running}")
        if running == to_digest:
            click.echo("rollback target matches running digest; nothing to do.")
            return

    _bootstrap_with_image(cfg, to_digest)
    if not _verify_health(cfg):
        raise click.ClickException(f"Rollback to {to_digest} did not become healthy.")
    _write_state(
        cfg,
        current_digest=to_digest,
        rolled_back_at=_now(),
        rolled_back_from=running,
    )
    click.echo(f"OK — {cfg.name} healthy on {to_digest}")


@cli.command("status")
@click.argument("name")
def status_cmd(name: str) -> None:
    """Show recorded state and the currently-running container digest."""
    cfg = load_finelog_config(name)
    _require_gcp(cfg)

    state = _read_state(cfg)
    click.echo(f"== status: {cfg.name} ==")
    click.echo(f"state file: {_state_path(cfg)}")
    if state:
        for key in (
            "current_digest",
            "previous_digest",
            "attempted_digest",
            "rollout_started_at",
            "rollout_succeeded_at",
            "rolled_back_from",
            "rolled_back_at",
        ):
            if key in state:
                click.echo(f"  {key}: {state[key]}")
    else:
        click.echo("  (no recorded state)")

    running = _running_repo_digest(cfg)
    click.echo(f"running digest: {running or '<unavailable>'}")


if __name__ == "__main__":
    cli()
