#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Safe finelog deploy with pre-flight, rollout, and rollback.

Wraps the GCE bootstrap path used by ``finelog deploy restart`` with:
  - a schema pre-flight that decides the deploy before it touches the host,
  - capture of the currently-running container's pinned image digest *before*
    the restart,
  - persistence of that digest under ``~/.cache/finelog/deploy-state/<name>.json``,
  - on health failure, an automatic re-bootstrap with the captured digest,
  - a separate ``rollback`` subcommand to restore the last good digest later.

The pre-flight reads each deployment's registered schemas and runs them through
the candidate image's own merge (``finelog-server check-schema``). A rollout
refuses to proceed when the image's schema would not register, because that
failure is invisible to the health gate: the server listens, ``/health`` is
green, and every write to the wedged namespace fails. After a successful
rollout the registered schemas are recorded under
``lib/finelog/deploy/registered_schemas/``, where a Rust test re-checks them on
every pull request that touches the server.

Usage:
    uv run python lib/finelog/scripts/safe_deploy.py preflight marin-dev
    uv run python lib/finelog/scripts/safe_deploy.py preflight --all
    uv run python lib/finelog/scripts/safe_deploy.py rollout marin-dev
    uv run python lib/finelog/scripts/safe_deploy.py rollback marin-dev
    uv run python lib/finelog/scripts/safe_deploy.py rollback marin-dev --to ghcr.io/...@sha256:...
    uv run python lib/finelog/scripts/safe_deploy.py status marin-dev
"""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import click
import httpx
from finelog.deploy._gcp import _ssh_args, _wait_health_via_ssh, apply_bootstrap, render_bootstrap_for
from finelog.deploy.bootstrap import CONTAINER_NAME, HEALTH_OK
from finelog.deploy.build import build_image as build_finelog_image
from finelog.deploy.config import FinelogConfig, bundled_config_names, load_finelog_config
from finelog.deploy.connection import open_log_client
from finelog.deploy.image import resolve_image_digest
from finelog.deploy.preflight import (
    Outcome,
    PreflightResult,
    blocks_rollout,
    check_image,
    load_golden,
    registered_schema_document,
    render_document,
    summarize,
)
from finelog.errors import StatsError
from finelog.schema import Schema
from rigging.auth import IapLoginRequired

STATE_DIR = Path.home() / ".cache" / "finelog" / "deploy-state"

# Recorded registered schemas, one file per deployment. Checked in: each is the
# last thing that deployment's catalog held, and `preflight::tests` re-decides
# every one of them against the binary on each pull request. A stale golden only
# over-reports — it can fail a change production would have accepted, never pass
# one production would reject — which is the safe direction to age in.
GOLDEN_DIR = Path(__file__).resolve().parents[1] / "deploy" / "registered_schemas"

# Long enough for an SSH or kubectl port-forward to a cold VM/pod to come up.
TUNNEL_TIMEOUT = 60.0


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
    return datetime.now(UTC).isoformat()


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
        raise click.ClickException("safe_deploy rollout/rollback only supports GCP deployments.")


def _golden_path(cfg: FinelogConfig) -> Path:
    return GOLDEN_DIR / f"{cfg.name}.json"


def _registered_schemas(cfg: FinelogConfig, name: str) -> dict[str, Schema] | None:
    """Read the registered schemas from the running server, or None if unreachable."""
    try:
        with open_log_client(cfg, name, TUNNEL_TIMEOUT) as client:
            return client.list_namespaces()
    # Three transports — IAP proxy, SSH tunnel, kubectl port-forward — fail in
    # their own vocabularies, and all of them mean the same thing here. A bug in
    # this script is not one of them, so `Exception` is too wide: it would read
    # as unreachability and quietly downgrade the decision.
    except (OSError, RuntimeError, StatsError, IapLoginRequired, httpx.HTTPError) as exc:
        click.echo(f"could not read registered schemas from {cfg.name}: {exc}", err=True)
        return None


def _preflight(cfg: FinelogConfig, name: str, image: str) -> PreflightResult:
    """Decide whether ``image`` would register against ``cfg``'s catalog."""
    live = _registered_schemas(cfg, name)
    if live is not None:
        document = registered_schema_document(
            deployment=cfg.name,
            namespaces=live,
            captured_at=_now(),
            captured_from=f"the live catalog of {cfg.name}",
        )
        source = "the live server"
    else:
        document = load_golden(_golden_path(cfg))
        if document is None:
            return PreflightResult(
                deployment=cfg.name,
                outcome=Outcome.UNKNOWN,
                source="nothing",
                report=f"no live server and no recorded golden at {_golden_path(cfg)}",
            )
        source = f"the recorded golden {_golden_path(cfg).name}"
    passed, report = check_image(image, document)
    return PreflightResult(
        deployment=cfg.name,
        outcome=Outcome.PASS if passed else Outcome.FAIL,
        source=source,
        report=report,
    )


def _record_golden(cfg: FinelogConfig, name: str, digest: str) -> Path | None:
    """Write the schemas ``cfg``'s catalog now holds to its checked-in golden."""
    namespaces = _registered_schemas(cfg, name)
    if namespaces is None:
        click.echo(f"could not record the deploy golden for {cfg.name}; it stays at its last value.", err=True)
        return None
    path = _golden_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_document(
            registered_schema_document(
                deployment=cfg.name,
                namespaces=namespaces,
                captured_at=_now(),
                captured_from=f"the live catalog of {cfg.name} on {digest}",
            )
        )
    )
    return path


def _bootstrap_with_image(cfg: FinelogConfig, image: str) -> bool:
    """Bootstrap the VM onto ``image``. Returns whether it came up healthy."""
    return apply_bootstrap(cfg, render_bootstrap_for(cfg, image))


def _verify_health(cfg: FinelogConfig) -> bool:
    assert cfg.deployment.gcp is not None
    health = _wait_health_via_ssh(cfg, cfg.port)
    if health != HEALTH_OK:
        click.echo(f"finelog is not ingesting: {health}", err=True)
        return False
    return True


@click.group()
def cli() -> None:
    """Safe finelog deploy: pre-flight, rollout with auto-rollback, explicit rollback."""


@cli.command("preflight")
@click.argument("names", nargs=-1)
@click.option("--all", "check_all", is_flag=True, default=False, help="Check every bundled deployment.")
@click.option(
    "--image",
    default=None,
    help="Image to decide. Default: the digest each config's image tag currently pins to.",
)
def preflight_cmd(names: tuple[str, ...], check_all: bool, image: str | None) -> None:
    """Decide whether an image's schemas would register, without touching a host.

    Reads each deployment's registered schemas from its running server and runs
    them through the candidate image's own merge. Every deployment keeps its own
    catalog and they can disagree, so a fleet-wide change is only safe when all
    of them accept it — that is what `--all` is for.
    """
    selected = list(names) + (bundled_config_names() if check_all else [])
    if not selected:
        raise click.UsageError("name at least one deployment, or pass --all")

    results = []
    for name in dict.fromkeys(selected):
        cfg = load_finelog_config(name)
        candidate = image or resolve_image_digest(cfg.image)
        if "@sha256:" not in candidate:
            raise click.ClickException(f"Could not pin {candidate} to a content digest; refusing to decide a tag.")
        click.echo(f"== preflight: {cfg.name} on {candidate} ==", err=True)
        results.append(_preflight(cfg, name, candidate))

    click.echo(summarize(results))
    if blocks_rollout(results):
        raise click.ClickException("This image would not register everywhere. Do not roll it out.")


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
@click.option(
    "--fast",
    is_flag=True,
    default=False,
    help="Build with the Rust `fast` profile (no LTO, parallel codegen) for a much "
    "quicker build. For dev/test clusters; the production `release` profile is the default.",
)
def rollout_cmd(name: str, auto_rollback: bool, force: bool, build: bool, fast: bool) -> None:
    """Roll forward to the digest pinned from cfg.image; capture the previous digest."""
    cfg = load_finelog_config(name)
    _require_gcp(cfg)

    click.echo(f"== rollout: {cfg.name} ==")

    if build:
        cargo_profile = "fast" if fast else "release"
        click.echo(f"building & pushing {cfg.image} (cargo profile: {cargo_profile})...")
        build_finelog_image(image=cfg.image, cargo_profile=cargo_profile)

    old_digest = _running_repo_digest(cfg)
    if old_digest is None:
        click.echo(
            "warning: no running container or digest unavailable; auto-rollback disabled.",
            err=True,
        )
        auto_rollback = False
    else:
        click.echo(f"captured running digest: {old_digest}")

    new_digest = resolve_image_digest(cfg.image)
    if "@sha256:" not in new_digest:
        raise click.ClickException(f"Could not pin {cfg.image} to a content digest; refusing to deploy a mutable tag.")
    click.echo(f"new pinned digest:       {new_digest}")

    if old_digest == new_digest and not force:
        click.echo("new digest matches running digest; nothing to do (pass --force to redeploy).")
        return
    if old_digest == new_digest:
        click.echo("--force: redeploying the same digest.")

    # Decide the deploy before it touches the host. A schema the catalog rejects
    # is invisible to every gate below it: the bootstrap's health poll, the
    # post-bootstrap probe, and the rollback trigger all pass while the wedged
    # namespace rejects every write.
    click.echo("pre-flight: deciding this image against the registered schemas...")
    result = _preflight(cfg, name, new_digest)
    click.echo(summarize([result]))
    if blocks_rollout([result]):
        raise click.ClickException(f"Refusing to roll {cfg.name} onto {new_digest}: its schemas would not register.")

    state_path = _write_state(
        cfg,
        previous_digest=old_digest,
        attempted_digest=new_digest,
        rollout_started_at=_now(),
    )
    click.echo(f"state recorded at {state_path}")

    click.echo("re-running bootstrap with new image...")
    # The remote bootstrap polls /health itself and exits non-zero on a
    # crash-loop or timeout. Treat that exactly like a failed post-bootstrap
    # health check — both flow into the auto-rollback path below. (Previously a
    # non-zero bootstrap raised here, short-circuiting the rollback entirely.)
    healthy = _bootstrap_with_image(cfg, new_digest)
    if healthy:
        click.echo("waiting for /health...")
        healthy = _verify_health(cfg)

    if healthy:
        _write_state(cfg, current_digest=new_digest, rollout_succeeded_at=_now())
        click.echo(f"OK — {cfg.name} healthy on {new_digest}")
        if old_digest:
            click.echo(f"rollback target preserved: {old_digest}")
        golden = _record_golden(cfg, name, new_digest)
        if golden is not None:
            click.echo(f"deploy golden written to {golden}; commit it so CI decides against it.")
        return

    click.echo("FAIL — finelog did not become healthy on the new image.", err=True)
    if not auto_rollback or old_digest is None:
        raise click.ClickException(
            "Health check failed. Run `safe_deploy rollback <name>` (optionally with --to) to recover."
        )

    click.echo(f"auto-rolling back to {old_digest}...", err=True)
    if not _bootstrap_with_image(cfg, old_digest) or not _verify_health(cfg):
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

    if not _bootstrap_with_image(cfg, to_digest) or not _verify_health(cfg):
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
