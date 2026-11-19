#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Staged deploy of the Finelog servers that take part in cross-cluster forwarding.

Each invocation runs one step, in the order marin-dev, then the hub (``marin``),
then the senders cw-us-east-08a, cw-us-east-02a and cw-rno2a:

    uv run python lib/finelog/scripts/deploy_forwarding.py preflight [SERVER...]
    uv run --all-packages --extra deploy python lib/finelog/scripts/deploy_forwarding.py rollout SERVER [--dry-run]
    uv run python lib/finelog/scripts/deploy_forwarding.py after [SERVER] [--minutes 60]

``preflight`` reads each server's running build and checks it against this
checkout with git: whether a binary rollback stays possible and whether a sender
needs its environment Secret re-synced. ``rollout`` gates on that, prints the
plan and the rollback command, waits for a typed ``yes``, deploys through
``safe_deploy.py`` (GCE) or ``marin-deploy finelog rollout`` (Kubernetes), then
runs the After checks. ``after`` runs the hub's forwarding health queries,
one at a time and each bounded in time.

``secret-presence`` is for the operator to run: it reports which environment
Secret keys exist, never a value.
"""

import json
import os
import subprocess
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import click
from finelog.deploy._k8s import kube_flags
from finelog.deploy.config import FinelogConfig, k8s_env_secret_name, load_finelog_config, tunnel_target_for
from finelog.deploy.connect import open_client
from finelog.deploy.image import resolve_image_digest
from finelog.deploy.rollout_checks import (
    BUCKET_MS,
    COUNTER_STALE_FAIL_SECONDS,
    GATE_COLUMNS,
    ROW_STALE_FAIL_SECONDS,
    ROW_STALE_WARN_SECONDS,
    STORE_SOURCE_DIR,
    FailureRow,
    Finding,
    GateRow,
    Status,
    VolumeRow,
    assess_failures,
    assess_staleness,
    assess_volume,
    gate_row,
    is_ancestor,
    order_blockers,
    runs_target,
    store_commits_in_range,
    worst,
)
from finelog.errors import StatsError
from rigging.tunnel import open_tunnel

REPO_ROOT = Path(__file__).resolve().parents[3]
SAFE_DEPLOY = Path(__file__).resolve().with_name("safe_deploy.py").relative_to(REPO_ROOT)
THIS_SCRIPT = Path(__file__).resolve().relative_to(REPO_ROOT)
CONFIG_DIR = "lib/finelog/config"
MARIN_DEPLOY_FINELOG = ("uv", "run", "--all-packages", "--extra", "deploy", "marin-deploy", "finelog")
DEPLOY_ORDER = ("marin-dev", "marin", "cw-us-east-08a", "cw-us-east-02a", "cw-rno2a")
HUB = "marin"
MAIN_REF = "origin/main"
# The hub takes the image already verified on marin-dev rather than rebuilding it.
PROMOTED_FROM = {"marin": "marin-dev"}
# Longer than the hub's own 30 s query deadline, so a slow query fails with the server's reason.
QUERY_TIMEOUT = 60.0
# A healthy hub answers the freshness query in about 2 s.
QUERY_LATENCY_WARN_SECONDS = 10.0
FRESHNESS_WINDOW_MS = 60 * 60 * 1000
COUNTER_WINDOW_MS = 15 * 60 * 1000
# Rows stamped a little after this process's clock still count as fresh.
CLOCK_SLACK_MS = 60 * 1000
TUNNEL_TIMEOUT = 60.0
AFTER_WINDOW_AFTER_ROLLOUT_MINUTES = 45
FORWARDER_ERRORS = (
    "hub permanently rejected the batch",
    "refusing to skip rows",
    "batch conflicts with the hub schema",
)
FORWARDER_LOG_WINDOW = "15m"
K8S_ROLLOUT_VERBS = (("patch", "deployments"), ("create", "pods/exec"))
GHCR_LOGIN_DISPLAY = 'gh auth token | docker login ghcr.io -u "$(gh api user -q .login)" --password-stdin'


def _is_k8s(server: str) -> bool:
    return load_finelog_config(server).deployment.k8s is not None


def _senders() -> list[str]:
    configs = [load_finelog_config(server) for server in DEPLOY_ORDER]
    return [cfg.forwarding.cluster for cfg in configs if cfg.forwarding is not None]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------------ reading builds


@dataclass(frozen=True)
class ServerBuild:
    """A server's ``/api/server`` ``.build`` block; ``raw`` is None when it couldn't be read."""

    raw: dict | None

    @property
    def commit(self) -> str | None:
        return (self.raw or {}).get("commit") or None


def running_build(server: str) -> ServerBuild:
    """Read the server's build through the same SSH or port-forward tunnel `finelog query` uses.

    The servers admit only local callers, so the request has to arrive through a tunnel.
    """
    cfg = load_finelog_config(server)
    try:
        with open_tunnel(tunnel_target_for(cfg), timeout=TUNNEL_TIMEOUT) as url:
            with urllib.request.urlopen(f"{url}/api/server", timeout=30) as response:
                return ServerBuild(json.load(response).get("build"))
    except (OSError, TimeoutError, RuntimeError) as error:
        click.echo(f"could not read {server}'s /api/server: {error}", err=True)
        return ServerBuild(None)


def _read_builds(servers: list[str]) -> dict[str, ServerBuild]:
    builds = {}
    for server in servers:
        build = running_build(server)
        click.echo(f"== {server}: {json.dumps(build.raw) if build.raw else '<no build info>'}")
        builds[server] = build
    return builds


def _gate(server: str, build: ServerBuild, target: str) -> GateRow:
    cfg = load_finelog_config(server)
    return gate_row(
        REPO_ROOT,
        server=server,
        deployed=build.commit,
        target=target,
        main_ref=MAIN_REF,
        config_path=f"{CONFIG_DIR}/{server}.yaml" if k8s_env_secret_name(cfg) else None,
    )


def _print_gate_table(rows: list[GateRow]) -> None:
    widths = (16, 12, 12, 13, 19, 11)
    click.echo("  ".join(column.ljust(width) for column, width in zip(GATE_COLUMNS, widths, strict=True)))
    for row in rows:
        click.echo("  ".join(cell.ljust(width) for cell, width in zip(row.cells(), widths, strict=True)))


# --------------------------------------------------------------------- deploy steps


def _builds_image(server: str) -> bool:
    return server not in PROMOTED_FROM


def deploy_argv(server: str) -> list[str]:
    if _is_k8s(server):
        return [*MARIN_DEPLOY_FINELOG, "rollout", server]
    argv = ["uv", "run", "python", str(SAFE_DEPLOY), "rollout", server]
    if not _builds_image(server):
        argv.append("--no-build")
    return argv


def rollback_command(server: str) -> str:
    if _is_k8s(server):
        return " ".join([*MARIN_DEPLOY_FINELOG, "rollback", server])
    return f"uv run python {SAFE_DEPLOY} rollback {server}"


NO_ROLLBACK = (
    "none. Once the new binary boots it migrates the catalog, and the older binary can't open it. "
    "Recover by deploying a corrected descendant against the same remote root."
)


def recovery_line(server: str, roll_forward_only: bool) -> str:
    return NO_ROLLBACK if roll_forward_only else rollback_command(server)


def docker_login_ghcr() -> None:
    """Log docker in to ghcr.io with the gh token.

    The token goes from ``gh auth token`` to ``docker login`` through an OS pipe,
    so it never passes through this process's memory or output.
    """
    user = subprocess.run(["gh", "api", "user", "-q", ".login"], check=True, capture_output=True, text=True)
    token = subprocess.Popen(["gh", "auth", "token"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert token.stdout is not None
    try:
        login = subprocess.run(
            ["docker", "login", "ghcr.io", "-u", user.stdout.strip(), "--password-stdin"],
            stdin=token.stdout,
            capture_output=True,
            text=True,
        )
    finally:
        token.stdout.close()
        token.wait()
    if token.returncode != 0 or login.returncode != 0:
        raise click.ClickException("docker login to ghcr.io failed; run `gh auth status` and retry")
    click.echo("docker is logged in to ghcr.io")


def _gce_running_digest(server: str) -> str | None:
    result = subprocess.run(
        ["uv", "run", "python", str(SAFE_DEPLOY), "status", server],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("running digest:"):
            digest = line.removeprefix("running digest:").strip()
            return digest if "@sha256:" in digest else None
    return None


def _promotion_blockers(server: str) -> list[str]:
    """The promoted image tag must still point at what its source server runs."""
    source = PROMOTED_FROM.get(server)
    if source is None:
        return []
    tag = load_finelog_config(server).image
    tagged = resolve_image_digest(tag)
    running = _gce_running_digest(source)
    click.echo(f"{tag} -> {tagged}")
    click.echo(f"{source} runs {running or '<unavailable>'}")
    if running is None or tagged != running:
        return [f"{tag} is no longer the image {source} runs; redeploy {source} first"]
    return []


def _kube_blockers(server: str) -> list[str]:
    cfg = load_finelog_config(server)
    k8s = cfg.deployment.k8s
    assert k8s is not None
    blockers = []
    if k8s.kubeconfig:
        wanted = Path(k8s.kubeconfig).expanduser().resolve()
        ambient = os.environ.get("KUBECONFIG")
        if not ambient or Path(ambient).expanduser().resolve() != wanted:
            blockers.append(f"the Pulumi rollout needs KUBECONFIG={wanted}; run `export KUBECONFIG={k8s.kubeconfig}`")
    for verb, resource in K8S_ROLLOUT_VERBS:
        result = subprocess.run(
            ["kubectl", *kube_flags(cfg), "-n", k8s.namespace, "auth", "can-i", verb, resource],
            capture_output=True,
            text=True,
        )
        click.echo(f"can-i {verb} {resource}: {result.stdout.strip() or 'no answer'}")
        if result.stdout.strip() != "yes":
            blockers.append(f"kubectl may not {verb} {resource} in {k8s.kube_context}")
    return blockers


def forwarder_errors(server: str) -> list[str]:
    """Forwarder error lines from the sender's own log."""
    cfg = load_finelog_config(server)
    assert cfg.deployment.k8s is not None
    result = subprocess.run(
        [
            "kubectl",
            *kube_flags(cfg),
            "-n",
            cfg.deployment.k8s.namespace,
            "logs",
            f"deployment/{cfg.name}",
            f"--since={FORWARDER_LOG_WINDOW}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if any(error in line for error in FORWARDER_ERRORS)]


def _report_forwarder_errors(server: str) -> list[Finding]:
    errors = forwarder_errors(server)
    click.echo(f"\n== {server} log: {len(errors)} forwarder error lines in the last {FORWARDER_LOG_WINDOW}")
    for line in errors[-20:]:
        click.echo(f"  {line}")
    return [Finding(Status.FAIL, server, f"{len(errors)} forwarder error lines")] if errors else []


def _plan_steps(server: str) -> list[str]:
    steps = [
        GHCR_LOGIN_DISPLAY if _builds_image(server) else f"(no build: reuse the image {PROMOTED_FROM[server]} runs)",
        " ".join(deploy_argv(server)),
    ]
    if _is_k8s(server):
        steps[-1] += (
            "\n     Pulumi asks for confirmation. Expected changes: the image (built from this checkout), "
            "an update to the Deployment (image and source-revision annotation), and a replace of `verify` "
            "(the /health check). Answer no to a delete or replace of anything else (Service, PVC, "
            "Namespace)."
        )
    steps.append("confirm the server reports this checkout's build commit")
    if _is_k8s(server):
        steps.append("check the sender's log for forwarder errors")
    steps.append("wait, then run the After checks")
    return steps


def _plan_notes(row: GateRow, target: str, secret_synced: bool) -> list[str]:
    notes = []
    if row.secret_sync_needed and secret_synced:
        secret = k8s_env_secret_name(load_finelog_config(row.server))
        notes.append(f"You confirmed the {secret} Secret was re-synced after its inputs changed.")
    if row.roll_forward_only:
        notes.append("WARN: roll-forward only. You accepted with --roll-forward-only; you'll be asked to confirm again.")
    elif row.deployed is not None:
        count = store_commits_in_range(REPO_ROOT, row.deployed, target)
        notes.append(
            f"{count} store commits in range; skim them for an object-state format change, which rollback "
            f"can't cross (infra/finelog/README.md): "
            f"git log --oneline {row.deployed}..{target[:10]} -- {STORE_SOURCE_DIR}"
        )
    return notes


def _rollout_blockers(
    server: str, target: str, *, allow_out_of_order: bool, secret_synced: bool, roll_forward_ok: bool
) -> tuple[GateRow, list[str]]:
    """The server's gate row, and everything that must hold before it is deployed from ``target``."""
    order = list(DEPLOY_ORDER)
    builds = _read_builds(order[: order.index(server) + 1])
    row = _gate(server, builds[server], target)
    _print_gate_table([row])
    blockers = row.blockers(secret_synced=secret_synced, roll_forward_ok=roll_forward_ok)
    if runs_target(REPO_ROOT, row.deployed, target):
        blockers.append("it already runs this checkout's build")
    running = {name: build.commit for name, build in builds.items()}
    out_of_order = order_blockers(REPO_ROOT, order, server, running, target)
    if allow_out_of_order:
        for reason in out_of_order:
            click.echo(f"warning: {reason}")
    else:
        blockers.extend(out_of_order)
    blockers.extend(_promotion_blockers(server))
    if _is_k8s(server):
        blockers.extend(_kube_blockers(server))
    return row, blockers


def _deploy(server: str, target: str, recovery: str) -> None:
    """Run the deploy and confirm the server reports ``target``'s build."""
    if _builds_image(server):
        docker_login_ghcr()
    deploy = subprocess.run(deploy_argv(server), cwd=REPO_ROOT)
    if deploy.returncode != 0:
        raise click.ClickException(
            f"the deploy failed (exit {deploy.returncode}); its own output says whether it restored the old "
            f"version. Otherwise, rollback: {recovery}"
        )
    deployed = running_build(server).commit
    if not runs_target(REPO_ROOT, deployed, target):
        raise click.ClickException(
            f"{server} reports build {deployed or 'UNKNOWN'}, not {target[:10]}. Rollback: {recovery}"
        )
    click.echo(f"{server} runs {deployed}")


# --------------------------------------------------------------------- after checks


@dataclass(frozen=True)
class AfterCheck:
    label: str
    title: str
    sql: str
    assess: Callable[[list[dict]], list[Finding]]
    # A failed freshness query says the hub is slow, not that forwarding broke; C2 covers it.
    error_status: Status = Status.FAIL


def _ages(rows: list[dict]) -> dict[str, float]:
    return {row["cluster"]: float(row["age_seconds"]) for row in rows}


def after_checks(senders: list[str], start_ms: int, end_ms: int) -> list[AfterCheck]:
    accepted = "json_get(attributes_json, 'outcome') = 'accepted'"
    sender_list = ", ".join(f"'{sender}'" for sender in senders)
    return [
        AfterCheck(
            "A",
            "Failed or rejected forwarding batches",
            f"""
SELECT "cluster", json_get(attributes_json, 'outcome') AS outcome,
       CAST(sum(value) AS BIGINT) AS batches, max(timestamp_ms) AS last_seen_ms
FROM "telemetry_v1.finelog"
WHERE name = 'forwarding_batches' AND json_get(attributes_json, 'outcome') <> 'accepted'
  AND timestamp_ms >= {start_ms} AND timestamp_ms < {end_ms}
GROUP BY 1, 2 ORDER BY 1, 2""",
            lambda rows: assess_failures(
                [FailureRow(r["cluster"], r["outcome"], r["batches"], r["last_seen_ms"]) for r in rows], end_ms
            ),
        ),
        AfterCheck(
            "B",
            "Accepted volume per sender per 15 minutes",
            f"""
SELECT (timestamp_ms / {BUCKET_MS}) * {BUCKET_MS} AS bucket_ms, "cluster",
       CAST(sum(CASE WHEN name = 'forwarding_seq_positions' THEN value ELSE 0 END) AS BIGINT) AS seq_positions,
       CAST(sum(CASE WHEN name = 'forwarding_batches' THEN value ELSE 0 END) AS BIGINT) AS batches
FROM "telemetry_v1.finelog"
WHERE name IN ('forwarding_seq_positions', 'forwarding_batches') AND {accepted}
  AND timestamp_ms >= {start_ms} AND timestamp_ms < {end_ms}
GROUP BY 1, 2 ORDER BY 1, 2""",
            lambda rows: assess_volume(
                [VolumeRow(r["bucket_ms"], r["cluster"], r["seq_positions"]) for r in rows],
                senders,
                end_ms,
            ),
        ),
        AfterCheck(
            "C",
            "Freshness: newest forwarded iris.task_state row per sender (also times the hub)",
            f"""
SELECT "cluster", to_unixtime(now()) - to_unixtime(max(ts)) AS age_seconds
FROM "iris.task_state"
WHERE "cluster" IN ({sender_list})
  AND ts >= to_timestamp_millis({end_ms - FRESHNESS_WINDOW_MS})
  AND ts < to_timestamp_millis({end_ms + CLOCK_SLACK_MS})
GROUP BY 1 ORDER BY 1""",
            lambda rows: assess_staleness(
                _ages(rows),
                senders,
                what="forwarded row",
                warn_seconds=ROW_STALE_WARN_SECONDS,
                fail_seconds=ROW_STALE_FAIL_SECONDS,
            ),
            error_status=Status.WARN,
        ),
        AfterCheck(
            "C2",
            "Newest accepted forwarding counter per sender",
            f"""
SELECT "cluster", to_unixtime(now()) - max(timestamp_ms) / 1000 AS age_seconds
FROM "telemetry_v1.finelog"
WHERE name = 'forwarding_batches' AND {accepted}
  AND timestamp_ms >= {end_ms - COUNTER_WINDOW_MS} AND timestamp_ms < {end_ms + CLOCK_SLACK_MS}
GROUP BY 1 ORDER BY 1""",
            lambda rows: assess_staleness(
                _ages(rows),
                senders,
                what="forwarding counter",
                warn_seconds=COUNTER_STALE_FAIL_SECONDS,
                fail_seconds=COUNTER_STALE_FAIL_SECONDS,
            ),
        ),
    ]


def _print_rows(rows: list[dict]) -> None:
    if not rows:
        click.echo("  (no rows)")
        return
    shown = [
        (
            {"bucket_utc": _utc(row["bucket_ms"]), **{k: v for k, v in row.items() if k != "bucket_ms"}}
            if "bucket_ms" in row
            else row
        )
        for row in rows
    ]
    columns = list(shown[0])
    widths = {c: max(len(c), *(len(str(row[c])) for row in shown)) for c in columns}
    click.echo("  " + "  ".join(c.ljust(widths[c]) for c in columns))
    for row in shown:
        click.echo("  " + "  ".join(str(row[c]).ljust(widths[c]) for c in columns))


def _utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%H:%M")


def run_after(minutes: int, server: str | None) -> Status:
    """Run the After checks on the hub, one at a time: the hub admits only two queries at once."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - minutes * 60_000
    click.echo(f"After checks on {HUB}, {_utc(start_ms)}-{_utc(end_ms)} UTC")
    findings: list[Finding] = []
    with open_client(load_finelog_config(HUB), HUB, request_timeout=QUERY_TIMEOUT) as client:
        for check in after_checks(_senders(), start_ms, end_ms):
            started = time.monotonic()
            try:
                rows = client.query(check.sql).to_pylist()
            except StatsError as error:
                click.echo(f"\n== {check.label}. {check.title} ({time.monotonic() - started:.1f} s)")
                click.echo(f"  error: {error}")
                findings.append(Finding(check.error_status, f"check {check.label}", str(error)))
                continue
            seconds = time.monotonic() - started
            click.echo(f"\n== {check.label}. {check.title} ({seconds:.1f} s)")
            _print_rows(rows)
            findings.extend(check.assess(rows))
            if check.label == "C" and seconds > QUERY_LATENCY_WARN_SECONDS:
                findings.append(Finding(Status.WARN, "hub", f"the freshness query took {seconds:.1f} s"))
    if server is not None and _is_k8s(server):
        findings.extend(_report_forwarder_errors(server))
    click.echo("\nVerdict:")
    for finding in findings:
        click.echo(f"  {finding.status.value:4}  {finding.subject}: {finding.message}")
    overall = worst(findings)
    click.echo(f"\nAfter checks: {overall.value}")
    return overall


# -------------------------------------------------------------------------- commands


@click.group()
def cli() -> None:
    """Staged deploy of the Finelog forwarding servers, one step per invocation."""


@cli.command("preflight")
@click.argument("servers", nargs=-1, type=click.Choice(DEPLOY_ORDER))
def preflight_cmd(servers: tuple[str, ...]) -> None:
    """Read each server's running build and check rollback safety against this checkout."""
    names = list(servers) or list(DEPLOY_ORDER)
    _git("fetch", "-q", "origin")
    target = _git("rev-parse", "HEAD")
    builds = _read_builds(names)
    rows = [_gate(server, builds[server], target) for server in names]
    click.echo(f"\ndeploying: {target}")
    _print_gate_table(rows)
    blocked = [(row.server, reason) for row in rows for reason in row.blockers()]
    for server, reason in blocked:
        click.echo(f"STOP {server}: {reason}")
    pending = [row.server for row in rows if not runs_target(REPO_ROOT, row.deployed, target)]
    if pending and not servers:
        click.echo(f"\nnext step: rollout {pending[0]}")
    if blocked:
        raise SystemExit(1)


@cli.command("rollout")
@click.argument("server", type=click.Choice(DEPLOY_ORDER))
@click.option("--dry-run", is_flag=True, help="Run the gate and print the plan; change nothing and run no hub queries.")
@click.option(
    "--allow-out-of-order",
    is_flag=True,
    help="Warn instead of stopping when an earlier server in the order doesn't run this checkout.",
)
@click.option(
    "--secret-synced",
    is_flag=True,
    help="Confirm the sender's environment Secret was re-synced after a secret-bearing config change.",
)
@click.option(
    "--roll-forward-only",
    "roll_forward_ok",
    is_flag=True,
    help="Accept a catalog migration that makes this deploy impossible to roll back.",
)
@click.option("--wait-minutes", type=int, default=15, show_default=True, help="Wait before the After checks.")
def rollout_cmd(
    server: str,
    dry_run: bool,
    allow_out_of_order: bool,
    secret_synced: bool,
    roll_forward_ok: bool,
    wait_minutes: int,
) -> None:
    """Gate, confirm, deploy one server, then run the After checks."""
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise click.ClickException("the checkout has uncommitted or untracked files; `dirty` would be stamped")
    _git("fetch", "-q", "origin")
    target = _git("rev-parse", "HEAD")
    if not is_ancestor(REPO_ROOT, target, MAIN_REF):
        raise click.ClickException(f"HEAD {target[:10]} is not on {MAIN_REF}; deploy from a main checkout")
    click.echo(f"== rollout {server} from {target[:10]}{' (dry run)' if dry_run else ''}")

    row, blockers = _rollout_blockers(
        server,
        target,
        allow_out_of_order=allow_out_of_order,
        secret_synced=secret_synced,
        roll_forward_ok=roll_forward_ok,
    )
    if blockers:
        for reason in blockers:
            click.echo(f"STOP: {reason}", err=True)
        raise SystemExit(1)

    click.echo("\nPlan:")
    for number, step in enumerate(_plan_steps(server), start=1):
        click.echo(f"  {number}. {step}")
    for note in _plan_notes(row, target, secret_synced):
        click.echo(f"\n{note}")
    recovery = recovery_line(server, row.roll_forward_only)
    click.echo(f"\nRollback: {recovery}\n")
    if dry_run:
        click.echo("dry run: nothing changed.")
        # The After checks take the hub's query slots, so the dry run leaves them to the operator.
        click.echo(f"For a 'before' snapshot: uv run python {THIS_SCRIPT} after {server}")
        return

    if click.prompt(f"Type yes to deploy {server}", default="", show_default=False) != "yes":
        raise click.ClickException("not confirmed; nothing changed")
    if row.roll_forward_only and click.prompt("Type roll-forward to continue", default="", show_default=False) != (
        "roll-forward"
    ):
        raise click.ClickException("not confirmed; nothing changed")
    _deploy(server, target, recovery)
    if _is_k8s(server):
        _report_forwarder_errors(server)
    click.echo(f"waiting {wait_minutes} minutes for a full counter bucket (Ctrl-C to check now)...")
    try:
        time.sleep(wait_minutes * 60)
    except KeyboardInterrupt:
        click.echo("checking now")
    status = run_after(AFTER_WINDOW_AFTER_ROLLOUT_MINUTES, server)
    click.echo(f"\nRollback, if the After checks say so: {recovery}")
    if status is Status.FAIL:
        raise SystemExit(1)


@cli.command("after")
@click.argument("server", required=False, type=click.Choice(DEPLOY_ORDER))
@click.option("--minutes", type=int, default=60, show_default=True, help="Window for checks A and B.")
def after_cmd(server: str | None, minutes: int) -> None:
    """Forwarding health on the hub; with a sender, also its forwarder log."""
    if run_after(minutes, server) is Status.FAIL:
        raise SystemExit(1)


# Key names only: these mirror the variables `finelog deploy sync-secret` writes.
SECRET_KEYS_S3 = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL")
SECRET_KEY_FORWARDING = "FINELOG_SIGNING_KEY"


@cli.command("secret-presence")
@click.argument("server", type=click.Choice([s for s in DEPLOY_ORDER if _is_k8s(s)]))
def secret_presence_cmd(server: str) -> None:
    """Report which Secret keys and R2 shell variables are set (the operator runs this; prints no value)."""
    cfg: FinelogConfig = load_finelog_config(server)
    k8s = cfg.deployment.k8s
    assert k8s is not None
    secret = k8s_env_secret_name(cfg)
    assert secret is not None
    for variable in ("R2_KEY_ID", "R2_KEY_SECRET"):
        click.echo(f"shell {variable}: {'set' if os.environ.get(variable) else 'missing'}")
    # The template prints key names only, so no value reaches this process.
    result = subprocess.run(
        [
            "kubectl",
            *kube_flags(cfg),
            "-n",
            k8s.namespace,
            "get",
            "secret",
            secret,
            "-o",
            'go-template={{range $k, $v := .data}}{{$k}}{{"\\n"}}{{end}}',
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise click.ClickException(f"could not read Secret {secret}'s key names")
    present = set(result.stdout.split())
    expected = [*SECRET_KEYS_S3, *([SECRET_KEY_FORWARDING] if cfg.forwarding is not None else [])]
    for key in expected:
        click.echo(f"Secret {secret} {key}: {'set' if key in present else 'missing'}")


if __name__ == "__main__":
    cli()
