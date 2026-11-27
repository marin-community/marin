# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decision logic for a staged Finelog forwarding rollout.

Two kinds of verdict:

- The **pre-flight gate**: given a server's deployed build commit and the commit
  about to be deployed, decide from this clone's git history alone whether a binary rollback stays
  possible (no catalog migration in between) and whether the pod's environment
  Secret must be re-synced first (a secret-bearing config field changed).
- The **After checks**: given the hub's forwarding counters and freshness rows,
  already queried by the caller, decide per sender whether forwarding is healthy
  after a restart.

``lib/finelog/scripts/deploy_forwarding.py`` reads the servers and the hub and calls these.
"""

import re
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

CATALOG_MIGRATIONS_DIR = "lib/finelog/rust/src/store/catalog/migrations/"
STORE_SOURCE_DIR = "lib/finelog/rust/src/store"
# Config fields whose values end up in a sender's environment Secret.
SECRET_FIELDS_PATTERN = re.compile(r"signing_key|remote_log_dir|object_storage_endpoint")

BUCKET_MS = 15 * 60 * 1000
ROW_STALE_WARN_SECONDS = 120
ROW_STALE_FAIL_SECONDS = 300
# Senders report counters every ~5 minutes, so an older counter means a missed report.
COUNTER_STALE_FAIL_SECONDS = 7 * 60
# Retries during a restart are expected; retries this recent mean it hasn't recovered.
GRACE_MINUTES = 15
RETRYABLE_GRACE_MS = GRACE_MINUTES * 60 * 1000
# Outcomes that never clear by themselves.
PERMANENT_OUTCOMES = frozenset({"permanent_rejection", "schema_conflict", "retention_eviction"})


class Status(StrEnum):
    OK = "ok"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class Finding:
    status: Status
    subject: str
    message: str


def worst(findings: list[Finding]) -> Status:
    statuses = {f.status for f in findings}
    if Status.FAIL in statuses:
        return Status.FAIL
    if Status.WARN in statuses:
        return Status.WARN
    return Status.OK


# ----------------------------------------------------------------------------- gate


class BuildState(StrEnum):
    UNKNOWN = "UNKNOWN"  # the server reported no build commit
    NOT_FETCHED = "not-fetched"  # the commit is not in this clone
    KNOWN = "known"


GATE_COLUMNS = ("server", "deployed", "on-main", "has-last-mig", "migration-in-range", "sync-secret")


@dataclass(frozen=True)
class GateRow:
    """One server's pre-flight verdict against the commit being deployed."""

    server: str
    deployed: str | None
    state: BuildState
    on_main: bool | None = None
    has_last_migration: bool | None = None
    migrations_differ: bool | None = None
    # None for servers without an environment Secret (GCE).
    secret_sync_needed: bool | None = None

    @property
    def roll_forward_only(self) -> bool:
        """Whether the older binary could no longer open the catalog once the target boots."""
        return self.state is BuildState.KNOWN and bool(self.migrations_differ or not self.has_last_migration)

    def blockers(self, *, secret_synced: bool = False, roll_forward_ok: bool = False) -> list[str]:
        """Reasons a rollout of this server must not start. Empty means clear.

        ``secret_synced`` records that the operator re-synced the environment Secret
        after its inputs changed; ``roll_forward_ok`` accepts a deploy that can't be
        rolled back.
        """
        if self.state is BuildState.UNKNOWN:
            return ["the server reports no build commit, so rollback safety can't be checked"]
        if self.state is BuildState.NOT_FETCHED:
            return [f"deployed commit {self.deployed} is not in this clone; fetch it and rerun"]
        reasons = []
        if self.roll_forward_only and not roll_forward_ok:
            reasons.append(
                "the catalog migrations differ between the deployed and target builds, so rollback stops working "
                "once the new binary boots; rerun with --roll-forward-only to accept that"
            )
        if self.secret_sync_needed and not secret_synced:
            reasons.append(
                "a secret-bearing config field changed: run `deploy_forwarding.py secret-presence "
                f"{self.server}`, then `finelog deploy sync-secret {self.server}`, then rerun with --secret-synced"
            )
        return reasons

    def cells(self) -> tuple[str, ...]:
        deployed = (self.deployed or "UNKNOWN")[:10]
        if self.state is not BuildState.KNOWN:
            return (self.server, deployed, self.state.value, "?", "?", "?")
        secret = "n/a" if self.secret_sync_needed is None else ("YES" if self.secret_sync_needed else "no")
        return (
            self.server,
            deployed,
            "yes" if self.on_main else "no",
            "yes" if self.has_last_migration else "no",
            "YES" if self.migrations_differ else "none",
            secret,
        )


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)


def _git_out(repo: Path, *args: str) -> str:
    result = _git(repo, *args)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def resolve_commit(repo: Path, rev: str) -> str | None:
    """Full SHA for ``rev``, or None when this clone doesn't have it."""
    result = _git(repo, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
    return result.stdout.strip() if result.returncode == 0 else None


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return _git(repo, "merge-base", "--is-ancestor", ancestor, descendant).returncode == 0


def last_migration_commit(repo: Path, target: str) -> str:
    """The newest commit on ``target``'s history that touched the catalog migrations."""
    commit = _git_out(repo, "log", "-1", "--format=%H", target, "--", CATALOG_MIGRATIONS_DIR)
    if not commit:
        raise RuntimeError(f"no commit under {CATALOG_MIGRATIONS_DIR} in the history of {target}")
    return commit


def secret_fields_changed(repo: Path, deployed: str, target: str, config_path: str) -> bool:
    """Whether a changed line in ``config_path`` between the two commits names a secret-bearing field."""
    diff = _git_out(repo, "diff", deployed, target, "--", config_path)
    for line in diff.splitlines():
        if line.startswith(("+++", "---")) or not line.startswith(("+", "-")):
            continue
        if SECRET_FIELDS_PATTERN.search(line):
            return True
    return False


def gate_row(
    repo: Path,
    *,
    server: str,
    deployed: str | None,
    target: str,
    main_ref: str,
    config_path: str | None,
) -> GateRow:
    """Assess one server. ``config_path`` is set only for servers with an environment Secret.

    Ancestry alone misses a build deployed from a branch before it merged, so the
    migration check compares trees: any difference in the migrations directory,
    in either direction, means the older binary can't open the newer catalog.
    """
    if not deployed:
        return GateRow(server=server, deployed=None, state=BuildState.UNKNOWN)
    commit = resolve_commit(repo, deployed)
    if commit is None:
        return GateRow(server=server, deployed=deployed, state=BuildState.NOT_FETCHED)
    return GateRow(
        server=server,
        deployed=deployed,
        state=BuildState.KNOWN,
        on_main=is_ancestor(repo, commit, main_ref),
        has_last_migration=is_ancestor(repo, last_migration_commit(repo, target), commit),
        migrations_differ=_git(repo, "diff", "--quiet", commit, target, "--", CATALOG_MIGRATIONS_DIR).returncode != 0,
        secret_sync_needed=None if config_path is None else secret_fields_changed(repo, commit, target, config_path),
    )


def runs_target(repo: Path, deployed: str | None, target: str) -> bool:
    """Whether a server's reported build commit is ``target``."""
    commit = resolve_commit(repo, deployed) if deployed else None
    return commit is not None and commit == resolve_commit(repo, target)


def store_commits_in_range(repo: Path, deployed: str, target: str) -> int:
    """Commits on ``target`` but not ``deployed`` that touch the store, where object-state formats live."""
    out = _git_out(repo, "log", "--format=%H", f"{deployed}..{target}", "--", STORE_SOURCE_DIR)
    return len(out.splitlines())


def order_blockers(repo: Path, order: list[str], server: str, running: dict[str, str | None], target: str) -> list[str]:
    """Servers earlier in ``order`` than ``server`` that don't yet run ``target``."""
    return [
        f"{earlier} runs {running.get(earlier) or 'UNKNOWN'}, not the target; deploy it first"
        for earlier in order[: order.index(server)]
        if not runs_target(repo, running.get(earlier), target)
    ]


# ---------------------------------------------------------------------- after checks


@dataclass(frozen=True)
class FailureRow:
    cluster: str
    outcome: str
    batches: int
    last_seen_ms: int


@dataclass(frozen=True)
class VolumeRow:
    bucket_ms: int
    cluster: str
    seq_positions: int


def assess_failures(rows: list[FailureRow], end_ms: int) -> list[Finding]:
    """Check A: permanent outcomes fail; retries warn while they continue and pass once they stop."""
    findings = []
    for row in rows:
        if row.outcome in PERMANENT_OUTCOMES:
            findings.append(Finding(Status.FAIL, row.cluster, f"{row.batches} {row.outcome} batches"))
        elif row.last_seen_ms >= end_ms - RETRYABLE_GRACE_MS:
            findings.append(
                Finding(
                    Status.WARN,
                    row.cluster,
                    f"{row.outcome} in the last {GRACE_MINUTES} minutes ({row.batches} batches)",
                )
            )
        else:
            findings.append(Finding(Status.OK, row.cluster, f"{row.outcome} ended over {GRACE_MINUTES} minutes ago"))
    return findings


def last_complete_buckets(end_ms: int, count: int = 2) -> list[int]:
    """Start times of the newest ``count`` 15-minute buckets that end at or before ``end_ms``, oldest first."""
    newest = (end_ms // BUCKET_MS) * BUCKET_MS - BUCKET_MS
    return [newest - i * BUCKET_MS for i in reversed(range(count))]


def assess_volume(rows: list[VolumeRow], senders: list[str], end_ms: int) -> list[Finding]:
    """Check B: a sender with no accepted rows in the last two complete buckets has stopped forwarding."""
    previous, latest = last_complete_buckets(end_ms)
    by_key = {(row.cluster, row.bucket_ms): row.seq_positions for row in rows}
    findings = []
    for sender in senders:
        before, last = by_key.get((sender, previous), 0), by_key.get((sender, latest), 0)
        message = f"accepted seq positions in the last two complete buckets: {before:,}, {last:,}"
        if before == 0 and last == 0:
            findings.append(Finding(Status.FAIL, sender, message))
        elif last == 0:
            findings.append(Finding(Status.WARN, sender, message))
        else:
            findings.append(Finding(Status.OK, sender, message))
    return findings


def assess_staleness(
    staleness: dict[str, float],
    senders: list[str],
    *,
    what: str,
    warn_seconds: float,
    fail_seconds: float,
) -> list[Finding]:
    """Checks C and C2: how far each sender's newest row on the hub lags the hub's clock."""
    findings = []
    for sender in senders:
        age = staleness.get(sender)
        if age is None:
            findings.append(Finding(Status.FAIL, sender, f"no {what} in the window"))
            continue
        status = Status.FAIL if age > fail_seconds else Status.WARN if age > warn_seconds else Status.OK
        findings.append(Finding(status, sender, f"newest {what} is {age:.0f} s old"))
    return findings
