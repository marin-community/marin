#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drive one table's legacy-to-object-native migration on a live deployment.

Subcommands compose into the rollout loop for one namespace:

    uv run python lib/finelog/scripts/migrate_table.py baseline marin iris.profile
    uv run python lib/finelog/scripts/migrate_table.py migrate  marin iris.profile
    uv run python lib/finelog/scripts/migrate_table.py watch    marin iris.profile
    uv run python lib/finelog/scripts/migrate_table.py validate marin iris.profile
    uv run python lib/finelog/scripts/migrate_table.py abort    marin iris.profile

``baseline`` freezes the namespace's current ``max_seq`` and records, under
``~/.cache/finelog/migration-state/<name>/<namespace>.json``:

  - the full row count at ``seq <= frozen`` (informational on tables under
    eviction pressure — the legacy live window shrinks from the oldest seq),
  - order-independent per-column digests over the newest ``--window`` seqs
    ending at the frozen mark (the hard equality contract: eviction eats the
    oldest rows first, so the freshest window survives the migration),
  - wall-time samples for three generic query shapes (count, recent fetch,
    group-by), for before/after latency comparison.

``migrate`` registers TableSpec version 1 over the deployed schema and storage
policy — the server classifies this as a version-0 import and background
maintenance runs the backfill. ``watch`` polls the migration phase until the
spec activates. ``validate`` recomputes the frozen-window digests (must match
exactly), reports full-count and latency drift, and exits non-zero on any
digest mismatch. ``abort`` calls AbortTableMigration.
"""

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import click
from finelog.client.log_client import LogClient
from finelog.deploy.cli import DEFAULT_TUNNEL_TIMEOUT, _open_cli_client
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.table_spec import TableSpec

STATE_DIR = Path.home() / ".cache" / "finelog" / "migration-state"
LATENCY_REPS = 3
DEFAULT_WINDOW = 2_000_000
DEFAULT_REQUEST_TIMEOUT = 30.0
WATCH_POLL_SECONDS = 15.0

NUMERIC_TYPES = {
    stats_pb2.COLUMN_TYPE_INT32,
    stats_pb2.COLUMN_TYPE_INT64,
    stats_pb2.COLUMN_TYPE_FLOAT64,
}
# min/max and DISTINCT are not defined for nested values; count is.
NESTED_TYPES = {
    stats_pb2.COLUMN_TYPE_FLOAT64_LIST,
    stats_pb2.COLUMN_TYPE_INT64_LIST,
    stats_pb2.COLUMN_TYPE_MAP,
}


@dataclass(frozen=True)
class MigrationBaseline:
    captured_at: str
    frozen_max_seq: int
    min_seq_at_baseline: int
    window_start: int
    bytes_window_start: int
    row_count_reported: int
    full_count_at_frozen: int
    column_digests: dict[str, dict[str, object]]
    latency_probes: dict[str, str]
    latency_ms: dict[str, list[float]]
    active_version_at_baseline: int

    @classmethod
    def read(cls, path: Path) -> "MigrationBaseline":
        return cls(**json.loads(path.read_text()))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))


def _state_path(name: str, namespace: str) -> Path:
    return STATE_DIR / name / f"{namespace}.json"


def _quoted(namespace: str) -> str:
    return '"' + namespace + '"'


def _namespace_info(client: LogClient, namespace: str):
    for info in client.list_namespaces():
        if info.namespace == namespace:
            return info
    raise click.ClickException(f"namespace {namespace!r} not found on this deployment")


def _scalar(client: LogClient, sql: str):
    table = client.query(sql)
    if table.num_rows != 1 or table.num_columns != 1:
        raise click.ClickException(f"expected one scalar from {sql!r}, got {table.num_rows}x{table.num_columns}")
    return table.column(0)[0].as_py()


def _full_count(client: LogClient, namespace: str, min_seq: int, frozen: int) -> int:
    """Count rows with ``min_seq <= seq <= frozen``, bisecting on deadline.

    One count over a multi-billion-row table exceeds the server's query
    deadline, and fixed-size seq chunks are hopeless against sparse seq spaces
    (telemetry namespaces carry sentinel min_seqs quintillions below their
    data). Try the whole range; when the server reports a deadline, split it
    and recurse — empty halves prune by segment metadata and return fast.
    """
    lo, hi = min_seq - 1, frozen
    if lo >= hi:
        return 0
    try:
        return _scalar(
            client,
            f"SELECT count(*) FROM {_quoted(namespace)} WHERE seq > {lo} AND seq <= {hi}",
        )
    except Exception as error:
        if "deadline" not in str(error).lower() or hi - lo <= 1:
            raise
    mid = lo + (hi - lo) // 2
    return _full_count(client, namespace, lo + 1, mid) + _full_count(client, namespace, mid + 1, hi)


def _column_digests(
    client: LogClient, namespace: str, window_start: int, bytes_window_start: int, frozen: int
) -> dict[str, dict]:
    """Order-independent per-column facts over ``window_start < seq <= frozen``.

    BYTES columns scan ``bytes_window_start < seq <= frozen`` instead: payload
    blobs are orders of magnitude wider than scalar columns and a full-window
    scan blows the server's query deadline.
    """
    schema = client.get_table_schema(namespace)
    digests: dict[str, dict] = {}
    where = f"WHERE seq > {window_start} AND seq <= {frozen}"
    bytes_where = f"WHERE seq > {bytes_window_start} AND seq <= {frozen}"
    for column in schema.columns:
        name = column.name
        quoted_col = '"' + name + '"'
        parts = [f"count({quoted_col}) AS non_null"]
        if column.type in NESTED_TYPES:
            pass
        elif column.type == stats_pb2.COLUMN_TYPE_BYTES:
            # DISTINCT/min/max over large payload blobs blows the server's query
            # deadline; total length still catches truncation and loss.
            parts.append(f"cast(sum(length({quoted_col})) AS varchar) AS total_bytes")
        elif column.type in NUMERIC_TYPES:
            # sum/min/max are order-independent; cast to string for stable JSON.
            parts += [
                f"cast(sum({quoted_col}) AS varchar) AS total",
                f"cast(min({quoted_col}) AS varchar) AS low",
                f"cast(max({quoted_col}) AS varchar) AS high",
            ]
        else:
            parts += [
                f"count(DISTINCT {quoted_col}) AS distinct_values",
                f"cast(min({quoted_col}) AS varchar) AS low",
                f"cast(max({quoted_col}) AS varchar) AS high",
            ]
        active_where = bytes_where if column.type == stats_pb2.COLUMN_TYPE_BYTES else where
        sql = f"SELECT {', '.join(parts)} FROM {_quoted(namespace)} {active_where}"
        table = client.query(sql)
        digests[name] = {field: table.column(field)[0].as_py() for field in table.schema.names}
    return digests


def _latency_probes(namespace: str, frozen: int) -> dict[str, str]:
    ns = _quoted(namespace)
    return {
        "recent_count": f"SELECT count(*) FROM {ns} WHERE seq > {frozen - 100_000}",
        "recent_fetch": f"SELECT * FROM {ns} WHERE seq > {frozen - 1_000} ORDER BY seq DESC LIMIT 200",
        "window_scan": f"SELECT count(*), min(seq), max(seq) FROM {ns} WHERE seq > {frozen - 1_000_000}",
    }


def _measure_latencies(client: LogClient, probes: dict[str, str]) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {}
    for label, sql in probes.items():
        runs = []
        for _ in range(LATENCY_REPS):
            start = time.monotonic()
            client.query(sql)
            runs.append(round((time.monotonic() - start) * 1000.0, 1))
        samples[label] = runs
    return samples


FLOAT_TOTAL_RELATIVE_TOLERANCE = 1e-9


def _digests_equal(expected: dict | None, actual: dict | None) -> bool:
    """Exact equality, except float ``total`` fields compare with relative
    tolerance: the rewrite re-sorts rows, so IEEE summation order differs and
    the last couple of digits legitimately move."""
    if expected is None or actual is None or expected.keys() != actual.keys():
        return expected == actual
    for field, left in expected.items():
        right = actual[field]
        if left == right:
            continue
        if field != "total" or not isinstance(left, str) or not isinstance(right, str):
            return False
        if not any(mark in left for mark in (".", "e", "E")):
            return False
        try:
            a, b = float(left), float(right)
        except ValueError:
            return False
        if abs(a - b) > FLOAT_TOTAL_RELATIVE_TOLERANCE * max(abs(a), abs(b), 1.0):
            return False
    return True


def _print_status(client: LogClient, namespace: str) -> None:
    status = client.get_table_status(namespace)
    click.echo(
        f"{namespace}: active_version={status.active_version} desired_version={status.desired_version} "
        f"phase={status.migration_phase} catalog_generation={status.catalog_generation}"
    )


@click.group()
def cli() -> None:
    """Baseline, trigger, watch, and validate one table's object-native migration."""


@cli.command("baseline")
@click.argument("name")
@click.argument("namespace")
@click.option("--window", type=int, default=DEFAULT_WINDOW, show_default=True, help="Digest window size in seqs.")
@click.option("--timeout", "request_timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT, show_default=True)
def baseline_cmd(name: str, namespace: str, window: int, request_timeout: float) -> None:
    """Freeze max_seq and record digests + latency samples for later validation."""
    with _open_cli_client(name, DEFAULT_TUNNEL_TIMEOUT, request_timeout) as client:
        info = _namespace_info(client, namespace)
        frozen = info.max_seq
        window_start = max(info.min_seq, frozen - window)
        bytes_window_start = max(info.min_seq, frozen - window // 40)
        click.echo(
            f"frozen max_seq={frozen} min_seq={info.min_seq} "
            f"window=({window_start}, {frozen}] bytes_window=({bytes_window_start}, {frozen}]"
        )
        full_count = _full_count(client, namespace, info.min_seq, frozen)
        digests = _column_digests(client, namespace, window_start, bytes_window_start, frozen)
        probes = _latency_probes(namespace, frozen)
        latencies = _measure_latencies(client, probes)
        # A pre-rollout binary has no GetTableStatus RPC; every table is then
        # an unmigrated legacy table (spec version 0).
        try:
            active_version = client.get_table_status(namespace).active_version
        except Exception as error:
            if "method not found" not in str(error):
                raise
            active_version = 0

    state = MigrationBaseline(
        captured_at=datetime.now(UTC).isoformat(),
        frozen_max_seq=frozen,
        min_seq_at_baseline=info.min_seq,
        window_start=window_start,
        bytes_window_start=bytes_window_start,
        row_count_reported=info.row_count,
        full_count_at_frozen=full_count,
        column_digests=digests,
        latency_probes=probes,
        latency_ms=latencies,
        active_version_at_baseline=active_version,
    )
    path = _state_path(name, namespace)
    state.write(path)
    click.echo(f"baseline written to {path}")
    click.echo(f"full_count_at_frozen={full_count}")
    for label, runs in latencies.items():
        click.echo(f"latency {label}: {runs} ms")


@cli.command("migrate")
@click.argument("name")
@click.argument("namespace")
@click.option("--timeout", "request_timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT, show_default=True)
def migrate_cmd(name: str, namespace: str, request_timeout: float) -> None:
    """Register TableSpec version 1: the server starts the version-0 import."""
    if not _state_path(name, namespace).is_file():
        raise click.ClickException(f"no baseline for {namespace!r}; run `baseline {name} {namespace}` first")
    with _open_cli_client(name, DEFAULT_TUNNEL_TIMEOUT, request_timeout) as client:
        status = client.get_table_status(namespace)
        if status.active_version != 0 or status.desired_version is not None:
            raise click.ClickException(
                f"{namespace!r} is not an unmigrated legacy table: active_version={status.active_version} "
                f"desired_version={status.desired_version} phase={status.migration_phase}"
            )
        info = _namespace_info(client, namespace)
        schema = client.get_table_schema(namespace)
        click.echo(f"registering TableSpec version 1 over {len(schema.columns)} columns...")
        client._register_table(namespace, schema, info.storage_policy, TableSpec(version=1))
        _print_status(client, namespace)


@cli.command("watch")
@click.argument("name")
@click.argument("namespace")
@click.option("--budget", type=float, default=3600.0, show_default=True, help="Seconds before giving up.")
@click.option("--timeout", "request_timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT, show_default=True)
def watch_cmd(name: str, namespace: str, budget: float, request_timeout: float) -> None:
    """Poll the migration phase until the object spec activates."""
    deadline = time.monotonic() + budget
    last = None
    with _open_cli_client(name, DEFAULT_TUNNEL_TIMEOUT, request_timeout) as client:
        while time.monotonic() < deadline:
            status = client.get_table_status(namespace)
            line = f"active={status.active_version} desired={status.desired_version} phase={status.migration_phase}"
            if line != last:
                click.echo(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {line}")
                last = line
            if status.active_version >= 1:
                click.echo("activated")
                return
            time.sleep(WATCH_POLL_SECONDS)
    raise click.ClickException(f"migration did not activate within {budget:.0f}s (last: {last})")


@cli.command("validate")
@click.argument("name")
@click.argument("namespace")
@click.option("--timeout", "request_timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT, show_default=True)
def validate_cmd(name: str, namespace: str, request_timeout: float) -> None:
    """Recompute the frozen-window digests and compare latencies against the baseline."""
    path = _state_path(name, namespace)
    if not path.is_file():
        raise click.ClickException(f"no baseline recorded at {path}")
    state = MigrationBaseline.read(path)
    frozen = state.frozen_max_seq
    window_start = state.window_start
    bytes_window_start = state.bytes_window_start

    with _open_cli_client(name, DEFAULT_TUNNEL_TIMEOUT, request_timeout) as client:
        _print_status(client, namespace)
        info = _namespace_info(client, namespace)
        if info.min_seq > window_start:
            click.echo(
                f"WARNING: min_seq advanced past the digest window start ({info.min_seq} > {window_start}); "
                "window digests are no longer comparable"
            )
        count_floor = min(info.min_seq, state.min_seq_at_baseline)
        full_count = _full_count(client, namespace, count_floor, frozen)
        digests = _column_digests(client, namespace, window_start, bytes_window_start, frozen)
        latencies = _measure_latencies(client, state.latency_probes)

    drift = full_count - state.full_count_at_frozen
    click.echo(f"full_count_at_frozen: baseline={state.full_count_at_frozen} now={full_count} drift={drift}")

    mismatches = []
    for column, expected in state.column_digests.items():
        actual = digests.get(column)
        if not _digests_equal(expected, actual):
            mismatches.append(column)
            click.echo(f"DIGEST MISMATCH {column}:\n  baseline: {expected}\n  now:      {actual}")
    if not mismatches:
        click.echo(f"window digests match on all {len(digests)} columns")

    for label, runs in latencies.items():
        click.echo(f"latency {label}: baseline={state.latency_ms[label]} now={runs} ms")

    if mismatches:
        raise click.ClickException(f"digest mismatch on: {', '.join(mismatches)}")


@cli.command("abort")
@click.argument("name")
@click.argument("namespace")
@click.option("--timeout", "request_timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT, show_default=True)
def abort_cmd(name: str, namespace: str, request_timeout: float) -> None:
    """Abort the in-flight migration and restore the source version."""
    with _open_cli_client(name, DEFAULT_TUNNEL_TIMEOUT, request_timeout) as client:
        status = client.abort_table_migration(namespace)
        click.echo(f"aborted; active_version={status.active_version} phase={status.migration_phase}")


@cli.command("status")
@click.argument("name")
@click.argument("namespace")
@click.option("--timeout", "request_timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT, show_default=True)
def status_cmd(name: str, namespace: str, request_timeout: float) -> None:
    """Print the table's spec versions and migration phase."""
    with _open_cli_client(name, DEFAULT_TUNNEL_TIMEOUT, request_timeout) as client:
        _print_status(client, namespace)


if __name__ == "__main__":
    cli()
