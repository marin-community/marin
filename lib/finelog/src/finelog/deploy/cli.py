# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""finelog deploy CLI — config-driven deployment management.

Each subcommand takes a logical config name (or path), loads it via
`load_finelog_config`, and dispatches to either the GCE or Kubernetes
backend based on which `deployment.*` block the config sets. The CLI
itself is platform-agnostic; finelog owns the platform decision via its
config schema, mirroring how `iris cluster start` decides backend from
cluster yaml.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
from datetime import datetime
from enum import StrEnum

import click
import duckdb
import fsspec
import pyarrow as pa
from rigging.log_setup import configure_logging
from rigging.tunnel import GcpSshForwardTarget, K8sPortForwardTarget, TunnelTarget, open_tunnel

from finelog.client.log_client import LogClient
from finelog.deploy import _gcp, _k8s
from finelog.deploy.build import build_image as build_finelog_image
from finelog.deploy.config import FinelogConfig, load_finelog_config

_SEGMENT_FILENAME_RE = re.compile(r"seg_L\d+_\d+\.parquet$")


def _tunnel_target(cfg: FinelogConfig) -> TunnelTarget:
    """Translate a finelog deployment block into a rigging tunnel target.

    The GCP path forwards ``deployment.gcp.service_account`` as the SSH
    impersonation principal — matching the deploy CLI's own SSH calls
    (see ``_gcp._ssh_args``), so this command works wherever
    ``finelog deploy status`` does.
    """
    if cfg.deployment.gcp is not None:
        gcp = cfg.deployment.gcp
        return GcpSshForwardTarget(
            project=gcp.project,
            zone=gcp.zone,
            instance=cfg.name,
            port=cfg.port,
            impersonate_service_account=gcp.service_account,
        )
    assert cfg.deployment.k8s is not None
    k8s = cfg.deployment.k8s
    return K8sPortForwardTarget(namespace=k8s.namespace, service=cfg.name, port=cfg.port)


def _dispatch_up(cfg: FinelogConfig) -> None:
    if cfg.deployment.gcp is not None:
        _gcp.gcp_up(cfg)
    else:
        _k8s.k8s_up(cfg)


def _dispatch_down(cfg: FinelogConfig, *, yes: bool) -> None:
    if cfg.deployment.gcp is not None:
        _gcp.gcp_down(cfg, yes=yes)
    else:
        _k8s.k8s_down(cfg, yes=yes)


def _dispatch_restart(cfg: FinelogConfig) -> None:
    if cfg.deployment.gcp is not None:
        _gcp.gcp_restart(cfg)
    else:
        _k8s.k8s_restart(cfg)


def _dispatch_status(cfg: FinelogConfig) -> None:
    if cfg.deployment.gcp is not None:
        _gcp.gcp_status(cfg)
    else:
        _k8s.k8s_status(cfg)


def _dispatch_logs(cfg: FinelogConfig, *, tail: int, follow: bool) -> None:
    if cfg.deployment.gcp is not None:
        _gcp.gcp_logs(cfg, tail=tail, follow=follow)
    else:
        _k8s.k8s_logs(cfg, tail=tail, follow=follow)


@click.group()
def cli() -> None:
    """Manage finelog deployments."""


@cli.group("deploy")
def deploy() -> None:
    """Provision and manage a finelog deployment from a config file."""


@deploy.command("up")
@click.argument("name")
@click.option(
    "--build/--no-build",
    "build",
    default=True,
    show_default=True,
    help="Build and push the finelog image (using cfg.image as the tag) before provisioning.",
)
def up_cmd(name: str, build: bool) -> None:
    """Provision the finelog deployment described by `<name>` (idempotent)."""
    cfg = load_finelog_config(name)
    if build:
        build_finelog_image(image=cfg.image)
    _dispatch_up(cfg)


@deploy.command("down")
@click.argument("name")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation; for k8s also deletes the PVC.")
def down_cmd(name: str, yes: bool) -> None:
    """Tear down the finelog deployment described by `<name>`."""
    cfg = load_finelog_config(name)
    _dispatch_down(cfg, yes=yes)


@deploy.command("restart")
@click.argument("name")
@click.option(
    "--build/--no-build",
    "build",
    default=True,
    show_default=True,
    help="Build and push the finelog image (using cfg.image as the tag) before restarting.",
)
def restart_cmd(name: str, build: bool) -> None:
    """Restart the finelog deployment in place (refresh the container/image)."""
    cfg = load_finelog_config(name)
    if build:
        build_finelog_image(image=cfg.image)
    _dispatch_restart(cfg)


@deploy.command("status")
@click.argument("name")
def status_cmd(name: str) -> None:
    """Show status of the finelog deployment."""
    cfg = load_finelog_config(name)
    _dispatch_status(cfg)


class OutputFormat(StrEnum):
    TABLE = "table"
    JSON = "json"
    CSV = "csv"


def _print_table(table: pa.Table) -> None:
    """Render an Arrow table as fixed-width columns to stdout."""
    if table.num_rows == 0:
        click.echo(f"(0 rows; columns: {', '.join(table.schema.names)})")
        return
    rows = [[_format_cell(v) for v in row.values()] for row in table.to_pylist()]
    headers = list(table.schema.names)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = " | "
    click.echo(sep.join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    click.echo("-+-".join("-" * w for w in widths))
    for row in rows:
        click.echo(sep.join(row[i].ljust(widths[i]) for i in range(len(headers))))
    click.echo(f"({table.num_rows} rows)")


def _format_cell(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, bytes):
        return v.hex()
    return str(v)


def _print_json(table: pa.Table) -> None:
    json.dump(table.to_pylist(), sys.stdout, default=str, indent=2)
    sys.stdout.write("\n")


def _print_csv(table: pa.Table) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(table.schema.names)
    for row in table.to_pylist():
        writer.writerow([_format_cell(v) for v in row.values()])


_PRINTERS = {
    OutputFormat.TABLE: _print_table,
    OutputFormat.JSON: _print_json,
    OutputFormat.CSV: _print_csv,
}


@cli.command("query")
@click.argument("name")
@click.argument("sql")
@click.option(
    "--format",
    "output_format",
    type=click.Choice([f.value for f in OutputFormat]),
    default=OutputFormat.TABLE.value,
    show_default=True,
    help="Output format for the result.",
)
@click.option(
    "--max-rows",
    type=int,
    default=100_000,
    show_default=True,
    help="Reject results larger than this (use LIMIT or raise this cap).",
)
@click.option(
    "--tunnel-timeout",
    type=float,
    default=60.0,
    show_default=True,
    help="Seconds to wait for the local tunnel to become reachable.",
)
def query_cmd(name: str, sql: str, output_format: str, max_rows: int, tunnel_timeout: float) -> None:
    """Run SQL against the deployed finelog `<name>` via a tunnel.

    Opens an IAP tunnel (GCP) or `kubectl port-forward` (k8s) to the
    configured finelog server, runs `<sql>` through `StatsService.Query`,
    and prints results in `--format` (table/json/csv).
    """
    configure_logging(level=logging.INFO)
    cfg = load_finelog_config(name)
    target = _tunnel_target(cfg)
    with open_tunnel(target, timeout=tunnel_timeout) as url:
        client = LogClient.connect(url)
        try:
            table = client.query(sql, max_rows=max_rows)
        finally:
            client.close()
    _PRINTERS[OutputFormat(output_format)](table)


def _list_namespace_dirs(remote_log_dir: str, fs: fsspec.AbstractFileSystem) -> list[str]:
    """Return namespace names under ``remote_log_dir`` that hold parquet segments.

    A namespace directory is "real" if at least one ``seg_L*.parquet`` lives
    under it — finelog never writes other parquet shapes into a namespace dir,
    so this check filters out stray top-level files without descending deeper.
    """
    base = remote_log_dir.rstrip("/")
    listing = fs.ls(base, detail=True)
    found: list[str] = []
    for entry in listing:
        if entry.get("type") != "directory":
            continue
        ns = entry["name"].rstrip("/").rsplit("/", 1)[-1]
        if fs.glob(f"{base}/{ns}/seg_L*.parquet"):
            found.append(ns)
    return sorted(found)


def _info_time_created_ms(info: dict[str, object]) -> int | None:
    """Best-effort extraction of an object's creation time in epoch_ms.

    Different fsspec backends expose this under different keys: gcsfs uses
    ``timeCreated`` (ISO 8601 string); ``LocalFileSystem`` uses ``created``
    (float seconds). Returns ``None`` when no usable timestamp is present.
    """
    raw = info.get("timeCreated") or info.get("created") or info.get("ctime")
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return int(raw.timestamp() * 1000)
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw * 1000)
    if isinstance(raw, str):
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        return int(dt.timestamp() * 1000)
    return None


def _list_namespace_segments(
    remote_log_dir: str,
    namespace: str,
    fs: fsspec.AbstractFileSystem,
    *,
    created_since_ms: int | None,
    created_until_ms: int | None,
) -> list[str]:
    """Enumerate ``seg_L*.parquet`` URIs under one namespace, filtered by mtime.

    Pre-filtering by object ``timeCreated`` is conservative for the canonical
    ``epoch_ms`` predicate: a segment's data is ingested *before* upload, so
    ``time_created`` is an upper bound on row ``epoch_ms`` in that file —
    files with ``time_created < epoch_floor`` cannot satisfy the query. Pad
    ``--created-until-ms`` by enough to cover L0→L1 compaction lag (a few
    hours is usually sufficient).
    """
    ns_dir = f"{remote_log_dir.rstrip('/')}/{namespace}"
    listing = fs.ls(ns_dir, detail=True)
    has_window = created_since_ms is not None or created_until_ms is not None
    out: list[str] = []
    for entry in listing:
        name = str(entry.get("name", ""))
        if not _SEGMENT_FILENAME_RE.search(name):
            continue
        if has_window:
            tc = _info_time_created_ms(entry)
            # Skip entries with unknown timestamps when a window is set —
            # safer to drop than to include a file we can't reason about.
            if tc is None:
                continue
            if created_since_ms is not None and tc < created_since_ms:
                continue
            if created_until_ms is not None and tc > created_until_ms:
                continue
        out.append(fs.unstrip_protocol(name))
    return sorted(out)


def _register_namespace_views(
    conn: duckdb.DuckDBPyConnection,
    remote_log_dir: str,
    namespaces: list[str],
    *,
    fs: fsspec.AbstractFileSystem | None = None,
    created_since_ms: int | None = None,
    created_until_ms: int | None = None,
) -> None:
    """Create one ``CREATE VIEW <ns>`` per namespace.

    Without a time window, the view body is
    ``SELECT * FROM read_parquet('<remote_log_dir>/<ns>/seg_L*.parquet')`` —
    DuckDB defers globbing and schema inference until the view is queried.

    With a time window, the glob is replaced by an explicit file list
    filtered by GCS object ``time_created``; this skips the parquet-footer
    fetch for segments that cannot contain matching rows. For namespaces
    with thousands of segments the footer prune is the only practical way
    to keep egress bounded. When the filter drops every file the view is
    not created at all, so referencing SQL fails fast.

    Namespace names with dots (e.g. ``iris.worker``) require double-quoted
    identifiers in user SQL.
    """
    base = remote_log_dir.rstrip("/")
    has_window = created_since_ms is not None or created_until_ms is not None
    if has_window and fs is None:
        raise ValueError("fs is required when a time window is set")
    for ns in namespaces:
        # CREATE VIEW does not accept prepared parameters, so paths are
        # inlined. Single-quote escaping guards against unusual paths;
        # the double-quoted identifier guards the view name.
        if has_window:
            assert fs is not None
            files = _list_namespace_segments(
                base,
                ns,
                fs,
                created_since_ms=created_since_ms,
                created_until_ms=created_until_ms,
            )
            if not files:
                continue
            list_literal = "[" + ", ".join(f"'{f.replace(chr(39), chr(39) * 2)}'" for f in files) + "]"
            conn.execute(f'CREATE OR REPLACE VIEW "{ns}" AS SELECT * FROM read_parquet({list_literal})')
        else:
            glob = f"{base}/{ns}/seg_L*.parquet".replace("'", "''")
            conn.execute(f"CREATE OR REPLACE VIEW \"{ns}\" AS SELECT * FROM read_parquet('{glob}')")


@cli.command("gcs-query")
@click.argument("name")
@click.argument("sql")
@click.option(
    "--namespace",
    "namespaces",
    multiple=True,
    help="Restrict views to these namespaces (default: every namespace discovered under remote_log_dir).",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice([f.value for f in OutputFormat]),
    default=OutputFormat.TABLE.value,
    show_default=True,
    help="Output format for the result.",
)
@click.option(
    "--max-rows",
    type=int,
    default=100_000,
    show_default=True,
    help="Reject results larger than this (use LIMIT or raise this cap).",
)
@click.option(
    "--created-since-ms",
    type=int,
    default=None,
    help=(
        "Only include parquet files whose GCS time_created is >= this epoch_ms. "
        "Use to skip the parquet-footer fetch for clearly-too-old segments; safe to set to "
        "the same lower bound you use for epoch_ms in the WHERE clause."
    ),
)
@click.option(
    "--created-until-ms",
    type=int,
    default=None,
    help=(
        "Only include parquet files whose GCS time_created is <= this epoch_ms. "
        "Pad above your epoch_ms upper bound by L0->L1 compaction lag (a few hours typically)."
    ),
)
def gcs_query_cmd(
    name: str,
    sql: str,
    namespaces: tuple[str, ...],
    output_format: str,
    max_rows: int,
    created_since_ms: int | None,
    created_until_ms: int | None,
) -> None:
    """Run SQL against the GCS-archived parquet for finelog ``<name>``.

    Use when ``FetchLogs`` returns empty because the live deque has already
    evicted segments to ``REMOTE`` — the parquet still lives in
    ``cfg.remote_log_dir`` and this command queries it directly. Each
    namespace directory under ``remote_log_dir`` is registered as a DuckDB
    view named after the namespace; reference it in the FROM clause, e.g.
    ``select * from log where key like '/ryan/%'`` (use double-quotes for
    names with dots: ``from "iris.worker"``).
    """
    configure_logging(level=logging.INFO)
    cfg = load_finelog_config(name)
    if not cfg.remote_log_dir:
        raise click.UsageError(f"finelog config {name!r} has no remote_log_dir; nothing to query")

    fs, _ = fsspec.url_to_fs(cfg.remote_log_dir)
    discovered = _list_namespace_dirs(cfg.remote_log_dir, fs)
    if not discovered:
        raise click.UsageError(f"no namespaces with parquet segments found under {cfg.remote_log_dir}")

    selected = list(namespaces) if namespaces else discovered
    unknown = sorted(set(selected) - set(discovered))
    if unknown:
        raise click.UsageError(f"namespace(s) not found: {unknown} (available: {discovered})")

    conn = duckdb.connect()
    conn.register_filesystem(fs)
    _register_namespace_views(
        conn,
        cfg.remote_log_dir,
        selected,
        fs=fs,
        created_since_ms=created_since_ms,
        created_until_ms=created_until_ms,
    )

    # ``.arrow()`` returns a streaming RecordBatchReader in duckdb >= 1.4;
    # ``.fetch_arrow_table()`` materializes a pa.Table so we can size-check.
    table = conn.execute(sql).fetch_arrow_table()
    if table.num_rows > max_rows:
        raise click.UsageError(
            f"query returned {table.num_rows} rows, exceeds --max-rows={max_rows} " f"(add a LIMIT or raise the cap)"
        )
    _PRINTERS[OutputFormat(output_format)](table)


@deploy.command("logs")
@click.argument("name")
@click.option("--tail", type=int, default=200, show_default=True)
@click.option("-f", "--follow", is_flag=True, help="Stream logs")
def logs_cmd(name: str, tail: int, follow: bool) -> None:
    """Tail logs from the finelog deployment."""
    cfg = load_finelog_config(name)
    _dispatch_logs(cfg, tail=tail, follow=follow)


if __name__ == "__main__":
    cli()
