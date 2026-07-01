# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""`ducky query` — run a SQL query against a ducky service from the CLI.

ducky is reached through the controller endpoint proxy, which sits behind IAP. The
CLI hides that: with ``--cluster <name>`` it drives ``iris cluster dashboard`` to
open a local tunnel, resolves the ``/proxy/<endpoint>`` path itself, runs the query,
and tears the tunnel down — so you just run::

    ducky query --cluster marin "SELECT count(*) FROM read_parquet('gs://…/*.parquet')"

Alternatively pass ``--base-url`` to target an already-open tunnel (or set
``DUCKY_BASE_URL``). The command submits, polls until done, prints the capped
preview as a table, and writes a stats line (rows, time, result size, cached, and
the full-result GCS path) to stderr.
"""

from __future__ import annotations

import json
import os
import time

import click
import httpx

from ducky.tunnel import cluster_tunnel

DEFAULT_BASE_URL = "http://127.0.0.1:10000/proxy/ducky"
# Per-request HTTP timeout. Each submit/poll call is fast (the query runs async on the
# server), so this only bounds a single round-trip — not the query.
_HTTP_TIMEOUT = 30


def _render_table(columns: list[str], rows: list[list]) -> str:
    """Format columns/rows as a simple aligned text table."""
    str_rows = [["NULL" if cell is None else str(cell) for cell in row] for row in rows]
    widths = [len(c) for c in columns]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [fmt(columns), "-+-".join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in str_rows)
    return "\n".join(lines)


def _error_message(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", response.text))
    except (ValueError, KeyError):
        return f"HTTP {response.status_code}: {response.text[:200]}"


def _run_query(base: str, sql: str, output_format: str, poll_interval: float, timeout: int) -> None:
    base = base.rstrip("/")
    submit = httpx.post(f"{base}/query", json={"sql": sql}, timeout=_HTTP_TIMEOUT)
    if submit.status_code != 202:
        raise click.ClickException(_error_message(submit))
    query_id = submit.json()["query_id"]

    deadline = time.monotonic() + timeout
    while True:
        resp = httpx.get(f"{base}/result/{query_id}", timeout=_HTTP_TIMEOUT)
        if resp.status_code != 200:
            raise click.ClickException(_error_message(resp))
        result = resp.json()
        if result.get("status") != "running":
            break
        if time.monotonic() > deadline:
            raise click.ClickException(f"Query still running after {timeout}s (id {query_id}).")
        time.sleep(poll_interval)

    if result["status"] == "error":
        raise click.ClickException(result["error"])

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(_render_table(result["columns"], result["rows"]))
    count = f"{len(result['rows'])} of {result['total_rows']}" if result["truncated"] else str(result["total_rows"])
    cached = "cached" if result["cached"] else "computed"
    click.echo(f"\n{count} rows · {result['elapsed_ms']} ms · {result['result_bytes']} B · {cached}", err=True)
    click.echo(f"full result: {result['result_path']}", err=True)


@click.command("query")
@click.argument("sql", required=False)
@click.option("--cluster", default=None, help="Iris cluster to auto-tunnel to (hides the tunnel/proxy).")
@click.option("--endpoint", default="ducky", show_default=True, help="ducky endpoint name behind the controller proxy.")
@click.option(
    "--base-url",
    default=None,
    help=f"Explicit ducky base URL (default $DUCKY_BASE_URL or {DEFAULT_BASE_URL}); mutually exclusive with --cluster.",
)
@click.option("--format", "output_format", type=click.Choice(["table", "json"]), default="table", show_default=True)
@click.option("--poll-interval", default=1.0, show_default=True, help="Seconds between status polls.")
@click.option("--timeout", default=3600, show_default=True, help="Max seconds to wait for the query to finish.")
def query(
    sql: str | None,
    cluster: str | None,
    endpoint: str,
    base_url: str | None,
    output_format: str,
    poll_interval: float,
    timeout: int,
) -> None:
    """Run SQL against a ducky service and print the result. SQL comes from the argument or stdin."""
    if cluster and base_url:
        raise click.UsageError("Pass --cluster or --base-url, not both.")
    if not sql:
        sql = click.get_text_stream("stdin").read()
    sql = sql.strip()
    if not sql:
        raise click.UsageError("No SQL provided — pass it as an argument or via stdin.")

    if cluster:
        with cluster_tunnel(cluster) as controller_url:
            _run_query(f"{controller_url}/proxy/{endpoint}", sql, output_format, poll_interval, timeout)
    else:
        base = base_url or os.environ.get("DUCKY_BASE_URL", DEFAULT_BASE_URL)
        _run_query(base, sql, output_format, poll_interval, timeout)


if __name__ == "__main__":
    query()
