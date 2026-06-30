# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""`ducky query` — run a SQL query against a ducky service from the CLI.

ducky is reached through the controller endpoint proxy, which sits behind IAP; the
simplest way in is a local tunnel. Open one with ``iris --cluster=<name> cluster
dashboard`` (holds a tunnel at ``http://127.0.0.1:10000``), then::

    ducky query "SELECT count(*) FROM read_parquet('gs://marin-us-east5/…/*.parquet')"

The default ``--base-url`` points at that tunnel. The command submits the query,
polls until it finishes, prints the capped preview as a table, and writes a stats
line (rows, time, result size, cached, the full-result GCS path) to stderr.
"""

from __future__ import annotations

import json
import os
import time

import click
import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:10000/proxy/ducky"


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


@click.command("query")
@click.argument("sql", required=False)
@click.option(
    "--base-url",
    default=lambda: os.environ.get("DUCKY_BASE_URL", DEFAULT_BASE_URL),
    help=f"ducky base URL (default: {DEFAULT_BASE_URL}; open a tunnel with `iris cluster dashboard`).",
)
@click.option("--format", "output_format", type=click.Choice(["table", "json"]), default="table", show_default=True)
@click.option("--poll-interval", default=1.0, show_default=True, help="Seconds between status polls.")
@click.option("--timeout", default=3600, show_default=True, help="Max seconds to wait for the query to finish.")
def query(sql: str | None, base_url: str, output_format: str, poll_interval: float, timeout: int) -> None:
    """Run SQL against a ducky service and print the result. SQL comes from the argument or stdin."""
    if not sql:
        sql = click.get_text_stream("stdin").read()
    sql = sql.strip()
    if not sql:
        raise click.UsageError("No SQL provided — pass it as an argument or via stdin.")

    base = base_url.rstrip("/")
    submit = httpx.post(f"{base}/query", json={"sql": sql}, timeout=30)
    if submit.status_code != 202:
        raise click.ClickException(_error_message(submit))
    query_id = submit.json()["query_id"]

    deadline = time.monotonic() + timeout
    while True:
        result = httpx.get(f"{base}/result/{query_id}", timeout=30).json()
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
    shown = len(result["rows"])
    count = f"{shown} of {result['total_rows']}" if result["truncated"] else str(result["total_rows"])
    cached = "cached" if result["cached"] else "computed"
    click.echo(
        f"\n{count} rows · {result['elapsed_ms']} ms · {result['result_bytes']} B · {cached}",
        err=True,
    )
    click.echo(f"full result: {result['result_path']}", err=True)


def _error_message(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", response.text))
    except (ValueError, KeyError):
        return f"HTTP {response.status_code}: {response.text[:200]}"


if __name__ == "__main__":
    query()
