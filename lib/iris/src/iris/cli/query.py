# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI command for executing raw SQL queries against the controller."""

import csv
import io
import json

import click
from tabulate import tabulate

from iris.cli.main import require_controller_url, rpc_client
from iris.rpc import query_pb2


def _parse_rows(response_rows: list[str]) -> list[list[object]]:
    """Decode JSON-encoded row arrays from the query response."""
    return [json.loads(row) for row in response_rows]


def _format_table(columns: list[query_pb2.ColumnMeta], rows: list[list[object]]) -> str:
    headers = [c.name for c in columns]
    return tabulate(rows, headers=headers, tablefmt="plain")


def _format_json(columns: list[query_pb2.ColumnMeta], rows: list[list[object]]) -> str:
    headers = [c.name for c in columns]
    records = [dict(zip(headers, row, strict=True)) for row in rows]
    return json.dumps(records, indent=2)


def _format_csv(columns: list[query_pb2.ColumnMeta], rows: list[list[object]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([c.name for c in columns])
    for row in rows:
        writer.writerow(row)
    return buf.getvalue().rstrip("\n")


_FORMATTERS = {
    "table": _format_table,
    "json": _format_json,
    "csv": _format_csv,
}


@click.command("query")
@click.argument("sql")
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format",
)
@click.pass_context
def query_cmd(ctx: click.Context, sql: str, fmt: str) -> None:
    """Execute a raw SQL query against the controller database.

    \b
    Examples:
      iris query "SELECT * FROM jobs LIMIT 10"
      iris query "SELECT count(*) FROM jobs"
      iris query -f json "SELECT job_id, state FROM jobs"
    """
    controller_url = require_controller_url(ctx)
    token_provider = ctx.obj.get("token_provider") if ctx.obj else None

    with rpc_client(controller_url, token_provider) as client:
        request = query_pb2.RawQueryRequest(sql=sql)
        response = client.execute_raw_query(request)

    columns = list(response.columns)
    rows = _parse_rows(list(response.rows))
    formatter = _FORMATTERS[fmt]
    output = formatter(columns, rows)

    if output:
        click.echo(output)
