# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI command for executing structured and raw SQL queries against the controller."""

import csv
import io
import json

import click
from google.protobuf import json_format
from tabulate import tabulate

from iris.cli.main import _make_authenticated_client, require_controller_url
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
@click.argument("query_json", required=False)
@click.option("--raw", "raw_sql", default=None, help="Raw SQL query (admin-only)")
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format",
)
@click.pass_context
def query_cmd(ctx: click.Context, query_json: str | None, raw_sql: str | None, fmt: str) -> None:
    """Execute a query against the controller database.

    Pass a JSON query structure as a positional argument, or use --raw for
    direct SQL (admin-only).

    \b
    Examples:
      iris query '{"from": {"name": "jobs"}, "limit": 10}'
      iris query --raw "SELECT count(*) FROM jobs"
      iris query -f json '{"from": {"name": "workers"}}'
    """
    if not query_json and not raw_sql:
        raise click.UsageError("Provide a query JSON argument or --raw SQL")
    if query_json and raw_sql:
        raise click.UsageError("Cannot use both a query JSON argument and --raw")

    controller_url = require_controller_url(ctx)
    token_provider = ctx.obj.get("token_provider") if ctx.obj else None
    client = _make_authenticated_client(controller_url, token_provider)

    try:
        total_count = 0
        if raw_sql:
            request = query_pb2.RawQueryRequest(sql=raw_sql)
            response = client.execute_raw_query(request)
        else:
            query_dict = json.loads(query_json)  # type: ignore[arg-type]
            query = json_format.ParseDict(query_dict, query_pb2.Query())
            request = query_pb2.QueryRequest(query=query)
            response = client.execute_query(request)
            total_count = response.total_count

        columns = list(response.columns)
        rows = _parse_rows(list(response.rows))
        formatter = _FORMATTERS[fmt]
        output = formatter(columns, rows)

        if output:
            click.echo(output)

        if fmt == "table" and total_count > len(rows):
            click.echo(f"\n({len(rows)} of {total_count} rows)")
    finally:
        client.close()
