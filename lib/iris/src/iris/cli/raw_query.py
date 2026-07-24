# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for CLI commands that read the controller DB over ExecuteRawQuery.

The RPC returns each row as a JSON-encoded array alongside a separate column
list, so every caller has to decode and, usually, zip the two back together.
"""

import json

from iris.rpc import query_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync


def sql_literal(value: str) -> str:
    """Quote *value* as a SQL string literal."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def parse_rows(response: query_pb2.RawQueryResponse) -> list[list]:
    """Decode a raw-query response into positional row values."""
    return [json.loads(row) for row in response.rows]


def query_dicts(client: ControllerServiceClientSync, sql: str) -> list[dict]:
    """Run *sql* against the controller and return rows keyed by column name."""
    response = client.execute_raw_query(query_pb2.RawQueryRequest(sql=sql))
    names = [column.name for column in response.columns]
    return [dict(zip(names, values, strict=True)) for values in parse_rows(response)]
