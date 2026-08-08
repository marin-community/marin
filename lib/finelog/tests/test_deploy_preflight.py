# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The deploy pre-flight's pure halves: the capture format and the rollout gate.

The decision itself belongs to the server binary (`rust/src/preflight.rs`); what
is testable here is the document handed to it and the verdict read back out.
"""

import json
from pathlib import Path

import pytest
from finelog.client.log_client import LogClient
from finelog.deploy.preflight import (
    Outcome,
    PreflightResult,
    blocks_rollout,
    registered_schema_document,
    render_document,
    schema_to_catalog_json,
    summarize,
)
from finelog.embedded import is_available, require_embedded_server
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.schema import Column, CoveringProjection, GroupedExtrema, Schema


def _result(deployment: str, outcome: Outcome) -> PreflightResult:
    return PreflightResult(deployment=deployment, outcome=outcome, report="report body")


def test_catalog_json_carries_every_field_the_merge_reads() -> None:
    # The document is the seam between this capture and the server's
    # `schema_from_json`. A field dropped here silently narrows what the
    # pre-flight decides, so pin the whole shape.
    schema = Schema(
        columns=(
            Column(
                name="name",
                type=stats_pb2.COLUMN_TYPE_STRING,
                nullable=False,
                trigram_index=True,
                exact_values=("step", "phase"),
                value_counts=True,
            ),
        ),
        key_column="name",
        projections=(
            CoveringProjection(
                name="training-status",
                predicate_column="name",
                predicate_values=("step", "phase"),
                columns=("name", "value"),
            ),
        ),
        grouped_extrema=(
            GroupedExtrema(
                filter_column="service",
                group_json_column="resource_attributes_json",
                group_json_key="job_id",
                extrema_column="timestamp_ms",
            ),
        ),
    )

    assert schema_to_catalog_json(schema) == {
        "key_column": "name",
        "columns": [
            {
                "name": "name",
                "type": "COLUMN_TYPE_STRING",
                "nullable": False,
                "index": {"trigram": True, "exact_values": ["step", "phase"], "value_counts": True},
            }
        ],
        "projections": [
            {
                "name": "training-status",
                "predicate_column": "name",
                "predicate_values": ["step", "phase"],
                "columns": ["name", "value"],
            }
        ],
        # The store names these roles `json_column` / `json_key`; the proto
        # prefixes both with `group_`. The document speaks the store's names.
        "grouped_extrema": [
            {
                "filter_column": "service",
                "json_column": "resource_attributes_json",
                "json_key": "job_id",
                "extrema_column": "timestamp_ms",
            }
        ],
    }


def test_a_failing_deployment_blocks_the_rollout_and_ends_the_summary() -> None:
    results = [
        _result("finelog-marin", Outcome.PASS),
        _result("finelog-cw-rno2a", Outcome.FAIL),
        _result("finelog-marin-dev", Outcome.PASS),
    ]

    assert blocks_rollout(results)
    summary = summarize(results)
    # A failure that scrolls past is a failure that gets missed: passes first,
    # then the failing deployment, then the verdict.
    assert summary.index("finelog-marin ") < summary.index("finelog-cw-rno2a ")
    assert summary.strip().endswith("PREFLIGHT FAIL: finelog-cw-rno2a")


@pytest.mark.skipif(not is_available(), reason="needs the native finelog server extension")
def test_a_document_captured_from_a_real_server_describes_the_registered_schema(tmp_path: Path) -> None:
    # End to end over the real RPC: what `safe_deploy` records is what a server
    # actually holds. `log` is registered by the store itself, so every finelog
    # has it and its shape does not move between extension versions.
    server = require_embedded_server()(log_dir=str(tmp_path))
    try:
        client = LogClient.connect(server.address)
        namespaces = client.list_namespaces()
        client.close()
    finally:
        server.stop()

    document = registered_schema_document(
        deployment="finelog-test",
        namespaces=namespaces,
        captured_at="2026-08-08T00:00:00+00:00",
    )
    captured = json.loads(render_document(document))["namespaces"]

    log = captured["log"]
    assert log["key_column"] == "key"
    assert {column["name"] for column in log["columns"]} >= {"key", "source", "data", "epoch_ms", "level"}
    data = next(column for column in log["columns"] if column["name"] == "data")
    assert data["type"] == "COLUMN_TYPE_STRING"
    assert data["index"]["trigram"] is True
    # The wire strips the server-assigned `seq`; the binary restores it before
    # merging, so a document carrying one would describe a schema no client can
    # register.
    assert "seq" not in {column["name"] for column in log["columns"]}
