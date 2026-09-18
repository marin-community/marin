# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zephyr executions applet backend against an embedded Finelog server.

The fixture dataclasses carry only the columns the applet queries.
"""

import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from finelog.client import LogClient
from finelog.embedded import EmbeddedServer

APPLET = Path(__file__).resolve().parents[1] / "applets" / "zephyr-executions"
EXECUTION = "20260910-195311-32c9cebb"
STAGE = "stage2-Reduce → Write"
T0 = datetime(2026, 9, 10, 19, 53)


def _load(name: str):
    package = sys.modules.get("zephyr_applet_server")
    if package is None:
        package = types.ModuleType("zephyr_applet_server")
        package.__path__ = [str(APPLET / "server")]
        sys.modules["zephyr_applet_server"] = package
    spec = importlib.util.spec_from_file_location(f"zephyr_applet_server.{name}", APPLET / "server" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class ExecutionRow:
    key_column: ClassVar[str] = "root_job_id"
    execution_id: str
    root_job_id: str
    coordinator_job_id: str
    ts: datetime
    input_shards: int
    stages_json: str


@dataclass
class StageRow:
    key_column: ClassVar[str] = "execution_id"
    execution_id: str
    stage_name: str
    status: str
    ts: datetime
    elapsed: float
    items: int
    total_shards: int
    mem_peak_bytes_max: int


@dataclass
class ShuffleRow:
    key_column: ClassVar[str] = "execution_id"
    execution_id: str
    stage_name: str
    target_shard: int
    num_targets: int
    attempt: int
    input_rows: int | None
    payload_bytes: int | None
    num_sources: int | None
    ts: datetime
    job_id: str


@dataclass
class WorkerRow:
    key_column: ClassVar[str] = "execution_id"
    execution_id: str
    stage_name: str
    shard_idx: int
    attempt_id: str
    status: str
    ts: datetime
    items: int


PLAN = [
    {
        "stage_name": "stage0-Map → Scatter",
        "label": "Map → Scatter",
        "stage_type": "map",
        "has_reduce": False,
        "dependencies": [],
    },
    {
        "stage_name": STAGE,
        "label": "Reduce → Write",
        "stage_type": "reduce",
        "has_reduce": True,
        "dependencies": ["stage0-Map → Scatter"],
    },
]


def _shuffle(target: int, attempt: int, rows: int | None, seconds: float, job: str = "/karan/job/worker") -> ShuffleRow:
    return ShuffleRow(
        execution_id=EXECUTION,
        stage_name=STAGE,
        target_shard=target,
        num_targets=25,
        attempt=attempt,
        input_rows=rows,
        payload_bytes=None if rows is None else rows * 33,
        num_sources=None if rows is None else 1,
        ts=T0 + timedelta(seconds=seconds),
        job_id=job,
    )


@pytest.fixture(scope="module")
def finelog(tmp_path_factory):
    server = EmbeddedServer(log_dir=str(tmp_path_factory.mktemp("finelog")))
    url = f"http://127.0.0.1:{server.port}"
    client = LogClient.connect(url)
    executions = client.get_table("zephyr.execution", ExecutionRow)
    stages = client.get_table("zephyr.stage", StageRow)
    shuffle = client.get_table("zephyr.shuffle", ShuffleRow)
    workers = client.get_table("zephyr.worker", WorkerRow)

    executions.write(
        [
            ExecutionRow(EXECUTION, "/karan/job", "/karan/job/coord", T0, 8, json.dumps(PLAN)),
            ExecutionRow(EXECUTION, "/karan/job", "/karan/job/coord", T0 + timedelta(seconds=1), 8, json.dumps(PLAN)),
            ExecutionRow("older", "/other/job", "/other/job/coord", T0 - timedelta(days=30), 1, "[]"),
            ExecutionRow("broken", "/karan/job", "/karan/job/coord", T0 + timedelta(minutes=1), 1, "{"),
        ]
    )
    stages.write(
        [
            StageRow(EXECUTION, STAGE, "FAILED", T0 + timedelta(seconds=5), 1.0, 1, 25, 100),
            StageRow(EXECUTION, STAGE, "END", T0 + timedelta(seconds=9), 2.5, 128, 25, 299_000_000),
        ]
    )
    # Placeholders for every target, then measurements with the ordering edge cases:
    # target 0 measured twice (attempt 0 then 1); target 1 placeholder only; target 2
    # measured zero; target 3's measurement carries an older ts than its placeholder.
    rows = [_shuffle(target, 0, None, 0.0, "/karan/job/coord") for target in range(25)]
    rows += [_shuffle(0, 0, 10, 2.0), _shuffle(0, 1, 1, 3.0), _shuffle(2, 0, 0, 2.0), _shuffle(3, 0, 5, -1.0)]
    rows += [_shuffle(target, 0, 25 - target, 2.0) for target in range(4, 25)]
    shuffle.write(rows)
    workers.write(
        [
            WorkerRow(EXECUTION, STAGE, 0, "a", "START", T0 + timedelta(seconds=1), 0),
            WorkerRow(EXECUTION, STAGE, 0, "a", "FAILED", T0 + timedelta(seconds=2), 0),
            WorkerRow(EXECUTION, STAGE, 0, "b", "END", T0 + timedelta(seconds=4), 1),
            WorkerRow(EXECUTION, STAGE, 2, "c", "END", T0 + timedelta(seconds=4), 0),
        ]
    )
    for table in (executions, stages, shuffle, workers):
        assert table.flush(timeout=30) is not None
    yield url
    client.close()
    server.stop()


@pytest.fixture()
def api(finelog, monkeypatch):
    monkeypatch.setenv("ZEPHYR_APPLET_FINELOG_URL", finelog)
    monkeypatch.delenv("ZEPHYR_APPLET_IAP_CLUSTER", raising=False)
    app = _load("app")
    # Pin "now" one hour after the seeded rows so the listing window is stable.
    monkeypatch.setattr(app, "_now", lambda: (T0 + timedelta(hours=1)).replace(tzinfo=UTC))
    return TestClient(app.create_api(None))


def _stage_url(suffix: str) -> str:
    return f"/executions/{EXECUTION}/stages/{quote(STAGE, safe='')}/{suffix}"


def test_health_reports_source_and_namespaces(api, finelog):
    body = api.get("/health").json()
    assert body["plan_records"] is True
    assert {"zephyr.execution", "zephyr.stage", "zephyr.shuffle", "zephyr.worker"} <= set(body["namespaces"])
    assert body["finelog"].startswith(finelog)


def test_executions_dedupe_by_execution_and_filter_by_root_job(api):
    rows = api.get("/executions?days=14").json()
    ids = [row["execution_id"] for row in rows]
    assert ids == ["broken", EXECUTION], "newest first, one row per execution, 'older' outside the window"
    assert rows[1]["stages"] == PLAN
    assert rows[0]["stages"] == [] and rows[0]["plan_error"].startswith("Cannot read execution plan")
    assert api.get("/executions?days=14&root_job=/other/job").json() == []
    assert [row["execution_id"] for row in api.get("/executions?days=90&root_job=/other/job").json()] == ["older"]


def test_single_execution_lookup_ignores_the_listing_window(api):
    assert api.get("/executions/older").json()["root_job_id"] == "/other/job"
    assert api.get("/executions/missing").status_code == 404


def test_stage_stats_keep_the_latest_report_per_stage(api):
    rows = api.get(f"/executions/{EXECUTION}/stages").json()
    assert rows == [
        {
            "stage_name": STAGE,
            "status": "END",
            "elapsed": 2.5,
            "items": 128,
            "total_shards": 25,
            "mem_peak_bytes_max": 299_000_000,
        }
    ]


def test_reducer_rule_prefers_attempt_then_measurement_then_recency_and_joins_worker_status(api):
    page = api.get(_stage_url("reducers?page=0")).json()
    rows = page + api.get(_stage_url("reducers?page=1")).json()
    by_target = {row["target_shard"]: row for row in rows}
    assert (
        by_target[0]["attempt"] == 1 and by_target[0]["input_rows"] == 1
    ), "highest attempt wins over a bigger older one"
    assert by_target[0]["task_status"] == "END", "latest worker report by ts, not by attempt id"
    assert by_target[3]["input_rows"] == 5, "a measurement beats its placeholder even with an older ts"
    assert by_target[2]["input_rows"] == 0 and by_target[2]["task_status"] == "END", "an observed zero stays zero"
    assert [row["target_shard"] for row in page][:3] == [4, 5, 6], "largest payload first"
    assert len(page) == 20


def test_reducer_paging_puts_placeholders_last(api):
    second = api.get(_stage_url("reducers?page=1")).json()
    assert len(second) == 5
    assert second[-1]["target_shard"] == 1 and second[-1]["input_rows"] is None
    assert second[-1]["task_status"] is None


def test_summary_uses_fractional_medians_and_counts_unreported(api):
    body = api.get(_stage_url("summary")).json()
    assert body["expected_targets"] == 25
    assert body["persisted_targets"] == 25
    assert body["observed_targets"] == 24
    assert body["max_rows"] == 21
    # Observed rows are 0, 1, 1, 2, 3, ..., 21: the median falls between 9 and 10.
    assert body["median_rows"] == pytest.approx(9.5), "integer MEDIAN would truncate to 9"
    assert body["median_bytes"] == pytest.approx(9.5 * 33)
