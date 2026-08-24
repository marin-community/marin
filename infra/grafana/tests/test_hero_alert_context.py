# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from datetime import UTC, datetime

import duckdb
import pyarrow as pa
import pytest
from hero_alert_context import (
    HeroAlertContextAssembler,
    HeroAlertIdentity,
    hero_alert_identity,
    log_context_query,
    normalize_log_message,
    root_job_from_execution_uid,
    select_log_evidence,
)


def alert(*, job: str = "/power/hero-example-coord", run: str = "hero-example") -> dict:
    return {
        "status": "firing",
        "labels": {
            "alertname": "TrainingTelemetryGone",
            "notification": "hero-run",
            "operator_behavior": "hero",
            "cluster": "cw-a",
            "job": job,
            "run": run,
        },
    }


def test_hero_alert_identity_keeps_replacement_jobs_in_one_logical_run():
    identity = hero_alert_identity([alert(), alert(job="/power/hero-example-coord-1")])

    assert identity == HeroAlertIdentity(
        cluster="cw-a",
        run_id="hero-example",
        root_jobs=("/power/hero-example-coord", "/power/hero-example-coord-1"),
    )


def test_hero_alert_identity_rejects_a_mixed_or_spoofed_group():
    mixed = alert()
    mixed["labels"]["cluster"] = "cw-b"
    with pytest.raises(ValueError, match="more than one cluster"):
        hero_alert_identity([alert(), mixed])

    with pytest.raises(ValueError, match="disagree"):
        hero_alert_identity([alert(run="hero-someone-else")])

    wrong_behavior = alert()
    wrong_behavior["labels"]["operator_behavior"] = "default"
    with pytest.raises(ValueError, match="operator behavior"):
        hero_alert_identity([wrong_behavior])


def test_execution_uid_recovers_a_prior_coordinator_root():
    assert (
        root_job_from_execution_uid("iris:/rav/hero-example-coord-prior/grug-train-hero-example/0:attempt:2")
        == "/rav/hero-example-coord-prior"
    )
    assert root_job_from_execution_uid("opaque-attempt-id") is None


def test_log_selection_finds_novel_localized_failures_without_a_signature_catalog():
    anchor = datetime(2026, 8, 21, 4, 6, 30, tzinfo=UTC)
    rows = [
        {
            "anchor_at": anchor,
            "observed_at": anchor,
            "source": "stderr",
            "level": 4,
            "message": "[rank1] E0821 04:06:44.462579 820 allocator.cc:418] Stats: Limit: 138.22GiB",
            "occurrences": 1,
            "task_attempts": 1,
            "total_task_attempts": 8,
            "sample_key": "/power/hero-example-coord/train/9:0",
        },
        {
            "anchor_at": anchor,
            "observed_at": anchor,
            "source": "stderr",
            "level": 4,
            "message": (
                "[rank1] E0821 04:06:44.462569 820 transport.cc:414] "
                "collective transport entered poisoned state at 0x27e80b9200"
            ),
            "occurrences": 1,
            "task_attempts": 1,
            "total_task_attempts": 8,
            "sample_key": "/power/hero-example-coord/train/9:0",
        },
        {
            "anchor_at": anchor,
            "observed_at": anchor,
            "source": "stderr",
            "level": 4,
            "message": (
                "[rank3] E0821 04:06:44.470503 825 transport.cc:414] "
                "collective transport entered poisoned state at 0xab4edc200"
            ),
            "occurrences": 1,
            "task_attempts": 1,
            "total_task_attempts": 8,
            "sample_key": "/power/hero-example-coord/train/9:0",
        },
        {
            "anchor_at": anchor,
            "observed_at": anchor,
            "source": "stderr",
            "level": 4,
            "message": "collective barrier failed after a sibling exited",
            "occurrences": 256,
            "task_attempts": 8,
            "total_task_attempts": 8,
            "sample_key": "/power/hero-example-coord/train/0:0",
        },
    ]

    selected = select_log_evidence(rows)

    assert [item["message"] for item in selected] == [rows[0]["message"], rows[1]["message"]]
    assert normalize_log_message(rows[1]["message"]) == normalize_log_message(rows[2]["message"])


def test_nvlink_incident_backtest_is_recovered_by_the_generic_strategy():
    anchor = datetime(2026, 8, 21, 4, 6, 30, tzinfo=UTC)
    actual_incident_row = {
        "anchor_at": anchor,
        "observed_at": datetime(2026, 8, 21, 4, 6, 44, 462000, tzinfo=UTC),
        "source": "stderr",
        "level": 4,
        "message": (
            "[rank1] E0821 04:06:44.462569 820 gpu_cudamallocasync_allocator.cc:414] "
            "cudaFreeAsync failed to free 0x27e80b9200: INTERNAL: CUDA error: : "
            "CUDA_ERROR_NVLINK_UNCORRECTABLE: uncorrectable NVLink error detected during the execution"
        ),
        "occurrences": 1,
        "task_attempts": 1,
        "total_task_attempts": 7,
        "sample_key": "/power/hero-example-coord/train/9:0",
    }

    query = log_context_query(
        HeroAlertIdentity("cw-a", "hero-example", ("/power/hero-example-coord",)),
        [anchor],
    )
    assert "NVLINK" not in query and "CUDA" not in query and "Traceback" not in query
    database = duckdb.connect()
    database.execute("CREATE MACRO to_timestamp_millis(value) AS epoch_ms(value)")
    database.execute(
        'CREATE TABLE "log"(cluster VARCHAR, key VARCHAR, source VARCHAR, data VARCHAR, epoch_ms BIGINT, level INT)'
    )
    incident_ms = round(actual_incident_row["observed_at"].timestamp() * 1000)
    database.execute(
        'INSERT INTO "log" VALUES (?, ?, ?, ?, ?, ?)',
        [
            "cw-a",
            actual_incident_row["sample_key"],
            actual_incident_row["source"],
            actual_incident_row["message"],
            incident_ms,
            actual_incident_row["level"],
        ],
    )
    database.executemany(
        'INSERT INTO "log" VALUES (?, ?, ?, ?, ?, ?)',
        [
            (
                "cw-a",
                f"/power/hero-example-coord/train/{task}:0",
                "stderr",
                "collective barrier failed after a sibling exited",
                incident_ms + task,
                4,
            )
            for task in range(7)
        ],
    )

    selected = select_log_evidence(database.execute(query).fetch_arrow_table().to_pylist())

    assert [item["message"] for item in selected] == [actual_incident_row["message"]]


def test_log_context_redacts_credentials_before_prompt_injection():
    anchor = datetime(2026, 8, 21, 4, 6, 30, tzinfo=UTC)
    selected = select_log_evidence(
        [
            {
                "anchor_at": anchor,
                "observed_at": anchor,
                "source": "stderr",
                "level": 4,
                "message": (
                    "upload failed Authorization=top-secret Bearer eyJheader.payload.signature "
                    "at https://user:password@example.com with AKIA1234567890ABCDEF"
                ),
                "occurrences": 1,
                "task_attempts": 1,
                "total_task_attempts": 8,
                "sample_key": "/power/hero-example-coord/train/9:0",
            }
        ]
    )

    message = selected[0]["message"]
    assert "top-secret" not in message
    assert "eyJheader.payload.signature" not in message
    assert "user:password" not in message
    assert "AKIA1234567890ABCDEF" not in message


class _ContextSource:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def query(self, sql: str, *, max_rows: int) -> pa.Table:
        self.queries.append(sql)
        if 'FROM "telemetry_v1"' in sql:
            return pa.Table.from_pylist(
                [
                    {
                        "execution_uid": "iris:/rav/hero-example-coord-prior/grug-train-hero-example/0:attempt:0",
                        "name": "progress_time_seconds",
                        "value": 600.0,
                        "observed_at": datetime(2026, 8, 21, 4, 6, tzinfo=UTC),
                        "execution_first_at": datetime(2026, 8, 21, 3, 50, tzinfo=UTC),
                        "execution_last_at": datetime(2026, 8, 21, 4, 6, tzinfo=UTC),
                        "execution_rank": 1,
                    }
                ]
            )
        if 'FROM "iris.task_state"' in sql:
            return pa.Table.from_pylist([{"observed_at": datetime(2026, 8, 21, 4, 12, tzinfo=UTC), "running": 0}])
        if 'FROM "iris.task_event"' in sql:
            return pa.Table.from_pylist(
                [
                    {
                        "reason": "OOMKilled",
                        "type": "Normal",
                        "source": "k8s/container",
                        "first_at": datetime(2026, 8, 21, 4, 10, tzinfo=UTC),
                        "last_at": datetime(2026, 8, 21, 4, 10, tzinfo=UTC),
                        "event_count": 1,
                        "affected_tasks": 1,
                        "sample_message": "container exited while checkpointing",
                    },
                    {
                        "reason": "TaskRetryScheduled",
                        "type": "Normal",
                        "source": "iris/controller",
                        "first_at": datetime(2026, 8, 21, 4, 11, tzinfo=UTC),
                        "last_at": datetime(2026, 8, 21, 4, 11, tzinfo=UTC),
                        "event_count": 1,
                        "affected_tasks": 1,
                        "sample_message": "retrying failed task",
                    },
                ]
            )
        if 'FROM "log"' in sql:
            return pa.Table.from_pylist([])
        raise AssertionError(sql)


def test_context_assembler_spans_recent_executions_and_versions_its_schema():
    source = _ContextSource()

    context = asyncio.run(HeroAlertContextAssembler(source, max_rows=1_000).assemble([alert()]))

    assert context["schemaVersion"] == 1
    assert context["status"] == "complete"
    assert context["recentExecutions"][0]["executionUid"].startswith("iris:/rav/hero-example-coord-prior/")
    assert context["scope"]["alertRootJobs"] == ["/power/hero-example-coord"]
    assert context["scope"]["evidenceRootJobs"] == [
        "/power/hero-example-coord",
        "/rav/hero-example-coord-prior",
    ]
    assert [event["reason"] for event in context["taskEvents"]] == ["OOMKilled", "TaskRetryScheduled"]
    log_query = next(sql for sql in source.queries if 'FROM "log"' in sql)
    assert "2026-08-21 04:06:00" in log_query
    assert "/power/hero-example-coord/%" in log_query
    assert "/rav/hero-example-coord-prior/%" in log_query
    event_queries = [sql for sql in source.queries if 'FROM "iris.task_event"' in sql]
    assert len(event_queries) == 2
    assert "/rav/hero-example-coord-prior/%" in event_queries[-1]
