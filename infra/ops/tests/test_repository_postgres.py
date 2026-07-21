# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

import psycopg
import pytest
from ops_workflow.grafana_source import snapshot_from_rows
from ops_workflow.migrations import Connection as MigrationConnection
from ops_workflow.migrations import apply_migrations, migration_plan
from ops_workflow.repository import OpsRepository

DATABASE_URL = os.environ.get("OPS_TEST_DATABASE_URL")
MIGRATIONS = Path(__file__).parent.parent / "migrations"

pytestmark = pytest.mark.skipif(DATABASE_URL is None, reason="OPS_TEST_DATABASE_URL is not configured")


@pytest.fixture(autouse=True)
def migrated_database() -> None:
    assert DATABASE_URL is not None
    assert urlsplit(DATABASE_URL).path.endswith("_test"), "repository tests require a dedicated *_test database"
    with psycopg.connect(DATABASE_URL) as connection:
        connection.execute("DROP SCHEMA public CASCADE")
        connection.execute("CREATE SCHEMA public")
        connection.commit()
        apply_migrations(cast(MigrationConnection, connection), migration_plan(MIGRATIONS))


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _poll_row() -> dict[str, object]:
    return {
        "rule_org_id": 1,
        "rule_uid": "dns-config-forming",
        "labels": '[["cluster","cw-us-east-08a"],["kind","Pod"],["namespace","kube-system"]]',
        "labels_hash": "2b05ef3b1641c79a",
        "current_state_since": 1_784_647_897,
        "last_eval_time": 1_784_647_957,
        "fired_at": 1_784_647_897,
        "instance_annotations": '{"summary":"Nameserver limits were exceeded"}',
        "last_result": '{"values":{"A":6548,"C":1}}',
        "rule_title": "DNSConfigForming",
        "rule_labels": '{"severity":"warning"}',
        "rule_annotations": "{}",
    }


@pytest.mark.anyio
async def test_new_fingerprint_queues_one_follow_up_while_group_turn_is_running():
    assert DATABASE_URL is not None
    repository = OpsRepository(DATABASE_URL, repo_revision="test", skill_revision="test")
    now = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)
    first = await repository.reconcile_grafana_snapshot(snapshot_from_rows([_poll_row()], observed_at=now))
    assert len(first) == 1
    first_result = first[0]
    assert len(first_result.case_ids) == 1
    case_id = first_result.case_ids[0]

    turn = await repository.claim_next_turn()
    assert turn is not None
    await repository.turn_started(
        turn_id=str(turn["id"]),
        loom_session_id="loom-test",
        loom_session_url="https://loom.test/s/loom-test",
        loom_turn_number=0,
    )

    new_row = {
        **_poll_row(),
        "labels_hash": "83b8bf0c924cbb8e",
        "labels": (
            '[["cluster","cw-us-east-08a"],["kind","Pod"],'
            '["name","dirty-frag-mit-f4826"],["namespace","kube-system"]]'
        ),
        "instance_annotations": '{"summary":"Nameserver limits exceeded for dirty-frag-mit"}',
    }
    second_results = await repository.reconcile_grafana_snapshot(
        snapshot_from_rows([_poll_row(), new_row], observed_at=now + timedelta(minutes=1))
    )
    assert len(second_results) == 1
    second = second_results[0]

    assert second.case_ids == (case_id,)
    assert second.signal_dispositions["1:dns-config-forming:83b8bf0c924cbb8e"] == "created"
    assert second.queued_case_ids == (case_id,)
    detail = await repository.case_detail(case_id)
    assert detail is not None
    turns = detail["turns"]
    assert isinstance(turns, list)
    assert [turn["state"] for turn in turns] == ["running", "queued"]


@pytest.mark.anyio
async def test_successful_snapshots_deduplicate_resolve_and_reopen_a_grafana_instance():
    assert DATABASE_URL is not None
    repository = OpsRepository(DATABASE_URL, repo_revision="test", skill_revision="test")
    now = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)

    first = await repository.reconcile_grafana_snapshot(snapshot_from_rows([_poll_row()], observed_at=now))
    assert len(first) == 1
    assert first[0].signal_dispositions == {"1:dns-config-forming:2b05ef3b1641c79a": "created"}
    assert len(first[0].queued_case_ids) == 1
    case_id = first[0].case_ids[0]

    unchanged = await repository.reconcile_grafana_snapshot(
        snapshot_from_rows([_poll_row()], observed_at=now + timedelta(minutes=1))
    )
    assert unchanged[0].case_ids == (case_id,)
    assert unchanged[0].signal_dispositions == {"1:dns-config-forming:2b05ef3b1641c79a": "updated"}
    assert unchanged[0].queued_case_ids == ()

    first_absence = await repository.reconcile_grafana_snapshot(
        snapshot_from_rows([], observed_at=now + timedelta(minutes=2))
    )
    assert first_absence == ()
    detail = await repository.case_detail(case_id)
    assert detail is not None
    signals = detail["signals"]
    assert isinstance(signals, list)
    assert signals[0]["state"] == "firing"
    assert signals[0]["missing_successful_polls"] == 1

    resolved = await repository.reconcile_grafana_snapshot(
        snapshot_from_rows([], observed_at=now + timedelta(minutes=3))
    )
    assert resolved[0].signal_dispositions == {"1:dns-config-forming:2b05ef3b1641c79a": "resolved"}
    detail = await repository.case_detail(case_id)
    assert detail is not None
    signals = detail["signals"]
    assert isinstance(signals, list)
    assert signals[0]["state"] == "resolved"

    reopened = await repository.reconcile_grafana_snapshot(
        snapshot_from_rows([_poll_row()], observed_at=now + timedelta(minutes=4))
    )
    assert reopened[0].signal_dispositions == {"1:dns-config-forming:2b05ef3b1641c79a": "reopened"}
    assert reopened[0].queued_case_ids == (case_id,)
    detail = await repository.case_detail(case_id)
    assert detail is not None
    signals = detail["signals"]
    assert isinstance(signals, list)
    assert signals[0]["generation"] == 2


@pytest.mark.anyio
async def test_only_one_service_instance_reconciles_each_poll_minute():
    assert DATABASE_URL is not None
    repository = OpsRepository(DATABASE_URL, repo_revision="test", skill_revision="test")
    now = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)

    first = await repository.reconcile_grafana_snapshot(snapshot_from_rows([_poll_row()], observed_at=now))
    competing = await repository.reconcile_grafana_snapshot(
        snapshot_from_rows([], observed_at=now + timedelta(seconds=20))
    )

    assert len(first) == 1
    assert competing == ()
    detail = await repository.case_detail(first[0].case_ids[0])
    assert detail is not None
    signals = detail["signals"]
    assert isinstance(signals, list)
    assert signals[0]["missing_successful_polls"] == 0
