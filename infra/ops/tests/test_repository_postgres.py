# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

import psycopg
import pytest
from ops_workflow.grafana_source import snapshot_from_api_alerts
from ops_workflow.migrations import Connection as MigrationConnection
from ops_workflow.migrations import apply_migrations, migration_plan
from ops_workflow.repository import ArchiveResult, OpsRepository
from ops_workflow.result import parse_ops_result
from ops_workflow.slack import escalation_draft

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


@pytest.fixture
async def repository() -> AsyncIterator[OpsRepository]:
    assert DATABASE_URL is not None
    value = OpsRepository(DATABASE_URL, repo_revision="test", skill_revision="test")
    try:
        yield value
    finally:
        await value.close()


def _api_alert() -> dict[str, object]:
    return {
        "annotations": {"summary": "Nameserver limits were exceeded"},
        "endsAt": "2026-07-21T17:00:00Z",
        "fingerprint": "2b05ef3b1641c79a",
        "generatorURL": "https://grafana.oa.dev/alerting/grafana/dns-config-forming/view?orgId=1",
        "labels": {
            "alertname": "DNSConfigForming",
            "cluster": "cw-us-east-08a",
            "kind": "Pod",
            "namespace": "kube-system",
            "severity": "warning",
        },
        "receivers": [{"name": "ops-critical"}],
        "startsAt": "2026-07-21T15:31:37Z",
        "status": {"inhibitedBy": [], "silencedBy": [], "state": "active"},
        "updatedAt": "2026-07-21T15:32:37Z",
    }


@pytest.mark.anyio
async def test_new_fingerprint_queues_one_follow_up_while_group_turn_is_running(repository: OpsRepository):
    now = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)
    first = await repository.reconcile_grafana_snapshot(snapshot_from_api_alerts([_api_alert()], observed_at=now))
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

    original_labels = _api_alert()["labels"]
    assert isinstance(original_labels, dict)
    new_alert = {
        **_api_alert(),
        "fingerprint": "83b8bf0c924cbb8e",
        "labels": {**original_labels, "name": "dirty-frag-mit-f4826"},
        "annotations": {"summary": "Nameserver limits exceeded for dirty-frag-mit"},
    }
    second_results = await repository.reconcile_grafana_snapshot(
        snapshot_from_api_alerts([_api_alert(), new_alert], observed_at=now + timedelta(minutes=1))
    )
    assert len(second_results) == 1
    second = second_results[0]

    assert second.case_ids == (case_id,)
    assert second.signal_dispositions["83b8bf0c924cbb8e"] == "created"
    assert second.queued_case_ids == (case_id,)
    detail = await repository.case_detail(case_id)
    assert detail is not None
    turns = detail["turns"]
    assert isinstance(turns, list)
    assert [turn["state"] for turn in turns] == ["running", "queued"]


@pytest.mark.anyio
async def test_successful_snapshots_deduplicate_resolve_and_reopen_a_grafana_instance(repository: OpsRepository):
    now = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)

    first = await repository.reconcile_grafana_snapshot(snapshot_from_api_alerts([_api_alert()], observed_at=now))
    assert len(first) == 1
    assert first[0].signal_dispositions == {"2b05ef3b1641c79a": "created"}
    assert len(first[0].queued_case_ids) == 1
    case_id = first[0].case_ids[0]

    unchanged = await repository.reconcile_grafana_snapshot(
        snapshot_from_api_alerts([_api_alert()], observed_at=now + timedelta(minutes=1))
    )
    assert unchanged[0].case_ids == (case_id,)
    assert unchanged[0].signal_dispositions == {"2b05ef3b1641c79a": "updated"}
    assert unchanged[0].queued_case_ids == ()

    first_absence = await repository.reconcile_grafana_snapshot(
        snapshot_from_api_alerts([], observed_at=now + timedelta(minutes=2))
    )
    assert first_absence == ()
    detail = await repository.case_detail(case_id)
    assert detail is not None
    signals = detail["signals"]
    assert isinstance(signals, list)
    assert signals[0]["state"] == "firing"
    assert signals[0]["missing_successful_polls"] == 1

    resolved = await repository.reconcile_grafana_snapshot(
        snapshot_from_api_alerts([], observed_at=now + timedelta(minutes=3))
    )
    assert resolved[0].signal_dispositions == {"2b05ef3b1641c79a": "resolved"}
    detail = await repository.case_detail(case_id)
    assert detail is not None
    signals = detail["signals"]
    assert isinstance(signals, list)
    assert signals[0]["state"] == "resolved"

    reopened = await repository.reconcile_grafana_snapshot(
        snapshot_from_api_alerts([_api_alert()], observed_at=now + timedelta(minutes=4))
    )
    assert reopened[0].signal_dispositions == {"2b05ef3b1641c79a": "reopened"}
    assert reopened[0].queued_case_ids == (case_id,)
    detail = await repository.case_detail(case_id)
    assert detail is not None
    signals = detail["signals"]
    assert isinstance(signals, list)
    assert signals[0]["generation"] == 2


@pytest.mark.anyio
async def test_only_one_service_instance_reconciles_each_poll_minute(repository: OpsRepository):
    now = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)

    first = await repository.reconcile_grafana_snapshot(snapshot_from_api_alerts([_api_alert()], observed_at=now))
    competing = await repository.reconcile_grafana_snapshot(
        snapshot_from_api_alerts([], observed_at=now + timedelta(seconds=20))
    )

    assert len(first) == 1
    assert competing == ()
    detail = await repository.case_detail(first[0].case_ids[0])
    assert detail is not None
    signals = detail["signals"]
    assert isinstance(signals, list)
    assert signals[0]["missing_successful_polls"] == 0


@pytest.mark.anyio
async def test_archive_distinguishes_active_already_archived_and_missing_cases(repository: OpsRepository):
    now = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)

    result = await repository.reconcile_grafana_snapshot(snapshot_from_api_alerts([_api_alert()], observed_at=now))
    case_id = result[0].case_ids[0]
    assert await repository.archive_case(case_id=case_id, actor="test@example.com") == ArchiveResult.ARCHIVED
    assert await repository.archive_case(case_id=case_id, actor="test@example.com") == ArchiveResult.ALREADY_ARCHIVED
    assert await repository.archive_case(case_id="00000000-0000-0000-0000-000000000000", actor="test@example.com") == (
        ArchiveResult.NOT_FOUND
    )
    detail = await repository.case_detail(case_id)
    assert detail is not None
    turns = detail["turns"]
    assert isinstance(turns, list)
    assert turns[0]["state"] == "cancelled"

    active_case_id = await repository.create_question(text="Is the cluster healthy?", actor="test@example.com")
    turn = await repository.claim_next_turn()
    assert turn is not None
    assert await repository.archive_case(case_id=active_case_id, actor="test@example.com") == ArchiveResult.ACTIVE_TURN


@pytest.mark.anyio
async def test_slack_escalation_is_durable_and_deduplicated_by_signal_generation(repository: OpsRepository):
    now = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)
    ingested = await repository.reconcile_grafana_snapshot(snapshot_from_api_alerts([_api_alert()], observed_at=now))
    case_id = ingested[0].case_ids[0]

    turn = await repository.claim_next_turn()
    assert turn is not None
    turn_id = str(turn["id"])
    await repository.turn_started(
        turn_id=turn_id,
        loom_session_id="loom-test",
        loom_session_url="https://loom.test/s/loom-test",
        loom_turn_number=0,
    )
    await _finish_with_escalation(repository, case_id=case_id, turn_id=turn_id, artifact_revision=1)

    detail = await repository.case_detail(case_id)
    assert detail is not None
    case = cast(Mapping[str, object], detail["case"])
    escalations = cast(Sequence[Mapping[str, object]], detail["escalations"])
    assert case["outcome"] == "action_recommended"
    assert len(escalations) == 1
    delivery = await repository.claim_slack_escalation()
    assert delivery is not None
    assert delivery.attempts == 1
    await repository.slack_escalation_sent(delivery.id)

    follow_up_id = await repository.enqueue_follow_up(
        case_id=case_id,
        text="Recheck the same incident.",
        actor="operator@example.com",
    )
    follow_up = await repository.claim_next_turn()
    assert follow_up is not None
    assert str(follow_up["id"]) == follow_up_id
    await repository.turn_started(
        turn_id=follow_up_id,
        loom_session_id="loom-test",
        loom_session_url="https://loom.test/s/loom-test",
        loom_turn_number=1,
    )
    await _finish_with_escalation(repository, case_id=case_id, turn_id=follow_up_id, artifact_revision=2)

    detail = await repository.case_detail(case_id)
    assert detail is not None
    escalations = cast(Sequence[Mapping[str, object]], detail["escalations"])
    assert [item["state"] for item in escalations] == ["sent"]
    assert await repository.claim_slack_escalation() is None


async def _finish_with_escalation(
    repository: OpsRepository, *, case_id: str, turn_id: str, artifact_revision: int
) -> None:
    result = parse_ops_result(
        json.dumps(
            {
                "schema_version": 2,
                "case_id": case_id,
                "ops_turn_id": turn_id,
                "outcome": "action_recommended",
                "summary": "The warning requires operator attention.",
                "evidence": [],
                "action_taken": "none",
                "recommended_next_step": "Inspect disk consumers.",
                "escalation": {"severity": "error", "reason": "Automated cleanup freed no space."},
            }
        ),
        case_id=case_id,
        turn_id=turn_id,
    )
    detail = await repository.case_detail(case_id)
    assert detail is not None
    case = cast(Mapping[str, object], detail["case"])
    signals = cast(Sequence[Mapping[str, object]], detail["signals"])
    draft = escalation_draft(
        result=result,
        case=case,
        signals=signals,
        public_url="https://ops.oa.dev",
    )
    assert draft is not None
    await repository.finish_turn(
        turn_id=turn_id,
        result=result,
        artifact_revision=artifact_revision,
        escalation=draft,
    )
