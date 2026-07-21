# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import hmac
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

import psycopg
import pytest
from ops_workflow.grafana import verify_grafana_webhook
from ops_workflow.migrations import Connection as MigrationConnection
from ops_workflow.migrations import apply_migrations, migration_plan
from ops_workflow.repository import OpsRepository

DATABASE_URL = os.environ.get("OPS_TEST_DATABASE_URL")
FIXTURE = Path(__file__).parent.parent / "fixtures" / "dns-warning-firing.json"
MIGRATIONS = Path(__file__).parent.parent / "migrations"
SECRET = b"repository-test-secret"

pytestmark = pytest.mark.skipif(DATABASE_URL is None, reason="OPS_TEST_DATABASE_URL is not configured")


@pytest.fixture(scope="module", autouse=True)
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


def _verified(payload: dict[str, object], now: datetime):
    body = json.dumps(payload, sort_keys=True).encode()
    timestamp = str(int(now.timestamp()))
    signature = hmac.new(SECRET, timestamp.encode() + b":" + body, hashlib.sha256).hexdigest()
    return verify_grafana_webhook(
        body,
        signature=signature,
        timestamp=timestamp,
        secret=SECRET,
        now=now,
    )


@pytest.mark.anyio
async def test_new_fingerprint_queues_one_follow_up_while_group_turn_is_running():
    assert DATABASE_URL is not None
    repository = OpsRepository(DATABASE_URL, repo_revision="test", skill_revision="test")
    first_payload = json.loads(FIXTURE.read_bytes())
    now = datetime.now(UTC)
    first = await repository.ingest(_verified(first_payload, now), key_id="test-key")
    assert len(first.case_ids) == 1
    case_id = first.case_ids[0]

    turn = await repository.claim_next_turn()
    assert turn is not None
    await repository.turn_started(
        turn_id=str(turn["id"]),
        loom_session_id="loom-test",
        loom_session_url="https://loom.test/s/loom-test",
        loom_turn_number=0,
    )

    second_payload = json.loads(FIXTURE.read_bytes())
    third_alert = dict(second_payload["alerts"][0])
    third_alert["fingerprint"] = "83b8bf0c924cbb8e"
    third_alert["labels"] = {
        **third_alert["labels"],
        "namespace": "kube-system",
        "name": "dirty-frag-mit-f4826",
    }
    third_alert["annotations"] = {"summary": "Nameserver limits exceeded for dirty-frag-mit"}
    second_payload["alerts"].append(third_alert)
    second_payload["title"] = "[FIRING:3] DNSConfigForming cw-us-east-08a (warning)"

    second = await repository.ingest(_verified(second_payload, now + timedelta(seconds=1)), key_id="test-key")

    assert second.case_ids == (case_id,)
    assert second.signal_dispositions["83b8bf0c924cbb8e"] == "created"
    assert second.queued_case_ids == (case_id,)
    detail = await repository.case_detail(case_id)
    assert detail is not None
    turns = detail["turns"]
    assert isinstance(turns, list)
    assert [turn["state"] for turn in turns] == ["running", "queued"]
