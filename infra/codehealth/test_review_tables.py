# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Round-trip tests for the code-health Finelog tables.

Runs against the in-process native server so the wire contract, schema
registration, and SQL reads are the real ones. Skips when the native extension
is not built.
"""

import dataclasses
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pytest
from finelog.client import LogClient
from finelog.embedded import is_available, require_embedded_server
from finelog.errors import SchemaValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent))

import log_stats
from backfill_from_wandb import to_human_comment, to_invocation
from review import LATEST_HUMAN_COMMENTS_SQL, query_rows
from review_tables import (
    FINDINGS_NAMESPACE,
    HUMAN_COMMENTS_NAMESPACE,
    INVOCATIONS_NAMESPACE,
    HumanComment,
    append_rows,
)

CODEHEALTH_DIR = Path(__file__).resolve().parent


@pytest.fixture
def client(tmp_path):
    if not is_available():
        pytest.skip("finelog native server extension (finelog_server) not available")
    server = require_embedded_server()(log_dir=str(tmp_path / "log-server"))
    try:
        yield LogClient.connect(server.address)
    finally:
        server.stop()


def _event(invocation_id: str = "inv-1", findings: list | None = None) -> dict:
    return {
        "invocation_id": invocation_id,
        "ts": "2026-08-27T12:00:00Z",
        "tool": "pre-commit-review",
        "invocation": {
            "variant": "compose",
            "trigger": "local",
            "agent_cli": "codex",
            "pr_number": 8731,
            "diff_files": 4,
            "elapsed": 91.25,
            "agent_exit_code": 0,
            "timed_out": False,
        },
        "findings": findings if findings is not None else [["a.py", 42, "ml-magic-constant", 0.85, "bare 600"]],
    }


def test_event_round_trips_through_both_namespaces(client):
    """A shipped event is readable back as the values the caller supplied."""
    log_stats.write_event(client, log_stats.fill_defaults(_event()))

    inv = client.query(f'SELECT * FROM "{INVOCATIONS_NAMESPACE}"').to_pylist()
    assert len(inv) == 1
    assert inv[0]["invocation_id"] == "inv-1"
    assert inv[0]["pr_number"] == 8731
    assert inv[0]["elapsed"] == pytest.approx(91.25)
    assert inv[0]["timed_out"] is False
    # finding_count comes from fill_defaults counting the findings list; the event omits it.
    assert inv[0]["finding_count"] == 1

    found = client.query(f'SELECT * FROM "{FINDINGS_NAMESPACE}"').to_pylist()
    assert [(f["code"], f["line"], f["confidence"]) for f in found] == [("ml-magic-constant", 42, pytest.approx(0.85))]
    assert found[0]["invocation_id"] == inv[0]["invocation_id"]


def test_clean_run_with_no_findings_is_still_recorded(client):
    """A clean run still writes an invocation row, with no findings namespace."""
    log_stats.write_event(client, log_stats.fill_defaults(_event(findings=[])))

    inv = client.query(f'SELECT finding_count FROM "{INVOCATIONS_NAMESPACE}"').to_pylist()
    assert inv == [{"finding_count": 0}]
    with pytest.raises(SchemaValidationError, match="not found"):
        client.query(f'SELECT * FROM "{FINDINGS_NAMESPACE}"')


def test_append_rows_raises_when_the_server_rejects_the_schema(client):
    """A rejected registration must fail loudly.

    `Table.flush` reports that the client queue drained, so a schema the server
    refuses is otherwise logged by the flush thread and dropped silently.
    """

    @dataclasses.dataclass(frozen=True)
    class NoKeyColumn:
        ts: dt.datetime
        name: str

    with pytest.raises(RuntimeError, match="rejected the schema"):
        append_rows(
            client,
            "codehealth.autolint.rejected",
            NoKeyColumn,
            [NoKeyColumn(ts=dt.datetime.now(dt.UTC), name="x")],
            flush_timeout=15.0,
        )


def test_query_rows_is_empty_before_the_first_write(client):
    """query_rows returns [] for a namespace that has never been written."""
    assert query_rows(client, LATEST_HUMAN_COMMENTS_SQL, HUMAN_COMMENTS_NAMESPACE) == []


def test_reader_takes_the_newest_row_per_comment(client):
    """Both re-emitted rows are stored; the read returns only the newest per comment."""
    base = dict(
        pr_number=8731,
        pr_title="A PR",
        merged_at=None,
        author="someone",
        comment_id=55,
        comment_type="inline",
        file="a.py",
        line=10,
        body="please rename this",
        catchable_strict=False,
        confidence=0.5,
        reason="first pass",
    )
    first = HumanComment(
        ts=dt.datetime(2026, 8, 1, tzinfo=dt.UTC), comment_class="other", catchable_generous=False, **base
    )
    second = HumanComment(
        ts=dt.datetime(2026, 8, 2, tzinfo=dt.UTC), comment_class="lint", catchable_generous=True, **base
    )
    append_rows(client, HUMAN_COMMENTS_NAMESPACE, HumanComment, [first])
    append_rows(client, HUMAN_COMMENTS_NAMESPACE, HumanComment, [second])

    assert client.query(f'SELECT count(*) AS n FROM "{HUMAN_COMMENTS_NAMESPACE}"').to_pylist()[0]["n"] == 2
    latest = query_rows(client, LATEST_HUMAN_COMMENTS_SQL, HUMAN_COMMENTS_NAMESPACE)
    assert [(r["comment_id"], r["comment_class"], r["catchable_generous"]) for r in latest] == [(55, "lint", True)]
    # query_rows normalizes finelog's naive timestamps so window filters work.
    assert latest[0]["ts"] == dt.datetime(2026, 8, 2, tzinfo=dt.UTC)


def test_disable_flag_skips_the_write_without_contacting_a_server():
    """MARIN_REVIEW_STATS=0 must short-circuit before any connection attempt."""
    result = subprocess.run(
        [sys.executable, str(CODEHEALTH_DIR / "log_stats.py"), "--finelog-url", "http://127.0.0.1:1"],
        input=json.dumps(_event()).encode(),
        capture_output=True,
        env={"PATH": "/usr/bin:/bin", "MARIN_REVIEW_STATS": "0"},
    )
    assert result.returncode == 0, result.stderr.decode()


def test_legacy_wandb_rows_convert_to_typed_rows():
    """W&B stored every number as a float and blanks as empty strings."""
    invocation = to_invocation(
        {
            "ts": "2026-08-27T12:00:00Z",
            "invocation_id": "inv-9",
            "tool": "pre-commit-review",
            "variant": "",
            "pr_number": None,
            "diff_files": 4.0,
            "elapsed": 91.25,
            "agent_exit_code": 0.0,
            "timed_out": False,
        }
    )
    assert invocation.diff_files == 4
    assert invocation.agent_exit_code == 0
    assert invocation.variant is None
    assert invocation.pr_number is None
    assert invocation.ts == dt.datetime(2026, 8, 27, 12, 0, tzinfo=dt.UTC)

    comment = to_human_comment(
        {
            "ts": "2026-08-27T12:00:00Z",
            "pr_number": 8731.0,
            "comment_id": 55.0,
            "comment_type": "inline",
            "class": "lint",
            "catchable_strict": True,
            "catchable_generous": False,
            "confidence": 0.9,
            "merged_at": "",
        }
    )
    # The `class` column is renamed on the way in; `class` is a Python keyword.
    assert comment.comment_class == "lint"
    assert comment.pr_number == 8731
    assert comment.merged_at is None
