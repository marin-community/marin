# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import sqlalchemy
from sqlalchemy.pool import StaticPool

from infra.codehealth import refinement_tools, review_store, rule_probe
from infra.codehealth.github_review_corpus import (
    ChangedFileRecord,
    PullRequestBundle,
    PullRequestRecord,
    ReviewEventRecord,
)
from infra.lint.catalog import DEFAULT_CATALOG_DIR, load_catalog, render_lane

REPOSITORY = "marin-community/marin"
NOW = dt.datetime(2026, 8, 31, 12, tzinfo=dt.UTC)


@pytest.fixture
def engine() -> sqlalchemy.Engine:
    database = sqlalchemy.create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    review_store.create_schema(database)
    return database


def _bundle(pr_number: int, *, body: str = "let exceptions flow") -> PullRequestBundle:
    head_sha = f"head-{pr_number}"
    pull_request = PullRequestRecord(
        repository=REPOSITORY,
        number=pr_number,
        node_id=f"PR_{pr_number}",
        url=f"https://github.com/{REPOSITORY}/pull/{pr_number}",
        title=f"Pull request {pr_number}",
        body="Complete pull-request body",
        state="open",
        draft=False,
        author="author",
        author_type="User",
        author_association="MEMBER",
        created_at="2026-08-01T00:00:00Z",
        updated_at="2026-08-30T00:00:00Z",
        closed_at=None,
        merged_at=None,
        base_ref="main",
        base_sha="base",
        head_ref="feature",
        head_sha=head_sha,
        additions=1,
        deletions=1,
        changed_files=0,
        commits=0,
        review_comments=1,
        issue_comments=0,
        commit_shas=(),
        diff_path=f"diffs/{pr_number}.diff",
    )
    event = ReviewEventRecord(
        event_id=f"{REPOSITORY}:inline_comment:{pr_number}01",
        kind="inline_comment",
        database_id=pr_number * 100 + 1,
        node_id=f"comment-{pr_number}",
        repository=REPOSITORY,
        pr_number=pr_number,
        pr_author="author",
        author="rjpower",
        author_type="User",
        author_association="MEMBER",
        body=body,
        state=None,
        created_at="2026-08-30T01:00:00Z",
        updated_at="2026-08-30T01:00:00Z",
        submitted_at=None,
        source_url=f"https://github.com/{REPOSITORY}/pull/{pr_number}#discussion_r{pr_number}01",
        review_id=55,
        thread_id=f"thread-{pr_number}",
        parent_comment_id=None,
        thread_is_resolved=False,
        thread_is_outdated=False,
        thread_resolved_by=None,
        path="example.py",
        side="RIGHT",
        line=150,
        original_line=150,
        start_side=None,
        start_line=None,
        original_start_line=None,
        commit_id=head_sha,
        original_commit_id=head_sha,
        diff_hunk="@@ -149,2 +149,2 @@\n-old\n+new",
        is_bot=False,
        is_agent_marked=False,
        is_human=True,
        in_window=True,
    )
    return PullRequestBundle(
        pull_request=pull_request,
        events=(event,),
        threads=(),
        files=(),
        commits=(),
        diff="diff --git a/example.py b/example.py\n-old\n+new\n",
    )


def _sync(engine: sqlalchemy.Engine) -> review_store.SyncRun:
    return review_store.start_or_resume_sync(engine, REPOSITORY, now=NOW)


def test_store_preserves_versions_and_joins_human_and_lint_activity(engine: sqlalchemy.Engine) -> None:
    run = _sync(engine)
    first = _bundle(8629).model_copy(
        update={
            "files": (
                ChangedFileRecord(
                    pr_number=8629, filename="old.py", status="modified", additions=1, deletions=1, changes=2
                ),
            )
        }
    )
    second = _bundle(8629, body="this try/catch doesn't add anything").model_copy(
        update={
            "files": (
                ChangedFileRecord(
                    pr_number=8629, filename="new.py", status="added", additions=1, deletions=0, changes=1
                ),
            )
        }
    )
    review_store.store_bundle(engine, run.sync_id, first, observed_at=NOW)
    review_store.store_bundle(engine, run.sync_id, second, observed_at=NOW + dt.timedelta(minutes=1))
    review_store.store_telemetry(
        engine,
        [
            {
                "invocation_id": "run-1",
                "ts": "2026-08-30T02:00:00Z",
                "pr_number": 8629,
                "head_sha": "head-8629",
                "lint_catalog_sha": "catalog-a",
                "agent_exit_code": 0,
                "timed_out": False,
                "finding_count": 1,
            }
        ],
        [
            {
                "invocation_id": "run-1",
                "ts": "2026-08-30T02:00:00Z",
                "pr_number": 8629,
                "code": "ml-exception-swallow",
            }
        ],
    )

    rows = review_store.list_pull_request_activity(
        engine,
        start=NOW - dt.timedelta(days=30),
        end=NOW,
        repository=REPOSITORY,
        require_human=True,
        require_lint=True,
    )
    assert [(row.number, row.human_events, row.lint_runs, row.rule_codes) for row in rows] == [
        (8629, 1, 1, ("ml-exception-swallow",))
    ]
    context = review_store.review_context(engine, second.events[0].event_id)
    assert context.event.body == "this try/catch doesn't add anything"
    assert context.diff == second.diff
    with engine.begin() as connection:
        versions = connection.execute(
            sqlalchemy.select(sqlalchemy.func.count()).select_from(review_store.review_event_versions)
        ).scalar_one()
        current_files = connection.execute(sqlalchemy.select(review_store.changed_files.c.filename)).scalars().all()
    assert versions == 2
    assert current_files == ["new.py"]


def test_failed_sync_resumes_same_window_after_last_committed_pull_request(engine: sqlalchemy.Engine) -> None:
    first = _sync(engine)
    review_store.store_bundle(engine, first.sync_id, _bundle(1), observed_at=NOW)
    review_store.fail_sync(engine, first.sync_id, "transient GitHub failure")

    resumed = review_store.start_or_resume_sync(engine, REPOSITORY, now=NOW + dt.timedelta(hours=1))
    assert resumed.sync_id == first.sync_id
    assert resumed.window_start == first.window_start.replace(tzinfo=None)
    assert resumed.window_end == first.window_end.replace(tzinfo=None)
    assert review_store.completed_pull_requests(engine, resumed.sync_id) == {1}

    review_store.store_bundle(engine, resumed.sync_id, _bundle(2), observed_at=NOW + dt.timedelta(hours=1))
    review_store.complete_sync(
        engine,
        resumed.sync_id,
        candidate_pull_requests=2,
        github_usage={"graphql_points": 4, "rest_requests": 2},
        finelog_watermark={"deployment": "test"},
        completed_at=NOW + dt.timedelta(hours=1),
    )
    with engine.begin() as connection:
        runs = connection.execute(sqlalchemy.select(review_store.sync_runs)).mappings().all()
    assert len(runs) == 1
    assert runs[0]["status"] == review_store.SyncStatus.COMPLETE.value
    assert runs[0]["window_start"] == (NOW - dt.timedelta(days=30)).replace(tzinfo=None)
    assert review_store.completed_pull_requests(engine, resumed.sync_id) == {1, 2}


def test_context_fetches_and_caches_bounded_source_window(
    engine: sqlalchemy.Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _sync(engine)
    bundle = _bundle(10)
    review_store.store_bundle(engine, run.sync_id, bundle, observed_at=NOW)
    source = "\n".join(f"line {line}" for line in range(1, 301))
    calls = 0

    def fake_fetch(_repository: str, _path: str, _commit_sha: str) -> str:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(refinement_tools, "_fetch_github_file", fake_fetch)
    stored_context = review_store.review_context(engine, bundle.events[0].event_id)
    context = refinement_tools.ensure_source_context(engine, stored_context)
    cached = refinement_tools.ensure_source_context(engine, context)

    assert calls == 1
    assert context.source_start_line == 50
    assert context.source_end_line == 250
    assert context.source is not None
    assert "    50  line 50" in context.source
    assert "   250  line 250" in context.source
    assert cached.context_sha == context.context_sha


def test_structured_catalog_is_lossless_and_independently_editable(tmp_path: Path) -> None:
    catalog = load_catalog()
    assert len(catalog.rules) == 63
    rendered = render_lane(catalog, "robustness")
    assert "`ml-exception-swallow`" in rendered
    assert catalog.shared_prompt not in rendered

    copied = tmp_path / "lint"
    shutil.copytree(DEFAULT_CATALOG_DIR, copied)
    path = copied / "rules" / "robustness" / "ml-exception-swallow.yaml"
    path.write_text(
        path.read_text().replace(
            "'`except Exception` returning `None` / a default'",
            "'Exception boundary hides the original error'",
        )
    )
    edited = load_catalog(copied)
    assert edited.rule("ml-exception-swallow").title == "Exception boundary hides the original error"


def test_rule_probe_records_model_rule_context_and_idempotent_result(
    engine: sqlalchemy.Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _sync(engine)
    bundle = _bundle(22)
    review_store.store_bundle(engine, run.sync_id, bundle, observed_at=NOW)
    context = review_store.review_context(engine, bundle.events[0].event_id)
    prompt = rule_probe.build_probe_prompt(load_catalog().rule("ml-exception-swallow"), context)
    assert bundle.events[0].body not in prompt
    assert "\\n-old\\n+new" in prompt
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        output = Path(args[args.index("--output-last-message") + 1])
        output.write_text(json.dumps({"fired": True, "confidence": 0.9, "finding": "wrapper hides native error"}))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(rule_probe.subprocess, "run", fake_run)
    first = rule_probe.run_rule_probe(
        engine,
        load_catalog(),
        context,
        rule_code="ml-exception-swallow",
        model="gpt-5.6-luna",
        effort="low",
        idempotency_key="probe-22",
    )
    second = rule_probe.run_rule_probe(
        engine,
        load_catalog(),
        context,
        rule_code="ml-exception-swallow",
        model="gpt-5.6-luna",
        effort="low",
        idempotency_key="probe-22",
    )

    assert len(calls) == 1
    assert "gpt-5.6-luna" in calls[0]
    assert first == second
    assert first.context_sha == context.context_sha
    assert first.catalog_sha
