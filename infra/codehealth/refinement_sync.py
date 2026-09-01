# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resumable weekly synchronization of review and lint activity into PostgreSQL."""

from __future__ import annotations

import datetime as dt
import json

import click
from pydantic import BaseModel, ConfigDict
from sqlalchemy.engine import Engine

from infra.lint.catalog import LintCatalog, catalog_sha, load_catalog

from .github_review_corpus import GitHubClient, GitHubUsage, collect_corpus
from .review_store import (
    DEFAULT_BACKFILL_DAYS,
    LintFindingRecord,
    LintInvocationRecord,
    LintRecordRows,
    cached_pull_request_fingerprints,
    checkpoint_reused_pull_request,
    complete_sync,
    completed_pull_requests,
    database_config_from_environment,
    database_engine,
    fail_sync,
    start_or_resume_sync,
    store_bundle,
    store_catalog_snapshot,
    store_telemetry,
    utc_iso,
)
from .review_tables import (
    DEFAULT_BOT_LOGINS,
    DEFAULT_DEPLOYMENT,
    DEFAULT_REPOSITORY,
    FINDINGS_NAMESPACE,
    INVOCATIONS_NAMESPACE,
    open_tables_client,
    query_rows,
)

SQL_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


class SyncResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sync_id: str
    repository: str
    window_start: str
    window_end: str
    candidate_pull_requests: int
    persisted_pull_requests: int
    reused_pull_requests: int
    github_usage: GitHubUsage
    lint_invocations: int
    lint_findings: int


def load_lint_telemetry(deployment: str, start: dt.datetime, end: dt.datetime) -> LintRecordRows:
    """Read the exact Finelog window mirrored into the review store."""
    start_sql = start.astimezone(dt.UTC).strftime(SQL_TIMESTAMP_FORMAT)
    end_sql = end.astimezone(dt.UTC).strftime(SQL_TIMESTAMP_FORMAT)
    with open_tables_client(deployment) as client:
        invocations = query_rows(
            client,
            f'SELECT * FROM "{INVOCATIONS_NAMESPACE}" '
            f"WHERE ts >= TIMESTAMP '{start_sql}' AND ts < TIMESTAMP '{end_sql}'",
            INVOCATIONS_NAMESPACE,
        )
        findings = query_rows(
            client,
            f'SELECT * FROM "{FINDINGS_NAMESPACE}" '
            f"WHERE ts >= TIMESTAMP '{start_sql}' AND ts < TIMESTAMP '{end_sql}'",
            FINDINGS_NAMESPACE,
        )
    return LintRecordRows(
        invocations=[LintInvocationRecord.model_validate(row) for row in invocations],
        findings=[LintFindingRecord.model_validate(row) for row in findings],
    )


def _catalog_record(catalog: LintCatalog) -> dict[str, object]:
    return {
        "shared_prompt": catalog.shared_prompt,
        "lanes": [
            {
                "name": lane.name,
                "prompt": lane.prompt,
                "include_complexity_leads": lane.include_complexity_leads,
                "min_diff_lines": lane.min_diff_lines,
                "rules": [
                    {
                        "code": rule.code,
                        "title": rule.title,
                        "prompt": rule.prompt,
                        "minimum_confidence": rule.minimum_confidence,
                    }
                    for rule in lane.rules
                ],
            }
            for lane in catalog.lanes
        ],
    }


def sync_review_activity(
    engine: Engine,
    *,
    repository: str,
    deployment: str,
    days: int = DEFAULT_BACKFILL_DAYS,
    now: dt.datetime | None = None,
    github_client: GitHubClient | None = None,
    telemetry: LintRecordRows | None = None,
) -> SyncResult:
    """Run or resume one fixed review window and checkpoint every reconciled PR batch."""
    current = now or dt.datetime.now(dt.UTC)
    run = start_or_resume_sync(engine, repository, now=current, days=days)
    completed = completed_pull_requests(engine, run.sync_id)
    cached_fingerprints = cached_pull_request_fingerprints(engine, repository)
    try:
        result = collect_corpus(
            repository,
            run.window_start,
            run.window_end,
            bot_logins=set(DEFAULT_BOT_LOGINS),
            client=github_client,
            checkpointed_pr_numbers=completed,
            cached_fingerprints=cached_fingerprints,
            bundle_sink=lambda bundle: store_bundle(engine, run.sync_id, bundle, observed_at=dt.datetime.now(dt.UTC)),
            reused_pull_request_sink=lambda pr_number: checkpoint_reused_pull_request(
                engine,
                run.sync_id,
                pr_number,
                observed_at=dt.datetime.now(dt.UTC),
            ),
        )
        lint_rows = telemetry or load_lint_telemetry(deployment, run.window_start, run.window_end)
        store_telemetry(engine, repository, lint_rows.invocations, lint_rows.findings)
        catalog = load_catalog()
        store_catalog_snapshot(
            engine,
            catalog_sha(catalog),
            _catalog_record(catalog),
            observed_at=dt.datetime.now(dt.UTC),
        )
        watermark = {
            "deployment": deployment,
            "window_start": utc_iso(run.window_start),
            "window_end": utc_iso(run.window_end),
        }
        usage = result.usage
        complete_sync(
            engine,
            run.sync_id,
            candidate_pull_requests=result.candidate_pull_requests,
            reused_pull_requests=result.reused_pull_requests,
            github_usage=usage,
            finelog_watermark=watermark,
            completed_at=dt.datetime.now(dt.UTC),
        )
    except Exception as error:
        fail_sync(engine, run.sync_id, str(error))
        raise
    persisted = completed_pull_requests(engine, run.sync_id)
    return SyncResult(
        sync_id=run.sync_id,
        repository=repository,
        window_start=utc_iso(run.window_start),
        window_end=utc_iso(run.window_end),
        candidate_pull_requests=result.candidate_pull_requests,
        persisted_pull_requests=len(persisted),
        reused_pull_requests=result.reused_pull_requests,
        github_usage=usage,
        lint_invocations=len(lint_rows.invocations),
        lint_findings=len(lint_rows.findings),
    )


@click.command()
@click.option("--repository", default=DEFAULT_REPOSITORY, show_default=True)
@click.option("--deployment", default=DEFAULT_DEPLOYMENT, show_default=True)
@click.option("--days", type=click.IntRange(min=1), default=DEFAULT_BACKFILL_DAYS, show_default=True)
def cli(repository: str, deployment: str, days: int) -> None:
    """Synchronize review activity into Marin's existing metadata database."""
    with database_engine(database_config_from_environment()) as engine:
        result = sync_review_activity(engine, repository=repository, deployment=deployment, days=days)
        click.echo(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    cli()
