# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded CLI tools for an agent exploring persisted review activity."""

from __future__ import annotations

import base64
import datetime as dt
import json
import subprocess
import urllib.parse
from pathlib import Path

import click
from pydantic import BaseModel
from sqlalchemy.engine import Engine

from infra.lint.catalog import DEFAULT_CATALOG_DIR, catalog_sha, load_catalog

from .review_store import (
    ReviewContext,
    catalog_snapshot_shas,
    database_config_from_environment,
    database_engine,
    latest_sync_status,
    lint_activity,
    list_pr_review_events,
    list_pull_request_activity,
    review_context,
    store_source_context,
)
from .review_tables import DEFAULT_REPOSITORY
from .rule_probe import run_rule_probe

SOURCE_CONTEXT_LINES = 100


def _json(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _fetch_github_file(repository: str, path: str, commit_sha: str) -> str:
    endpoint = f"repos/{repository}/contents/{urllib.parse.quote(path, safe='/')}"
    result = subprocess.run(
        ["gh", "api", "--method", "GET", endpoint, "-f", f"ref={commit_sha}"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    if payload.get("encoding") != "base64" or not isinstance(payload.get("content"), str):
        raise ValueError(f"GitHub returned non-file content for {path}@{commit_sha}")
    return base64.b64decode(payload["content"].replace("\n", "")).decode("utf-8")


def ensure_source_context(engine: Engine, context: ReviewContext) -> ReviewContext:
    """Fetch and persist a missing ±100-line source window for one inline event."""
    event = context.event
    if context.source is not None or event.path is None:
        return context
    commit_sha = event.original_commit_id or event.commit_id
    anchor_line = event.original_line or event.line
    if commit_sha is None or anchor_line is None:
        return context
    start_line = max(1, anchor_line - SOURCE_CONTEXT_LINES)
    end_line = anchor_line + SOURCE_CONTEXT_LINES
    try:
        lines = _fetch_github_file(event.repository, event.path, commit_sha).splitlines()
        end_line = min(end_line, len(lines))
        text = "\n".join(f"{number:>6}  {lines[number - 1]}" for number in range(start_line, end_line + 1))
        reason = None
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError, ValueError) as error:
        text = None
        detail = str(error)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            detail = f"{detail}: {error.stderr.strip()}"
        reason = f"{type(error).__name__}: {detail}"[:500]
    store_source_context(
        engine,
        event_id=event.event_id,
        commit_sha=commit_sha,
        path=event.path,
        anchor_line=anchor_line,
        start_line=start_line,
        end_line=end_line,
        text=text,
        unavailable_reason=reason,
        fetched_at=dt.datetime.now(dt.UTC),
    )
    return review_context(engine, event.event_id)


@click.group()
def cli() -> None:
    """Explore review data, inspect rules, run probes, and publish reports."""


@cli.command("sync-status")
@click.option("--repository", default=DEFAULT_REPOSITORY, show_default=True)
def sync_status(repository: str) -> None:
    """Show whether the latest fixed-window sync is safe to query."""
    with database_engine(database_config_from_environment()) as engine:
        status = latest_sync_status(engine, repository)
        click.echo(_json(None if status is None else status.model_dump(mode="json")))


@cli.command("list-prs")
@click.option("--days", type=click.IntRange(min=1, max=365), default=30, show_default=True)
@click.option("--repository", default=DEFAULT_REPOSITORY, show_default=True)
@click.option("--human", "require_human", is_flag=True)
@click.option("--lint", "require_lint", is_flag=True)
@click.option("--limit", type=click.IntRange(min=1, max=500), default=100, show_default=True)
def list_prs(days: int, repository: str, require_human: bool, require_lint: bool, limit: int) -> None:
    """List PRs with joined human-review and lint activity."""
    with database_engine(database_config_from_environment()) as engine:
        end = dt.datetime.now(dt.UTC)
        rows = list_pull_request_activity(
            engine,
            start=end - dt.timedelta(days=days),
            end=end,
            repository=repository,
            require_human=require_human,
            require_lint=require_lint,
            limit=limit,
        )
        click.echo(_json([row.model_dump(mode="json") for row in rows]))


@cli.command("list-comments")
@click.option("--repository", default=DEFAULT_REPOSITORY, show_default=True)
@click.option("--pr", "pr_number", type=click.IntRange(min=1), required=True)
def list_comments(repository: str, pr_number: int) -> None:
    """List review events and lint totals for one PR."""
    with database_engine(database_config_from_environment()) as engine:
        rows = list_pr_review_events(engine, repository, pr_number)
        click.echo(_json([row.model_dump(mode="json") for row in rows]))


@cli.command("context")
@click.option("--event-id", required=True)
def context_command(event_id: str) -> None:
    """Fetch one comment's thread, diff, source window, and lint activity."""
    with database_engine(database_config_from_environment()) as engine:
        context = ensure_source_context(engine, review_context(engine, event_id))
        click.echo(_json(context))


@cli.command("list-rules")
@click.option("--lane")
def list_rules(lane: str | None) -> None:
    """List structured rules from the current working tree."""
    catalog = load_catalog()
    rules = catalog.lane(lane).rules if lane else catalog.rules
    click.echo(
        _json(
            [
                {
                    "code": rule.code,
                    "lane": rule.lane,
                    "title": rule.title,
                    "minimum_confidence": rule.minimum_confidence,
                    "path": str(rule.path.relative_to(DEFAULT_CATALOG_DIR.parent.parent)),
                }
                for rule in rules
            ]
        )
    )


@cli.command("get-rule")
@click.option("--code", required=True)
def get_rule(code: str) -> None:
    """Read one complete rule from the current working tree."""
    rule = load_catalog().rule(code)
    click.echo(
        _json(
            {
                "code": rule.code,
                "lane": rule.lane,
                "title": rule.title,
                "minimum_confidence": rule.minimum_confidence,
                "prompt": rule.prompt,
                "path": str(rule.path),
            }
        )
    )


@cli.command("validate-rules")
def validate_rules() -> None:
    """Validate and fingerprint the working-tree lint catalog."""
    catalog = load_catalog()
    click.echo(_json({"catalog_sha": catalog_sha(catalog), "rules": len(catalog.rules), "lanes": len(catalog.lanes)}))


@cli.command("rule-activity")
@click.option("--days", type=click.IntRange(min=1, max=365), default=30, show_default=True)
@click.option("--repository", default=DEFAULT_REPOSITORY, show_default=True)
def rule_activity(days: int, repository: str) -> None:
    """List current-catalog exposure and findings alongside whole-window findings."""
    with database_engine(database_config_from_environment()) as engine:
        end = dt.datetime.now(dt.UTC)
        activity = lint_activity(engine, repository=repository, start=end - dt.timedelta(days=days), end=end)
        catalog = load_catalog()
        current_sha = catalog_sha(catalog)
        known_catalogs = catalog_snapshot_shas(engine)
        current_runs = [row for row in activity.invocations if row.lint_catalog_sha == current_sha]
        current_ids = {row.invocation_id for row in current_runs}
        rows = []
        for rule in catalog.rules:
            lane = catalog.lane(rule.lane)
            eligible = [
                row
                for row in current_runs
                if lane.min_diff_lines == 0
                or int(row.diff_added_lines or 0) + int(row.diff_removed_lines or 0) > lane.min_diff_lines
            ]
            rows.append(
                {
                    "code": rule.code,
                    "lane": rule.lane,
                    "current_catalog_eligible_runs": len(eligible),
                    "current_catalog_findings": sum(
                        finding.code == rule.code and finding.invocation_id in current_ids
                        for finding in activity.findings
                    ),
                    "window_findings_all_catalog_versions": sum(
                        finding.code == rule.code for finding in activity.findings
                    ),
                }
            )
        click.echo(
            _json(
                {
                    "days": days,
                    "catalog_sha": current_sha,
                    "successful_runs_all_catalog_versions": len(activity.invocations),
                    "successful_runs_current_catalog": len(current_runs),
                    "catalog_versions": [
                        {
                            "catalog_sha": sha,
                            "successful_runs": sum(row.lint_catalog_sha == sha for row in activity.invocations),
                            "snapshot_available": sha in known_catalogs,
                        }
                        for sha in sorted({row.lint_catalog_sha for row in activity.invocations if row.lint_catalog_sha})
                    ],
                    "rules": rows,
                }
            )
        )


@cli.command("probe")
@click.option("--event-id", required=True)
@click.option("--rule", "rule_code", required=True)
@click.option("--model", required=True)
@click.option("--effort", required=True)
@click.option("--idempotency-key", required=True)
def probe(event_id: str, rule_code: str, model: str, effort: str, idempotency_key: str) -> None:
    """Probe one rule against one stored comment context."""
    with database_engine(database_config_from_environment()) as engine:
        context = ensure_source_context(engine, review_context(engine, event_id))
        result = run_rule_probe(
            engine,
            load_catalog(),
            context,
            rule_code=rule_code,
            model=model,
            effort=effort,
            idempotency_key=idempotency_key,
        )
        click.echo(_json(result))


@cli.command("post-report")
@click.option("--name", default="codehealth-refinement-report", show_default=True)
@click.option("--title", required=True)
@click.option("--report", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--summary", required=True)
@click.option("--idempotency-key", required=True)
def post_report(name: str, title: str, report: Path, summary: str, idempotency_key: str) -> None:
    """Publish agent-authored Markdown and announce it on the durable channel."""
    write = subprocess.run(
        ["loom", "artifacts", "write", name, str(report), "--title", title],
        check=True,
        capture_output=True,
        text=True,
    )
    output = write.stdout.strip().split()
    if not output or not output[0].startswith(("http://", "https://")):
        raise RuntimeError(f"loom artifact write returned no URL: {write.stdout[:500]}")
    artifact_url = output[0]
    message = f"{summary.rstrip()}\nReport: {artifact_url}"
    subprocess.run(
        [
            "loom",
            "channels",
            "send",
            "--kind",
            "result",
            "--idempotency-key",
            idempotency_key,
            message,
        ],
        check=True,
    )
    click.echo(_json({"artifact_url": artifact_url, "summary": summary}))


if __name__ == "__main__":
    cli()
