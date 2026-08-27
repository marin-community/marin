# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-shot import of the legacy W&B review-stats tables into Finelog.

The code-health tables lived in a single persistent W&B run
(``marin-community/marin-review-stats``) as four flat ``wandb.Table``
artifacts, re-logged in full on every append. This replays them into the
``codehealth.autolint.*`` Finelog namespaces that ``log_stats.py`` and
``review.py`` now write directly.

Defaults to a dry run that reports what it would write. Run it once against
each target; a second run appends the same rows again, because Finelog
namespaces are append-only and this script does no deduplication.

    uv run infra/codehealth/backfill_from_wandb.py --dry-run
    uv run infra/codehealth/backfill_from_wandb.py --no-dry-run --deployment marin
"""

import datetime as dt
import logging
import math
from collections.abc import Callable
from typing import Any

import click
from review_tables import (
    DEFAULT_DEPLOYMENT,
    FINDINGS_NAMESPACE,
    HUMAN_COMMENTS_NAMESPACE,
    INVOCATIONS_NAMESPACE,
    PR_REVIEW_OUTCOMES_NAMESPACE,
    Finding,
    HumanComment,
    Invocation,
    PrReviewOutcome,
    append_rows,
    open_tables_client,
    parse_utc,
)

logger = logging.getLogger("codehealth.backfill")

WANDB_ENTITY = "marin-community"
WANDB_PROJECT = "marin-review-stats"
WANDB_RUN_ID = "review-stats"

FLUSH_TIMEOUT = 120.0


def _text(value: Any) -> str | None:
    """Normalize a W&B cell to a string, mapping blanks and NaN to None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _integer(value: Any) -> int | None:
    """W&B stores every number as a float; recover the integers."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return int(float(stripped))
    return int(value)


def _number(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return float(stripped)
    return float(value)


def _flag(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1"):
            return True
        if lowered in ("false", "0"):
            return False
        return None
    return bool(value)


def _moment(value: Any) -> dt.datetime | None:
    """Parse an ISO-8601 cell into a tz-aware UTC datetime."""
    text = _text(value)
    return parse_utc(text) if text is not None else None


def _required_moment(value: Any) -> dt.datetime:
    """A row with no usable timestamp cannot be placed in an append-only table."""
    moment = _moment(value)
    if moment is None:
        raise ValueError(f"row has no parseable timestamp: {value!r}")
    return moment


def to_invocation(row: dict) -> Invocation:
    return Invocation(
        ts=_required_moment(row.get("ts")),
        invocation_id=_text(row.get("invocation_id")) or "",
        tool=_text(row.get("tool")) or "",
        variant=_text(row.get("variant")),
        trigger=_text(row.get("trigger")),
        agent_cli=_text(row.get("agent_cli")),
        git_branch=_text(row.get("git_branch")),
        merge_base_sha=_text(row.get("merge_base_sha")),
        head_sha=_text(row.get("head_sha")),
        pr_number=_integer(row.get("pr_number")),
        marin_user=_text(row.get("marin_user")),
        lint_catalog_sha=_text(row.get("lint_catalog_sha")),
        diff_files=_integer(row.get("diff_files")),
        diff_added_lines=_integer(row.get("diff_added_lines")),
        diff_removed_lines=_integer(row.get("diff_removed_lines")),
        finding_count=_integer(row.get("finding_count")),
        elapsed=_number(row.get("elapsed")),
        agent_exit_code=_integer(row.get("agent_exit_code")),
        timed_out=_flag(row.get("timed_out")),
    )


def to_finding(row: dict) -> Finding:
    return Finding(
        ts=_required_moment(row.get("ts")),
        invocation_id=_text(row.get("invocation_id")) or "",
        tool=_text(row.get("tool")) or "",
        pr_number=_integer(row.get("pr_number")),
        git_branch=_text(row.get("git_branch")),
        head_sha=_text(row.get("head_sha")),
        marin_user=_text(row.get("marin_user")),
        file=_text(row.get("file")),
        line=_integer(row.get("line")),
        code=_text(row.get("code")),
        confidence=_number(row.get("confidence")),
        message=_text(row.get("message")),
    )


def to_human_comment(row: dict) -> HumanComment:
    return HumanComment(
        ts=_required_moment(row.get("ts")),
        pr_number=_integer(row.get("pr_number")) or 0,
        pr_title=_text(row.get("pr_title")),
        merged_at=_moment(row.get("merged_at")),
        author=_text(row.get("author")),
        comment_id=_integer(row.get("comment_id")) or 0,
        comment_type=_text(row.get("comment_type")) or "",
        file=_text(row.get("file")),
        line=_integer(row.get("line")),
        body=_text(row.get("body")),
        comment_class=_text(row.get("class")),
        catchable_strict=_flag(row.get("catchable_strict")),
        catchable_generous=_flag(row.get("catchable_generous")),
        confidence=_number(row.get("confidence")),
        reason=_text(row.get("reason")),
    )


def to_pr_review_outcome(row: dict) -> PrReviewOutcome:
    return PrReviewOutcome(
        ts=_required_moment(row.get("ts")),
        pr_number=_integer(row.get("pr_number")) or 0,
        pr_title=_text(row.get("pr_title")),
        merged_at=_moment(row.get("merged_at")),
        author=_text(row.get("author")),
        head_sha=_text(row.get("head_sha")),
        base_sha=_text(row.get("base_sha")),
        total_human_comments=_integer(row.get("total_human_comments")),
        by_class_json=_text(row.get("by_class_json")),
        catchable_strict_count=_integer(row.get("catchable_strict_count")),
        catchable_generous_count=_integer(row.get("catchable_generous_count")),
        bot_findings_count=_integer(row.get("bot_findings_count")),
        overlap_count=_integer(row.get("overlap_count")),
    )


# W&B table key -> (finelog namespace, row dataclass, converter).
TABLES: dict[str, tuple[str, type, Callable[[dict], Any]]] = {
    "invocations": (INVOCATIONS_NAMESPACE, Invocation, to_invocation),
    "findings": (FINDINGS_NAMESPACE, Finding, to_finding),
    "human_comments": (HUMAN_COMMENTS_NAMESPACE, HumanComment, to_human_comment),
    "pr_review_outcomes": (PR_REVIEW_OUTCOMES_NAMESPACE, PrReviewOutcome, to_pr_review_outcome),
}


def read_wandb_table(entity: str, project: str, run_id: str, key: str) -> list[dict]:
    """Fetch one ``wandb.Table`` artifact as a list of column-keyed dicts."""
    import wandb  # noqa: PLC0415  # guarded: only the backfill needs the legacy client

    api = wandb.Api()
    artifact = api.artifact(f"{entity}/{project}/run-{run_id}-{key}:latest")
    table = artifact.get(key)
    columns = list(table.columns)
    return [dict(zip(columns, row, strict=False)) for row in table.data]


@click.command(help=__doc__)
@click.option("--deployment", default=DEFAULT_DEPLOYMENT, show_default=True, help="Finelog deployment to write to.")
@click.option("--finelog-url", default=None, help="Connect directly to this Finelog URL instead of a deployment.")
@click.option("--entity", default=WANDB_ENTITY, show_default=True)
@click.option("--project", default=WANDB_PROJECT, show_default=True)
@click.option("--run-id", default=WANDB_RUN_ID, show_default=True)
@click.option("--dry-run/--no-dry-run", default=True, show_default=True, help="Report rows without writing them.")
def main(
    deployment: str,
    finelog_url: str | None,
    entity: str,
    project: str,
    run_id: str,
    dry_run: bool,
) -> None:
    """Replay the four legacy W&B tables into Finelog."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    converted: dict[str, list] = {}
    for key, (namespace, _row_type, convert) in TABLES.items():
        raw = read_wandb_table(entity, project, run_id, key)
        rows = [convert(row) for row in raw]
        converted[key] = rows
        span = ""
        if rows:
            moments = sorted(row.ts for row in rows)
            span = f"  {moments[0].date()} .. {moments[-1].date()}"
        logger.info("%-22s %5d rows -> %s%s", key, len(rows), namespace, span)

    total = sum(len(rows) for rows in converted.values())
    if dry_run:
        logger.info("Dry run: %d rows converted, nothing written.", total)
        return

    with open_tables_client(deployment, finelog_url) as client:
        for key, rows in converted.items():
            if not rows:
                continue
            namespace, row_type, _convert = TABLES[key]
            total_rows = append_rows(client, namespace, row_type, rows, flush_timeout=FLUSH_TIMEOUT)
            logger.info("Wrote %d rows to %s (%d total)", len(rows), namespace, total_rows)
    logger.info("Backfill complete: %d rows.", total)


if __name__ == "__main__":
    main()
