#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Storage prep workflow: scan objects and compute summaries.

Usage:
    uv run scripts/storage/prep.py run [--from X] [--to Y] [--force] [--scan-workers N]
    uv run scripts/storage/prep.py plan
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from typing import Any

import click

from scripts.storage.scan import scan_objects
from scripts.storage.db import (
    DEFAULT_CATALOG,
    PLAN_FINGERPRINT,
    REPO_ROOT,
    Context,
    StepSpec,
    ensure_output_dirs,
    marker_exists,
    marker_matches,
    materialize_dir_summary,
    print_summary,
    timestamp_string,
    write_marker,
)

# ===========================================================================
# Step runners
# ===========================================================================


def _scan_objects_step(ctx: Context, action: StepSpec) -> None:
    scan_objects(ctx, action)


def _materialize_dir_summary_step(ctx: Context, action: StepSpec) -> None:
    fingerprint = hashlib.sha256((PLAN_FINGERPRINT + "dir_summary").encode()).hexdigest()
    if not ctx.force and marker_matches(ctx.conn, action.action_id, fingerprint):
        print_summary(f"skip {action.action_id}: marker is current")
        return
    total_dirs = materialize_dir_summary(ctx.conn)
    print_summary(f"{action.action_id}: materialized {total_dirs} directory summary rows")
    write_marker(ctx.conn, action.action_id, fingerprint, dry_run=ctx.dry_run)


# ===========================================================================
# Step registry
# ===========================================================================

STEPS: list[StepSpec] = [
    StepSpec(
        action_id="prep.scan_objects",
        group_name="prep",
        command_name="scan-objects",
        description="Scan bucket objects into the DuckDB catalog.",
        help_text=(
            "List every object in each bucket and insert into the local DuckDB database. "
            "Fans out over top-level prefixes with `--scan-workers` concurrent threads. "
            "Skips already-scanned prefixes unless `--force` is given."
        ),
        mutating=False,
        runner=_scan_objects_step,
        scan_workers=True,
    ),
    StepSpec(
        action_id="prep.materialize_dir_summary",
        group_name="prep",
        command_name="materialize-dir-summary",
        description="Aggregate objects into per-directory summary rows.",
        help_text=(
            "Group all objects by their parent directory and compute aggregate counts/bytes. "
            "Uses the most expensive storage class per directory for conservative cost estimation."
        ),
        mutating=False,
        runner=_materialize_dir_summary_step,
        predecessors=("prep.scan_objects",),
    ),
]

STEP_INDEX = {step.action_id: step for step in STEPS}


# ===========================================================================
# CLI helpers
# ===========================================================================


def selected_steps(
    *,
    from_action: str | None,
    to_action: str | None,
    only_action: str | None,
) -> list[StepSpec]:
    if only_action is not None:
        return [STEP_INDEX[only_action]]
    start_index = 0 if from_action is None else next(i for i, s in enumerate(STEPS) if s.action_id == from_action)
    end_index = len(STEPS) - 1 if to_action is None else next(i for i, s in enumerate(STEPS) if s.action_id == to_action)
    return STEPS[start_index : end_index + 1]


def assert_step_predecessors(ctx: Context, step: StepSpec) -> None:
    if ctx.force:
        return
    missing = [p for p in step.predecessors if not marker_exists(ctx.conn, p)]
    if missing:
        raise RuntimeError(
            f"{step.action_id} requires these predecessor steps to be complete first: {', '.join(missing)}"
        )


def build_context(
    *,
    dry_run: bool,
    force: bool,
    scan_workers: int,
    log_prefix: str,
    project: str | None,
) -> Context:
    catalog = DEFAULT_CATALOG
    conn = ensure_output_dirs(catalog)
    return Context(
        conn=conn,
        db_lock=threading.Lock(),
        dry_run=dry_run,
        force=force,
        include_optional=False,
        scan_workers=scan_workers,
        settle_hours=0,
        log_path=catalog.log_dir / f"{log_prefix}_{timestamp_string()}.log",
        timestamp=timestamp_string(),
        project=project,
    )


def runtime_options(
    *,
    scan_workers: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        options: list[Callable[[Callable[..., Any]], Callable[..., Any]]] = [
            click.option("--force", is_flag=True, help="Ignore cached markers and recompute."),
            click.option("--dry-run", is_flag=True, help="Read-only mode: inspect but never mutate remote state."),
            click.option("--project", help="Override the GCP project for Cloud Storage API calls."),
        ]
        if scan_workers:
            options.append(
                click.option("--scan-workers", default=64, show_default=True, type=int, help="Concurrent scan workers.")
            )
        for option in reversed(options):
            func = option(func)
        return func

    return decorator


# ===========================================================================
# CLI
# ===========================================================================


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Storage prep workflow: scan bucket objects and materialize "
        "summaries for downstream cost analysis. Run `plan` to see step status, or use the "
        "prep subcommands to execute individual steps."
    ),
)
def cli() -> None:
    pass


@cli.command("plan", help="Show ordered step list and completion status.")
def plan_cli() -> None:
    conn = ensure_output_dirs()
    print("Ordered storage prep steps:")
    for step in STEPS:
        status = "done" if marker_exists(conn, step.action_id) else "pending"
        print(f"  {status:7}  {step.action_id}")


@cli.command(
    "run",
    help="Execute a contiguous slice of the ordered workflow.",
)
@runtime_options(scan_workers=True)
@click.option("--from", "from_action", type=click.Choice(sorted(STEP_INDEX)), help="Start from this step.")
@click.option("--to", "to_action", type=click.Choice(sorted(STEP_INDEX)), help="Stop after this step.")
@click.option("--only", "only_action", type=click.Choice(sorted(STEP_INDEX)), help="Run exactly one step.")
def run_cli(
    dry_run: bool,
    force: bool,
    scan_workers: int,
    project: str | None,
    from_action: str | None,
    to_action: str | None,
    only_action: str | None,
) -> None:
    ctx = build_context(
        dry_run=dry_run,
        force=force,
        scan_workers=scan_workers,
        log_prefix="prep_run",
        project=project,
    )
    steps = selected_steps(
        from_action=from_action,
        to_action=to_action,
        only_action=only_action,
    )
    print_summary(f"running {len(steps)} steps; log: {ctx.log_path.relative_to(REPO_ROOT)}")
    for step in steps:
        print_summary(f"==> {step.action_id}: {step.description}")
        assert_step_predecessors(ctx, step)
        step.run(ctx)
    print_summary("completed selected steps")


@cli.group(help="Preparation commands: scan objects, materialize summaries.")
def prep() -> None:
    pass


def register_step_command(group: click.Group, step: StepSpec) -> None:
    @runtime_options(scan_workers=step.scan_workers)
    def command(
        dry_run: bool,
        force: bool,
        project: str | None,
        scan_workers: int = 64,
    ) -> None:
        ctx = build_context(
            dry_run=dry_run,
            force=force,
            scan_workers=scan_workers,
            log_prefix=step.action_id.replace(".", "__"),
            project=project,
        )
        print_summary(f"running 1 step; log: {ctx.log_path.relative_to(REPO_ROOT)}")
        print_summary(f"==> {step.action_id}: {step.description}")
        assert_step_predecessors(ctx, step)
        step.run(ctx)
        print_summary("completed selected steps")

    command.__name__ = step.action_id.replace(".", "_").replace("-", "_")
    group.command(name=step.command_name, help=step.help_text)(command)


for _step in STEPS:
    register_step_command(prep, _step)


if __name__ == "__main__":
    cli()
