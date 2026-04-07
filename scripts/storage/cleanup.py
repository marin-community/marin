#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Storage cleanup workflow: delete objects from a pre-computed deletion manifest with soft-delete safety net.

The deletion manifest is produced by compute.py and lists directory prefixes to
delete. This workflow enables soft-delete, deletes all objects under those
prefixes, then optionally disables soft-delete after a safety window.

Usage:
    uv run scripts/storage/cleanup.py run [--from X] [--to Y] [--force] [--dry-run]
    uv run scripts/storage/cleanup.py plan
"""

from __future__ import annotations

import csv
import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

import click
from google.cloud import storage
from tqdm.auto import tqdm

from scripts.storage.db import (
    DEFAULT_CATALOG,
    DELETE_BATCH_SIZE,
    PLAN_FINGERPRINT,
    REPO_ROOT,
    SOFT_DELETE_RETENTION_SECONDS,
    Context,
    StepSpec,
    _fetchall_dicts,
    ensure_output_dirs,
    file_digest,
    human_bytes,
    marker_exists,
    marker_matches,
    now_utc,
    plan_rows,
    print_summary,
    read_marker_extra,
    timestamp_string,
    write_marker,
)
from scripts.storage.scan import (
    bucket_soft_delete_seconds,
    gcloud_bucket_describe,
    run_subprocess,
    storage_client,
)

log = logging.getLogger(__name__)


# ===========================================================================
# CLEANUP steps
# ===========================================================================


def enable_soft_delete(ctx: Context, action: StepSpec) -> None:
    fingerprint = PLAN_FINGERPRINT
    remote_summary: dict[str, Any] = {"regions": {}}
    for row in plan_rows():
        region = row["region"]
        bucket_url = f"gs://{row['bucket']}"
        metadata = gcloud_bucket_describe(ctx, bucket_url)
        current_seconds = bucket_soft_delete_seconds(metadata)
        if current_seconds >= SOFT_DELETE_RETENTION_SECONDS and not ctx.force:
            print_summary(
                f"{action.action_id}: {bucket_url} already has soft-delete >= {SOFT_DELETE_RETENTION_SECONDS}s"
            )
            remote_summary["regions"][region] = {"soft_delete_seconds": current_seconds, "action": "already_enabled"}
            continue
        print_summary(
            f"{action.action_id}: {'would enable' if ctx.dry_run else 'enabling'} "
            f"soft-delete ({SOFT_DELETE_RETENTION_SECONDS}s) on {bucket_url}"
        )
        if not ctx.dry_run:
            run_subprocess(
                ctx,
                [
                    "gcloud",
                    "storage",
                    "buckets",
                    "update",
                    bucket_url,
                    f"--soft-delete-duration={SOFT_DELETE_RETENTION_SECONDS}s",
                ],
            )
            after = gcloud_bucket_describe(ctx, bucket_url)
            after_seconds = bucket_soft_delete_seconds(after)
            if after_seconds < SOFT_DELETE_RETENTION_SECONDS:
                raise RuntimeError(
                    f"Soft-delete on {bucket_url} is {after_seconds}s, expected >= {SOFT_DELETE_RETENTION_SECONDS}s"
                )
            remote_summary["regions"][region] = {"soft_delete_seconds": after_seconds, "action": "enabled"}
        else:
            remote_summary["regions"][region] = {"action": "dry_run"}
    if not ctx.dry_run:
        write_marker(
            ctx.conn, action.action_id, fingerprint, dry_run=ctx.dry_run, extra={"remote_summary": remote_summary}
        )


def _load_manifest(ctx: Context) -> dict[str, list[str]]:
    """Load the deletion manifest CSV and return prefixes grouped by bucket.

    Also validates the CSV fingerprint against the compute step's stored fingerprint.
    """
    catalog = DEFAULT_CATALOG
    manifest_path = catalog.deletion_manifest_csv
    if not manifest_path.exists():
        raise RuntimeError(
            f"Deletion manifest not found at {manifest_path.relative_to(REPO_ROOT)}. "
            "Run `uv run scripts/storage/compute.py run` first."
        )

    # Validate fingerprint
    compute_extra = read_marker_extra(ctx.conn, "compute.materialize_deletion_set")
    if compute_extra is not None:
        expected_fingerprint = compute_extra.get("csv_fingerprint")
        actual_fingerprint = file_digest(manifest_path)
        if expected_fingerprint and actual_fingerprint != expected_fingerprint:
            raise RuntimeError(
                "Deletion manifest has been modified since the compute step ran. "
                "Re-run `uv run scripts/storage/compute.py run --force` to regenerate."
            )

    by_bucket: dict[str, list[str]] = defaultdict(list)
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            by_bucket[row["bucket"]].append(row["prefix"])
    return dict(by_bucket)


def _delete_manifest_prefix(
    ctx: Context,
    bucket_name: str,
    prefix: str,
) -> tuple[int, int, dict[str, int]]:
    """Delete all objects under a manifest prefix from GCS.

    Returns (count, bytes, by_class).
    """
    with ctx.db_lock:
        rows = _fetchall_dicts(
            ctx.conn.execute(
                """
            SELECT o.name, o.size_bytes, sc.name as storage_class
            FROM objects o
            JOIN storage_classes sc ON o.storage_class_id = sc.id
            WHERE o.bucket = ?
              AND o.name LIKE ? || '%'
            ORDER BY o.name
            """,
                (bucket_name, prefix),
            )
        )

    if not rows:
        return 0, 0, {}

    client = storage_client(ctx)
    bucket_obj = client.bucket(bucket_name)
    deleted_count = 0
    deleted_bytes = 0
    deleted_by_class: dict[str, int] = defaultdict(int)
    batch: list[storage.Blob] = []

    for row in rows:
        deleted_count += 1
        deleted_bytes += int(row["size_bytes"])
        deleted_by_class[row["storage_class"]] += 1
        batch.append(bucket_obj.blob(row["name"]))

        if len(batch) >= DELETE_BATCH_SIZE:
            if not ctx.dry_run:
                with client.batch():
                    for b in batch:
                        b.delete()
            batch.clear()

    if batch and not ctx.dry_run:
        with client.batch():
            for b in batch:
                b.delete()

    return deleted_count, deleted_bytes, dict(deleted_by_class)


def delete_from_manifest(ctx: Context, action: StepSpec) -> None:
    """Delete objects listed in the pre-computed deletion manifest."""
    fingerprint = PLAN_FINGERPRINT
    if not ctx.force and marker_matches(ctx.conn, action.action_id, fingerprint):
        print_summary(f"skip {action.action_id}: already completed")
        return

    manifest = _load_manifest(ctx)
    remote_summary: dict[str, Any] = {"regions": {}}

    for row in plan_rows():
        region = row["region"]
        bucket_name = row["bucket"]
        prefixes = manifest.get(bucket_name, [])

        if not prefixes:
            print_summary(f"{action.action_id}: {bucket_name}: no prefixes in manifest, skipping")
            remote_summary["regions"][region] = {"deleted_count": 0, "deleted_bytes": 0}
            continue

        print_summary(
            f"{action.action_id}: deleting objects from {bucket_name} "
            f"({len(prefixes)} manifest prefixes, {ctx.scan_workers} workers)"
        )

        total_deleted_count = 0
        total_deleted_bytes = 0
        total_deleted_by_class: dict[str, int] = defaultdict(int)

        progress = tqdm(total=len(prefixes), desc=f"delete {bucket_name}", unit="prefix", leave=True)
        workers = max(1, min(ctx.scan_workers, len(prefixes) or 1))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_prefix = {executor.submit(_delete_manifest_prefix, ctx, bucket_name, p): p for p in prefixes}
            for future in as_completed(future_to_prefix):
                p = future_to_prefix[future]
                count, nbytes, by_class = future.result()
                total_deleted_count += count
                total_deleted_bytes += nbytes
                for sc, c in by_class.items():
                    total_deleted_by_class[sc] += c
                progress.set_postfix_str(f"{p[:50]} ({count} del)")
                progress.update(1)
        progress.close()

        remote_summary["regions"][region] = {
            "deleted_count": total_deleted_count,
            "deleted_bytes": total_deleted_bytes,
        }
        verb = "would delete" if ctx.dry_run else "deleted"
        print_summary(f"{region}: {verb} {total_deleted_count} objects ({human_bytes(total_deleted_bytes)})")
    if not ctx.dry_run:
        write_marker(
            ctx.conn, action.action_id, fingerprint, dry_run=ctx.dry_run, extra={"remote_summary": remote_summary}
        )


def wait_for_soft_delete_window(ctx: Context, action: StepSpec) -> None:
    if not marker_exists(ctx.conn, "cleanup.delete_cold_objects"):
        raise RuntimeError("cleanup.wait_for_safety_window requires cleanup.delete_cold_objects to be complete first")
    fingerprint = PLAN_FINGERPRINT
    settle_deadline = now_utc() + timedelta(hours=ctx.settle_hours)

    existing_extra = read_marker_extra(ctx.conn, action.action_id)
    if existing_extra is not None and not ctx.force:
        existing_deadline = datetime.fromisoformat(existing_extra["settle_deadline"])
        if now_utc() < existing_deadline:
            remaining = existing_deadline - now_utc()
            raise RuntimeError(
                f"Soft-delete safety window still open until {existing_deadline.isoformat()} "
                f"({remaining} remaining). Rerun after the deadline or use --force to override."
            )
        print_summary(f"{action.action_id}: soft-delete safety window already elapsed")
        return

    print_summary(
        f"{action.action_id}: recording safety window for {ctx.settle_hours} hours "
        f"(until {settle_deadline.isoformat()})"
    )
    write_marker(
        ctx.conn,
        action.action_id,
        fingerprint,
        dry_run=ctx.dry_run,
        extra={"settle_deadline": settle_deadline.isoformat()},
    )


def disable_soft_delete(ctx: Context, action: StepSpec) -> None:
    fingerprint = PLAN_FINGERPRINT
    remote_summary: dict[str, Any] = {"regions": {}}
    for row in plan_rows():
        region = row["region"]
        bucket_url = f"gs://{row['bucket']}"
        metadata = gcloud_bucket_describe(ctx, bucket_url)
        current_seconds = bucket_soft_delete_seconds(metadata)
        if current_seconds == 0 and not ctx.force:
            print_summary(f"{action.action_id}: {bucket_url} already has soft-delete disabled")
            remote_summary["regions"][region] = {"soft_delete_seconds": 0, "action": "already_disabled"}
            continue
        print_summary(
            f"{action.action_id}: {'would disable' if ctx.dry_run else 'disabling'} soft-delete on {bucket_url}"
        )
        if not ctx.dry_run:
            run_subprocess(ctx, ["gcloud", "storage", "buckets", "update", bucket_url, "--clear-soft-delete"])
            after = gcloud_bucket_describe(ctx, bucket_url)
            after_seconds = bucket_soft_delete_seconds(after)
            if after_seconds != 0:
                raise RuntimeError(f"Soft-delete on {bucket_url} is still {after_seconds}s after clear")
            remote_summary["regions"][region] = {"soft_delete_seconds": 0, "action": "disabled"}
        else:
            remote_summary["regions"][region] = {"action": "dry_run"}
    if not ctx.dry_run:
        write_marker(
            ctx.conn, action.action_id, fingerprint, dry_run=ctx.dry_run, extra={"remote_summary": remote_summary}
        )


# ===========================================================================
# Step registry
# ===========================================================================

STEPS: list[StepSpec] = [
    StepSpec(
        action_id="cleanup.enable_soft_delete",
        group_name="cleanup",
        command_name="enable-soft-delete",
        description="Enable 3-day soft-delete retention on source buckets.",
        help_text=(
            "Enable soft-delete on each source bucket with a 3-day retention window. "
            "This ensures deleted objects can be recovered if something goes wrong. "
            "Safe to re-run; skips buckets that already have the required retention."
        ),
        mutating=True,
        runner=enable_soft_delete,
    ),
    StepSpec(
        action_id="cleanup.delete_cold_objects",
        group_name="cleanup",
        command_name="delete-cold-objects",
        description="Delete objects listed in the pre-computed deletion manifest.",
        help_text=(
            "Read the deletion manifest produced by compute.py and delete all objects "
            "under the listed prefixes. Validates the manifest fingerprint before proceeding. "
            "Fans out over manifest prefixes with --scan-workers concurrent threads."
        ),
        mutating=True,
        runner=delete_from_manifest,
        predecessors=("cleanup.enable_soft_delete", "compute.materialize_deletion_set"),
        scan_workers=True,
    ),
    StepSpec(
        action_id="cleanup.wait_for_safety_window",
        group_name="cleanup",
        command_name="wait-for-safety-window",
        description="Record and honor the soft-delete safety window.",
        help_text=(
            "Record the soft-delete safety window and refuse to disable soft-delete until that window "
            "has elapsed. This is a checkpoint, not a sleep. Use `--settle-hours` to adjust the window."
        ),
        mutating=False,
        runner=wait_for_soft_delete_window,
        predecessors=("cleanup.delete_cold_objects",),
        settle_hours=True,
    ),
    StepSpec(
        action_id="cleanup.disable_soft_delete",
        group_name="cleanup",
        command_name="disable-soft-delete",
        description="Disable soft-delete after the safety window has elapsed.",
        help_text=(
            "Disable soft-delete on each source bucket, permanently removing the soft-deleted objects. "
            "Only run this after the safety window has elapsed and you have confirmed no important "
            "data was accidentally deleted."
        ),
        mutating=True,
        runner=disable_soft_delete,
        predecessors=("cleanup.wait_for_safety_window",),
        optional=True,
    ),
]

STEP_INDEX = {step.action_id: step for step in STEPS}

# ===========================================================================
# CLI
# ===========================================================================


def selected_steps(
    *,
    from_action: str | None,
    to_action: str | None,
    only_action: str | None,
    include_optional: bool,
) -> list[StepSpec]:
    if only_action is not None:
        return [STEP_INDEX[only_action]]
    start_index = 0 if from_action is None else next(i for i, s in enumerate(STEPS) if s.action_id == from_action)
    end_index = len(STEPS) - 1 if to_action is None else next(i for i, s in enumerate(STEPS) if s.action_id == to_action)
    steps = STEPS[start_index : end_index + 1]
    if include_optional:
        return steps
    if to_action is not None and STEP_INDEX[to_action].optional:
        return steps
    return [s for s in steps if not s.optional]


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
    include_optional: bool,
    scan_workers: int,
    settle_hours: int,
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
        include_optional=include_optional,
        scan_workers=scan_workers,
        settle_hours=settle_hours,
        log_path=catalog.log_dir / f"{log_prefix}_{timestamp_string()}.log",
        timestamp=timestamp_string(),
        project=project,
    )


def runtime_options(
    *,
    scan_workers: bool = False,
    settle_hours: bool = False,
    include_optional: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        options: list[Callable[[Callable[..., Any]], Callable[..., Any]]] = [
            click.option("--force", is_flag=True, help="Ignore cached markers and recompute."),
            click.option("--dry-run", is_flag=True, help="Read-only mode: inspect but never mutate remote state."),
            click.option("--project", help="Override the GCP project for Cloud Storage API calls."),
        ]
        if include_optional:
            options.append(click.option("--include-optional", is_flag=True, help="Include optional cleanup steps."))
        if scan_workers:
            options.append(
                click.option(
                    "--scan-workers", default=64, show_default=True, type=int, help="Concurrent scan/delete workers."
                )
            )
        if settle_hours:
            options.append(
                click.option(
                    "--settle-hours", default=72, show_default=True, type=int, help="Soft-delete safety window (hours)."
                )
            )
        for option in reversed(options):
            func = option(func)
        return func

    return decorator


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Storage cleanup workflow: delete objects matching delete rules from Marin buckets "
        "with a soft-delete safety net. Run `plan` to see step status, or use the "
        "cleanup subcommands to execute individual steps."
    ),
)
def cli() -> None:
    pass


@cli.command("plan", help="Show ordered step list and completion status.")
def plan_cli() -> None:
    conn = ensure_output_dirs()
    print("Ordered cleanup steps:")
    for step in STEPS:
        suffix = " [optional]" if step.optional else ""
        status = "done" if marker_exists(conn, step.action_id) else "pending"
        print(f"  {status:7}  {step.action_id}{suffix}")


@cli.command(
    "run",
    help="Execute a contiguous slice of the ordered workflow.",
)
@runtime_options(scan_workers=True, settle_hours=True, include_optional=True)
@click.option("--from", "from_action", type=click.Choice(sorted(STEP_INDEX)), help="Start from this step.")
@click.option("--to", "to_action", type=click.Choice(sorted(STEP_INDEX)), help="Stop after this step.")
@click.option("--only", "only_action", type=click.Choice(sorted(STEP_INDEX)), help="Run exactly one step.")
def run_cli(
    dry_run: bool,
    force: bool,
    include_optional: bool,
    scan_workers: int,
    settle_hours: int,
    project: str | None,
    from_action: str | None,
    to_action: str | None,
    only_action: str | None,
) -> None:
    ctx = build_context(
        dry_run=dry_run,
        force=force,
        include_optional=include_optional,
        scan_workers=scan_workers,
        settle_hours=settle_hours,
        log_prefix="run",
        project=project,
    )
    steps = selected_steps(
        from_action=from_action,
        to_action=to_action,
        only_action=only_action,
        include_optional=include_optional,
    )
    print_summary(f"running {len(steps)} steps; log: {ctx.log_path.relative_to(REPO_ROOT)}")
    for step in steps:
        print_summary(f"==> {step.action_id}: {step.description}")
        assert_step_predecessors(ctx, step)
        step.run(ctx)
    print_summary("completed selected steps")


@cli.group(help="Cleanup commands: enable soft-delete, delete objects matching rules, finalize.")
def cleanup() -> None:
    pass


GROUPS: dict[str, click.Group] = {"cleanup": cleanup}


def register_step_command(group: click.Group, step: StepSpec) -> None:
    @runtime_options(
        scan_workers=step.scan_workers,
        settle_hours=step.settle_hours,
    )
    def command(
        dry_run: bool,
        force: bool,
        project: str | None,
        scan_workers: int = 64,
        settle_hours: int = 72,
    ) -> None:
        ctx = build_context(
            dry_run=dry_run,
            force=force,
            include_optional=False,
            scan_workers=scan_workers,
            settle_hours=settle_hours,
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
    register_step_command(GROUPS[_step.group_name], _step)


if __name__ == "__main__":
    cli()
