#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute the deletion set from protect/delete rules and output a collapsed CSV manifest.

The manifest lists directory prefixes to delete, collapsed so that if all children
of a parent are marked for deletion, only the parent appears. This keeps the output
compact and human-reviewable before feeding it to the cleanup workflow.

Usage:
    uv run scripts/storage/compute.py plan
    uv run scripts/storage/compute.py run [--force]
"""

from __future__ import annotations

import csv
import hashlib
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import click

from scripts.storage.db import (
    DEFAULT_CATALOG,
    GCS_DISCOUNT,
    PLAN_FINGERPRINT,
    REPO_ROOT,
    Context,
    StepSpec,
    _fetchall_dicts,
    continent_for_region,
    ensure_output_dirs,
    file_digest,
    human_bytes,
    marker_exists,
    marker_matches,
    plan_rows,
    print_summary,
    timestamp_string,
    write_marker,
)

# ---------------------------------------------------------------------------
# Collapsing algorithm
# ---------------------------------------------------------------------------


@dataclass
class DirEntry:
    bucket: str
    prefix: str
    status: str  # "delete" or "keep"
    matched_rule: str
    standard_count: int
    standard_bytes: int
    nearline_count: int
    nearline_bytes: int
    coldline_count: int
    coldline_bytes: int
    archive_count: int
    archive_bytes: int

    @property
    def object_count(self) -> int:
        return self.standard_count + self.nearline_count + self.coldline_count + self.archive_count

    @property
    def total_bytes(self) -> int:
        return self.standard_bytes + self.nearline_bytes + self.coldline_bytes + self.archive_bytes

    @property
    def storage_class_breakdown(self) -> str:
        parts = []
        for name, count in [
            ("STANDARD", self.standard_count),
            ("NEARLINE", self.nearline_count),
            ("COLDLINE", self.coldline_count),
            ("ARCHIVE", self.archive_count),
        ]:
            if count > 0:
                parts.append(f"{name}:{count}")
        return ";".join(parts)

    @property
    def depth(self) -> int:
        return self.prefix.rstrip("/").count("/") + 1


def _parent_prefix(prefix: str) -> str | None:
    """Return the parent directory prefix, or None if already at root."""
    stripped = prefix.rstrip("/")
    idx = stripped.rfind("/")
    if idx < 0:
        return None
    return stripped[: idx + 1]


def _collapse_deletions(entries: list[DirEntry]) -> list[DirEntry]:
    """Collapse delete-marked directories bottom-up.

    If all children of a parent (within the same bucket) are "delete", replace
    them with a single entry for the parent with summed stats. Repeat until stable.
    """
    # Group by bucket so entries from different buckets never interact.
    by_bucket: dict[str, list[DirEntry]] = defaultdict(list)
    for e in entries:
        by_bucket[e.bucket].append(e)

    result: list[DirEntry] = []
    for bucket, bucket_entries in sorted(by_bucket.items()):
        result.extend(_collapse_bucket(bucket, bucket_entries))
    return result


def _collapse_bucket(bucket: str, entries: list[DirEntry]) -> list[DirEntry]:
    """Collapse within a single bucket."""
    by_prefix: dict[str, DirEntry] = {e.prefix: e for e in entries}

    changed = True
    while changed:
        changed = False

        children_of: dict[str, list[str]] = defaultdict(list)
        for prefix in by_prefix:
            parent = _parent_prefix(prefix)
            if parent is not None:
                children_of[parent].append(prefix)

        for parent, child_prefixes in children_of.items():
            if len(child_prefixes) < 2:
                continue

            children = [by_prefix[cp] for cp in child_prefixes if cp in by_prefix]
            if len(children) != len(child_prefixes):
                continue
            if not all(c.status == "delete" for c in children):
                continue

            if parent in by_prefix and by_prefix[parent].status == "keep":
                continue

            merged = DirEntry(
                bucket=bucket,
                prefix=parent,
                status="delete",
                matched_rule=_most_common_rule(children),
                standard_count=sum(c.standard_count for c in children),
                standard_bytes=sum(c.standard_bytes for c in children),
                nearline_count=sum(c.nearline_count for c in children),
                nearline_bytes=sum(c.nearline_bytes for c in children),
                coldline_count=sum(c.coldline_count for c in children),
                coldline_bytes=sum(c.coldline_bytes for c in children),
                archive_count=sum(c.archive_count for c in children),
                archive_bytes=sum(c.archive_bytes for c in children),
            )

            if parent in by_prefix and by_prefix[parent].status == "delete":
                existing = by_prefix[parent]
                merged.standard_count += existing.standard_count
                merged.standard_bytes += existing.standard_bytes
                merged.nearline_count += existing.nearline_count
                merged.nearline_bytes += existing.nearline_bytes
                merged.coldline_count += existing.coldline_count
                merged.coldline_bytes += existing.coldline_bytes
                merged.archive_count += existing.archive_count
                merged.archive_bytes += existing.archive_bytes

            for cp in child_prefixes:
                del by_prefix[cp]
            by_prefix[parent] = merged
            changed = True

    return sorted(by_prefix.values(), key=lambda e: e.prefix)


def _most_common_rule(entries: list[DirEntry]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for e in entries:
        counts[e.matched_rule] += 1
    return max(counts, key=counts.get)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Query + materialize
# ---------------------------------------------------------------------------

_STATUS_QUERY = """
    SELECT d.bucket, d.dir_prefix,
           d.standard_count, d.standard_bytes,
           d.nearline_count, d.nearline_bytes,
           d.coldline_count, d.coldline_bytes,
           d.archive_count, d.archive_bytes,
           CASE WHEN p.dir_prefix IS NOT NULL THEN 'keep'
                WHEN del.dir_prefix IS NOT NULL THEN 'delete'
                ELSE 'keep'
           END AS status,
           COALESCE(del.matched_rule, '') AS matched_rule
    FROM dir_summary d
    LEFT JOIN (
        SELECT DISTINCT d2.bucket, d2.dir_prefix
        FROM protect_rules p
        JOIN dir_summary d2
            ON d2.dir_prefix LIKE p.pattern
            AND (d2.bucket = p.bucket OR p.bucket = '*')
        WHERE d2.bucket = ?
    ) p USING (bucket, dir_prefix)
    LEFT JOIN (
        SELECT DISTINCT ON (d3.bucket, d3.dir_prefix)
               d3.bucket, d3.dir_prefix, dr.pattern AS matched_rule
        FROM delete_rules dr
        JOIN dir_summary d3 ON d3.dir_prefix LIKE dr.pattern
        WHERE d3.bucket = ?
          AND dr.storage_class IS NULL
    ) del USING (bucket, dir_prefix)
    WHERE d.bucket = ?
    ORDER BY d.dir_prefix
"""


def _compute_deletion_entries(conn: Any, bucket: str) -> list[DirEntry]:
    """Query dir_summary with delete/keep status and return delete-marked entries."""
    rows = _fetchall_dicts(conn.execute(_STATUS_QUERY, (bucket, bucket, bucket)))

    all_entries = [
        DirEntry(
            bucket=r["bucket"],
            prefix=r["dir_prefix"],
            status=r["status"],
            matched_rule=r["matched_rule"],
            standard_count=int(r["standard_count"] or 0),
            standard_bytes=int(r["standard_bytes"] or 0),
            nearline_count=int(r["nearline_count"] or 0),
            nearline_bytes=int(r["nearline_bytes"] or 0),
            coldline_count=int(r["coldline_count"] or 0),
            coldline_bytes=int(r["coldline_bytes"] or 0),
            archive_count=int(r["archive_count"] or 0),
            archive_bytes=int(r["archive_bytes"] or 0),
        )
        for r in rows
    ]

    collapsed = _collapse_deletions(all_entries)
    return [e for e in collapsed if e.status == "delete"]


def _compute_fingerprint(catalog: Any) -> str:
    """Fingerprint from plan + rule file contents."""
    h = hashlib.sha256(PLAN_FINGERPRINT.encode())
    if catalog.protect_rules_json.exists():
        h.update(file_digest(catalog.protect_rules_json).encode())
    if catalog.delete_rules_json.exists():
        h.update(file_digest(catalog.delete_rules_json).encode())
    return h.hexdigest()


CSV_COLUMNS = [
    "bucket",
    "prefix",
    "object_count",
    "total_bytes",
    "bytes_human",
    "storage_class_breakdown",
    "matched_rule",
]


def _write_manifest(entries: list[DirEntry], path: Any) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for e in entries:
            writer.writerow(
                {
                    "bucket": e.bucket,
                    "prefix": e.prefix,
                    "object_count": e.object_count,
                    "total_bytes": e.total_bytes,
                    "bytes_human": human_bytes(e.total_bytes),
                    "storage_class_breakdown": e.storage_class_breakdown,
                    "matched_rule": e.matched_rule,
                }
            )


def _print_summary(entries: list[DirEntry]) -> None:
    """Print a human-readable summary of the deletion manifest."""
    if not entries:
        print_summary("  no directories marked for deletion")
        return

    # Per-region summary
    by_region: dict[str, list[DirEntry]] = defaultdict(list)
    for e in entries:
        region = e.bucket.removeprefix("marin-")
        by_region[region].append(e)

    prices_us = {"STANDARD": 0.020, "NEARLINE": 0.010, "COLDLINE": 0.004, "ARCHIVE": 0.0012}
    prices_eu = {"STANDARD": 0.023, "NEARLINE": 0.013, "COLDLINE": 0.006, "ARCHIVE": 0.0025}
    discount = 1.0 - GCS_DISCOUNT

    grand_prefixes = 0
    grand_objects = 0
    grand_bytes = 0
    grand_cost = 0.0

    for region in sorted(by_region):
        region_entries = by_region[region]
        continent = continent_for_region(region)
        prices = prices_us if continent == "US" else prices_eu

        region_objects = sum(e.object_count for e in region_entries)
        region_bytes = sum(e.total_bytes for e in region_entries)
        region_cost = 0.0
        for e in region_entries:
            for sc, byte_val in [
                ("STANDARD", e.standard_bytes),
                ("NEARLINE", e.nearline_bytes),
                ("COLDLINE", e.coldline_bytes),
                ("ARCHIVE", e.archive_bytes),
            ]:
                region_cost += byte_val / (1024**3) * prices[sc] * discount

        print_summary(
            f"  {region}: {len(region_entries)} prefixes, "
            f"{region_objects:,} objects, {human_bytes(region_bytes)}, "
            f"~${region_cost:,.2f}/mo"
        )
        grand_prefixes += len(region_entries)
        grand_objects += region_objects
        grand_bytes += region_bytes
        grand_cost += region_cost

    print_summary(
        f"  total: {grand_prefixes} prefixes, "
        f"{grand_objects:,} objects, {human_bytes(grand_bytes)}, "
        f"~${grand_cost:,.2f}/mo"
    )

    # Top 10 largest
    top = sorted(entries, key=lambda e: e.total_bytes, reverse=True)[:10]
    print_summary("  top 10 by size:")
    for e in top:
        print_summary(f"    {e.bucket}/{e.prefix}  {human_bytes(e.total_bytes)}  ({e.object_count:,} objects)")


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------


def _materialize_deletion_set(ctx: Context, action: StepSpec) -> None:
    catalog = DEFAULT_CATALOG
    fingerprint = _compute_fingerprint(catalog)
    if not ctx.force and marker_matches(ctx.conn, action.action_id, fingerprint):
        print_summary(f"skip {action.action_id}: marker is current")
        return

    all_entries: list[DirEntry] = []
    for plan_row in plan_rows():
        bucket = plan_row["bucket"]
        print_summary(f"  computing deletion set for {bucket}...")
        entries = _compute_deletion_entries(ctx.conn, bucket)
        all_entries.extend(entries)

    manifest_path = catalog.deletion_manifest_csv
    _write_manifest(all_entries, manifest_path)
    print_summary(f"  wrote {len(all_entries)} prefixes to {manifest_path.relative_to(REPO_ROOT)}")

    _print_summary(all_entries)

    csv_fingerprint = file_digest(manifest_path) if manifest_path.exists() else ""
    write_marker(
        ctx.conn,
        action.action_id,
        fingerprint,
        dry_run=ctx.dry_run,
        extra={
            "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
            "csv_fingerprint": csv_fingerprint,
            "prefix_count": len(all_entries),
            "total_objects": sum(e.object_count for e in all_entries),
            "total_bytes": sum(e.total_bytes for e in all_entries),
        },
    )


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

STEPS: list[StepSpec] = [
    StepSpec(
        action_id="compute.materialize_deletion_set",
        group_name="compute",
        command_name="materialize-deletion-set",
        description="Compute and collapse the deletion set into a CSV manifest.",
        help_text=(
            "Evaluate delete_rules against protect_rules and the dir_summary table. "
            "Collapse directories where all children are marked for deletion. "
            "Output a CSV manifest for human inspection before cleanup."
        ),
        mutating=False,
        runner=_materialize_deletion_set,
        predecessors=("prep.materialize_dir_summary",),
    ),
]

STEP_INDEX = {step.action_id: step for step in STEPS}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_context(
    *,
    dry_run: bool,
    force: bool,
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
        scan_workers=1,
        settle_hours=0,
        log_path=catalog.log_dir / f"{log_prefix}_{timestamp_string()}.log",
        timestamp=timestamp_string(),
        project=project,
    )


def runtime_options() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        options: list[Callable[[Callable[..., Any]], Callable[..., Any]]] = [
            click.option("--force", is_flag=True, help="Ignore cached markers and recompute."),
            click.option("--dry-run", is_flag=True, help="Read-only mode: inspect but never write the step marker."),
            click.option("--project", help="Override the GCP project for Cloud Storage API calls."),
        ]
        for option in reversed(options):
            func = option(func)
        return func

    return decorator


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Compute the deletion set: evaluate delete and protect rules against the "
        "scanned object catalog, collapse directories, and output a CSV manifest "
        "for human review."
    ),
)
def cli() -> None:
    pass


@cli.command("plan", help="Show ordered step list and completion status.")
def plan_cli() -> None:
    conn = ensure_output_dirs()
    print("Ordered compute steps:")
    for step in STEPS:
        status = "done" if marker_exists(conn, step.action_id) else "pending"
        print(f"  {status:7}  {step.action_id}")


@cli.command("run", help="Compute the deletion manifest.")
@runtime_options()
def run_cli(dry_run: bool, force: bool, project: str | None) -> None:
    ctx = build_context(dry_run=dry_run, force=force, log_prefix="compute_run", project=project)
    for step in STEPS:
        missing = [p for p in step.predecessors if not marker_exists(ctx.conn, p)]
        if missing and not force:
            raise RuntimeError(
                f"{step.action_id} requires these predecessor steps to be complete first: {', '.join(missing)}"
            )
        print_summary(f"==> {step.action_id}: {step.description}")
        step.run(ctx)
    print_summary("completed")


@cli.group(help="Compute commands.")
def compute() -> None:
    pass


@compute.command("materialize-deletion-set", help=STEPS[0].help_text)
@runtime_options()
def materialize_cmd(dry_run: bool, force: bool, project: str | None) -> None:
    ctx = build_context(dry_run=dry_run, force=force, log_prefix="compute__materialize", project=project)
    step = STEPS[0]
    missing = [p for p in step.predecessors if not marker_exists(ctx.conn, p)]
    if missing and not force:
        raise RuntimeError(
            f"{step.action_id} requires these predecessor steps to be complete first: {', '.join(missing)}"
        )
    print_summary(f"==> {step.action_id}: {step.description}")
    step.run(ctx)
    print_summary("completed")


if __name__ == "__main__":
    cli()
