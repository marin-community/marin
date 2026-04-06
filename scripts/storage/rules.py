# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rule loading and consolidation for the storage purge workflow."""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict

import pyarrow as pa

from scripts.storage.db import (
    DEFAULT_CATALOG,
    MARIN_BUCKETS,
    Context,
    StepSpec,
    file_digest,
    glob_to_like,
    marker_matches,
    normalize_relative_prefix,
    normalized_prefix_url,
    print_summary,
    read_csv_rows,
    url_object_path,
    write_marker,
)

log = logging.getLogger(__name__)

_WILDCARD_PREFIXES = ("raw/", "tokenized/")


def _consolidate_wildcard_rules(
    rows: list[tuple[str, str, str, str, str, str]],
    all_buckets: frozenset[str],
) -> list[tuple[str, str, str, str, str, str]]:
    """Consolidate per-bucket rules into wildcard (bucket='*') rules where possible.

    A pattern is consolidated when it appears with identical owners, reasons,
    and sources across every bucket. Additionally, patterns under raw/ or
    tokenized/ are always promoted to wildcard rules since those prefixes
    exist in every bucket.
    """
    # Group rows by pattern
    by_pattern: dict[str, list[tuple[str, str, str, str, str, str]]] = defaultdict(list)
    for r in rows:
        by_pattern[r[1]].append(r)

    result: list[tuple[str, str, str, str, str, str]] = []
    for pattern, group in by_pattern.items():
        buckets_present = {r[0] for r in group}

        # Check if all buckets have this pattern with identical metadata
        representative = group[0]
        identical_across_all = buckets_present == all_buckets and all(
            r[3] == representative[3] and r[4] == representative[4] for r in group
        )

        # Promote to wildcard if present in all buckets or under raw/tokenized
        is_wildcard_prefix = any(pattern.startswith(wp) for wp in _WILDCARD_PREFIXES)

        if identical_across_all or is_wildcard_prefix:
            result.append(("*", pattern, representative[2], representative[3], representative[4], representative[5]))
        else:
            result.extend(group)

    return result


def load_protect_rules(ctx: Context, action: StepSpec) -> None:
    """Load protect globs and direct prefixes into the protect_rules DB table.

    Reads both classified and direct CSVs. Direct prefixes become pattern_type='prefix',
    wildcard globs become pattern_type='like' with * → % conversion. No GCS calls needed.
    """
    protect_dir = DEFAULT_CATALOG.protect_dir
    classified_path = protect_dir / "protect_prefixes_classified.csv"
    direct_path = protect_dir / "protect_prefixes_direct.csv"
    fingerprint = hashlib.sha256((file_digest(classified_path) + file_digest(direct_path)).encode()).hexdigest()
    if not ctx.force and marker_matches(ctx.conn, action.action_id, fingerprint):
        print_summary(f"skip {action.action_id}: marker is current")
        return

    classified_rows = read_csv_rows(classified_path)
    direct_rows = read_csv_rows(direct_path)

    # Build DB rows: (bucket, pattern, pattern_type, owners, reasons, sources)
    # All patterns are normalized to LIKE format (relative path, % wildcard).
    db_rows: list[tuple[str, str, str, str, str, str]] = []

    for row in direct_rows:
        rel = normalize_relative_prefix(url_object_path(normalized_prefix_url(row["sts_prefix"])))
        like_pattern = rel + "%"
        db_rows.append((row["bucket"], like_pattern, "like", row["owners"], row["reasons"], row["sources"]))

    for row in classified_rows:
        if row["classification"] != "sts_prefix_via_listing":
            continue
        like_pattern = glob_to_like(row["normalized_glob"])
        # Strip gs://bucket/ prefix if present
        like_pattern = url_object_path(like_pattern) if like_pattern.startswith("gs://") else like_pattern
        db_rows.append((row["bucket"], like_pattern, "like", row["owners"], row["reasons"], row["sources"]))

    # Deduplicate by (bucket, pattern), keeping the first occurrence
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str, str, str, str, str]] = []
    for r in db_rows:
        key = (r[0], r[1])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    # Consolidate rules that appear identically across all buckets into
    # a single wildcard rule (bucket='*').  Also create wildcard rules for
    # the raw/ and tokenized/ top-level prefixes which exist in every bucket.
    all_bucket_set = frozenset(MARIN_BUCKETS)
    deduped = _consolidate_wildcard_rules(deduped, all_bucket_set)

    # Write to DB
    conn = ctx.conn
    conn.execute("DELETE FROM rule_costs")
    conn.execute("DELETE FROM protect_rules")
    if deduped:
        arrow_table = pa.table(
            {
                "bucket": [r[0] for r in deduped],
                "pattern": [r[1] for r in deduped],
                "pattern_type": [r[2] for r in deduped],
                "owners": [r[3] for r in deduped],
                "reasons": [r[4] for r in deduped],
                "sources": [r[5] for r in deduped],
            }
        )
        conn.register("_protect_stage", arrow_table)
        conn.execute(
            """
            INSERT INTO protect_rules (bucket, pattern, pattern_type, owners, reasons, sources)
            SELECT bucket, pattern, pattern_type, owners, reasons, sources FROM _protect_stage
            ON CONFLICT (bucket, pattern) DO UPDATE SET
                pattern_type = EXCLUDED.pattern_type,
                owners = EXCLUDED.owners,
                reasons = EXCLUDED.reasons,
                sources = EXCLUDED.sources
            """
        )
        conn.unregister("_protect_stage")

    wildcard_count = sum(1 for r in deduped if r[0] == "*")
    like_count = sum(1 for r in deduped if r[2] == "like")
    print_summary(
        f"{action.action_id}: loaded {len(deduped)} protect rules "
        f"({wildcard_count} wildcard, {like_count} like) into DB"
    )
    write_marker(ctx.conn, action.action_id, fingerprint, dry_run=ctx.dry_run)
