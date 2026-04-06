# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rule consolidation helpers for the storage purge workflow."""

from __future__ import annotations

from collections import defaultdict

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
