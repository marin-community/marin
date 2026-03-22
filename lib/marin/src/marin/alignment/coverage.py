# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Covering array generation for variation axes.

Given a set of axes (each with a discrete spectrum of values), generate the
smallest set of configurations such that every t-tuple of axis-value assignments
appears in at least one configuration. Uses a greedy set-cover algorithm.

t=2 gives pairwise covering, t=3 gives 3-way covering, etc.

"""

from __future__ import annotations

import random
import re
from itertools import combinations, product
from typing import Any


def generate_covering_configs(
    variation_axes: list[dict[str, Any]],
    t: int = 2,
    seed: int = 42,
    n_candidates: int = 200,
) -> list[dict[str, str]]:
    """Generate t-way covering array over all axes.

    Returns list of configs, each mapping axis_name -> spectrum_value.
    Guarantees every t-tuple (axis_i1=val1, ..., axis_it=valt) appears in at
    least one config.

    Algorithm: greedy set cover
    1. Enumerate all uncovered t-tuples
    2. While uncovered tuples remain, generate N random candidates and pick
       the one covering the most uncovered tuples
    3. Return the array
    """
    rng = random.Random(seed)

    axes_names = [ax["axis"] for ax in variation_axes]
    axes_values = [ax["spectrum"] for ax in variation_axes]
    n_axes = len(axes_names)

    if n_axes < t:
        if n_axes == 0:
            return []
        all_combos = list(product(*axes_values))
        return [{axes_names[i]: combo[i] for i in range(n_axes)} for combo in all_combos]

    # Build set of all t-tuples that need covering.
    uncovered: set[tuple[tuple[int, int], ...]] = set()
    for axis_indices in combinations(range(n_axes), t):
        val_ranges = [range(len(axes_values[ai])) for ai in axis_indices]
        for val_combo in product(*val_ranges):
            tup = tuple((axis_indices[k], val_combo[k]) for k in range(t))
            uncovered.add(tup)

    configs: list[dict[str, str]] = []
    all_axis_combos = list(combinations(range(n_axes), t))

    while uncovered:
        best_config_indices: list[int] | None = None
        best_count = 0

        for _ in range(n_candidates):
            candidate = [rng.randrange(len(vals)) for vals in axes_values]

            count = 0
            for axis_indices in all_axis_combos:
                tup = tuple((ai, candidate[ai]) for ai in axis_indices)
                if tup in uncovered:
                    count += 1

            if count > best_count:
                best_count = count
                best_config_indices = candidate

        if best_config_indices is None:
            best_config_indices = [rng.randrange(len(vals)) for vals in axes_values]

        to_remove = set()
        for axis_indices in all_axis_combos:
            tup = tuple((ai, best_config_indices[ai]) for ai in axis_indices)
            if tup in uncovered:
                to_remove.add(tup)
        uncovered -= to_remove

        config = {axes_names[i]: axes_values[i][best_config_indices[i]] for i in range(n_axes)}
        configs.append(config)

    return configs


def compute_coverage_stats(
    configs: list[dict[str, str]],
    axes: list[dict[str, Any]],
    t: int = 2,
) -> dict[str, Any]:
    """Return coverage statistics: total t-tuples, covered t-tuples, per-axis value counts."""
    axes_names = [ax["axis"] for ax in axes]
    axes_values = [ax["spectrum"] for ax in axes]
    n_axes = len(axes_names)

    total_tuples = 0
    for axis_indices in combinations(range(n_axes), t):
        count = 1
        for ai in axis_indices:
            count *= len(axes_values[ai])
        total_tuples += count

    covered_tuples: set[tuple[tuple[str, str], ...]] = set()
    for config in configs:
        for axis_indices in combinations(range(n_axes), t):
            tup = tuple((axes_names[ai], config[axes_names[ai]]) for ai in axis_indices)
            covered_tuples.add(tup)

    # Single pass over configs to count per-axis value occurrences
    per_axis_value_counts: dict[str, dict[str, int]] = {ax["axis"]: {val: 0 for val in ax["spectrum"]} for ax in axes}
    for config in configs:
        for name in axes_names:
            val = config.get(name)
            if val is not None and val in per_axis_value_counts[name]:
                per_axis_value_counts[name][val] += 1

    return {
        "total_tuples": total_tuples,
        "covered_tuples": len(covered_tuples),
        "covering_strength": t,
        "num_configs": len(configs),
        "per_axis_value_counts": per_axis_value_counts,
    }


def slugify_axis_value(axis_name: str, value: str) -> str:
    """Convert axis name + value to a filterable tag string.

    Example: ("user_cultural_context", "East Asian") -> "culture:east_asian"
    """
    short_names: dict[str, str] = {
        "user_cultural_context": "culture",
        "user_demographic_identity": "demographic",
    }
    short = short_names.get(axis_name, axis_name)

    slug = value.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")

    if len(slug) > 40:
        slug = slug[:40].rstrip("_")

    return f"{short}:{slug}"


def make_tags(config: dict[str, str], axes: list[dict[str, Any]]) -> list[str]:
    """Generate flat tag list from an axis configuration for easy filtering."""
    tags = []
    for ax in axes:
        name = ax["axis"]
        if name in config:
            tags.append(slugify_axis_value(name, config[name]))
    return tags
