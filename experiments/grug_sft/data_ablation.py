# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stable identities and mixture transforms for SFT source ablations."""

import hashlib
import math
import re
from collections.abc import Collection, Mapping

_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def canonical_exclusions(excluded_groups: Collection[str]) -> tuple[str, ...]:
    """Return a validated, order-independent exclusion tuple."""
    if len(excluded_groups) != len(set(excluded_groups)):
        raise ValueError("Excluded SFT groups must be unique")
    if any(not group for group in excluded_groups):
        raise ValueError("Excluded SFT groups must be nonempty")
    return tuple(sorted(excluded_groups))


def ablated_group_weights(group_weights: Mapping[str, float], excluded_groups: Collection[str]) -> dict[str, float]:
    """Remove groups and renormalize the remaining groups to the original SFT share."""
    excluded = canonical_exclusions(excluded_groups)
    unknown = set(excluded) - group_weights.keys()
    if unknown:
        raise ValueError(f"Unknown SFT groups: {', '.join(sorted(unknown))}")
    selected = {name: weight for name, weight in group_weights.items() if name not in excluded}
    if not selected:
        raise ValueError("An ablation must retain at least one SFT group")
    if any(weight <= 0 for weight in group_weights.values()):
        raise ValueError("SFT group weights must be positive")
    original_share = math.fsum(group_weights.values())
    selected_share = math.fsum(selected.values())
    return {name: weight * original_share / selected_share for name, weight in selected.items()}


def versioned_run_id(base_run_id: str, version: str, excluded_groups: Collection[str]) -> str:
    """Build a stable run ID whose ablation identity is independent of CLI ordering."""
    if not _SAFE_SEGMENT.fullmatch(version):
        raise ValueError("Version must contain only letters, numbers, '.', '_', and '-'")
    excluded = canonical_exclusions(excluded_groups)
    if not excluded:
        return f"{base_run_id}-{version}"
    digest = hashlib.sha256("\n".join(excluded).encode()).hexdigest()[:10]
    return f"{base_run_id}-drop-{digest}-{version}"
