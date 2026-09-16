# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Weighted subsampling of the hero data mixture to a fixed token budget.

To retokenize ~100B tokens (~0.5% of the 23T store) at the step-108000 mixture distribution
without a global two-pass count, keep each surviving document independently with a per-cell
Bernoulli probability

    p[cell] = min(1, target_tokens * weight[cell] / available_tokens[cell])

where ``weight`` is the step-108k ("main") phase weight over the 200 ``(cluster, quality)`` cells
and ``available_tokens`` is the store's per-cell token count. The expected sampled tokens per cell
is then ``p[cell] * available_tokens[cell] ≈ target_tokens * weight[cell]``, i.e. the mixture
distribution, and the expected total is ``target_tokens``. This is embarrassingly parallel: each
shard decides independently, no coordination.
"""

from __future__ import annotations

import json

CELL_CLUSTER_DIGITS = slice(1, 3)  # "c27q0" -> cluster 27
CELL_QUALITY_INDEX = 4  # "c27q0" -> quality 0


def cell_name(cluster: int, quality: int) -> str:
    return f"c{cluster:02d}q{quality}"


def keep_probabilities(mix_json_path: str, target_tokens: int, phase: str = "main") -> dict[str, float]:
    """Per-cell Bernoulli keep probability for a ``target_tokens`` weighted sample.

    ``phase`` selects the phase whose weights define the target distribution (the step-108k
    distribution is the "main" phase of the September mixture).
    """
    with open(mix_json_path) as handle:
        spec = json.load(handle)
    available = spec["available_tokens"]
    phases = {p["name"]: p["weights"] for p in spec["phases"]}
    if phase not in phases:
        raise ValueError(f"phase {phase!r} not in {sorted(phases)}")
    weights = phases[phase]
    if set(weights) != set(available):
        raise ValueError("weight cells do not match available_tokens cells")
    return {
        cell: 0.0 if available[cell] <= 0 else min(1.0, target_tokens * weights[cell] / available[cell])
        for cell in available
    }


def expected_sampled_tokens(keep_probs: dict[str, float], mix_json_path: str) -> float:
    with open(mix_json_path) as handle:
        available = json.load(handle)["available_tokens"]
    return sum(keep_probs[c] * available[c] for c in keep_probs)
