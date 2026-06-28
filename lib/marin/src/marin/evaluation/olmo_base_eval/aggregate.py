# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate per-task BPB into the 51 Table 9 components and the macro.

The macro is the unweighted mean over the 51 components. The four MMLU buckets
are size-weighted micro-averages over their subjects' ``_rc`` BPB. This matches
OLMix's Table 9 convention and reproduces SC's offline ``table9_macro_bpb`` to
floating-point epsilon (verified against the SC oracle panel).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from marin.evaluation.olmo_base_eval.components import (
    MMLU_BUCKETS,
    MMLU_CATEGORY_WEIGHTS,
    NUM_TABLE9_COMPONENTS,
    leaf_components,
    table9_components,
)


def collapse_mmlu(subject_bpb: Mapping[str, float]) -> dict[str, float]:
    """Collapse 57 MMLU subject BPB into the 4 Table 9 buckets.

    Each bucket is ``sum(weight[subject] * subject_bpb[subject])`` over the
    bucket's subjects, with weights summing to 1. ``subject_bpb`` keys are the
    bare subject names (e.g. ``mmlu_abstract_algebra``), not the ``_rc`` form.
    """
    buckets: dict[str, float] = {}
    for bucket in MMLU_BUCKETS:
        weights = MMLU_CATEGORY_WEIGHTS[bucket]
        if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"MMLU weights for {bucket} do not sum to 1")
        missing = sorted(set(weights).difference(subject_bpb))
        if missing:
            raise ValueError(f"missing MMLU subject BPB for {bucket}: {missing}")
        buckets[bucket] = sum(weight * subject_bpb[subject] for subject, weight in weights.items())
    return buckets


def assemble_table9(leaf_bpb: Mapping[str, float], mmlu_subject_bpb: Mapping[str, float]) -> dict[str, float]:
    """Build the 51-component BPB dict from 47 leaf scores + 57 MMLU subject scores."""
    missing_leaves = sorted(set(leaf_components()).difference(leaf_bpb))
    if missing_leaves:
        raise ValueError(f"missing leaf component BPB: {missing_leaves}")
    buckets = collapse_mmlu(mmlu_subject_bpb)
    return {
        component: buckets[component] if component in MMLU_BUCKETS else leaf_bpb[component]
        for component in table9_components()
    }


def table9_macro(component_bpb: Mapping[str, float]) -> float:
    """Return the macro BPB: the unweighted mean over the 51 Table 9 components."""
    components = table9_components()
    missing = sorted(set(components).difference(component_bpb))
    if missing:
        raise ValueError(f"missing Table 9 components for macro: {missing}")
    return sum(component_bpb[component] for component in components) / NUM_TABLE9_COMPONENTS
