# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stable W&B metric names for the Table 9 BPB evaluator.

Two namespaces are emitted:

- The Marin-native Table 9 names requested for this evaluator:
  ``olmo_base_easy/table9/<component>/bpb`` and ``olmo_base_easy/table9_macro_bpb``
  (with the explicit ``..._51_component_macro_bpb`` alias used by downstream
  mixture configs).
- The SC-compatible per-task keys ``olmo_base_eval/easy_bpb/<task>/bpb`` for the
  47 leaf tasks and 57 MMLU subjects (subjects carry the ``_rc`` suffix), so
  existing SC analyses and W&B writeback keep working unchanged.

These exact strings are a contract; tests assert them to catch naming drift.
"""

from __future__ import annotations

from collections.abc import Mapping

from marin.evaluation.olmo_base_eval.components import table9_components

TABLE9_PREFIX = "olmo_base_easy/table9"
TABLE9_MACRO_KEY = "olmo_base_easy/table9_macro_bpb"
TABLE9_MACRO_ALIAS_KEY = "olmo_base_easy/table9_51_component_macro_bpb"
SC_PREFIX = "olmo_base_eval/easy_bpb"


def table9_component_key(component: str) -> str:
    """W&B key for one Table 9 component, e.g. ``olmo_base_easy/table9/arc_easy/bpb``."""
    return f"{TABLE9_PREFIX}/{component}/bpb"


def sc_task_key(task: str) -> str:
    """SC-compatible per-task key, e.g. ``olmo_base_eval/easy_bpb/arc_easy/bpb``."""
    return f"{SC_PREFIX}/{task}/bpb"


def sc_mmlu_subject_key(subject: str) -> str:
    """SC-compatible key for an MMLU subject (SC suffixes subjects with ``_rc``)."""
    return f"{SC_PREFIX}/{subject}_rc/bpb"


def build_wandb_metrics(
    *,
    component_bpb: Mapping[str, float],
    leaf_bpb: Mapping[str, float],
    mmlu_subject_bpb: Mapping[str, float],
    macro_bpb: float,
) -> dict[str, float]:
    """Assemble the full W&B metric dict (Table 9 names + macro + SC-compat keys)."""
    metrics: dict[str, float] = {}
    for component in table9_components():
        metrics[table9_component_key(component)] = component_bpb[component]
    metrics[TABLE9_MACRO_KEY] = macro_bpb
    metrics[TABLE9_MACRO_ALIAS_KEY] = macro_bpb
    for task, value in leaf_bpb.items():
        metrics[sc_task_key(task)] = value
    for subject, value in mmlu_subject_bpb.items():
        metrics[sc_mmlu_subject_key(subject)] = value
    return metrics
