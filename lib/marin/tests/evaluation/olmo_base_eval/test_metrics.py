# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""W&B metric naming is a stable contract.

These exact key strings are consumed by downstream mixture configs and SC
analyses, so a rename is a silent break. Guards the "metric naming drift" class.
"""

from __future__ import annotations

from marin.evaluation.olmo_base_eval.components import leaf_components, mmlu_subjects, table9_components
from marin.evaluation.olmo_base_eval.metrics import (
    TABLE9_MACRO_ALIAS_KEY,
    TABLE9_MACRO_KEY,
    build_wandb_metrics,
    sc_mmlu_subject_key,
    sc_task_key,
    table9_component_key,
)


def test_table9_and_sc_key_formats_are_exact():
    assert table9_component_key("arc_easy") == "olmo_base_easy/table9/arc_easy/bpb"
    assert table9_component_key("mmlu_stem") == "olmo_base_easy/table9/mmlu_stem/bpb"
    assert TABLE9_MACRO_KEY == "olmo_base_easy/table9_macro_bpb"
    assert TABLE9_MACRO_ALIAS_KEY == "olmo_base_easy/table9_51_component_macro_bpb"
    assert sc_task_key("arc_easy") == "olmo_base_eval/easy_bpb/arc_easy/bpb"
    # SC suffixes MMLU subjects with _rc (this is how the oracle columns are named).
    assert sc_mmlu_subject_key("mmlu_abstract_algebra") == "olmo_base_eval/easy_bpb/mmlu_abstract_algebra_rc/bpb"


def test_build_wandb_metrics_emits_all_components_macro_and_sc_keys():
    component_bpb = {component: 1.0 for component in table9_components()}
    leaf_bpb = {task: 1.0 for task in leaf_components()}
    mmlu_subject_bpb = {subject: 1.0 for subject in mmlu_subjects()}
    metrics = build_wandb_metrics(
        component_bpb=component_bpb,
        leaf_bpb=leaf_bpb,
        mmlu_subject_bpb=mmlu_subject_bpb,
        macro_bpb=1.0,
    )
    # All 51 Table 9 component keys present.
    for component in table9_components():
        assert table9_component_key(component) in metrics
    # Macro (and its explicit alias) present.
    assert TABLE9_MACRO_KEY in metrics
    assert TABLE9_MACRO_ALIAS_KEY in metrics
    # SC-compat keys for the 47 leaves and 57 MMLU subjects present.
    assert sc_task_key("lambada") in metrics
    assert sc_mmlu_subject_key("mmlu_world_religions") in metrics
    # Expected total: 51 components + 2 macro keys + 47 leaf + 57 subject = 157.
    assert len(metrics) == 51 + 2 + 47 + 57
