# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Table 9 component registry: exactly the 51 named components, in order.

Guards the "missing components" / "extra components" silent-mismatch class: if a
component is dropped, renamed, or duplicated, the macro denominator and the
emitted metric names silently change.
"""

from __future__ import annotations

from marin.evaluation.olmo_base_eval.components import (
    MMLU_BUCKETS,
    MMLU_CATEGORY_WEIGHTS,
    leaf_components,
    mmlu_subjects,
    scored_tasks,
    table9_components,
)

# The canonical 51 Table 9 components, transcribed independently from the OLMix
# Table 9 / SC `table9_macro_components.csv` oracle (bare names; MMLU collapsed
# to 4 buckets). This is the contract the implementation must match.
EXPECTED_TABLE9 = (
    "minerva_math_algebra",
    "minerva_math_counting_and_probability",
    "minerva_math_geometry",
    "minerva_math_intermediate_algebra",
    "minerva_math_number_theory",
    "minerva_math_prealgebra",
    "minerva_math_precalculus",
    "codex_humaneval",
    "mbpp",
    "mt_mbpp_bash",
    "mt_mbpp_c",
    "mt_mbpp_cpp",
    "mt_mbpp_csharp",
    "mt_mbpp_go",
    "mt_mbpp_haskell",
    "mt_mbpp_java",
    "mt_mbpp_javascript",
    "mt_mbpp_matlab",
    "mt_mbpp_php",
    "mt_mbpp_python",
    "mt_mbpp_r",
    "mt_mbpp_ruby",
    "mt_mbpp_rust",
    "mt_mbpp_scala",
    "mt_mbpp_swift",
    "mt_mbpp_typescript",
    "arc_easy",
    "arc_challenge",
    "mmlu_stem",
    "mmlu_humanities",
    "mmlu_social_sciences",
    "mmlu_other",
    "csqa",
    "hellaswag",
    "winogrande",
    "socialiqa",
    "piqa",
    "coqa",
    "drop",
    "jeopardy",
    "naturalqs",
    "squad",
    "sciq",
    "basic_skills_arithmetic",
    "basic_skills_coding",
    "basic_skills_common_knowledge",
    "basic_skills_logical_reasoning",
    "basic_skills_pattern",
    "basic_skills_string_operations",
    "lambada",
    "medmcqa",
)


def test_table9_components_are_the_canonical_51_in_order():
    assert table9_components() == EXPECTED_TABLE9
    assert len(table9_components()) == 51


def test_leaf_components_are_the_47_table9_tasks_excluding_mmlu_buckets():
    # Checked against the independent EXPECTED_TABLE9 oracle, not against
    # table9_components() (which leaf_components() is derived from).
    expected_leaves = tuple(component for component in EXPECTED_TABLE9 if component not in MMLU_BUCKETS)
    assert len(expected_leaves) == 47
    assert leaf_components() == expected_leaves


def test_mmlu_has_57_subjects_split_18_14_12_13():
    counts = {bucket: len(MMLU_CATEGORY_WEIGHTS[bucket]) for bucket in MMLU_BUCKETS}
    assert counts == {
        "mmlu_stem": 18,
        "mmlu_humanities": 13,
        "mmlu_social_sciences": 12,
        "mmlu_other": 14,
    }
    assert len(mmlu_subjects()) == 57
    assert len(set(mmlu_subjects())) == 57  # no subject shared across buckets


def test_scored_tasks_are_104_unique_tasks():
    tasks = scored_tasks()
    assert len(tasks) == 104
    # No task is scored twice — a duplicate would double-count its BPB in the macro.
    assert len(set(tasks)) == 104
