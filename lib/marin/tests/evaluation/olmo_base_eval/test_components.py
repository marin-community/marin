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


def test_leaf_components_exclude_the_four_mmlu_buckets():
    leaves = leaf_components()
    assert len(leaves) == 47
    assert not (set(leaves) & set(MMLU_BUCKETS))
    assert set(leaves) | set(MMLU_BUCKETS) == set(table9_components())


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


def test_scored_tasks_are_47_leaves_plus_57_subjects():
    assert len(scored_tasks()) == 104
    assert set(scored_tasks()) == set(leaf_components()) | set(mmlu_subjects())
