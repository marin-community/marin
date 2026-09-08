# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Table 9 BPB component registry for OLMoBaseEval Easy.

Defines the 51 Table 9 components, the 57 MMLU subjects and their category
collapse weights, and the canonical component ordering.

Ported verbatim from the OLMix paper's Table 9 ("Details of the BPB evaluation
suite") and OLMo-Eval. The MMLU category weights are copied from OLMix's
``aggregate_mmlu`` (per-subject instance-size proportions that sum to 1 within
each category); each of the four MMLU buckets enters Table 9 as one component
equal to the size-weighted micro-average of its subjects' reading-comprehension
(``_rc``) BPB. Every other Table 9 component is a leaf task scored directly.
"""

from __future__ import annotations

NUM_TABLE9_COMPONENTS = 51
NUM_MMLU_SUBJECTS = 57

MINERVA_SUBTASKS: tuple[str, ...] = (
    "minerva_math_algebra",
    "minerva_math_counting_and_probability",
    "minerva_math_geometry",
    "minerva_math_intermediate_algebra",
    "minerva_math_number_theory",
    "minerva_math_prealgebra",
    "minerva_math_precalculus",
)

MT_MBPP_SUBTASKS: tuple[str, ...] = (
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
)

BASIC_SKILLS_SUBTASKS: tuple[str, ...] = (
    "basic_skills_arithmetic",
    "basic_skills_coding",
    "basic_skills_common_knowledge",
    "basic_skills_logical_reasoning",
    "basic_skills_pattern",
    "basic_skills_string_operations",
)

# The 11 standalone QA tasks that follow the MMLU buckets in Table 9 order.
STANDALONE_QA_AFTER_MMLU: tuple[str, ...] = (
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
)

# The four derived MMLU components, in Table 9 order.
MMLU_BUCKETS: tuple[str, ...] = (
    "mmlu_stem",
    "mmlu_humanities",
    "mmlu_social_sciences",
    "mmlu_other",
)

# Per-subject collapse weights (instance-size proportions, sum to 1 per bucket).
# Copied verbatim from OLMix's aggregate_mmlu so the port stays standalone.
MMLU_CATEGORY_WEIGHTS: dict[str, dict[str, float]] = {
    "mmlu_stem": {
        "mmlu_abstract_algebra": 0.03313452617627568,
        "mmlu_astronomy": 0.05036447978793903,
        "mmlu_college_biology": 0.04771371769383698,
        "mmlu_college_chemistry": 0.03313452617627568,
        "mmlu_college_computer_science": 0.03313452617627568,
        "mmlu_college_mathematics": 0.03313452617627568,
        "mmlu_college_physics": 0.033797216699801194,
        "mmlu_computer_security": 0.03313452617627568,
        "mmlu_conceptual_physics": 0.07786613651424784,
        "mmlu_electrical_engineering": 0.04804506295559974,
        "mmlu_elementary_mathematics": 0.12524850894632206,
        "mmlu_high_school_biology": 0.10271703114645461,
        "mmlu_high_school_chemistry": 0.06726308813783963,
        "mmlu_high_school_computer_science": 0.03313452617627568,
        "mmlu_high_school_mathematics": 0.08946322067594434,
        "mmlu_high_school_physics": 0.050033134526176276,
        "mmlu_high_school_statistics": 0.07157057654075547,
        "mmlu_machine_learning": 0.03711066931742876,
    },
    "mmlu_other": {
        "mmlu_anatomy": 0.04164096236890808,
        "mmlu_business_ethics": 0.030845157310302282,
        "mmlu_clinical_knowledge": 0.08173966687230105,
        "mmlu_college_medicine": 0.05336212214682295,
        "mmlu_global_facts": 0.030845157310302282,
        "mmlu_human_aging": 0.06878470080197409,
        "mmlu_management": 0.03177051202961135,
        "mmlu_marketing": 0.07217766810610735,
        "mmlu_medical_genetics": 0.030845157310302282,
        "mmlu_miscellaneous": 0.24151758173966686,
        "mmlu_nutrition": 0.09438618136952498,
        "mmlu_professional_accounting": 0.08698334361505243,
        "mmlu_professional_medicine": 0.08389882788402221,
        "mmlu_virology": 0.05120296113510179,
    },
    "mmlu_social_sciences": {
        "mmlu_econometrics": 0.03704907377315567,
        "mmlu_high_school_geography": 0.06434839129021774,
        "mmlu_high_school_government_and_politics": 0.06272343191420214,
        "mmlu_high_school_macroeconomics": 0.12674683132921677,
        "mmlu_high_school_microeconomics": 0.07734806629834254,
        "mmlu_high_school_psychology": 0.17712057198570036,
        "mmlu_human_sexuality": 0.04257393565160871,
        "mmlu_professional_psychology": 0.19889502762430938,
        "mmlu_public_relations": 0.03574910627234319,
        "mmlu_security_studies": 0.07962300942476438,
        "mmlu_sociology": 0.0653233669158271,
        "mmlu_us_foreign_policy": 0.032499187520311994,
    },
    "mmlu_humanities": {
        "mmlu_formal_logic": 0.026780021253985122,
        "mmlu_high_school_european_history": 0.03506907545164718,
        "mmlu_high_school_us_history": 0.04335812964930925,
        "mmlu_high_school_world_history": 0.050371944739638685,
        "mmlu_international_law": 0.0257173219978746,
        "mmlu_jurisprudence": 0.022954303931987247,
        "mmlu_logical_fallacies": 0.034643995749202974,
        "mmlu_moral_disputes": 0.07353878852284804,
        "mmlu_moral_scenarios": 0.1902231668437832,
        "mmlu_philosophy": 0.06609989373007438,
        "mmlu_prehistory": 0.06886291179596174,
        "mmlu_professional_law": 0.32603613177470775,
        "mmlu_world_religions": 0.03634431455897981,
    },
}


def table9_components() -> tuple[str, ...]:
    """Return the 51 Table 9 component names in canonical (paper) order.

    47 leaf tasks plus the 4 derived MMLU buckets (``mmlu_stem``,
    ``mmlu_humanities``, ``mmlu_social_sciences``, ``mmlu_other``).
    """
    components = (
        *MINERVA_SUBTASKS,
        "codex_humaneval",
        "mbpp",
        *MT_MBPP_SUBTASKS,
        "arc_easy",
        "arc_challenge",
        *MMLU_BUCKETS,
        *STANDALONE_QA_AFTER_MMLU,
        *BASIC_SKILLS_SUBTASKS,
        "lambada",
        "medmcqa",
    )
    assert len(components) == NUM_TABLE9_COMPONENTS, f"expected 51 components, got {len(components)}"
    return components


def leaf_components() -> tuple[str, ...]:
    """Return the 47 Table 9 components scored directly (excludes MMLU buckets)."""
    return tuple(component for component in table9_components() if component not in MMLU_BUCKETS)


def mmlu_subjects() -> tuple[str, ...]:
    """Return the 57 MMLU subject task names (bare, without the ``_rc`` suffix)."""
    subjects = tuple(subject for bucket in MMLU_BUCKETS for subject in MMLU_CATEGORY_WEIGHTS[bucket])
    assert len(subjects) == NUM_MMLU_SUBJECTS, f"expected 57 MMLU subjects, got {len(subjects)}"
    return subjects


def scored_tasks() -> tuple[str, ...]:
    """Return every task scored directly by the evaluator: 47 leaves + 57 MMLU subjects."""
    return (*leaf_components(), *mmlu_subjects())
