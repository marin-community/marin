# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OLMoBaseEval easy-suite overlap tasks and derived metrics.

This module defines the subset of the OLMo 3 / Olmix base easy BPB suite that we
can run faithfully enough through Marin's current Levanter lm-eval path.

The main exclusions are:
- code BPB tasks, which rely on OLMES's `compute_gold_bpb` behavior rather than
  standard lm-eval code generation metrics,
- Minerva math BPB tasks, which currently require additional math extras in the
  runtime environment,
- generation / gen2mc task variants such as CoQA, DROP, NaturalQs, and SQuAD
  that are not close enough to the OLMES configuration in the current stack.

For MMLU, we run the standard 5-shot `mmlu` task and derive the four Olmix
category BPBs from subject-level BPB metrics using the official lm-eval subject
grouping.
"""

from __future__ import annotations

from collections import defaultdict

from marin.evaluation.evaluation_config import EvalTaskConfig

OLMO_BASE_EASY_OVERLAP_CACHE_PATH = "gs://marin-us-east5/raw/eval-datasets/olmo-base-easy-overlap-v1"

OLMO_BASE_EASY_OVERLAP_TASKS = (
    EvalTaskConfig("mmlu", 5, task_alias="mmlu_5shot"),
    EvalTaskConfig("arc_easy", 5, task_alias="arc_easy_5shot"),
    EvalTaskConfig("arc_challenge", 5, task_alias="arc_challenge_5shot"),
    EvalTaskConfig("commonsense_qa", 5, task_alias="csqa_5shot"),
    EvalTaskConfig("hellaswag", 5, task_alias="hellaswag_5shot"),
    EvalTaskConfig("winogrande", 5, task_alias="winogrande_5shot"),
    EvalTaskConfig("social_iqa", 5, task_alias="socialiqa_5shot"),
    EvalTaskConfig("piqa", 5, task_alias="piqa_5shot"),
    EvalTaskConfig("sciq", 5, task_alias="sciq_5shot"),
    EvalTaskConfig("lambada_openai", 0, task_alias="lambada_0shot"),
    EvalTaskConfig("medmcqa", 5, task_alias="medmcqa_5shot"),
)

# Copied from lm-eval's official MMLU subject grouping in
# lm_eval.tasks.mmlu._generate_configs.SUBJECTS.
MMLU_SUBJECT_TO_CATEGORY = {
    "abstract_algebra": "stem",
    "anatomy": "stem",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}

MMLU_CATEGORY_ORDER = ("stem", "humanities", "social_sciences", "other")


def add_olmo_base_easy_overlap_metrics(flat_metrics: dict[str, float]) -> dict[str, float]:
    """Derive OLMoBaseEval-overlap aggregate metrics from flat eval outputs."""
    derived: dict[str, float] = {}
    category_bpbs: dict[str, list[float]] = defaultdict(list)

    for subject, category in MMLU_SUBJECT_TO_CATEGORY.items():
        metric_key = f"lm_eval/mmlu_{subject}_5shot/bpb"
        value = flat_metrics.get(metric_key)
        if value is None:
            continue
        category_bpbs[category].append(value)

    for category in MMLU_CATEGORY_ORDER:
        values = category_bpbs.get(category)
        if not values:
            continue
        derived[f"lm_eval/mmlu_{category}_5shot/bpb"] = float(sum(values) / len(values))

    macro_metric_keys = [
        "lm_eval/arc_easy_5shot/bpb",
        "lm_eval/arc_challenge_5shot/bpb",
        "lm_eval/mmlu_stem_5shot/bpb",
        "lm_eval/mmlu_humanities_5shot/bpb",
        "lm_eval/mmlu_social_sciences_5shot/bpb",
        "lm_eval/mmlu_other_5shot/bpb",
        "lm_eval/csqa_5shot/bpb",
        "lm_eval/hellaswag_5shot/bpb",
        "lm_eval/winogrande_5shot/bpb",
        "lm_eval/socialiqa_5shot/bpb",
        "lm_eval/piqa_5shot/bpb",
        "lm_eval/sciq_5shot/bpb",
        "lm_eval/lambada_0shot/bpb",
        "lm_eval/medmcqa_5shot/bpb",
    ]

    macro_values = [flat_metrics.get(key, derived.get(key)) for key in macro_metric_keys]
    macro_values = [value for value in macro_values if value is not None]
    if macro_values:
        derived["lm_eval/olmo_base_easy_overlap/macro_bpb"] = float(sum(macro_values) / len(macro_values))
        derived["lm_eval/olmo_base_easy_overlap/task_count"] = float(len(macro_values))

    return derived
