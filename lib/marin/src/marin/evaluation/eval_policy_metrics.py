# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Metric names for the October 2026 publication-policy evaluation suite.

``evaluator_metric`` is the spelling declared by Evalchemy or Harbor. ``record_metric``
is the canonical key consumed by Evaldash. ``published_metric`` identifies the score
specified in issue #10197; it can name a subgroup or a separate cohort statistic.
The map describes the current campaign, not historical record migrations.
CRUXEval's record metric combines directions while its published metric uses only
output trials. SOTOPIA's published metric is computed across a model cohort after
the Harbor reward record is written.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class PolicyMetric:
    mechanism: str
    task: str | None
    evaluator_metric: str
    record_metric: str
    published_metric: str


def _evalchemy(task: str, evaluator: str, record: str, published: str | None = None) -> PolicyMetric:
    return PolicyMetric("evalchemy", task, evaluator, record, published or evaluator)


def _harbor(published: str = "reward") -> PolicyMetric:
    return PolicyMetric("harbor", None, "reward", "reward", published)


POLICY_METRICS_2026_10_16: Mapping[str, PolicyMetric] = MappingProxyType(
    {
        "math500": _evalchemy("MATH500", "accuracy", "accuracy"),
        "humanevalplus": _evalchemy("HumanEvalPlus", "pass@1", "pass_at_1"),
        "mbppplus": _evalchemy("MBPPPlus", "pass@1", "pass_at_1"),
        "olympiadbench": _evalchemy("OlympiadBench", "accuracy_avg", "accuracy"),
        "gsm8k-0shot": _evalchemy("gsm8k", "exact_match", "accuracy"),
        "piqa": _evalchemy("piqa", "acc_norm", "normalized_accuracy"),
        "winogrande": _evalchemy("winogrande", "acc", "accuracy"),
        "boolq": _evalchemy("boolq", "acc", "accuracy"),
        "truthfulqa": _evalchemy("truthfulqa_mc2", "acc", "accuracy"),
        "triviaqa": _evalchemy("triviaqa", "exact_match", "accuracy"),
        "aime24": _evalchemy("AIME24", "accuracy_avg", "accuracy"),
        "mmlu-pro": _evalchemy("MMLUPro", "accuracy_avg", "accuracy"),
        "gpqa-diamond": _evalchemy("GPQADiamond", "accuracy_avg", "accuracy"),
        "cruxeval": _evalchemy("CruxEval", "pass@1", "pass_at_1", "output_pass@1"),
        "financebench": _evalchemy("FinanceBench", "accuracy", "accuracy"),
        "ifbench": _evalchemy("IFBench", "strict_prompt_accuracy", "strict_prompt_accuracy"),
        "mrcr": _evalchemy("MRCR", "mrcr_accuracy", "mrcr_accuracy"),
        "nupa": _evalchemy("NUPA5K-Loose", "exact_match", "accuracy"),
        "swebench-verified": _harbor(),
        "ot-tblite-recovery": _harbor(),
        "tb2-recovery": _harbor(),
        "ds-1000-local": _harbor(),
        "bfclparity-pi": _harbor(),
        "bixbench-pi": _harbor(),
        "tau3-pi": _harbor(),
        "sotopia-hard": _harbor("length_controlled_overall"),
    }
)
