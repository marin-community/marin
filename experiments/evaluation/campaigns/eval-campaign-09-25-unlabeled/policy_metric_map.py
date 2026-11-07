# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curated and Harbor metric contracts for the publication campaign.

Curated evaluator and record keys use their protocol spelling. Published metrics
can select a declared subgroup; CRUXEval publishes output-direction accuracy.
Historical records retain their recorded benchmark metadata.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class PolicyMechanism(StrEnum):
    CURATED = "curated"
    HARBOR = "harbor"


@dataclass(frozen=True)
class PolicyMetric:
    mechanism: PolicyMechanism
    task: str | None
    evaluator_metric: str
    record_metric: str
    published_metric: str


def _curated(task: str, evaluator: str, record: str, published: str | None = None) -> PolicyMetric:
    return PolicyMetric(PolicyMechanism.CURATED, task, evaluator, record, published or evaluator)


def _harbor(metric: str = "reward") -> PolicyMetric:
    return PolicyMetric(PolicyMechanism.HARBOR, None, metric, metric, metric)


POLICY_METRICS_2026_10_16: Mapping[str, PolicyMetric] = MappingProxyType(
    {
        "math500": _curated("MATH500", "accuracy", "accuracy"),
        "humanevalplus": _curated("HumanEvalPlus", "accuracy", "accuracy"),
        "mbppplus": _curated("MBPPPlus", "accuracy", "accuracy"),
        "olympiadbench": _curated("OlympiadBench", "accuracy", "accuracy"),
        "gsm8k-0shot": _curated("gsm8k-0shot", "exact_match,strict-match", "exact_match,strict-match"),
        "piqa": _curated("piqa", "acc_norm,none", "acc_norm,none"),
        "winogrande": _curated("winogrande", "acc,none", "acc,none"),
        "boolq": _curated("boolq", "acc,none", "acc,none"),
        "truthfulqa": _curated("truthfulqa_mc2", "acc,none", "acc,none"),
        "triviaqa": _curated("triviaqa", "exact_match,strict_answer", "exact_match,strict_answer"),
        "aime24": _curated("AIME24", "accuracy", "accuracy"),
        "mmlu-pro": _curated("MMLUPro", "accuracy", "accuracy"),
        "gpqa-diamond": _curated("GPQADiamond", "accuracy", "accuracy"),
        "cruxeval": _curated("CruxEval", "accuracy", "accuracy", "direction=output/accuracy"),
        "financebench": _curated("FinanceBench", "accuracy", "accuracy"),
        "ifbench": _curated("IFBench", "strict_prompt_accuracy", "strict_prompt_accuracy"),
        "mrcr": _curated("MRCR", "accuracy", "accuracy"),
        "mrcr-32k": _curated("MRCR-32k", "accuracy", "accuracy"),
        "mrcr-65k": _curated("MRCR-65k", "accuracy", "accuracy"),
        "nupa": _curated("NUPA5K-Loose", "exact_match", "exact_match"),
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
