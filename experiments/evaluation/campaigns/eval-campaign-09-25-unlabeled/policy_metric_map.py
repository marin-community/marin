# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Metric contracts for the publication campaign.

Curated evaluator and record keys use their protocol spelling. Published metrics
can select a declared subgroup; CRUXEval publishes output-direction accuracy.
Historical records retain their recorded benchmark metadata.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class PolicyMetric:
    evaluator_metric: str
    record_metric: str
    published_metric: str


def _metric(evaluator: str, record: str, published: str | None = None) -> PolicyMetric:
    return PolicyMetric(evaluator, record, published or evaluator)


POLICY_METRICS_2026_10_16: Mapping[str, PolicyMetric] = MappingProxyType(
    {
        "math500": _metric("accuracy", "accuracy"),
        "humanevalplus": _metric("accuracy", "accuracy"),
        "mbppplus": _metric("accuracy", "accuracy"),
        "olympiadbench": _metric("accuracy", "accuracy"),
        "gsm8k-0shot": _metric("exact_match,flexible-extract", "exact_match,flexible-extract"),
        "piqa": _metric("acc_norm,none", "acc_norm,none"),
        "winogrande": _metric("acc,none", "acc,none"),
        "boolq": _metric("acc,none", "acc,none"),
        "truthfulqa": _metric("acc,none", "acc,none"),
        "triviaqa": _metric("exact_match,strict_answer", "exact_match,strict_answer"),
        "aime24": _metric("accuracy", "accuracy"),
        "mmlu-pro": _metric("accuracy", "accuracy"),
        "gpqa-diamond": _metric("accuracy", "accuracy"),
        "cruxeval": _metric("accuracy", "accuracy", "direction=output/accuracy"),
        "financebench": _metric("accuracy", "accuracy"),
        "ifbench": _metric("strict_prompt_accuracy", "strict_prompt_accuracy"),
        "mrcr": _metric("accuracy", "accuracy"),
        "mrcr-32k": _metric("accuracy", "accuracy"),
        "mrcr-65k": _metric("accuracy", "accuracy"),
        "nupa": _metric("exact_match", "exact_match"),
        "swebench-verified": _metric("reward", "reward"),
        "ot-tblite-recovery": _metric("reward", "reward"),
        "tb2-recovery": _metric("reward", "reward"),
        "ds-1000-local": _metric("reward", "reward"),
        "bfclparity-pi": _metric("reward", "reward"),
        "bixbench-pi": _metric("reward", "reward"),
    }
)
