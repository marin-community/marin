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
    evaluator_metric: str
    record_metric: str
    published_metric: str


def _curated(evaluator: str, record: str, published: str | None = None) -> PolicyMetric:
    return PolicyMetric(PolicyMechanism.CURATED, evaluator, record, published or evaluator)


def _harbor(metric: str = "reward") -> PolicyMetric:
    return PolicyMetric(PolicyMechanism.HARBOR, metric, metric, metric)


POLICY_METRICS_2026_10_16: Mapping[str, PolicyMetric] = MappingProxyType(
    {
        "math500": _curated("accuracy", "accuracy"),
        "humanevalplus": _curated("accuracy", "accuracy"),
        "mbppplus": _curated("accuracy", "accuracy"),
        "olympiadbench": _curated("accuracy", "accuracy"),
        "gsm8k-0shot": _curated("exact_match,strict-match", "exact_match,strict-match"),
        "piqa": _curated("acc_norm,none", "acc_norm,none"),
        "winogrande": _curated("acc,none", "acc,none"),
        "boolq": _curated("acc,none", "acc,none"),
        "truthfulqa": _curated("acc,none", "acc,none"),
        "triviaqa": _curated("exact_match,strict_answer", "exact_match,strict_answer"),
        "aime24": _curated("accuracy", "accuracy"),
        "mmlu-pro": _curated("accuracy", "accuracy"),
        "gpqa-diamond": _curated("accuracy", "accuracy"),
        "cruxeval": _curated("accuracy", "accuracy", "direction=output/accuracy"),
        "financebench": _curated("accuracy", "accuracy"),
        "ifbench": _curated("strict_prompt_accuracy", "strict_prompt_accuracy"),
        "mrcr": _curated("accuracy", "accuracy"),
        "mrcr-32k": _curated("accuracy", "accuracy"),
        "mrcr-65k": _curated("accuracy", "accuracy"),
        "nupa": _curated("exact_match", "exact_match"),
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
