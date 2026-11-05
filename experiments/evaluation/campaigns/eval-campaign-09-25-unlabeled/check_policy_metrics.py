#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit campaign primary metrics against immutable curated protocol declarations."""

from pathlib import Path

from experiments.evaluation.curated import CuratedDefinition
from experiments.post_training.task_curation.sources import evaluation_sources

CAMPAIGN = Path(__file__).resolve().parent
PRIMARY_METRICS = {
    "aime24": "accuracy",
    "boolq": "acc,none",
    "cruxeval": "accuracy",
    "financebench": "accuracy",
    "gpqa-diamond": "accuracy",
    "gsm8k-0shot": "exact_match,strict-match",
    "humanevalplus": "accuracy",
    "ifbench": "strict_prompt_accuracy",
    "math500": "accuracy",
    "mbppplus": "accuracy",
    "mmlu-pro": "accuracy",
    "mrcr": "accuracy",
    "mrcr-32k": "accuracy",
    "mrcr-65k": "accuracy",
    "nupa": "exact_match",
    "olympiadbench": "accuracy",
    "piqa": "acc_norm,none",
    "triviaqa": "exact_match,strict_answer",
    "truthfulqa": "acc,none",
    "winogrande": "acc,none",
}


def main() -> None:
    catalog = evaluation_sources()
    configurations = sorted((CAMPAIGN / "curated-configs").glob("*.yaml"))
    if {path.stem for path in configurations} != PRIMARY_METRICS.keys():
        raise ValueError("Campaign configurations differ from the curated metric contract")
    for path in configurations:
        definition = CuratedDefinition.from_file(path)
        for name in (*definition.sources, *definition.chat_sources):
            source = catalog[name].config
            assert source is not None
            protocol = source.protocol
            if protocol.primary_metric != PRIMARY_METRICS[path.stem]:
                raise ValueError(f"{path.stem}: primary metric differs from its declared campaign contract")
            if protocol.requires_judge and definition.judge is None:
                raise ValueError(f"{path.stem}: missing judge deployment identity")
    print(f"validated {len(configurations)} curated campaign metric contracts")


if __name__ == "__main__":
    main()
