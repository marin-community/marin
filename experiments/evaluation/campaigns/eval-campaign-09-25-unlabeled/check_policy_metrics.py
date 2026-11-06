#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit publication keys against curated protocols and Harbor metadata."""

import runpy
from pathlib import Path

import yaml
from taskcompendium.importers.evalchemy.code import CRUX_DIRECTIONS

from experiments.evaluation.curated import CuratedDefinition
from experiments.post_training.task_curation.sources import evaluation_sources

CAMPAIGN = Path(__file__).resolve().parent
ROOT = CAMPAIGN.parents[3]


def main() -> None:
    contract = runpy.run_path(str(CAMPAIGN / "policy_metric_map.py"))
    expected = contract["POLICY_METRICS_2026_10_16"]
    mechanism = contract["PolicyMechanism"]
    curated = {path.stem: path for path in (CAMPAIGN / "evalchemy-configs").glob("*.yaml")}
    harbor = {path.stem: path for path in (CAMPAIGN / "harbor-configs").glob("*.yaml")}
    if set(expected) != curated.keys() | harbor.keys():
        raise ValueError("Campaign configurations differ from the publication metric contract")
    catalog = evaluation_sources()
    harbor_types = runpy.run_path(str(ROOT / "vendor/harbor/src/harbor/models/benchmark.py"))
    reward = harbor_types["reward_benchmark_metadata"]("policy", n_benchmark=1, n_attempted=1)
    for name, metric in expected.items():
        if metric.mechanism is mechanism.CURATED:
            definition = CuratedDefinition.from_file(curated[name])
            for source in (*definition.sources, *definition.chat_sources):
                dataset = catalog[source].config
                assert dataset is not None
                protocol = dataset.protocol
                if (protocol.benchmark, protocol.primary_metric, protocol.primary_metric) != (
                    metric.task,
                    metric.evaluator_metric,
                    metric.record_metric,
                ):
                    raise ValueError(f"{name}: curated protocol differs from the publication contract")
                available = {item.name for item in protocol.metrics}
                if name == "cruxeval":
                    available.update(
                        f"direction={direction}/{item.name}"
                        for direction in CRUX_DIRECTIONS
                        for item in protocol.metrics
                    )
                if metric.published_metric not in available:
                    raise ValueError(f"{name}: undeclared publication metric {metric.published_metric}")
                if protocol.requires_judge and definition.judge is None:
                    raise ValueError(f"{name}: missing judge deployment identity")
            continue
        config = yaml.safe_load(harbor[name].read_text())
        metadata = (
            harbor_types["sotopia_benchmark_metadata"](reward)
            if any(agent["name"] == "sotopia" for agent in config["agents"])
            else reward
        )
        primary = next(item for item in metadata.metrics if item.name == metadata.primary_metric)
        if (primary.source_name, metadata.primary_metric) != (metric.evaluator_metric, metric.record_metric):
            raise ValueError(f"{name}: Harbor primary differs from the publication contract")
        if metric.published_metric not in {item.name for item in metadata.metrics}:
            raise ValueError(f"{name}: publication metric is absent from Harbor metadata")
    print(f"validated {len(expected)} publication metric contracts")


if __name__ == "__main__":
    main()
