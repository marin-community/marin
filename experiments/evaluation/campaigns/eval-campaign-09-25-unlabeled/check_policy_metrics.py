#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit publication keys against curated protocols."""

import runpy
from pathlib import Path

from evalrecords.historical_protocols import HISTORICAL_HARBOR_BENCHMARKS, HISTORICAL_PROTOCOLS
from taskcompendium.importers.evalchemy.code import CRUX_DIRECTIONS

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.evaluation.curated import load_evaluations
from experiments.evaluation.policy_finalization import POLICY_JUDGED_PROTOCOLS

CAMPAIGN = Path(__file__).resolve().parent


def historical_benchmark_matches(name: str, benchmark: str) -> bool:
    """Whether ``benchmark`` continues the Evaldash history of the campaign's evaluation ``name``.

    A nonagentic name has one historical protocol; an agentic one ran on Harbor, whose records name a
    dataset that Evaldash files under a curated benchmark.
    """
    if name in HISTORICAL_PROTOCOLS:
        return HISTORICAL_PROTOCOLS[name].benchmark == benchmark
    return benchmark in HISTORICAL_HARBOR_BENCHMARKS.values()


def main() -> None:
    contract = runpy.run_path(str(CAMPAIGN / "policy_metric_map.py"))
    expected = contract["POLICY_METRICS_2026_10_16"]
    curated = {selection.name: selection for selection in load_evaluations(CAMPAIGN / "evaluations.yaml")}
    if set(expected) != curated.keys():
        raise ValueError("Campaign configurations differ from the publication metric contract")
    catalog = evaluation_benchmarks()
    for name, metric in expected.items():
        definition = curated[name]
        for source in catalog[definition.benchmark].resolve(definition.protocol, definition.num_fewshot):
            dataset = source.config
            assert dataset is not None
            protocol = dataset.protocol
            if not historical_benchmark_matches(name, protocol.benchmark):
                raise ValueError(f"{name}: {protocol.benchmark} does not continue the evaluation's history")
            if (protocol.primary_metric, protocol.primary_metric) != (metric.evaluator_metric, metric.record_metric):
                raise ValueError(f"{name}: curated metric differs from the publication contract")
            available = {item.name for item in protocol.metrics}
            if name == "cruxeval":
                available.update(
                    f"direction={direction}/{item.name}" for direction in CRUX_DIRECTIONS for item in protocol.metrics
                )
            if metric.published_metric not in available:
                raise ValueError(f"{name}: undeclared publication metric {metric.published_metric}")
            policy_judged = (protocol.benchmark, dataset.protocol_name) in POLICY_JUDGED_PROTOCOLS
            if protocol.requires_judge and definition.run.judge is None and not policy_judged:
                raise ValueError(f"{name}: missing judge deployment identity")
    print(f"validated {len(expected)} publication metric contracts")


if __name__ == "__main__":
    main()
