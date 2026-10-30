#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the publication metric map against the pinned evaluator packages.

Run in the vendored Evalchemy environment so lm-eval's pinned task registry is
available. The unit workflow runs this check on pull requests and every day.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import NamedTuple

import yaml
from eval.contracts.benchmark_metadata import SourceMetric, resolve_metric_metadata
from eval.sources import SOURCES
from eval.task import TaskManager as CustomTaskManager
from lm_eval.tasks import TaskManager as LMEvalTaskManager
from policy_metric_map import POLICY_METRICS_2026_10_16, PolicyMechanism, PolicyMetric

CAMPAIGN = Path(__file__).resolve().parent
ROOT = CAMPAIGN.parents[3]
EVALCHEMY_CONFIGS = CAMPAIGN / "evalchemy-configs"
HARBOR_CONFIGS = CAMPAIGN / "harbor-configs"
MRCR_CONTEXT_VARIANTS = frozenset({"mrcr-32k", "mrcr-65k"})


class ResolvedMetric(NamedTuple):
    evaluator_metric: str
    record_metric: str
    available_source_names: frozenset[str]


def _campaign_config_names() -> set[str]:
    return {
        *(path.stem for path in EVALCHEMY_CONFIGS.glob("*.yaml") if path.stem not in MRCR_CONTEXT_VARIANTS),
        *(path.stem for path in HARBOR_CONFIGS.glob("*.yaml")),
    }


def _evalchemy_metric(
    metric: PolicyMetric, custom: CustomTaskManager, lm_eval: LMEvalTaskManager, selected_metric: str | None
) -> ResolvedMetric:
    assert metric.task is not None
    if metric.task in SOURCES:
        benchmark = custom.get_benchmark(metric.task)
        if benchmark is None:
            raise ValueError(f"{metric.task}: custom benchmark did not load")
        source_metrics = tuple(SourceMetric(name) for name in benchmark.benchmark_metrics())
        primary_metric = benchmark.benchmark_primary_metric()
        name_overrides = benchmark.METRIC_NAME_OVERRIDES
        kind_overrides = benchmark.METRIC_KIND_OVERRIDES
    else:
        entry = lm_eval.task_index.get(metric.task)
        if entry is None:
            raise ValueError(f"{metric.task}: no pinned lm-eval task definition")
        config = entry.cfg
        source_metrics = tuple(
            SourceMetric(str(item["metric"]), bool(item.get("higher_is_better", True)))
            for item in config.get("metric_list", ())
        )
        if not source_metrics:
            raise ValueError(f"{metric.task}: no metric list in pinned lm-eval task definition")
        metadata = config.get("metadata") or {}
        primary_metric = metadata.get("primary_metric")
        name_overrides = metadata.get("metric_name_overrides")
        kind_overrides = metadata.get("metric_kind_overrides")
    metrics, primary = resolve_metric_metadata(
        source_metrics,
        primary_metric=selected_metric or primary_metric,
        name_overrides=name_overrides,
        kind_overrides=kind_overrides,
    )
    source = next(item.source_name for item in metrics if item.name == primary)
    return ResolvedMetric(source, primary, frozenset(item.source_name for item in metrics))


def main() -> None:
    expected = POLICY_METRICS_2026_10_16
    config_names = _campaign_config_names()
    if set(expected) != config_names:
        raise ValueError(
            f"policy metric map differs from campaign configs: missing={config_names - set(expected)}, "
            f"extra={set(expected) - config_names}"
        )

    custom_names = [metric.task for metric in expected.values() if metric.task in SOURCES]
    custom = CustomTaskManager(task_list=custom_names)
    if custom.load_failures:
        raise ValueError(f"custom benchmark load failures: {custom.load_failures}")
    lm_eval = LMEvalTaskManager(include_path=[str(ROOT / "vendor/evalchemy/eval/lm_eval_tasks")])
    harbor_metadata = runpy.run_path(ROOT / "vendor/harbor/src/harbor/models/benchmark.py")
    harbor = harbor_metadata["reward_benchmark_metadata"]("policy-example", n_benchmark=1, n_attempted=1)

    for name, metric in expected.items():
        if metric.mechanism == PolicyMechanism.EVALCHEMY:
            config = yaml.safe_load((EVALCHEMY_CONFIGS / f"{name}.yaml").read_text())
            if config["tasks"] != [metric.task]:
                raise ValueError(f"{name}: configured task differs from metric map")
            selected_metric = config.get("task_options", {}).get(metric.task, {}).get("primary_metric")
            resolved = _evalchemy_metric(metric, custom, lm_eval, selected_metric)
            actual = resolved.evaluator_metric, resolved.record_metric
            if name != "cruxeval" and metric.published_metric not in resolved.available_source_names:
                raise ValueError(f"{name}: publication metric {metric.published_metric!r} is not declared")
        else:
            metadata = harbor_metadata["sotopia_benchmark_metadata"](harbor) if name == "sotopia-hard" else harbor
            primary = next(item for item in metadata.metrics if item.name == metadata.primary_metric)
            actual = primary.source_name, metadata.primary_metric
            if metric.published_metric not in {item.name for item in metadata.metrics}:
                raise ValueError(f"{name}: publication metric is missing from Harbor metadata")
        wanted = metric.evaluator_metric, metric.record_metric
        if actual != wanted:
            raise ValueError(f"{name}: evaluator/record metrics {actual} differ from policy map {wanted}")

    crux = custom.get_benchmark("CruxEval")
    assert crux is not None
    crux_publication = crux.REDUCER.groups["direction"].format(group="output", metric=crux.REDUCER.accuracy)
    if expected["cruxeval"].published_metric != crux_publication:
        raise ValueError("CRUXEval publication metric differs from the output-direction reducer")
    print(f"validated {len(expected)} policy metric contracts")


if __name__ == "__main__":
    main()
