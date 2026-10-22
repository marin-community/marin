#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the publication metric map against the pinned evaluator packages.

Run in the vendored Evalchemy environment so lm-eval's pinned task registry is
available. The unit workflow runs this check on pull requests and every day.
"""

from __future__ import annotations

import ast
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


def _evalchemy_metric(metric: PolicyMetric, custom: CustomTaskManager, lm_eval: LMEvalTaskManager) -> ResolvedMetric:
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
        primary_metric=primary_metric,
        name_overrides=name_overrides,
        kind_overrides=kind_overrides,
    )
    source = next(item.source_name for item in metrics if item.name == primary)
    return ResolvedMetric(source, primary, frozenset(item.source_name for item in metrics))


def _sotopia_score_fields() -> set[str]:
    path = CAMPAIGN / "analysis/sotopia_length_control/rank.py"
    tree = ast.parse(path.read_text())
    model_score = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelScore")
    return {
        node.target.id
        for node in model_score.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }


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
    harbor_factory = runpy.run_path(ROOT / "vendor/harbor/src/harbor/models/benchmark.py")["reward_benchmark_metadata"]
    harbor = harbor_factory("policy-example", n_benchmark=1, n_attempted=1)

    for name, metric in expected.items():
        if metric.mechanism == PolicyMechanism.EVALCHEMY:
            config = yaml.safe_load((EVALCHEMY_CONFIGS / f"{name}.yaml").read_text())
            if config["tasks"] != [metric.task]:
                raise ValueError(f"{name}: configured task differs from metric map")
            resolved = _evalchemy_metric(metric, custom, lm_eval)
            actual = resolved.evaluator_metric, resolved.record_metric
            if name != "cruxeval" and metric.published_metric not in resolved.available_source_names:
                raise ValueError(f"{name}: publication metric {metric.published_metric!r} is not declared")
        else:
            actual = harbor.metrics[0].source_name, harbor.primary_metric
        wanted = metric.evaluator_metric, metric.record_metric
        if actual != wanted:
            raise ValueError(f"{name}: evaluator/record metrics {actual} differ from policy map {wanted}")

    crux = custom.get_benchmark("CruxEval")
    assert crux is not None
    crux_publication = crux.REDUCER.groups["direction"].format(group="output", metric=crux.REDUCER.accuracy)
    if expected["cruxeval"].published_metric != crux_publication:
        raise ValueError("CRUXEval publication metric differs from the output-direction reducer")
    if expected["sotopia-hard"].published_metric not in _sotopia_score_fields():
        raise ValueError("SOTOPIA publication metric is missing from its cohort score")
    print(f"validated {len(expected)} policy metric contracts")


if __name__ == "__main__":
    main()
