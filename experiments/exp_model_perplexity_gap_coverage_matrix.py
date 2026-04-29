# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the registered perplexity-score coverage matrix for #5005.

This expands the canonical bundle/model registry into:

- one cached single-model score step per model/bundle
- the default pairwise gap reports derived from those cached scores
"""

from fray.v2.types import ResourceConfig

from experiments.evals.perplexity_gap_registry import build_registered_perplexity_gap_coverage_plan
from marin.execution.executor import executor_main

RESOURCE_CONFIG = ResourceConfig.with_tpu("v5p-8", regions=["us-central1"])
PLAN = build_registered_perplexity_gap_coverage_plan(resource_config=RESOURCE_CONFIG)
STEPS = [*PLAN.score_steps.values(), *PLAN.pairwise_gap_steps.values()]


if __name__ == "__main__":
    executor_main(
        STEPS,
        description="Run the registered model-perplexity coverage matrix and derived gap reports for #5005.",
    )
