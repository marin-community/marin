# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one cached-score mega bundle for all currently runnable raw eval sources.

This is the practical "everything so far" launcher for the current PR stack:

- base raw defaults (Paloma + Uncheatable + default capability slices)
- FineWeb2 multilingual
- currently self-contained issue #5005 long-tail families

It uses the cached single-model scoring path and then derives the default
pairwise gap reports from those score artifacts.
"""

from fray import ResourceConfig

from experiments.evals.perplexity_gap_mega_bundle import mega_available_raw_validation_sets
from experiments.evals.perplexity_gap_registry import (
    PerplexityGapBundle,
    build_registered_perplexity_gap_coverage_plan,
)
from marin.execution.executor import executor_main

RESOURCE_CONFIG = ResourceConfig.with_tpu("v5p-8", regions=["us-central1"])

MEGA_BUNDLE = PerplexityGapBundle(
    key="mega_available_raw",
    description="All currently runnable raw eval sources in the mega branch.",
    datasets_factory=mega_available_raw_validation_sets,
    max_docs_per_dataset=128,
)

PLAN = build_registered_perplexity_gap_coverage_plan(
    resource_config=RESOURCE_CONFIG,
    bundles=(MEGA_BUNDLE,),
)
STEPS = [*PLAN.score_steps.values(), *PLAN.pairwise_gap_steps.values()]


if __name__ == "__main__":
    executor_main(
        STEPS,
        description=(
            "Run the mega cached-score matrix over all currently runnable raw eval "
            "sources and derive the default pairwise gap reports."
        ),
    )
