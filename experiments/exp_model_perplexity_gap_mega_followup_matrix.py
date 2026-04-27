# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the post-mega follow-up raw eval slices added in #5053, #5059, and #5061.

This is the cheap incremental cached-score rerun after the first successful
mega matrix and the separate Common Crawl / game-music delta run. It only
scores the newly added LM-eval bridge, structured-text, and package-metadata
slices so the report/dashboard can be refreshed without repeating the full mega
bundle.
"""

from fray import ResourceConfig

from experiments.evals.perplexity_gap_mega_bundle import mega_followup_raw_validation_sets
from experiments.evals.perplexity_gap_registry import (
    PerplexityGapBundle,
    build_registered_perplexity_gap_coverage_plan,
)
from marin.execution.executor import executor_main

RESOURCE_CONFIG = ResourceConfig.with_tpu("v5p-8", regions=["us-central1"])

MEGA_FOLLOWUP_BUNDLE = PerplexityGapBundle(
    key="mega_followup_raw",
    description="Incremental post-mega raw slices for #5053, #5059, and #5061.",
    datasets_factory=mega_followup_raw_validation_sets,
    max_docs_per_dataset=128,
)

PLAN = build_registered_perplexity_gap_coverage_plan(
    resource_config=RESOURCE_CONFIG,
    bundles=(MEGA_FOLLOWUP_BUNDLE,),
)
STEPS = [*PLAN.score_steps.values(), *PLAN.pairwise_gap_steps.values()]


if __name__ == "__main__":
    executor_main(
        STEPS,
        description=(
            "Run the incremental cached-score matrix over the #5053/#5059/#5061 "
            "follow-up raw eval slices and derive the default pairwise gaps."
        ),
    )
