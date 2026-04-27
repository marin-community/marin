# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fray import ResourceConfig

from experiments.evals.perplexity_gap_registry import build_registered_perplexity_gap_coverage_plan


def test_registered_coverage_plan_shares_score_steps_across_pairwise_gaps():
    plan = build_registered_perplexity_gap_coverage_plan(
        resource_config=ResourceConfig.with_tpu("v5p-8", regions=["us-central1"])
    )

    marin_score = plan.score_steps[("base_raw", "marin_8b")]
    llama_gap = plan.pairwise_gap_steps[("base_raw", "marin_8b", "llama3_1_8b")]
    qwen_gap = plan.pairwise_gap_steps[("base_raw", "marin_8b", "qwen3_8b")]

    assert llama_gap.config.model_a_scores_path.step is marin_score
    assert qwen_gap.config.model_a_scores_path.step is marin_score


def test_registered_coverage_plan_uses_bundle_scoped_step_names():
    plan = build_registered_perplexity_gap_coverage_plan(
        resource_config=ResourceConfig.with_tpu("v5p-8", regions=["us-central1"])
    )

    score_step = plan.score_steps[("multilingual_raw", "qwen3_8b")]
    gap_step = plan.pairwise_gap_steps[("multilingual_raw", "marin_8b", "qwen3_8b")]

    assert score_step.name == "analysis/model_perplexity_scores/multilingual_raw/qwen3_8b"
    assert gap_step.name == "analysis/perplexity_gap/multilingual_raw/marin_8b-vs-qwen3_8b"
    assert gap_step.config.model_b_scores_path.step is score_step


def test_registered_coverage_plan_uses_smaller_batches_for_32b_models():
    plan = build_registered_perplexity_gap_coverage_plan(
        resource_config=ResourceConfig.with_tpu("v5p-8", regions=["us-central1"])
    )

    marin_8b_step = plan.score_steps[("base_raw", "marin_8b")]
    marin_32b_step = plan.score_steps[("base_raw", "marin_32b")]

    assert marin_8b_step.config.per_device_batch_size == 4
    assert marin_32b_step.config.per_device_batch_size == 1
