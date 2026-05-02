# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fray.types import ResourceConfig

from experiments.evals.perplexity_gap_registry import build_registered_perplexity_gap_coverage_plan


def test_registered_coverage_plan_includes_bio_chem_bundle() -> None:
    plan = build_registered_perplexity_gap_coverage_plan(
        resource_config=ResourceConfig.with_tpu("v5p-8", regions=["us-central1"])
    )

    marin_score = plan.score_steps[("bio_chem", "marin_8b")]
    qwen_gap = plan.pairwise_gap_steps[("bio_chem", "marin_8b", "qwen3_8b")]

    assert "bio_chem/refseq/refseq_viral_fasta" in marin_score.config.datasets
    assert "bio_chem/chembl/chembl_sdf" in marin_score.config.datasets
    assert qwen_gap.config.model_a_scores_path.step is marin_score
