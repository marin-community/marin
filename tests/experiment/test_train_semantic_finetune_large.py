# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

import train_semantic_finetune_large as finetune  # noqa: E402

from experiments.datakit.cluster.quality.fast_transformer.model import (  # noqa: E402
    FastEmbeddingTransformer,
    FastTransformerConfig,
)


def small_model(seed: int) -> FastEmbeddingTransformer:
    config = FastTransformerConfig(
        vocab_size=16,
        max_tokens=8,
        pool_window=2,
        embed_dim=8,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        mlp_ratio=2,
        dropout=0.0,
    )
    return FastEmbeddingTransformer(config, output_dim=4, key=jax.random.PRNGKey(seed))


def validation(rank_fraction: float) -> dict:
    return {"geometry": {"effective_rank_fraction": rank_fraction}}


@pytest.mark.parametrize(
    ("candidate_rank", "expected"),
    [(0.3, True), (0.2999, False)],
)
def test_rank_preservation_decision_applies_base_rank_boundary(
    candidate_rank: float,
    expected: bool,
) -> None:
    decision = finetune.rank_preservation_decision(validation(0.4), validation(candidate_rank))

    assert decision["rank_preserved"] is expected


def test_best_passing_model_mix_rejects_larger_gain_that_loses_rank() -> None:
    rank_safe = finetune.ModelMixCandidate(0.5, {}, {"passed": True, "semantic_mean_delta": 0.08})
    rank_loss = finetune.ModelMixCandidate(1.0, {}, {"passed": False, "semantic_mean_delta": 0.12})

    selected = finetune.best_passing_model_mix([rank_safe, rank_loss])

    assert selected.alpha == 0.5


def test_best_passing_model_mix_rejects_all_failed_candidates() -> None:
    candidates = [finetune.ModelMixCandidate(1.0, {}, {"passed": False, "semantic_mean_delta": 0.12})]

    with pytest.raises(ValueError, match="No model mix passed"):
        finetune.best_passing_model_mix(candidates)


def test_interpolate_models_keeps_exact_endpoint_behavior() -> None:
    base = small_model(0)
    fine_tuned = small_model(1)
    ids = jnp.asarray([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])

    base_endpoint = finetune.interpolate_models(base, fine_tuned, 0.0)
    tuned_endpoint = finetune.interpolate_models(base, fine_tuned, 1.0)
    midpoint = finetune.interpolate_models(base, fine_tuned, 0.5)

    np.testing.assert_allclose(np.asarray(base_endpoint(ids)), np.asarray(base(ids)), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(tuned_endpoint(ids)), np.asarray(fine_tuned(ids)), rtol=0, atol=0)
    assert np.isfinite(np.asarray(midpoint(ids))).all()
