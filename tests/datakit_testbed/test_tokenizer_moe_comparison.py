# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest

from experiments.datakit_testbed.tokenizer_moe_comparison import (
    DIGITS_FAMILIES,
    GRUG_RUNGS,
    ComparisonCell,
    TokenizerMoeComparisonConfig,
    comparison_cells,
    grug_model,
    validate_same_region,
)


def _config(**overrides) -> TokenizerMoeComparisonConfig:
    config = TokenizerMoeComparisonConfig(
        cache_prefix="gs://marin-eu-west4/data/datakit/tokenized/tokenizer-sweep-20260526",
        output_prefix="gs://marin-eu-west4",
        tokenizer_run_id="tokenizer-sweep-20260526",
        region="europe-west4",
        tpu_type="v6e-8",
        version="test",
    )
    return dataclasses.replace(config, **overrides)


def test_digits_sweep_builds_all_matched_comparison_cells() -> None:
    cells = comparison_cells(_config())

    assert len(cells) == 8
    assert {(cell.family, cell.vocab_size, cell.rung.hidden_dim) for cell in cells} == {
        (family, vocab_size, hidden_dim)
        for family in DIGITS_FAMILIES
        for vocab_size in (8_192, 32_768)
        for hidden_dim in (512, 768)
    }


@pytest.mark.parametrize(
    ("hidden_dim", "expected_layers", "expected_intermediate_dim"),
    [(512, 6, 256), (768, 8, 384)],
)
def test_tokenizer_families_share_canonical_grug_architecture(
    hidden_dim: int,
    expected_layers: int,
    expected_intermediate_dim: int,
) -> None:
    models = [
        grug_model(ComparisonCell(family=family, vocab_size=vocab_size, rung=GRUG_RUNGS[hidden_dim]))
        for family in DIGITS_FAMILIES
        for vocab_size in (8_192, 32_768)
    ]

    for model in models:
        assert model.hidden_dim == hidden_dim
        assert model.num_layers == expected_layers
        assert model.intermediate_dim == expected_intermediate_dim
        assert model.num_experts == 64
        assert model.num_experts_per_token == 4

    without_vocab = [dataclasses.replace(model, vocab_size=8_192) for model in models]
    assert all(model == without_vocab[0] for model in without_vocab)


def test_region_validation_rejects_cross_region_compute_and_outputs() -> None:
    with pytest.raises(ValueError, match="TPU region"):
        validate_same_region(
            "gs://marin-eu-west4/data/cache",
            "gs://marin-eu-west4/checkpoints",
            "us-east5",
        )

    with pytest.raises(ValueError, match="must match"):
        validate_same_region(
            "gs://marin-eu-west4/data/cache",
            "gs://marin-us-east5/checkpoints",
            "europe-west4",
        )
