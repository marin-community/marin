# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest

from experiments.datakit_testbed.tokenizer_moe_comparison import (
    GRUG_RUNGS,
    TOKENIZER_FAMILIES,
    ComparisonCell,
    ComparisonResult,
    TokenizerMoeComparisonConfig,
    comparison_cells,
    grug_model,
    render_results_table,
    validate_comparison_results,
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


def test_default_sweep_builds_complete_matched_comparison_matrix() -> None:
    cells = comparison_cells(_config())

    assert len(cells) == 16
    assert {(cell.family, cell.vocab_size, cell.rung.hidden_dim) for cell in cells} == {
        (family, vocab_size, hidden_dim)
        for family in TOKENIZER_FAMILIES
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
        for family in TOKENIZER_FAMILIES
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


def _results(cells: list[ComparisonCell]) -> list[ComparisonResult]:
    return [
        ComparisonResult(
            family=cell.family,
            vocab_size=cell.vocab_size,
            hidden_dim=cell.rung.hidden_dim,
            final_step=cell.rung.steps,
            parameter_count=100_000_000 + cell.vocab_size * cell.rung.hidden_dim,
            byte_weighted_bpb=1.25,
            byte_weighted_macro_bpb=1.3,
            wandb_url=f"https://wandb.example/{cell.name}",
        )
        for cell in cells
    ]


def test_result_table_contains_every_validated_cell() -> None:
    cells = comparison_cells(_config())
    results = _results(cells)

    validate_comparison_results(cells, results)
    table = render_results_table(results)

    assert table.count("https://wandb.example/") == 16
    assert "| gpt-oss-place-digits | 32k | d768 | 1292 |" in table
    assert "globally byte-weighted" in table


def test_result_validation_rejects_parameter_mismatch() -> None:
    cells = comparison_cells(_config())
    results = _results(cells)
    results[0] = dataclasses.replace(results[0], parameter_count=results[0].parameter_count + 1)

    with pytest.raises(ValueError, match="mismatched parameter counts"):
        validate_comparison_results(cells, results)
