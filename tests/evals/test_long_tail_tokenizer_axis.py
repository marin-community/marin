# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.evals.long_tail_tokenizer_axis import (
    TokenizerAxisBackend,
    TokenizerAxisSliceMetrics,
    TokenizerAxisSliceSample,
    TokenizerAxisTokenizerSpec,
    compute_tokenizer_axis_slice_metrics,
    compute_whitespace_inflation_correlations,
    default_tokenizer_axis_slices,
    default_tokenizer_axis_step,
    summarize_slice_byte_stats,
)


def test_compute_tokenizer_axis_slice_metrics_tracks_tokens_and_byte_shares():
    runnable_slice = next(slice_spec for slice_spec in default_tokenizer_axis_slices() if slice_spec.is_runnable)
    sample = TokenizerAxisSliceSample(
        slice_spec=runnable_slice,
        texts=("A  b\n", "x,yé"),
        docs_scanned=2,
        docs_used=2,
        docs_skipped_missing_text_key=0,
        docs_skipped_non_string=0,
        byte_stats=summarize_slice_byte_stats(("A  b\n", "x,yé")),
    )
    tokenizer_spec = TokenizerAxisTokenizerSpec(
        name="byte_level_fake",
        tokenizer_id="fake/byte-level",
        backend=TokenizerAxisBackend.LEVANTER_HF,
    )

    row = compute_tokenizer_axis_slice_metrics(sample, tokenizer_spec=tokenizer_spec, encode=lambda text: text.encode())

    assert row.token_count == 10
    assert row.byte_count == 10
    assert row.tokens_per_byte == pytest.approx(1.0)
    assert row.bytes_per_token == pytest.approx(1.0)
    assert row.whitespace_byte_share == pytest.approx(0.3)
    assert row.punctuation_byte_share == pytest.approx(0.1)
    assert row.non_ascii_byte_share == pytest.approx(0.2)


def test_compute_whitespace_inflation_correlations_uses_tpb_delta_against_baseline():
    baseline = "llama3_1_8b"
    rows = [
        TokenizerAxisSliceMetrics(
            tokenizer_name=baseline,
            tokenizer_id="meta-llama/Llama-3.1-8B",
            tokenizer_backend=TokenizerAxisBackend.LEVANTER_HF,
            slice_registry_key="slice/a",
            family=next(iter(default_tokenizer_axis_slices())).family,
            docs_used=2,
            docs_scanned=2,
            docs_skipped_missing_text_key=0,
            docs_skipped_non_string=0,
            token_count=20,
            byte_count=100,
            tokens_per_byte=0.2,
            bytes_per_token=5.0,
            whitespace_byte_share=0.1,
            punctuation_byte_share=0.05,
            non_ascii_byte_share=0.01,
        ),
        TokenizerAxisSliceMetrics(
            tokenizer_name=baseline,
            tokenizer_id="meta-llama/Llama-3.1-8B",
            tokenizer_backend=TokenizerAxisBackend.LEVANTER_HF,
            slice_registry_key="slice/b",
            family=next(iter(default_tokenizer_axis_slices())).family,
            docs_used=2,
            docs_scanned=2,
            docs_skipped_missing_text_key=0,
            docs_skipped_non_string=0,
            token_count=20,
            byte_count=100,
            tokens_per_byte=0.2,
            bytes_per_token=5.0,
            whitespace_byte_share=0.4,
            punctuation_byte_share=0.05,
            non_ascii_byte_share=0.01,
        ),
        TokenizerAxisSliceMetrics(
            tokenizer_name="qwen3_8b",
            tokenizer_id="Qwen/Qwen3-8B",
            tokenizer_backend=TokenizerAxisBackend.LEVANTER_HF,
            slice_registry_key="slice/a",
            family=next(iter(default_tokenizer_axis_slices())).family,
            docs_used=2,
            docs_scanned=2,
            docs_skipped_missing_text_key=0,
            docs_skipped_non_string=0,
            token_count=30,
            byte_count=100,
            tokens_per_byte=0.3,
            bytes_per_token=3.333333,
            whitespace_byte_share=0.1,
            punctuation_byte_share=0.05,
            non_ascii_byte_share=0.01,
        ),
        TokenizerAxisSliceMetrics(
            tokenizer_name="qwen3_8b",
            tokenizer_id="Qwen/Qwen3-8B",
            tokenizer_backend=TokenizerAxisBackend.LEVANTER_HF,
            slice_registry_key="slice/b",
            family=next(iter(default_tokenizer_axis_slices())).family,
            docs_used=2,
            docs_scanned=2,
            docs_skipped_missing_text_key=0,
            docs_skipped_non_string=0,
            token_count=60,
            byte_count=100,
            tokens_per_byte=0.6,
            bytes_per_token=1.666667,
            whitespace_byte_share=0.4,
            punctuation_byte_share=0.05,
            non_ascii_byte_share=0.01,
        ),
    ]

    correlations = compute_whitespace_inflation_correlations(metrics=rows, baseline_tokenizer_name=baseline)

    assert len(correlations) == 1
    assert correlations[0].tokenizer_name == "qwen3_8b"
    assert correlations[0].points == 2
    assert correlations[0].mean_tokens_per_byte_delta == pytest.approx(0.25)
    assert correlations[0].whitespace_vs_tpb_delta_pearson == pytest.approx(1.0)


def test_default_tokenizer_axis_registry_marks_planned_symbolic_slices_non_runnable():
    slices = default_tokenizer_axis_slices(include_planned_raw_slices=True)

    planned = [slice_spec for slice_spec in slices if not slice_spec.is_runnable]
    runnable = [slice_spec for slice_spec in slices if slice_spec.is_runnable]

    assert runnable
    assert planned
    assert any(slice_spec.registry_key == "long_tail_ppl/game_music/lichess_pgn" for slice_spec in planned)
    assert all(slice_spec.hf_dataset is None for slice_spec in planned)


def test_default_tokenizer_axis_step_exposes_config_for_dry_run():
    step = default_tokenizer_axis_step(
        name="unit-test",
        max_docs_per_slice=8,
        max_doc_bytes=1024,
        include_planned_raw_slices=True,
    )
    config = step.config

    assert step.name == "analysis/tokenizer_axis/unit-test"
    assert config.max_docs_per_slice == 8
    assert config.max_doc_bytes == 1024
    assert config.include_planned_raw_slices is True
