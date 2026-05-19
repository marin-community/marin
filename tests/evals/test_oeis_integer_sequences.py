# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fray.types import ResourceConfig
from marin.datakit.download.oeis import _format_record, _iter_name_rows, _iter_sequence_rows

from experiments.evals.oeis_integer_sequences import (
    OEIS_INTEGER_SEQUENCES_KEY,
    oeis_integer_sequence_raw_validation_sets,
)
from experiments.evals.perplexity_gap_registry import build_registered_perplexity_gap_coverage_plan


def test_oeis_parsers_skip_headers_and_preserve_sequence_text() -> None:
    names = dict(_iter_name_rows(["# header\n", "A000001 Number of groups of order n.\n"]))
    sequences = dict(_iter_sequence_rows(["# header\n", "A000001 ,0,1,1,2,\n"]))

    assert names == {"A000001": "Number of groups of order n."}
    assert sequences == {"A000001": "0,1,1,2"}
    assert _format_record("A000001", names["A000001"], sequences["A000001"]) == (
        "OEIS ID: A000001\nName: Number of groups of order n.\nTerms:\n0,1,1,2"
    )


def test_oeis_raw_validation_set_points_at_eval_step_glob() -> None:
    datasets = oeis_integer_sequence_raw_validation_sets()
    dataset = datasets[OEIS_INTEGER_SEQUENCES_KEY]

    assert dataset.input_path is not None
    assert dataset.text_key == "text"
    assert dataset.tags == (
        "oeis",
        "integer_sequences",
        "issue:5770",
        "max_sequences:50000",
        "records_per_doc:16",
    )


def test_registered_coverage_plan_includes_oeis_bundle() -> None:
    plan = build_registered_perplexity_gap_coverage_plan(
        resource_config=ResourceConfig.with_tpu("v5p-8", regions=["us-central1"])
    )

    marin_score = plan.score_steps[("oeis_integer_sequences", "marin_8b")]
    qwen_gap = plan.pairwise_gap_steps[("oeis_integer_sequences", "marin_8b", "qwen3_8b")]

    assert OEIS_INTEGER_SEQUENCES_KEY in marin_score.config.datasets
    assert qwen_gap.config.model_a_scores_path.step is marin_score
