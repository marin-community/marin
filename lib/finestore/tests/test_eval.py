# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the evaluation archive contract shipped in marin-finestore."""

from finestore.eval import (
    ARCHIVE_SAMPLES_TABLE,
    EvalSample,
    EvaluationStore,
    SampleKind,
    sample_from_archive_row,
    samples_from_lm_eval,
)
from finestore.reader import ReadView


def test_evaluation_store_round_trips_a_normalized_sample(tmp_path):
    root = str(tmp_path / "archive")
    with EvaluationStore.open(root, writer_id="evalchemy") as store:
        store.add_sample(
            EvalSample(
                task="gsm8k",
                doc_id="7",
                kind=SampleKind.GENERATION,
                prompt_text="2 + 2?",
                output="4",
                extracted="4",
                metrics={"exact_match": 1.0},
                correct=True,
            )
        )
        store.seal()

    table = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
    assert table is not None
    [row] = table.to_pylist(maps_as_pydicts="strict")
    sample = sample_from_archive_row(row)
    assert (sample.task, sample.doc_id, sample.output, sample.correct) == ("gsm8k", "7", "4", True)


def test_samples_from_lm_eval_expands_coalesced_filter_variants():
    raw = {
        "doc_id": 0,
        "doc": {"question": "2 + 2?"},
        "target": "4",
        "arguments": [["2 + 2?"]],
        "resps": [["4"]],
        "filtered_resps": ["invalid"],
        "filter": "strict-match",
        "metrics": ["exact_match"],
        "exact_match": 0.0,
        "filter_variants": [
            {
                "filter": "strict-match",
                "filtered_resps": ["invalid"],
                "metrics": {"exact_match": 0.0},
            },
            {
                "filter": "flexible-extract",
                "filtered_resps": ["4"],
                "metrics": {"exact_match": 1.0},
            },
        ],
    }

    samples = samples_from_lm_eval("gsm8k", raw)

    assert [(sample.grading.filter, sample.extracted, sample.correct) for sample in samples] == [
        ("strict-match", "invalid", False),
        ("flexible-extract", "4", True),
    ]
