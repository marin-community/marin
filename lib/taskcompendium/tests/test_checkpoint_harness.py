# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint likelihood prompts retain the pinned harness request boundaries."""

import csv

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.evaluation.spec import ChoiceLikelihood
from taskcompendium.importers.checkpoint_harness import CheckpointTemplate, read_checkpoint_file
from taskcompendium.pipeline.inputs import ConversionContext


@pytest.mark.parametrize(
    "family,document,expected",
    [
        (
            "wsc273",
            {
                "text": "Alice thanked Beth because she helped.",
                "pronoun": "she",
                "pronoun_loc": 27,
                "options": ["Alice", "Beth"],
                "label": 0,
            },
            [("Alice thanked Beth because Alice", "  helped."), ("Alice thanked Beth because Beth", "  helped.")],
        ),
        (
            "copa",
            {
                "premise": "It rained.",
                "question": "effect",
                "choice1": "The road got wet.",
                "choice2": "The road stayed dry.",
                "label": 0,
            },
            [("It rained therefore", "  the road got wet."), ("It rained therefore", "  the road stayed dry.")],
        ),
    ],
)
def test_checkpoint_multiple_choice_preserves_legacy_request_delimiters(tmp_path, family, document, expected):
    path = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([document]), path)
    (row,) = read_checkpoint_file(
        StoragePath(str(path)),
        ConversionContext(inputs={}, grader_environment=None),
        template=CheckpointTemplate(family),
    )
    operation = ChoiceLikelihood.model_validate(row["operation"])
    assert [(request.context, request.continuation) for request in operation.requests] == expected
    assert operation.gold == (0,)


def test_checkpoint_gpqa_cleans_answers_before_shuffling_and_selecting_gold(tmp_path):
    document = {
        "Question": "Pick the correct text.",
        "Correct Answer": " Correct [title] Answer [ignored] Text ",
        "Incorrect Answer 1": "First",
        "Incorrect Answer 2": "Second",
        "Incorrect Answer 3": "Third",
    }
    path = tmp_path / "source.csv"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(document))
        writer.writeheader()
        writer.writerow(document)
    (row,) = read_checkpoint_file(
        StoragePath(str(path)),
        ConversionContext(inputs={}, grader_environment=None),
        template=CheckpointTemplate("leaderboard_gpqa"),
    )
    operation = ChoiceLikelihood.model_validate(row["operation"])
    assert operation.gold == (3,)
    assert operation.requests[3].continuation == " (D)"
    assert "(D) Correct. Answer Text\nAnswer: " in operation.requests[3].context
    assert "[ignored]" not in operation.requests[3].context
