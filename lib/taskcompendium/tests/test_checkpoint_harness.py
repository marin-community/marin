# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint likelihood prompts retain the pinned harness request boundaries."""

import csv

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.evaluation.spec import ChoiceLikelihood
from taskcompendium.importers.evalchemy.code import read_humaneval_fewshot
from taskcompendium.importers.evalchemy.commonsense import commonsense_qa_template, copa_template, wsc_template
from taskcompendium.importers.evalchemy.reasoning import agieval_template, gpqa_likelihood_template
from taskcompendium.importers.evaluation_harness import HarnessTemplate, compile_choice_documents
from taskcompendium.importers.fewshot import read_choice_file
from taskcompendium.pipeline.inputs import ConversionContext


@pytest.mark.parametrize(
    "template,document,expected",
    [
        (
            wsc_template(),
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
            copa_template(),
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
def test_checkpoint_multiple_choice_preserves_legacy_request_delimiters(tmp_path, template, document, expected):
    path = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([document]), path)
    (row,) = read_choice_file(
        StoragePath(str(path)),
        ConversionContext(inputs={}, grader_environment=None),
        template=template,
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
    (row,) = read_choice_file(
        StoragePath(str(path)),
        ConversionContext(inputs={}, grader_environment=None),
        template=gpqa_likelihood_template(),
    )
    operation = ChoiceLikelihood.model_validate(row["operation"])
    assert operation.gold == (3,)
    assert operation.requests[3].continuation == " (D)"
    assert "(D) Correct. Answer Text\nAnswer: " in operation.requests[3].context
    assert "[ignored]" not in operation.requests[3].context


def test_winogrande_fewshots_precede_both_candidate_contexts():
    document = {"sentence": "Alice _ the race.", "option1": "won", "option2": "lost", "answer": "1"}
    example = {"sentence": "Bob _ the ball.", "option1": "caught", "option2": "dropped", "answer": "1"}
    (row,) = compile_choice_documents([document], [example, example], HarnessTemplate("winogrande", 1))
    operation = ChoiceLikelihood.model_validate(row["operation"])
    assert [(request.context, request.continuation) for request in operation.requests] == [
        ("Bob caught the ball.\n\nAlice won", " the race."),
        ("Bob caught the ball.\n\nAlice lost", " the race."),
    ]


@pytest.mark.parametrize("family", ["boolq", "piqa", "truthfulqa_mc2"])
def test_validation_fewshots_follow_pinned_harness_draws_across_documents(family):
    documents = [
        {
            "question": f"Question {i}",
            "passage": f"Passage {i}",
            "goal": f"Goal {i}",
            "sol1": "first",
            "sol2": "second",
            "label": 0,
            "mc2_targets": {"choices": ["first", "second"], "labels": [1, 0]},
        }
        for i in range(6)
    ]
    # ContextSampler at the pinned harness commit, Random(1234), n=2.
    # Training pools still draw n+1 when both configured split names are unset.
    pool = documents if family == "truthfulqa_mc2" else [{**doc, "question": "training"} for doc in documents]
    rows = list(compile_choice_documents(documents[:3], pool, HarnessTemplate(family, 2)))
    assert [row["fewshot_indices"] for row in rows] == (
        [[3, 4], [0, 4], [5, 0]] if family == "truthfulqa_mc2" else [[3, 0], [0, 4], [5, 0]]
    )


@pytest.mark.parametrize(
    "family,expected",
    [("commonsense_qa", [[3, 4], [0, 4], [5, 0]]), ("agieval_lsat_ar", [[3, 0], [0, 5], [4, 0]])],
)
def test_checkpoint_fewshots_use_split_configuration_instead_of_pool_identity(tmp_path, family, expected):
    documents = [
        {
            "question": f"Question {i}",
            "query": f"Question {i}",
            "choices": (
                {"text": ["red", "blue", "green", "white", "black"]} if family == "commonsense_qa" else ["red", "blue"]
            ),
            "answerKey": "A",
            "gold": [0],
        }
        for i in range(6)
    ]
    path = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(documents), path)
    rows = list(
        read_choice_file(
            StoragePath(str(path)),
            ConversionContext(inputs={}, grader_environment=None),
            template=commonsense_qa_template(2) if family == "commonsense_qa" else agieval_template(2),
        )
    )
    assert [row["fewshot_indices"] for row in rows[:3]] == expected


def test_humaneval_ten_shot_prompts_follow_the_pinned_test_split_draws(tmp_path):
    documents = [{"prompt": f"problem{i}", "test": f"test{i}", "entry_point": f"fn{i}"} for i in range(12)]
    path = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(documents), path)
    rows = list(read_humaneval_fewshot(StoragePath(str(path)), ConversionContext({}, None)))
    for row, indices in zip(rows[:2], ([7, 1, 0, 10, 9, 5, 6, 11, 4, 8], [3, 0, 10, 9, 5, 7, 4, 11, 6, 8]), strict=True):
        expected = "".join(f"problem{i} test{i}\ncheck(fn{i})\n\n" for i in indices)
        assert row["evaluation_prompt"] == expected + row["prompt"]
