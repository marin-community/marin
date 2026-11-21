# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from taskcompendium.convert.preference import binary_preference_task, pairwise_preference_task
from taskcompendium.models import NoGrader, Source, TaskSpec, TextMessage
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, RawRow

ROW = RawRow("preference-row", Source(dataset="fixture", revision="pin", row="0", importer_revision="1"), {})
HISTORY = [
    {"role": "system", "content": "Keep full context"},
    {"role": "user", "content": "Earlier request"},
    {"role": "assistant", "content": "Earlier response"},
    {"role": "user", "content": "Final request"},
]


def messages(*turns: tuple[str, str]) -> tuple[TextMessage, ...]:
    return tuple(TextMessage(role=role, content=content) for role, content in turns)


def test_pairwise_preference_shows_shared_history_and_keeps_candidates_with_the_grader():
    history = (("user", "Earlier request"), ("assistant", "Earlier response"), ("user", "Final request"))
    task = pairwise_preference_task(
        ROW, chosen=messages(*history, ("assistant", "Chosen")), rejected=messages(*history, ("assistant", "Rejected"))
    )
    assert isinstance(task, TaskSpec)
    assert task.context.events == messages(*history)
    assert isinstance(task.grader, NoGrader)
    contract = task.grader.contract
    assert [message["content"] for message in contract["chosen"] + contract["rejected"]] == ["Chosen", "Rejected"]


@pytest.mark.parametrize(
    ("chosen", "rejected", "reason"),
    [
        (
            messages(("user", "Request"), ("assistant", "Chosen")),
            messages(("user", "Other request"), ("assistant", "Rejected")),
            "preference_prompt_conflict",
        ),
        (
            messages(("user", "Request"), ("assistant", "Chosen")),
            messages(("user", "Request")),
            "missing_preference_completion",
        ),
        (
            messages(("assistant", "Greeting"), ("assistant", "Chosen")),
            messages(("assistant", "Greeting"), ("assistant", "Rejected")),
            "invalid_preference_prompt",
        ),
    ],
)
def test_pairwise_preference_rejects_candidates_without_one_shared_request(chosen, rejected, reason):
    result = pairwise_preference_task(ROW, chosen=chosen, rejected=rejected)
    assert isinstance(result, ImportRejection)
    assert (result.kind, result.reason) == (ImportFailureKind.SOURCE_DEFECT, reason)


def test_binary_preference_keeps_the_whole_prompt_and_the_labeled_completion():
    completion = [{"role": "assistant", "content": "Labeled answer"}]
    task = binary_preference_task(
        ROW, prompt=HISTORY, completion=completion, label=False, evidence={"component": "fixture"}
    )
    assert isinstance(task, TaskSpec)
    assert [event.content for event in task.context.events] == [message["content"] for message in HISTORY]
    assert isinstance(task.grader, NoGrader)
    assert task.grader.contract["preferred"] is False
    assert [event["content"] for event in task.grader.contract["response"]] == ["Labeled answer"]


@pytest.mark.parametrize(
    ("prompt", "completion", "label", "reason"),
    [
        (HISTORY, [{"role": "assistant", "content": "Answer"}], "false", "invalid_binary_preference"),
        ("Final request", [{"role": "assistant", "content": "Answer"}], True, "invalid_binary_preference"),
        ([*HISTORY, {"role": "assistant", "content": "Answer"}], [], True, "missing_preference_messages"),
    ],
)
def test_binary_preference_rejects_malformed_observations(prompt, completion, label, reason):
    result = binary_preference_task(ROW, prompt=prompt, completion=completion, label=label)
    assert isinstance(result, ImportRejection)
    assert (result.kind, result.reason) == (ImportFailureKind.SOURCE_DEFECT, reason)
