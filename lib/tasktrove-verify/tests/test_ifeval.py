# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from tasktrove_verify.grade import InvalidTask, Status
from tasktrove_verify.modes import grade_ifeval
from tasktrove_verify.spec import Constraint, IfevalSpec

# One passing and one failing response per constraint, written the way a model would answer.
CASES = [
    (
        "length_constraints:number_paragraphs",
        {"num_paragraphs": 2},
        "The first thing to know.\n\nThe second thing to know.",
        "Only one paragraph here.",
    ),
    (
        "length_constraints:number_words",
        {"relation": "at least", "num_words": 6},
        "One two three four five six seven",
        "Far too short",
    ),
    (
        "length_constraints:number_sentences",
        {"relation": "at most", "num_sentences": 2},
        "First sentence. Second sentence.",
        "First sentence. Second sentence. Third sentence.",
    ),
    (
        "length_constraints:nth_paragraph_first_word",
        {"num_paragraphs": 3, "nth_paragraph": 2, "first_word": "crash"},
        "Opening thoughts.\n\nCrash reports arrived late.\n\nClosing thoughts.",
        "Opening thoughts.\n\nReports arrived late.\n\nClosing thoughts.",
    ),
    (
        "keywords:forbidden_words",
        {"forbidden_words": ["banana", "mango"]},
        "I would rather have an apple with breakfast.",
        "I would rather have a banana with breakfast.",
    ),
    (
        "keywords:existence",
        {"keywords": ["quantum", "entangled"]},
        "The quantum pair stayed entangled across the lab.",
        "The quantum pair stayed together across the lab.",
    ),
    (
        "keywords:frequency",
        {"keyword": "cat", "relation": "at least", "frequency": 2},
        "The cat watched another cat from the window.",
        "The cat watched a dog from the window.",
    ),
    (
        "keywords:letter_frequency",
        {"letter": "z", "let_relation": "at least", "let_frequency": 3},
        "A zebra dozed in the zoo.",
        "A zebra slept in the yard.",
    ),
    (
        "change_case:english_lowercase",
        {},
        "everything here stays lowercase, as asked.",
        "Everything here stays lowercase, as asked.",
    ),
    (
        "change_case:english_capital",
        {},
        "EVERYTHING HERE SHOUTS.",
        "Everything here shouts.",
    ),
    (
        "punctuation:no_comma",
        {},
        "No commas appear anywhere in this reply.",
        "Commas, sadly, appear here.",
    ),
    (
        "startend:end_checker",
        {"end_phrase": "Any other questions?"},
        "That covers the deployment. Any other questions?",
        "That covers the deployment. Let me know.",
    ),
    (
        "detectable_format:number_bullet_lists",
        {"num_bullets": 3},
        "* first\n* second\n* third",
        "* first\n* second",
    ),
    (
        "detectable_format:title",
        {},
        "<<The Lantern Keeper>>\n\nShe lit the wick at dusk.",
        "The Lantern Keeper\n\nShe lit the wick at dusk.",
    ),
    (
        "detectable_format:json_format",
        {},
        '{"status": "ok", "items": [1, 2]}',
        "status: ok, items: 1 and 2",
    ),
    (
        "detectable_content:number_placeholders",
        {"num_placeholders": 2},
        "Send it to [name] at [address] before Friday.",
        "Send it to [name] before Friday.",
    ),
    (
        "first_word:first_word_answer",
        {"first_word": "crash"},
        "Crash barriers were installed overnight.",
        "The crash barriers were installed overnight.",
    ),
    (
        "last_word:last_word_answer",
        {"last_word": "contest"},
        "Everyone practised hard before the contest",
        "Everyone practised hard before the match",
    ),
    (
        "count:lowercase_counting",
        {"N": 4},
        "these five words stay lowercase",
        "These Words Are Capitalised",
    ),
    (
        "combination:two_responses",
        {},
        "First take on the question.\n******\nSecond take on the question.",
        "Only one take on the question.",
    ),
]


@pytest.fixture
def workspace(tmp_path):
    directory = tmp_path / "app"
    directory.mkdir()
    return directory


def answer(workspace, text):
    (workspace / "answer.txt").write_text(text)


def reward_for(workspace, tests_dir, constraints, text=None):
    if text is not None:
        answer(workspace, text)
    return grade_ifeval.grade(IfevalSpec(constraints=constraints), tests_dir, workspace)


@pytest.mark.parametrize("name, params, passing, failing", CASES, ids=[case[0] for case in CASES])
def test_constraint_separates_a_satisfying_response_from_a_violating_one(
    tmp_path, workspace, name, params, passing, failing
):
    constraints = (Constraint(name, params),)
    assert reward_for(workspace, tmp_path, constraints, passing).reward == 1.0
    violated = reward_for(workspace, tmp_path, constraints, failing)
    assert violated.reward == 0.0
    assert violated.detail["failed"] == [name]


def test_every_constraint_must_hold_for_a_full_reward(tmp_path, workspace):
    # The exemplar task: four paragraphs whose third starts with "crash", ending on "contest".
    constraints = (
        Constraint(
            "length_constraints:nth_paragraph_first_word",
            {"num_paragraphs": 4, "nth_paragraph": 3, "first_word": "crash"},
        ),
        Constraint("last_word:last_word_answer", {"last_word": "contest"}),
    )
    body = "One.\n\nTwo.\n\nCrash tests came third.\n\nWe entered the contest"
    assert reward_for(workspace, tmp_path, constraints, body).reward == 1.0

    wrong_ending = reward_for(workspace, tmp_path, constraints, body.replace("contest", "raffle"))
    assert wrong_ending.reward == 0.0
    assert wrong_ending.detail["failed"] == ["last_word:last_word_answer"]

    out_of_order = "Crash tests came first.\n\nTwo.\n\nThree.\n\nWe entered the contest"
    reordered = reward_for(workspace, tmp_path, constraints, out_of_order)
    assert reordered.detail["failed"] == ["length_constraints:nth_paragraph_first_word"]


def test_detail_records_a_verdict_for_every_constraint(tmp_path, workspace):
    constraints = (Constraint("punctuation:no_comma", {}), Constraint("keywords:word_once", {"keyword": "otter"}))
    reward = reward_for(workspace, tmp_path, constraints, "An otter swam past, twice.")
    assert [(entry["name"], entry["passed"]) for entry in reward.detail["constraints"]] == [
        ("punctuation:no_comma", False),
        ("keywords:word_once", True),
    ]


@pytest.mark.parametrize("text", [None, "", "  \n"])
def test_absent_or_blank_output_scores_zero_with_no_output(tmp_path, workspace, text):
    if text is not None:
        answer(workspace, text)
    reward = reward_for(workspace, tmp_path, (Constraint("punctuation:no_comma", {}),))
    assert (reward.reward, reward.status) == (0.0, Status.SCORED)
    assert reward.detail == {"reason": "no_output"}


def test_unknown_constraint_is_an_invalid_task(tmp_path, workspace):
    answer(workspace, "anything at all")
    with pytest.raises(InvalidTask, match="unknown ifeval constraints"):
        reward_for(workspace, tmp_path, (Constraint("keywords:no_such_check", {}),))


def test_spec_without_constraints_is_an_invalid_task(tmp_path, workspace):
    answer(workspace, "anything at all")
    with pytest.raises(InvalidTask, match="no constraints"):
        reward_for(workspace, tmp_path, ())


def test_constraint_missing_its_parameters_fails_the_candidate(tmp_path, workspace):
    constraints = (Constraint("length_constraints:number_paragraphs", {}),)
    reward = reward_for(workspace, tmp_path, constraints, "Two.\n\nParagraphs.")
    assert reward.reward == 0.0
    assert reward.detail["constraints"][0]["detail"] == "missing num_paragraphs"
