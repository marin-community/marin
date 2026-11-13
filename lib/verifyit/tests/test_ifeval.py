# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from verifyit.grade import InvalidTask, Status
from verifyit.instruction_observations import prepare_extended_instruction_observations
from verifyit.modes import grade_ifeval
from verifyit.spec import Constraint, EmptyOutputPolicy, IfevalSpec

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


@pytest.mark.parametrize(
    "text,expected",
    [
        ("First answer.******Second answer.", 1.0),
        ("******First answer.******Second answer.******", 1.0),
        ("Same answer.****** Same answer. ", 0.0),
        ("First.******Second.******Third.", 0.0),
        ("First.************Second.", 0.0),
    ],
)
def test_two_responses_requires_exactly_two_distinct_answers(tmp_path, workspace, text, expected):
    result = reward_for(workspace, tmp_path, (Constraint("combination:two_responses", {}),), text)
    assert (result.reward, result.status) == (expected, Status.SCORED)


@pytest.mark.parametrize("policy", [EmptyOutputPolicy.ZERO, EmptyOutputPolicy.GRADE])
def test_direct_candidate_preserves_file_empty_and_constraint_results(tmp_path, policy):
    spec = IfevalSpec((Constraint("punctuation:no_comma"),), empty_output=policy)
    for candidate in ["plain", "one,two", ""]:
        (tmp_path / "answer.txt").write_text(candidate)
        direct = grade_ifeval.grade_ifeval_candidate(spec, candidate)
        from_file = grade_ifeval.grade(spec, tmp_path, tmp_path)
        assert direct == from_file
        assert direct.reward == float("," not in candidate and (bool(candidate) or policy is EmptyOutputPolicy.GRADE))


def test_custom_constraint_cannot_replace_builtin_scoring():
    spec = IfevalSpec((Constraint("punctuation:no_comma"),))
    with pytest.raises(InvalidTask):
        grade_ifeval.grade_ifeval_candidate(
            spec, "one,two", registry={"punctuation:no_comma": lambda text, params: (True, "")}
        )
    assert grade_ifeval.grade_ifeval_candidate(spec, "one,two").reward == 0


@pytest.mark.parametrize("value,detail", [(1, ""), ("passed", ""), (None, ""), (True, {})])
def test_custom_constraint_truthy_malformed_result_cannot_award_credit(value, detail):
    spec = IfevalSpec((Constraint("custom:decision"),))
    with pytest.raises(RuntimeError):
        grade_ifeval.grade_ifeval_candidate(
            spec, "candidate", registry={"custom:decision": lambda text, params: (value, detail)}
        )


@pytest.mark.parametrize(
    "name,key", [("keywords:existence", "keywords"), ("keywords:forbidden_words", "forbidden_words")]
)
def test_keyword_substring_policy_preserves_lowercase_and_word_default(name, key):
    expected = name == "keywords:existence"
    default = IfevalSpec((Constraint(name, {key: ["cat"]}),))
    substring = IfevalSpec((Constraint(name, {key: ["cat"], "word_boundary": False}),))
    assert grade_ifeval.grade_ifeval_candidate(default, "SCATTER").reward == float(not expected)
    assert grade_ifeval.grade_ifeval_candidate(substring, "SCATTER").reward == float(expected)
    malformed = IfevalSpec((Constraint(name, {key: [], "word_boundary": "false"}),))
    assert grade_ifeval.grade_ifeval_candidate(malformed, "candidate").reward == 0


@pytest.mark.parametrize(
    "name,key,valid",
    [("keywords:existence", "keywords", "candidate"), ("keywords:forbidden_words", "forbidden_words", "absent")],
)
def test_malformed_keyword_members_cannot_silently_disappear(name, key, valid):
    for words in ([None], [valid, None]):
        spec = IfevalSpec((Constraint(name, {key: words}),))
        assert grade_ifeval.grade_ifeval_candidate(spec, "candidate").reward == 0
    for words in ([], [valid]):
        spec = IfevalSpec((Constraint(name, {key: words}),))
        assert grade_ifeval.grade_ifeval_candidate(spec, "candidate").reward == 1


@pytest.mark.parametrize(
    "text,language,expected",
    [("YES THIS IS ENGLISH", "en", 1.0), ("Yes this is English", "en", 0.0), ("OUI", "fr", 0.0)],
)
def test_prepared_observations_require_both_constraint_and_tool_result(text, language, expected):
    verdict = grade_ifeval.grade_instruction_observations(
        [
            (IfevalSpec((Constraint("change_case:english_capital"),)), text),
            ({"type": "string", "const": "en"}, language),
        ]
    )
    assert (verdict.status, verdict.reward) == (Status.SCORED, expected)


def test_invalid_prepared_reference_cannot_preserve_a_passing_observation():
    with pytest.raises(InvalidTask):
        grade_ifeval.grade_instruction_observations(
            [({"const": "valid"}, "valid"), ({"type": "array", "minItems": "many"}, [])]
        )
    empty = grade_ifeval.grade_instruction_observations([])
    assert (empty.status, empty.reward) == (Status.INVALID_TASK, 0.0)


@pytest.mark.parametrize("identifier", ["detectable_format:sentence_hyphens", "keywords:start_end"])
def test_empty_instruction_tokens_score_zero_without_invalidating_task(identifier):
    tools = SimpleNamespace(split_into_sentences=str.split, nltk=SimpleNamespace(word_tokenize=str.split))
    observations = prepare_extended_instruction_observations(identifier, {}, tools, "")
    verdict = grade_ifeval.grade_instruction_observations(observations)
    assert (verdict.status, verdict.reward) == (Status.SCORED, 0.0)


def test_repeated_phrase_requires_exactly_one_changed_word_per_match():
    args = {"phrase": "red warm soft fox", "small_n": 2}
    for candidate, expected in [
        ("red cold soft fox red warm furry fox", 1.0),
        ("red cold furry fox red warm furry fox", 0.0),
        ("red warm soft fox red warm furry fox", 0.0),
        ("red cold soft fox", 0.0),
    ]:
        observations = prepare_extended_instruction_observations("copy:repeat_phrase", args, None, candidate)
        assert grade_ifeval.grade_instruction_observations(observations).reward == expected


def test_source_copy_span_uses_exclusive_character_end():
    args = {"prompt_to_repeat": "A small bird", "n_start": 2, "n_end": 7}
    for candidate, expected in [("SMALL", 1.0), ("small b", 0.0), ("small bird", 0.0)]:
        observations = prepare_extended_instruction_observations("new:copy_span_idx", args, None, candidate)
        assert grade_ifeval.grade_instruction_observations(observations).reward == expected
