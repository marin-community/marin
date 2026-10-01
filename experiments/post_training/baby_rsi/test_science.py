# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter

import pytest
from levanter.models.snowball import SnowballConfig
from marin.execution.lazy import materialized_config
from marin.experiment.namespacing import user_owned_name
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.baby_rsi.science import (
    LETTERS,
    GenerateScienceConfig,
    apply_verification,
    parse_question_batch,
    shuffle_options,
)
from experiments.post_training.baby_rsi.science_trial import (
    CURRICULUM_IDS,
    QUESTIONS_VERSION,
    build_generation,
    build_science_trial,
    build_self_distill,
)
from experiments.post_training.baby_rsi.self_distill import AnswerCheck, answer_matches
from experiments.post_training.baby_rsi.trial import S3_TRIAL_PREFIX

PACKET = {
    "capability_id": "d03.example",
    "sampling_facets": [{"id": "f1", "description": "Lorentz transformations of events"}],
    "includes": [],
}

MUON_QUESTION = {
    "question": (
        "A muon with proper lifetime 2.2 microseconds moves at 0.8c relative to the lab. "
        "How far does it travel in the lab during one proper lifetime?"
    ),
    "correct_option": "880 m",
    "distractors": ["528 m", "660 m", "1170 m"],
    "rationale": "gamma = 5/3, so the lab lifetime is 3.67 us and the distance is 0.8c times that.",
}


def _question_response(index: int, arguments: dict, finish_reason: str = "tool_calls") -> dict:
    return {
        "custom_id": f"question-{index:05d}",
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "finish_reason": finish_reason,
                        "message": {
                            "tool_calls": [{"function": {"name": "submit_question", "arguments": json.dumps(arguments)}}]
                        },
                    }
                ]
            },
        },
    }


def _answer_response(request_id: str, content: str) -> dict:
    return {
        "custom_id": request_id,
        "response": {
            "status_code": 200,
            "body": {"choices": [{"finish_reason": "stop", "message": {"content": content, "reasoning": "Work."}}]},
        },
    }


def _jsonl(responses: list[dict]) -> str:
    return "\n".join(json.dumps(response) for response in responses)


def _config(requested: int, verify_samples: int = 3) -> GenerateScienceConfig:
    return GenerateScienceConfig(
        catalog_path="unused",
        output_path="unused",
        capability_id="d03.example",
        requested=requested,
        verify_samples=verify_samples,
        seed=17,
        max_completion_tokens=1024,
        relay_job="unused",
    )


def test_shuffle_puts_the_correct_option_at_the_recorded_letter_deterministically():
    distractors = ["528 m", "660 m", "1170 m"]
    letters = []
    for index in range(400):
        options, letter = shuffle_options("880 m", distractors, seed=17, index=index)
        assert options[LETTERS.index(letter)] == "880 m"
        assert sorted(options) == sorted(["880 m", *distractors])
        assert shuffle_options("880 m", distractors, seed=17, index=index) == (options, letter)
        letters.append(letter)

    # Uniform over four letters: each expects 100 of 400; 60 is more than five standard deviations below.
    assert min(Counter(letters)[letter] for letter in LETTERS) > 60


def test_verification_keeps_majority_agreement_and_drops_disagreement_and_malformed_options():
    leaky = {
        **MUON_QUESTION,
        "question": "Why does the muon reach the ground? Because of time dilation of its clock.",
        "correct_option": "time dilation of its clock",
    }
    responses = [
        _question_response(0, MUON_QUESTION),
        _question_response(1, {**MUON_QUESTION, "question": "At 0.6c, how far does the muon go?"}),
        _question_response(
            2, {**MUON_QUESTION, "question": "Same muon at 0.9c?", "distractors": ["528 m", "528  M", "1170 m"]}
        ),
        _question_response(3, leaky),
        _question_response(4, {**MUON_QUESTION, "question": "A pion..."}, finish_reason="length"),
    ]
    config = _config(requested=5)
    records = parse_question_batch(_jsonl(responses), config, PACKET)

    assert [record["rejection_reason"] for record in records] == [
        None,
        None,
        "duplicate_options",
        "answer_in_question",
        "truncated",
    ]
    assert "\\boxed{}" in records[0]["problem"]
    assert f"{records[0]['answer']}) 880 m" in records[0]["problem"]

    key0, key1 = records[0]["answer"], records[1]["answer"]
    wrong1 = next(letter for letter in LETTERS if letter != key1)
    answers = [
        _answer_response("question-00000-v00", f"So \\boxed{{{key0}}}."),
        _answer_response("question-00000-v01", f"So \\boxed{{({key0.lower()})}}."),
        _answer_response("question-00000-v02", "I am not sure."),
        _answer_response("question-00001-v00", f"So \\boxed{{{key1}}}."),
        _answer_response("question-00001-v01", f"So \\boxed{{{wrong1}}}."),
        _answer_response("question-00001-v02", f"So \\boxed{{{wrong1}}}."),
    ]
    verified = apply_verification(_jsonl(answers), config, records)

    assert [record["accepted"] for record in verified] == [True, False, False, False, False]
    assert [record["agreeing_samples"] for record in verified[:2]] == [2, 1]
    assert verified[1]["rejection_reason"] == "solver_disagrees"
    assert answer_matches(AnswerCheck.CHOICE, verified[0]["answer"], f"({key0.lower()})")


@pytest.fixture
def avoid_hf_config_fetch(monkeypatch):
    monkeypatch.setattr("marin.experiment.checkpoints.resolve_lm_config", lambda *_: SnowballConfig())


def test_generation_builds_one_verified_question_step_per_capability():
    steps = build_generation()

    assert set(steps) == set(CURRICULUM_IDS)
    for capability_id, step in steps.items():
        config = materialized_config(step, "s3://test-prefix")
        assert config.capability_id == capability_id
        assert config.verify_samples > 1


def test_self_distill_grades_letters_from_questions_adopted_from_the_source_prefix():
    config = materialized_config(build_self_distill(), S3_TRIAL_PREFIX)
    source_prefix = "s3://marin-us-east-02a/tmp/ttl=30d/curriculum-math-20260924"

    assert config.answer_check is AnswerCheck.CHOICE
    assert config.problems_paths == {
        capability_id: prefix_join(
            source_prefix,
            user_owned_name(f"documents/curriculum-sft/{capability_id}/science-questions/{QUESTIONS_VERSION}"),
        )
        for capability_id in CURRICULUM_IDS
    }


@pytest.mark.usefixtures("avoid_hf_config_fetch")
def test_trial_trains_on_self_distilled_rows_and_evaluates_gpqa_before_and_after():
    trial = build_science_trial("2026.09.26", learning_rate=1e-6, warmup=1)

    train_config = materialized_config(trial["train"], "s3://test-prefix").train_config
    assert set(train_config.data.components) == set(CURRICULUM_IDS)
    assert trial["after"].deps == (trial["train"],)
    for stage in ("baseline", "after"):
        assert "gpqa-diamond" in materialized_config(trial[stage], "s3://test-prefix").evals
