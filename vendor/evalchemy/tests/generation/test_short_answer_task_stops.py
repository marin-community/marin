"""Regression coverage for NQ-Open and TriviaQA chat stop policies."""

from collections import Counter
from functools import cache
from pathlib import Path
import sys

import pytest
from lm_eval.api.instance import Instance
from lm_eval.api.metrics import exact_match_hf_evaluate
from lm_eval.filters import build_filter_ensemble
from lm_eval.models.openai_completions import LocalChatCompletion
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval import robust_api
from eval.completion_response import CompletionClassification, CompletionContentPolicy
from eval.eval import DEFAULT_LM_EVAL_INCLUDE_DIR
from eval.generation_stops import SHORT_ANSWER_STOP_SEQUENCES, truncate_at_stop
from eval.lm_eval_tasks.short_answer_extraction import (
    INVALID_SHORT_ANSWER,
    ShortAnswerFormat,
    extract_marked_short_answer,
    extract_short_answer,
)

_TASKS = ("nq_open", "triviaqa")


@cache
def _task_config(task_name: str) -> dict:
    task_manager = TaskManager(include_path=[DEFAULT_LM_EVAL_INCLUDE_DIR])
    entry = task_manager.task_index[task_name]
    assert entry.yaml_path is not None
    assert Path(entry.yaml_path).resolve().is_relative_to(
        _REPO_ROOT / "eval" / "lm_eval_tasks"
    ), f"{task_name} did not resolve to its Evalchemy override"
    return load_yaml(entry.yaml_path, resolve_func=True)


def _adapter() -> LocalChatCompletion:
    adapter = object.__new__(LocalChatCompletion)
    adapter.completion_content_policy = CompletionContentPolicy.COMBINE
    adapter.completion_responses = []
    return adapter


def _filtered_response(
    task_name: str, message: dict[str, str | None], filter_name: str = "extract_answer"
) -> str:
    config = _task_config(task_name)
    filter_entry = next(
        entry for entry in config["filter_list"] if entry["name"] == filter_name
    )
    steps = [
        (
            step["function"],
            {key: value for key, value in step.items() if key != "function"},
        )
        for step in filter_entry["filter"]
    ]
    pipeline = build_filter_ensemble(filter_name, steps)
    instance = Instance("generate_until", {}, (), 0)
    instance.resps = [
        _adapter().parse_generations({"choices": [{"index": 0, "message": message}]})[0]
    ]
    pipeline.apply([instance])
    return instance.filtered_resps[filter_name]


@pytest.mark.parametrize(
    ("task_name", "message", "reference"),
    [
        (
            "nq_open",
            {
                "content": "\n\nAnswer: 14 December 1972",
                "reasoning_content": "The final Apollo mission returned from the Moon in December 1972.",
            },
            "14 December 1972",
        ),
        (
            "triviaqa",
            {
                "content": "\n\nAnswer: St. John's, Newfoundland",
                "reasoning_content": "The place name includes an apostrophe and a comma.",
            },
            "St Johns Newfoundland",
        ),
    ],
)
def test_short_answer_task_scores_marked_final_content_after_reasoning(task_name, message, reference):
    selected = _filtered_response(task_name, message)

    assert selected == message["content"].split(":", 1)[1].strip()
    assert exact_match_hf_evaluate(
        predictions=[selected],
        references=[reference],
        ignore_case=True,
        ignore_punctuation=True,
    )["exact_match"] == 1.0


@pytest.mark.parametrize("task_name", _TASKS)
def test_short_answer_task_uses_only_shared_turn_boundaries(task_name):
    config = _task_config(task_name)
    stops = config["generation_kwargs"]["until"]

    assert stops == SHORT_ANSWER_STOP_SEQUENCES
    assert not {"\n", ".", ","}.intersection(stops)
    assert truncate_at_stop("14 December 1972\nQuestion: repeated prompt", stops) == "14 December 1972"


@pytest.mark.parametrize("task_name", _TASKS)
def test_short_answer_task_establishes_and_extracts_its_answer_contract(task_name):
    config = _task_config(task_name)

    assert config["doc_to_text"].endswith('Respond with exactly "Answer: <short answer>".')
    assert config["fewshot_config"]["doc_to_target"].startswith("Answer:")
    assert [entry["name"] for entry in config["filter_list"]] == [
        "strict_answer",
        "extract_answer",
    ]


@pytest.mark.parametrize("task_name", _TASKS)
def test_short_answer_task_scores_explicit_answer_and_reports_contract_adherence(task_name):
    message = {
        "content": "The answer is (14 December 1972).",
        "reasoning_content": "I checked the Apollo mission timeline.",
    }

    selected = _filtered_response(task_name, message)
    strict_selected = _filtered_response(task_name, message, "strict_answer")

    assert selected == "14 December 1972"
    assert strict_selected == INVALID_SHORT_ANSWER
    assert exact_match_hf_evaluate(
        predictions=[selected],
        references=["14 December 1972"],
        ignore_case=True,
        ignore_punctuation=True,
    )["exact_match"] == 1.0


@pytest.mark.parametrize("task_name", _TASKS)
def test_short_answer_task_scores_a_bare_final_line(task_name):
    selected = _filtered_response(
        task_name,
        {
            "content": "<think>I checked the Apollo mission timeline.</think>\n[14 December 1972]",
            "reasoning_content": None,
        },
    )

    assert selected == "14 December 1972"


@pytest.mark.parametrize(
    ("response", "expected", "answer_format", "strict_expected"),
    [
        (
            "Reasoning\nAnswer: first\nFinal Answer: second",
            "second",
            ShortAnswerFormat.CONTRACT,
            "second",
        ),
        (
            "A: 14 December 1972",
            "14 December 1972",
            ShortAnswerFormat.CONTRACT,
            "14 December 1972",
        ),
        (
            "Answer: St. John's, Newfoundland\nQuestion: repeated prompt",
            "St. John's, Newfoundland",
            ShortAnswerFormat.CONTRACT,
            "St. John's, Newfoundland",
        ),
        (
            "The answer is (Leeds).",
            "Leeds",
            ShortAnswerFormat.EXPLICIT,
            INVALID_SHORT_ANSWER,
        ),
        (
            "The answer is Answer: car rental company",
            "car rental company",
            ShortAnswerFormat.EXPLICIT,
            INVALID_SHORT_ANSWER,
        ),
        (
            "<think>reasoning</think>\n[Miller Lite]",
            "Miller Lite",
            ShortAnswerFormat.BARE,
            INVALID_SHORT_ANSWER,
        ),
        (
            "<think>reasoning only</think>",
            INVALID_SHORT_ANSWER,
            ShortAnswerFormat.INVALID,
            INVALID_SHORT_ANSWER,
        ),
        (None, INVALID_SHORT_ANSWER, ShortAnswerFormat.INVALID, INVALID_SHORT_ANSWER),
    ],
)
def test_short_answer_extractor_classifies_shared_answer_formats(
    response, expected, answer_format, strict_expected
):
    extraction = extract_short_answer(response)

    assert extraction.answer == expected
    assert extraction.format == answer_format
    assert extract_marked_short_answer(response) == strict_expected


@pytest.mark.parametrize("task_name", _TASKS)
def test_short_answer_task_rejects_reasoning_only_truncation(task_name):
    _task_config(task_name)
    generated = _adapter().parse_generations(
        {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "message": {"content": None, "reasoning_content": "working"},
                }
            ]
        }
    )[0]

    assert generated.response.classification == CompletionClassification.REASONING_ONLY_TRUNCATED
    assert robust_api.completion_response_quality_invalid(Counter([generated.response.classification]))


@pytest.mark.parametrize("task_name", _TASKS)
def test_short_answer_task_rejects_successful_all_empty_chat_run(task_name):
    _task_config(task_name)
    responses = Counter(
        _adapter()
        .parse_generations(
            {
                "id": f"chatcmpl-{index}",
                "usage": {"completion_tokens": 2},
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"content": None}}],
            }
        )[0]
        .response.classification
        for index in range(3)
    )

    assert robust_api.completion_response_quality_invalid(responses)
