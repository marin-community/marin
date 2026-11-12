from pathlib import Path

import pytest
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml

from eval.eval import DEFAULT_LM_EVAL_INCLUDE_DIR
from eval.generation_stops import (
    GSM8K_STOP_SEQUENCES,
    OPENAI_COMPLETIONS_MAX_STOP_SEQUENCES,
    HUMANEVAL_REQUEST_STOP_SEQUENCES,
    bounded_request_stops,
    chat_request_stops,
    truncate_at_stop,
)
from eval.graders.answer_extraction import extract_final_boxed_answer
from eval.lm_eval_tasks.humaneval.scoring import build_predictions


def test_truncate_at_earliest_turn_boundary():
    output = "answer\nQuestion: repeated prompt<|im_end|>"
    assert truncate_at_stop(output) == "answer"
    assert truncate_at_stop("answer without a boundary") == "answer without a boundary"


def test_math_graders_ignore_boxed_answers_from_a_repeated_turn():
    output = (
        r"Reasoning. \boxed{17}" + "\nQuestion: unrelated\n" + r"Reasoning. \boxed{42}"
    )
    assert extract_final_boxed_answer(output) == "17"


@pytest.mark.parametrize(
    "response",
    [
        "    return 42\n```\nAssistant:\nHere is another answer",
        "    return 42\nprint(answer())\n",
    ],
)
def test_humaneval_scoring_discards_boundaries_omitted_from_the_request(response):
    docs = [{"prompt": "def answer():\n"}]
    responses = [[response]]
    assert build_predictions(responses, docs) == [["def answer():\n    return 42"]]


def test_humaneval_request_stops_fit_the_completions_api():
    assert len(HUMANEVAL_REQUEST_STOP_SEQUENCES) == OPENAI_COMPLETIONS_MAX_STOP_SEQUENCES


def test_bounded_request_stops_prefer_semantic_boundaries_to_token_sentinels():
    assert bounded_request_stops(GSM8K_STOP_SEQUENCES) == [
        "\nYou are an AI assistant",
        "Question:",
        "\nQ:",
        "\n[Question]",
    ]


def test_chat_request_stops_drop_textual_boundaries_that_truncate_reasoning():
    assert chat_request_stops(GSM8K_STOP_SEQUENCES) == [
        "<|im_end|>",
        "<|eot_id|>",
        "<|end_of_text|>",
        "<|endoftext|>",
    ]


def test_chat_request_stops_without_sentinels_request_no_stops():
    assert chat_request_stops(["Question:", "\nQ:", "\nUser:"]) == []


def test_generation_task_overrides_use_shared_stop_sequences():
    task_manager = TaskManager(include_path=[DEFAULT_LM_EVAL_INCLUDE_DIR])

    expected_stops = {
        "gsm8k": GSM8K_STOP_SEQUENCES,
        "humaneval": HUMANEVAL_REQUEST_STOP_SEQUENCES,
    }
    for task_name, stops in expected_stops.items():
        entry = task_manager.task_index[task_name]
        assert entry.yaml_path is not None
        assert (
            Path(entry.yaml_path)
            .resolve()
            .is_relative_to(Path(DEFAULT_LM_EVAL_INCLUDE_DIR).resolve())
        )
        config = load_yaml(entry.yaml_path, resolve_func=True)
        assert config["generation_kwargs"]["until"] == stops
