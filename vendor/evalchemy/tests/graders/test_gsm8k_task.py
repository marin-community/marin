"""Behavioral checks for GSM8K's task-level Minerva scorer."""

import pytest
from lm_eval.api.instance import Instance
from lm_eval.filters import build_filter_ensemble
from lm_eval.tasks._yaml_loader import load_yaml

from eval.lm_eval_tasks.gsm8k.utils import (
    extract_gsm8k_flexible_answer,
    gsm8k_flexible_extraction_filter,
    process_results,
)


@pytest.mark.parametrize(
    ("response", "reference", "expected"),
    [
        ("9.0", "9", 1.0),
        ("9", "9.0", 1.0),
        ("1,200", "1200", 1.0),
        ("$18", "18", 1.0),
        ("\\$235.00", "235", 1.0),
        ("24\\%", "24", 1.0),
        ("448 mL", "448", 1.0),
        ("\\$10,000\\ \\text{per day}", "10000", 1.0),
        ("10\\ \\text{days}", "10", 1.0),
        ("\\$15/week", "15", 1.0),
        ("\\frac{1}{2}", "0.5", 1.0),
        ("€77", "77", 1.0),
        ("There are 5 rats in each cage.", "5", 1.0),
        ("49^\\circ\\text{F}", "49", 1.0),
        ("\\text{Sid averaged a speed of 50 miles per hour.}", "50", 1.0),
        ("160\\text{ minutes}\\;(= 2\\text{ hours }40\\text{ minutes})", "160", 1.0),
        ("\\$25{,}000 \\text{ per year}", "25000", 1.0),
        ("50\\% - 25\\% = 25\\%", "25", 1.0),
        ("3 \\text{ onions} + 1 \\text{ potato}", "4", 1.0),
        ("16\\text{ chocolate bars} \\over 8\\text{ children}", "2", 1.0),
        ("0.5", "1/2", 1.0),
        ("0.5", "1", 0.0),
        ("9.1", "9", 0.0),
        ("[invalid]", "9", 0.0),
    ],
)
def test_gsm8k_minerva_grading_preserves_numeric_equivalence(response, reference, expected):
    doc = {"answer": f"reasoning\n#### {reference}"}

    assert process_results(doc, [response]) == {"exact_match": expected}


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (r"Start at \boxed{11:30} am.", "11:30"),
        (r"It costs \boxed{\$139}.", r"\$139"),
        (r"Each adult gets \boxed{\frac{1}{4}}.", r"\frac{1}{4}"),
    ],
)
def test_gsm8k_flexible_extraction_keeps_complete_final_box(response, expected):
    assert extract_gsm8k_flexible_answer(response) == expected


def test_gsm8k_task_scores_filtered_answer_with_minerva():
    config = load_yaml("eval/lm_eval_tasks/gsm8k/gsm8k.yaml")
    instance = Instance("generate_until", {}, (), 0)
    instance.resps = ["Reasoning used 3 and 6. Final answer: 9.0"]
    pipeline = build_filter_ensemble(
        "flexible-extract",
        [
            ("custom", {"filter_fn": gsm8k_flexible_extraction_filter}),
            ("take_first", {}),
        ],
    )
    pipeline.apply([instance])

    answer = instance.filtered_resps["flexible-extract"]
    assert config["process_results"]({"answer": "3 + 6 = 9\n#### 9"}, [answer]) == {"exact_match": 1.0}


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("The answer is \\boxed{2:00} pm.", 1.0),
        ("The answer is \\boxed{11:30} am.", 0.0),
    ],
)
def test_gsm8k_task_treats_whole_hour_clock_answer_as_reference_hour(response, expected):
    doc = {
        "question": "What is the latest time of day to start and finish by 5:00 pm?",
        "answer": "The latest start is 2:00 pm.\n#### 2",
    }
    prediction = extract_gsm8k_flexible_answer(response)

    assert process_results(doc, [prediction]) == {"exact_match": expected}


@pytest.mark.parametrize("prediction", ["\\frac{1}{4}", "1/4", "0.25"])
def test_gsm8k_task_accepts_fractional_form_for_percentage_question(prediction):
    doc = {
        "question": "What percentage of the watermelon does each adult get?",
        "answer": "Each adult gets 25 percent.\n#### 25",
    }

    assert process_results(doc, [prediction]) == {"exact_match": 1.0}


def test_gsm8k_task_accepts_fractional_form_for_pass_rate_question():
    doc = {
        "question": "What is the school's overall pass rate?",
        "answer": "The pass rate is 70 percent.\n#### 70",
    }

    assert process_results(doc, ["700/1000"]) == {"exact_match": 1.0}


@pytest.mark.parametrize("prediction", ["\\$600", "Friday", "Not enough information", "70"])
def test_gsm8k_task_rejects_wrong_box_even_if_reasoning_contains_reference(prediction):
    doc = {
        "question": "How many OLED TVs did the store sell?",
        "answer": "The store sold 25 OLED TVs.\n#### 25",
    }

    response = "Reasoning considered 25 TVs. Final answer: \\boxed{" + prediction + "}"
    extracted = extract_gsm8k_flexible_answer(response)

    assert process_results(doc, [extracted]) == {"exact_match": 0.0}
