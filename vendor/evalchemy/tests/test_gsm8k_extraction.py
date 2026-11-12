"""Regression coverage for GSM8K's flexible answer filter."""

import pytest

from eval.lm_eval_tasks.gsm8k.utils import gsm8k_flexible_extraction_filter
from eval.graders.gsm8k import extract_flexible
from lm_eval.api.instance import Instance
from lm_eval.filters import build_filter_ensemble


def _flexible_extract(response: str) -> tuple[Instance, str]:
    instance = Instance("generate_until", {}, (), 0)
    instance.resps = [response]
    pipeline = build_filter_ensemble(
        "flexible-extract",
        [
            ("custom", {"filter_fn": gsm8k_flexible_extraction_filter}),
            ("take_first", {}),
        ],
    )
    pipeline.apply([instance])
    return instance, instance.filtered_resps["flexible-extract"]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("$$\n\\boxed{18}\n$$", "18"),
        ("$18", "$18"),
        ("Answer: 18.", "18"),
        ("The calculation used 9 and 2. Final answer: 18.", "18"),
        ("\\boxed{-3.5}", "-3.5"),
        ("\\boxed{1,200}", "1,200"),
        ("\\boxed{18}. The calculation used 9 and 2.", "18"),
        ("The answer is 12.\nQuestion: unrelated question with 8088 rows", "12"),
        ("The answer is 12.\nQ: unrelated question with 8088 rows", "12"),
        ("The answer is 12.\n[Question] unrelated question with 8088 rows", "12"),
        ("$$", "[invalid]"),
    ],
)
def test_flexible_extract_uses_final_answer_syntax_before_numeric_fallback(response, expected):
    instance, selected = _flexible_extract(response)

    assert instance.resps == [response]
    assert selected == expected


def test_shared_gsm8k_extractor_uses_the_next_turn_boundary():
    response = "The answer is 12.\nQ: unrelated question with 8088 rows"

    assert extract_flexible(response) == "12."
