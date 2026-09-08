"""Frozen wire prompts must retain their answer instructions and thinking prefix."""

import hashlib
from dataclasses import replace

import pytest

from experiments.post_training.math_eval.contract import QWEN, SNOWBALL, render_prompt


@pytest.mark.parametrize(
    "template, env_class, expected",
    [
        (QWEN, "gsm8k", "a04c6c2f4a096049c6ee8eff0a78e82b39b984df7cc34f7a697c4c067dd340b3"),
        (QWEN, "aime", "61d58fafd743bf3e1ec3a19aa975e77caeff64018c105e2f6d95bc8506461e66"),
        (QWEN, "reasoning_gym", "61d58fafd743bf3e1ec3a19aa975e77caeff64018c105e2f6d95bc8506461e66"),
        (SNOWBALL, "gsm8k", "4fdf999d034dccb2ccc9cba072dafcf95a3665b77c9430c26774c22cc19f6c15"),
        (SNOWBALL, "aime", "256e10817333acac9602e2de571f0ee54f81677ceb3fbe55f303a5a8df939ebe"),
        (SNOWBALL, "reasoning_gym", "256e10817333acac9602e2de571f0ee54f81677ceb3fbe55f303a5a8df939ebe"),
    ],
)
def test_frozen_prompt_rendering_matches_recorded_wire_hash(template, env_class, expected):
    assert hashlib.sha256(render_prompt("What is 2 + 3?", env_class, template).encode()).hexdigest() == expected


def test_changed_template_is_rejected_before_rendering():
    with pytest.raises(ValueError, match="Template content changed"):
        render_prompt("What is 2 + 3?", "gsm8k", replace(QWEN, sha256="0" * 64))


def test_thinking_mode_change_cannot_silently_change_qwen_prefix():
    with pytest.raises(ValueError, match="Generation prefix changed"):
        render_prompt("What is 2 + 3?", "gsm8k", replace(QWEN, enable_thinking=True))
