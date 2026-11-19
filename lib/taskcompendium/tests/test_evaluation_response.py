# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from taskcompendium.evaluation.response import (
    InlineReasoning,
    ReasoningDelimiters,
    completion_text,
    inline_reasoning_message,
    split_inline_reasoning,
)

GRUG = ReasoningDelimiters("<|start_think|>", "<|end_think|>")


@pytest.mark.parametrize("reasoning_field", ["reasoning_content", "reasoning"])
@pytest.mark.parametrize(
    "content,stop_reason,expected",
    [
        ("Paris", "stop", "Paris"),
        # A complete reasoning trace is still not graded when the final answer is missing.
        ("", "stop", ""),
        ("", "length", ""),
    ],
)
def test_final_only_never_returns_reasoning(reasoning_field, content, stop_reason, expected):
    message = {"role": "assistant", "content": content, reasoning_field: "The capital of France is Paris."}
    assert completion_text(message, stop_reason, "final_only") == expected


@pytest.mark.parametrize(
    "text,prompt_opens_span,reasoning,content,closed",
    [
        ("<|start_think|>Count the twos.<|end_think|>5", False, "Count the twos.", "5", True),
        # Models may skip the opener; the text before the closer is still reasoning.
        ('Output should be "5".<|end_think|>5', False, 'Output should be "5".', "5", True),
        # A template that ends the prompt with the opener yields a reply with only a closer.
        ("Count the twos.<|end_think|>5", True, "Count the twos.", "5", True),
        # A length stop inside the prompt-opened span is unfinished reasoning, never the answer.
        ("Count the twos: 1, 2, 3", True, "Count the twos: 1, 2, 3", "", False),
        ("<|start_think|>Count the twos: 1, 2", False, "Count the twos: 1, 2", "", False),
        # Only text after the last closer is the answer.
        ("A<|end_think|>draft<|start_think|>B<|end_think|>5", False, "AdraftB", "5", True),
    ],
)
def test_split_inline_reasoning(text, prompt_opens_span, reasoning, content, closed):
    split = split_inline_reasoning(text, GRUG, prompt_opens_span=prompt_opens_span)
    assert split == InlineReasoning(reasoning, content, closed)


def test_reply_without_reasoning_is_not_split():
    assert split_inline_reasoning("5", GRUG, prompt_opens_span=False) is None


@pytest.mark.parametrize(
    "text,final,combined",
    [
        ("Think.<|end_think|>5", "5", "<|start_think|>Think.<|end_think|>5"),
        ("Think more", "", "<|start_think|>Think more"),
    ],
)
def test_inline_reasoning_selection(text, final, combined):
    split = split_inline_reasoning(text, GRUG, prompt_opens_span=True)
    assert split is not None
    message = inline_reasoning_message({"role": "assistant", "content": text}, split, GRUG)
    assert completion_text(message, "length", "final_only") == final
    assert completion_text(message, "length", "final_or_complete_reasoning") == final
    # Combine keeps one well-formed span, so scorers can still find the answer after the closer.
    assert completion_text(message, "length", "combine") == combined
