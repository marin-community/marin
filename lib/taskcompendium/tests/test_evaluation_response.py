# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from taskcompendium.evaluation.response import completion_text


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
