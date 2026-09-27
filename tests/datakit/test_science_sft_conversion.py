# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for science SFT conversion output validation."""

import pytest
from marin.datakit.chat_normalize import ChatChannel

from experiments.datakit.science_sft_conversion.conversion import Source, _document


def test_bold_markdown_conclusion_is_valid() -> None:
    source = Source("probe/physics", "", 0, 0)
    completion = {
        "user": "Calculate the final speed and present the result in Markdown bullets.",
        "reasoning_content": "F = ma gives a = 3 m/s²; v = at gives 9 m/s.",
        "answer": "- Mass: 2 kg\n- Force: 6 N\n\n**Conclusion:** Final speed is 9 m/s.",
    }

    record = _document(source, "test-2", 0, completion)

    assert record["messages"][1]["channel"] == ChatChannel.ANALYSIS
    assert record["messages"][1]["content"][0]["text"] == completion["reasoning_content"]
    assert record["messages"][2]["channel"] == ChatChannel.FINAL
    assert record["messages"][2]["content"][0]["text"] == completion["answer"]

    with pytest.raises(ValueError, match="Bullet answer lacks bullets or a conclusion"):
        _document(source, "test-2", 0, {**completion, "answer": "- Mass: 2 kg\n- Force: 6 N"})
