# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for science SFT conversion output validation."""

from experiments.datakit.science_sft_conversion.conversion import Source, _document


def test_bold_markdown_conclusion_is_valid() -> None:
    source = Source("probe/physics", "", 0, 0)
    completion = {
        "user": "Calculate the final speed and present the result in Markdown bullets.",
        "reasoning_content": "F = ma gives a = 3 m/s²; v = at gives 9 m/s.",
        "answer": "- Mass: 2 kg\n- Force: 6 N\n\n**Conclusion:** Final speed is 9 m/s.",
    }

    record = _document(source, "test-2", 0, completion)

    assert len(record["messages"]) == 3
