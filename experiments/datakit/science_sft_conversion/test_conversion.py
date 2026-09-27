# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for science SFT conversion output validation."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.chat_normalize import CHAT_SCHEMA, ChatChannel

from experiments.datakit.science_sft_conversion.audit import audit
from experiments.datakit.science_sft_conversion.conversion import Source, WorkItem, _document, _output_path


def test_bold_markdown_conclusion_is_valid() -> None:
    source = Source("probe/physics", "", 0, 0)
    passage = "A 2 kg object experiences a 6 N force for 3 seconds, starting from rest."
    completion = {
        "user": "Calculate the final speed and present the result in Markdown bullets.",
        "reasoning_content": "F = ma gives a = 3 m/s²; v = at gives 9 m/s.",
        "answer": "- Mass: 2 kg\n- Force: 6 N\n\n**Conclusion:** Final speed is 9 m/s.",
    }

    record = _document(source, "test-2", passage, 0, completion)

    assert passage in record["messages"][0]["content"][0]["text"]
    assert record["messages"][1]["channel"] == ChatChannel.ANALYSIS
    assert record["messages"][1]["content"][0]["text"] == completion["reasoning_content"]
    assert record["messages"][2]["channel"] == ChatChannel.FINAL
    assert record["messages"][2]["content"][0]["text"] == completion["answer"]

    with pytest.raises(ValueError):
        _document(source, "test-2", passage, 0, {**completion, "answer": "- Mass: 2 kg\n- Force: 6 N"})


def test_worked_solution_stays_out_of_the_user_turn() -> None:
    source = Source("swallow-math-v2/qa", "", 0, 0)
    passage = "Question: What is 2 + 2?\nAnswer: 4."
    completion = {
        "user": "What is 2 + 2? Give a short answer and then explain.",
        "reasoning_content": "Adding two and two gives four.",
        "answer": "Short answer: 4.\nAdding two and two gives four.",
    }

    record = _document(source, "test-2", passage, 0, completion)

    assert passage not in record["messages"][0]["content"][0]["text"]
    assert "What is 2 + 2?" in record["messages"][0]["content"][0]["text"]
    with pytest.raises(ValueError, match="omitted from the user turn"):
        _document(source, "test-2", passage, 0, {**completion, "user": "Using the source passage, solve 2 + 2."})
    with pytest.raises(ValueError, match="omitted from the user turn"):
        _document(source, "test-2", passage, 0, {**completion, "reasoning_content": "The passage says 2 + 2 = 4."})
    with pytest.raises(ValueError, match="withhold its solution"):
        _document(source, "test-2", passage, 0, {**completion, "user": "Set up 2 + 2, but do not solve it."})


def test_audit_detects_short_batch(tmp_path) -> None:
    source = Source("probe/physics", "", 0, 0)
    work = WorkItem(source, "unused.parquet", 0, 2)
    output = Path(_output_path(source, work.url, work.row_group, 0, str(tmp_path)))
    output.parent.mkdir(parents=True)
    record = {"id": "one", "messages": []}

    pq.write_table(pa.Table.from_pylist([record], schema=CHAT_SCHEMA), output)
    result = audit([work], str(tmp_path), workers=1)
    assert result.short_batches == 1
    assert result.missing_batches == 0

    pq.write_table(pa.Table.from_pylist([record, {**record, "id": "two"}], schema=CHAT_SCHEMA), output)
    result = audit([work], str(tmp_path), workers=1)
    assert result.complete
    assert result.source_rows == result.output_rows == 2
