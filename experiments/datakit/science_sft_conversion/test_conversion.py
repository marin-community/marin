# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for science SFT conversion output validation."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.chat_normalize import CHAT_SCHEMA, ChatChannel

from experiments.datakit.science_sft_conversion.audit import audit
from experiments.datakit.science_sft_conversion.conversion import (
    ConversionMode,
    Source,
    WorkItem,
    _document,
    _output_path,
    format_for,
    stratified_batches,
)


def validated_document(
    source: Source, source_id: str, passage: str, chunk_index: int, completion: dict, mode: ConversionMode
) -> dict:
    selected = format_for(source.name, source_id, chunk_index)
    return _document(source, source_id, passage, chunk_index, completion, selected, mode)


def test_bold_markdown_conclusion_is_valid() -> None:
    source = Source("probe/physics", "", 0, 0)
    passage = "A 2 kg object experiences a 6 N force for 3 seconds, starting from rest."
    completion = {
        "user": "Calculate the final speed and present the result in Markdown bullets.",
        "reasoning_content": "F = ma gives a = 3 m/s²; v = at gives 9 m/s.",
        "answer": "- Mass: 2 kg\n- Force: 6 N\n\n**Conclusion:** Final speed is 9 m/s.",
    }

    record = validated_document(source, "test-2", passage, 0, completion, ConversionMode.GROUNDED)

    assert passage in record["messages"][0]["content"][0]["text"]
    assert record["messages"][1]["channel"] == ChatChannel.ANALYSIS
    assert record["messages"][1]["content"][0]["text"] == completion["reasoning_content"]
    assert record["messages"][2]["channel"] == ChatChannel.FINAL
    assert record["messages"][2]["content"][0]["text"] == completion["answer"]

    with pytest.raises(ValueError):
        validated_document(
            source, "test-2", passage, 0, {**completion, "answer": "- Mass: 2 kg\n- Force: 6 N"}, ConversionMode.GROUNDED
        )


def test_worked_solution_stays_out_of_the_user_turn() -> None:
    source = Source("swallow-math-v2/qa", "", 0, 0)
    passage = "Question: What is 2 + 2?\nAnswer: 4."
    completion = {
        "user": "What is 2 + 2? Give a short answer and then explain.",
        "reasoning_content": "Adding two and two gives four.",
        "answer": "Short answer: 4.\nAdding two and two gives four.",
    }

    record = validated_document(source, "test-2", passage, 0, completion, ConversionMode.STANDALONE)

    assert passage not in record["messages"][0]["content"][0]["text"]
    assert "What is 2 + 2?" in record["messages"][0]["content"][0]["text"]
    grounded = validated_document(source, "test-2", passage, 0, completion, ConversionMode.GROUNDED)
    assert passage in grounded["messages"][0]["content"][0]["text"]
    with pytest.raises(ValueError, match="omitted from the user turn"):
        validated_document(
            source,
            "test-2",
            passage,
            0,
            {**completion, "user": "Using the source passage, solve 2 + 2."},
            ConversionMode.STANDALONE,
        )
    with pytest.raises(ValueError, match="omitted from the user turn"):
        validated_document(
            source,
            "test-2",
            passage,
            0,
            {**completion, "reasoning_content": "The passage says 2 + 2 = 4."},
            ConversionMode.STANDALONE,
        )
    with pytest.raises(ValueError, match="withhold its solution"):
        validated_document(
            source,
            "test-2",
            passage,
            0,
            {**completion, "user": "Set up 2 + 2, but do not solve it."},
            ConversionMode.STANDALONE,
        )
    with pytest.raises(ValueError, match="create an exercise"):
        validated_document(
            source,
            "test-2",
            passage,
            0,
            {**completion, "user": "Construct a self-contained exercise on addition. Then solve it."},
            ConversionMode.STANDALONE,
        )
    with pytest.raises(ValueError, match="withhold its solution"):
        validated_document(
            source,
            "test-2",
            passage,
            0,
            {**completion, "user": "What is 2 + 2? Do not provide the final answer."},
            ConversionMode.STANDALONE,
        )


def test_numbered_worked_solution_accepts_step_headings() -> None:
    source = Source("nemotron_specialized/math_textbooks", "", 0, 0)
    completion = {
        "user": "A coin has heads probability b or 1-b, where b > 1/2. Derive the posterior odds after one head.",
        "reasoning_content": "Use Bayes' theorem to update the prior odds.",
        "answer": "**Step 1 — Apply Bayes' theorem.**\nThe odds multiply by b/(1-b).\nFinal answer: updated odds.",
    }

    record = validated_document(
        source,
        "00001a321e0eff9c465f857eed9de1ee",
        "Worked answer: updated odds.",
        0,
        completion,
        ConversionMode.STANDALONE,
    )

    assert "Worked answer" not in record["messages"][0]["content"][0]["text"]
    assert record["messages"][2]["content"][0]["text"] == completion["answer"]


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


def test_early_conversion_covers_every_source_without_losing_batches() -> None:
    sources = [Source(f"source-{index}", "", 1, rows) for index, rows in enumerate([100000, 1024] + [3079] * 15)]
    items = [WorkItem(source, f"{source.name}.parquet", 0, source.rows) for source in sources]

    scheduled = stratified_batches(items, seed=20260927, max_batches=None)

    assert {batch.item.source.name for batch in scheduled[:17]} == {source.name for source in sources}
    assert len(scheduled) == 159
    identities = {(batch.item.url, batch.item.row_group, batch.batch_index) for batch in scheduled}
    assert len(identities) == 159
    assert {batch.batch_index for batch in scheduled if batch.item.source.name == "source-0"} == set(range(98))
    assert len([batch for batch in scheduled if batch.item.source.name == "source-1"]) == 1
    assert scheduled == stratified_batches(items, seed=20260927, max_batches=None)
