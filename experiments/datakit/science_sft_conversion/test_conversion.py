# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for science SFT conversion output validation."""

import asyncio
import json
import re
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.chat_normalize import CHAT_SCHEMA, ChatChannel

from experiments.datakit.science_sft_conversion import conversion, probe
from experiments.datakit.science_sft_conversion.audit import audit
from experiments.datakit.science_sft_conversion.conversion import (
    ConversionMode,
    Source,
    WorkBatch,
    WorkItem,
    _document,
    _output_path,
    convert_work_batches,
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


@pytest.mark.parametrize(
    "field, leaked_text",
    [
        ("reasoning_content", "The reasoning_content simply lists these source-grounded points."),
        ("reasoning_content", "The conversion task requires reporting those supplied references."),
        ("answer", "The answer field will report these explicitly stated facts."),
    ],
)
def test_conversion_process_leaks_are_rejected_without_blocking_unit_conversion(field, leaked_text) -> None:
    source = Source("probe/physics", "", 0, 0)
    passage = "One metre is 100 centimetres."
    completion = {
        "user": "Convert two metres to centimetres.",
        "reasoning_content": "The unit conversion process uses 100 centimetres per metre, so 2 * 100 = 200 centimetres.",
        "answer": "- 2 metres * 100 centimetres/metre = 200 centimetres.\nConclusion: 200 centimetres.",
    }
    with pytest.raises(ValueError, match="describes the conversion process"):
        validated_document(source, "test-2", passage, 0, {**completion, field: leaked_text}, ConversionMode.GROUNDED)

    record = validated_document(source, "test-2", passage, 0, completion, ConversionMode.GROUNDED)
    assert record["messages"][1]["content"][0]["text"] == completion["reasoning_content"]
    assert record["messages"][2]["content"][0]["text"] == completion["answer"]


@pytest.mark.parametrize("root_expression", ["√(2E/k)", "(2E/k)^0.5", "(2E/k)**(1/2)"])
def test_grounded_math_rejects_missing_radicals_but_standalone_exercises_can_derive_them(root_expression) -> None:
    source = Source("probe/physics", "", 0, 0)
    completion = {
        "user": "Report the stated amplitude of the oscillation.",
        "reasoning_content": f"The reported amplitude is {root_expression}.",
        "answer": f"- Amplitude: {root_expression}.\nConclusion: {root_expression}.",
    }
    with pytest.raises(ValueError, match="square-root notation absent"):
        validated_document(source, "test-2", "Amplitude (2E/k).", 0, completion, ConversionMode.GROUNDED)

    record = validated_document(source, "test-2", "Amplitude sqrt(2E/k).", 0, completion, ConversionMode.GROUNDED)
    assert record["messages"][2]["content"][0]["text"] == completion["answer"]
    exact_source = validated_document(
        source, "test-2", f"Amplitude {root_expression}.", 0, completion, ConversionMode.GROUNDED
    )
    assert exact_source["messages"][2]["content"][0]["text"] == completion["answer"]
    ambiguous = validated_document(
        source,
        "test-2",
        "Amplitude (2E/k).",
        0,
        {
            "user": "Quote the stated amplitude and flag missing notation.",
            "reasoning_content": "The extraction does not show square roots, so the original notation is ambiguous.",
            "answer": "- The extracted amplitude is (2E/k).\nConclusion: The square-root notation is ambiguous.",
        },
        ConversionMode.GROUNDED,
    )
    assert "square roots" in ambiguous["messages"][1]["content"][0]["text"]
    assert "(2E/k)" in ambiguous["messages"][2]["content"][0]["text"]
    exercise = validated_document(
        source,
        "test-2",
        "Energy E = k A² / 2; solve for positive A.",
        0,
        {**completion, "user": "Energy E = k A² / 2. Solve for positive A."},
        ConversionMode.STANDALONE,
    )
    assert exercise["messages"][2]["content"][0]["text"] == completion["answer"]


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
    input_path = tmp_path / "input.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"id": "one", "text": "First row."}, {"id": "two", "text": "Second row."}]), input_path
    )
    work = WorkItem(source, str(input_path), 0, 2)
    output = Path(_output_path(source, work.url, work.row_group, 0, str(tmp_path / "output")))
    output.parent.mkdir(parents=True)
    record = {"id": "one", "source_id": "probe/physics:one:0", "messages": []}

    pq.write_table(pa.Table.from_pylist([record], schema=CHAT_SCHEMA), output)
    result = audit([work], str(tmp_path / "output"), workers=1)
    assert result.short_batches == 1
    assert result.missing_batches == 0

    pq.write_table(
        pa.Table.from_pylist([record, {**record, "id": "two", "source_id": "probe/physics:two:0"}], schema=CHAT_SCHEMA),
        output,
    )
    result = audit([work], str(tmp_path / "output"), workers=1)
    assert result.complete
    assert result.source_rows == result.output_rows == 2


def test_audit_rejects_duplicate_replacing_missing_chunk(tmp_path) -> None:
    source = Source("probe/physics", "", 1, 2)
    input_path = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": "long", "text": "a" * 8001}, {"id": "short", "text": "b"}]), input_path)
    work = WorkItem(source, str(input_path), 0, 2)
    output_root = str(tmp_path / "output")
    output = Path(_output_path(source, work.url, 0, 0, output_root))
    output.parent.mkdir(parents=True)
    records = [
        {"id": "long-0", "source_id": "probe/physics:long:0", "messages": []},
        {"id": "long-1", "source_id": "probe/physics:long:1", "messages": []},
        {"id": "short-0", "source_id": "probe/physics:short:0", "messages": []},
    ]
    pq.write_table(pa.Table.from_pylist(records, schema=CHAT_SCHEMA), output)
    assert audit([work], output_root, workers=1).complete

    records[1] = records[0]
    pq.write_table(pa.Table.from_pylist(records, schema=CHAT_SCHEMA), output)
    result = audit([work], output_root, workers=1)
    assert result.output_rows == 3
    assert result.short_batches == result.missing_batches == result.wrong_schemas == 0
    assert result.wrong_lineage_batches == 1
    assert not result.complete


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


def test_probe_samples_rows_beyond_first_parquet_batch(tmp_path, monkeypatch) -> None:
    source = Source("probe/physics", "", 1, 512)
    input_path = tmp_path / "input.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"id": str(index), "text": f"Passage {index}."} for index in range(512)]), input_path
    )
    # Replace object-store discovery with one real local Parquet shard.
    monkeypatch.setattr(probe, "_source_files", lambda _: [str(input_path)])

    samples = probe._source_samples(source, count=8, seed=20260927)
    row_ids = [int(sample[1]) for sample in samples]

    assert row_ids[0] == 0
    assert len(set(row_ids)) == 8
    assert any(row_id >= 128 for row_id in row_ids)
    assert [sample[2] for sample in samples] == [f"Passage {row_id}." for row_id in row_ids]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "passage, evidence",
    [
        (
            "# Why This Chapter?\n\nAmplitude (2E/k).\n\nEnergy is conserved.\n\n"
            "Motion is periodic.\n\nThe spring stretches.",
            ["Amplitude (2E/k).", "Energy is conserved.", "Motion is periodic.", "The spring stretches."],
        ),
        ("# Amplitude (2E/k).", ["# Amplitude (2E/k)."]),
    ],
)
async def test_rejected_math_persists_verbatim_evidence_in_every_answer_format(
    tmp_path, monkeypatch, passage, evidence
) -> None:
    monkeypatch.setattr(conversion, "MAX_ATTEMPTS", 1)
    source = Source("probe/physics", "", 1, 6)
    row_ids = {}
    for index in range(1000):
        row_id = f"row-{index}"
        row_ids.setdefault(format_for(source.name, row_id, 0).name, row_id)
        if len(row_ids) == 6:
            break
    assert len(row_ids) == 6
    input_path = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": row_id, "text": passage} for row_id in row_ids.values()]), input_path)
    output_root = tmp_path / "output"
    (output_root / "outputs/main").mkdir(parents=True)
    work = [WorkBatch(WorkItem(source, str(input_path), 0, 6), 0)]
    reasoning = (
        "The supplied expression lacks an explicit root. "
        "Quoting it preserves the visible evidence without restoring notation."
    )

    async def response(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        schema = body["response_format"]["json_schema"]
        if schema["name"] == "source_evidence":
            paragraphs = json.loads(body["messages"][1]["content"])
            indices = [paragraph["index"] for paragraph in paragraphs if paragraph["paragraph"] in evidence]
            completion = {"paragraph_indices": indices, "reasoning_content": reasoning}
        else:
            answer = "Answer: √(2E/k)."
            if schema["schema"]["properties"]["answer"]["type"] == "object":
                answer = {"answer": "√(2E/k).", "evidence": [], "caveats": []}
            completion = {
                "user": "Report the supplied amplitude.",
                "reasoning_content": "The amplitude is √(2E/k).",
                "answer": answer,
            }
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(completion)}}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        await convert_work_batches(
            work, "http://test", client, concurrency=2, concurrent_batches=1, output_root=str(output_root)
        )

    records = pq.read_table(_output_path(source, str(input_path), 0, 0, str(output_root))).to_pylist()
    assert {record["source_id"] for record in records} == {f"{source.name}:{row_id}:0" for row_id in row_ids.values()}
    for record in records:
        assert passage in record["messages"][0]["content"][0]["text"]
        assert record["messages"][1]["channel"] == ChatChannel.ANALYSIS
        assert record["messages"][1]["content"][0]["text"] == reasoning
        answer = record["messages"][2]["content"][0]["text"]
        assert all(paragraph in answer for paragraph in evidence)
        assert "√" not in answer
        if record["source_id"] == f"{source.name}:{row_ids['json']}:0":
            assert json.loads(answer)["evidence"] == evidence


@pytest.mark.asyncio
async def test_later_batch_persists_while_first_response_waits(tmp_path) -> None:
    source = Source("probe/physics", "", 1, 8)
    input_path = tmp_path / "input.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"id": f"row-{index}", "text": f"row-{index}"} for index in range(8)]),
        input_path,
        row_group_size=4,
    )
    work = [WorkBatch(WorkItem(source, str(input_path), index, 4), 0) for index in range(2)]
    output_root = tmp_path / "output"
    (output_root / "outputs/main").mkdir(parents=True)
    release_first = asyncio.Event()
    active = 0
    peak_active = 0
    answers = {
        "paragraphs": "Explanation.\nAnswer: 2.",
        "numbered": "1. Add the numbers.\nFinal answer: 2.",
        "bullets": "- Add the numbers.\nConclusion: 2.",
        "short_then_detail": "Short answer: 2.\nAdd the numbers.",
        "table": "| Item | Value |\n| --- | --- |\n| Answer | 2 |",
        "json": {"answer": "2", "evidence": ['Addition: \\alpha + \\alpha\nA quoted "symbol".'], "caveats": []},
    }

    async def response(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak_active
        body = json.loads(request.content)
        prompt = body["messages"][1]["content"]
        match = re.search(r"<source_passage>\n(row-\d+)", prompt)
        assert match is not None
        row_id = match.group(1)
        selected = format_for(source.name, row_id, 0)
        active += 1
        peak_active = max(peak_active, active)
        try:
            if row_id == "row-0":
                await release_first.wait()
            else:
                # Let concurrent HTTP handlers run before completing this response.
                await asyncio.sleep(0)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {
                                        "user": "Add 1 and 1.",
                                        "reasoning_content": "1 + 1 = 2.",
                                        "answer": answers[selected.name],
                                    }
                                )
                            },
                        }
                    ]
                },
            )
        finally:
            active -= 1

    first_output = Path(_output_path(source, str(input_path), 0, 0, str(output_root)))
    second_output = Path(_output_path(source, str(input_path), 1, 0, str(output_root)))
    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        task = asyncio.create_task(
            convert_work_batches(
                work, "http://test", client, concurrency=2, concurrent_batches=2, output_root=str(output_root)
            )
        )

        async def wait_for_second_output() -> None:
            while not await asyncio.to_thread(second_output.exists):
                await asyncio.sleep(0)

        try:
            await asyncio.wait_for(wait_for_second_output(), timeout=5)
            assert not first_output.exists()
            assert {row["source_id"] for row in pq.read_table(second_output).to_pylist()} == {
                f"{source.name}:row-{index}:0" for index in range(4, 8)
            }
        finally:
            release_first.set()
            await task

    assert peak_active == 2
    assert {row["source_id"] for row in pq.read_table(first_output).to_pylist()} == {
        f"{source.name}:row-{index}:0" for index in range(4)
    }
    records = pq.read_table(first_output).to_pylist() + pq.read_table(second_output).to_pylist()
    json_records = [
        record for record in records if format_for(source.name, record["source_id"].split(":")[-2], 0).name == "json"
    ]
    assert json_records
    for record in json_records:
        assert json.loads(record["messages"][2]["content"][0]["text"]) == answers["json"]


@pytest.mark.asyncio
async def test_failed_batch_cancels_pending_response_without_writing_output(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(conversion, "MAX_ATTEMPTS", 1)
    source = Source("probe/physics", "", 1, 4)
    input_path = tmp_path / "input.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"id": text, "text": text} for text in ("failed", "waiting-a", "waiting-b", "waiting-c")]),
        input_path,
        row_group_size=2,
    )
    work = [WorkBatch(WorkItem(source, str(input_path), index, 2), 0) for index in range(2)]
    output_root = tmp_path / "output"
    (output_root / "outputs/main").mkdir(parents=True)
    waiting_started = asyncio.Event()
    started: set[str] = set()
    cancelled: set[str] = set()

    async def response(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][1]["content"]
        text = prompt.split("<source_passage>\n", 1)[1].split("\n</source_passage>", 1)[0]
        if text == "failed":
            await waiting_started.wait()
            return httpx.Response(400, json={"error": "request rejected"})
        started.add(text)
        if len(started) == 3:
            waiting_started.set()
        try:
            await asyncio.Event().wait()
            raise AssertionError("Pending response should be cancelled")
        finally:
            cancelled.add(text)

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        with pytest.raises(ExceptionGroup, match="unhandled errors in a TaskGroup") as errors:
            await asyncio.wait_for(
                convert_work_batches(
                    work, "http://test", client, concurrency=4, concurrent_batches=2, output_root=str(output_root)
                ),
                timeout=5,
            )
    assert errors.value.subgroup(RuntimeError) is not None
    assert cancelled == {"waiting-a", "waiting-b", "waiting-c"}
    assert not list(output_root.rglob("*.parquet"))
