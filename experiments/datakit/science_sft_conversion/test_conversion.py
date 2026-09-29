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
from marin.inference.openai_batch import BatchOutput, BatchSubmission

from experiments.datakit.science_sft_conversion import batch_transport, conversion, probe
from experiments.datakit.science_sft_conversion.audit import audit
from experiments.datakit.science_sft_conversion.batch_transport import GLMBatchChatClient
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


def test_glm_batch_transport_routes_out_of_order_responses_and_retries_missing_results(monkeypatch) -> None:
    class FakeBatchClient:
        def __init__(self, base_url: str, token: str, priority: str) -> None:
            assert base_url == "http://relay/bulk/v1"
            assert token == "batch-token"
            assert priority == "batch"
            self.requests: list[dict] = []

        def submit(self, requests: list[dict], filename: str) -> BatchSubmission:
            self.requests = requests
            return BatchSubmission("file-1", "batch-1")

        def wait(self, batch_id: str, poll_seconds: float, timeout_seconds: float) -> dict:
            return {"status": "completed"}

        def output(self, batch: dict) -> BatchOutput:
            second, first, _missing = self.requests
            rows = [
                {"custom_id": item["custom_id"], "response": {"status_code": 200, "body": {"answer": answer}}}
                for item, answer in ((first, "second"), (second, "first"))
            ]
            return BatchOutput("".join(json.dumps(row) + "\n" for row in rows), None)

    monkeypatch.setattr(batch_transport, "OpenAIBatchClient", FakeBatchClient)

    async def run() -> list[httpx.Response]:
        async with GLMBatchChatClient("http://relay", "batch-token", batch_size=3, workers=1) as client:
            return await asyncio.gather(
                *(client.post("http://relay/v1/chat/completions", json={"request": index}) for index in range(3))
            )

    first, second, missing = asyncio.run(run())
    assert first.json() == {"answer": "first"}
    assert second.json() == {"answer": "second"}
    assert missing.status_code == 502
    with pytest.raises(httpx.HTTPStatusError):
        missing.raise_for_status()


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


@pytest.mark.parametrize(
    "question, reasoning, answer",
    [
        (
            'Cell F12 contains the text string "February09", a named range. '
            'Explain why DSUM(INDIRECT(F12), "Profit", H1:H2) resolves that named range.',
            'INDIRECT converts the text "February09" into a range reference.',
            'Answer: INDIRECT resolves the text "February09" into the named range used by DSUM.',
        ),
        (
            'A1 contains "20240315". Use the text-parsing formula '
            "DATE(LEFT(A1,4), MID(A1,5,2), RIGHT(A1,2)) to convert the text in A1 into a date.",
            "LEFT extracts 2024, MID extracts 03, and RIGHT extracts 15.",
            "Answer: DATE constructs March 15, 2024.",
        ),
        (
            "A state enacted strict gun-control laws. Its violent-crime rate has decreased since "
            "the passage of those laws. Does this weaken the claim that repealing the laws would reduce crime?",
            "The passage of those laws preceded a decrease in violent crime. That weakens the repeal claim.",
            "Answer: Yes; the decrease is evidence against the stated claim.",
        ),
    ],
)
def test_teacher_exercises_accept_text_data_and_legislative_passage(question, reasoning, answer) -> None:
    # Production exhausted all formats on these ordinary uses of "text" and "passage".
    source = Source(conversion.SWALLOW_MATH_TEXTBOOK, "", 0, 0)
    private_reference = "Private teacher reference with a worked solution."
    completion = {"user": question, "reasoning_content": reasoning, "answer": answer}
    selected = next(f for f in conversion.FORMATS if f.name == "paragraphs")

    record = _document(source, "regression", private_reference, 0, completion, selected, ConversionMode.TEACHER_EXERCISE)

    assert record["messages"][0]["content"][0]["text"] == f"{question}\n\n{selected.instruction}"
    assert record["messages"][1]["content"][0]["text"] == reasoning
    assert record["messages"][2]["content"][0]["text"] == answer


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
        ("\nAmplitude (2E/k).\n", ["\nAmplitude (2E/k).\n"]),
        ("Amplitude (2E/k).\n\nEnergy is conserved.", ["Amplitude (2E/k).", "Energy is conserved."]),
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
            if len(paragraphs) == 2:
                indices = [0, 0]
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
        expected_evidence = [paragraph.strip() for paragraph in evidence]
        if record["source_id"] == f"{source.name}:{row_ids['json']}:0":
            fields = json.loads(answer)
            assert fields["evidence"] == evidence
            answer = fields["answer"]
        if record["source_id"] == f"{source.name}:{row_ids['table']}:0":
            expected_evidence = [paragraph.replace("\n", "<br>") for paragraph in evidence]
        assert all(paragraph in answer for paragraph in expected_evidence)
        assert "√" not in answer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_name,passage,question,reasoning,answer",
    [
        (
            conversion.BIO_INSTRUCTION,
            "Question: Complement <dna>ATGC</dna> using A-T and C-G pairing. Answer: TACG.",
            "Complement the DNA sequence <dna>ATGC</dna> using A-T and C-G pairing.",
            "Pair A with T, T with A, G with C, and C with G to obtain TACG.",
            "TACG",
        ),
        (
            conversion.BIO_INSTRUCTION,
            "Question: Locate the annotated CDS in <rna>CCAUGAAAUAA</rna>. " 'Reference: {"start":3,"end":8}.',
            "Locate the annotated CDS in <rna>CCAUGAAAUAA</rna>.",
            "The reference CDS is 3-8. Its first triplet is AUG and its last is AAA; "
            "sequence positions can be checked, but an independent annotation is not provided.",
            "Reference CDS 3-8",
        ),
        (
            conversion.SWALLOW_MATH_TEXTBOOK,
            "Theory: F = ma. Example: m = 2 kg, a = 3 m/s², so F = 6 N.",
            "Using F = ma, find the force for m = 2 kg and a = 3 m/s².",
            "Multiply the mass by the acceleration: 2 times 3 gives 6 N.",
            "6 N",
        ),
    ],
)
async def test_teacher_exercise_keeps_reference_private_after_rejected_response(
    tmp_path, monkeypatch, source_name, passage, question, reasoning, answer
) -> None:
    monkeypatch.setattr(conversion, "MAX_ATTEMPTS", 1)
    source = Source(source_name, "", 1, 1)
    input_path = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": "teacher-row", "text": passage}]), input_path)
    output_root = tmp_path / "output"
    (output_root / "outputs/main").mkdir(parents=True)
    work = [WorkBatch(WorkItem(source, str(input_path), 0, 1), 0)]
    first_response = True

    async def response(request: httpx.Request) -> httpx.Response:
        nonlocal first_response
        body = json.loads(request.content)
        assert passage in body["messages"][1]["content"]
        if "CCAUGAAAUAA" in passage:
            prompt = body["messages"][1]["content"]
            assert "First AUG occurrence: position 3" in prompt
            assert "first three=AUG, last three=AAA" in prompt
        selected = next(f for f in conversion.FORMATS if f"({f.name}):" in body["messages"][1]["content"])
        formatted_answers = {
            "paragraphs": f"Answer: {answer}",
            "numbered": f"1. {reasoning}\nFinal answer: {answer}",
            "bullets": f"- {reasoning}\nConclusion: {answer}",
            "short_then_detail": f"Short answer: {answer}\n{reasoning}",
            "table": f"| Result | Value |\n|---|---|\n| Answer | {answer} |\nConclusion: {answer}",
            "json": {"answer": answer, "evidence": [reasoning], "caveats": []},
        }
        completion = {"user": question, "reasoning_content": reasoning, "answer": formatted_answers[selected.name]}
        if first_response:
            first_response = False
            if source_name == conversion.BIO_INSTRUCTION:
                completion["user"] = re.sub(r"(<(?:dna|rna)>)([^<]+)", r"\1\2A", question)
            else:
                completion["answer"] = ""
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(completion)}}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        await convert_work_batches(
            work, "http://test", client, concurrency=1, concurrent_batches=1, output_root=str(output_root)
        )
    record = pq.read_table(_output_path(source, str(input_path), 0, 0, str(output_root))).to_pylist()[0]
    user = record["messages"][0]["content"][0]["text"]
    assert question in user
    assert passage not in user
    assert answer not in user
    assert record["messages"][1]["content"][0]["text"] == reasoning
    assert answer in record["messages"][2]["content"][0]["text"]


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


@pytest.mark.asyncio
async def test_biology_placeholder_preserves_long_input_and_keeps_reference_private(tmp_path) -> None:
    sequence = "ACGT" * 900
    passage = f"Question: count symbols in <dna>{sequence}</dna>. Private reference: 3600."
    source = Source(conversion.BIO_INSTRUCTION, "", 1, 1)
    input_path = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": "long", "text": passage}]), input_path)
    output_root = tmp_path / "output"

    async def response(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][1]["content"]
        assert f"[[MOLECULAR_INPUT_0]] = <dna>{sequence}</dna>" in prompt
        selected = next(f for f in conversion.FORMATS if f"({f.name}):" in prompt)
        completion = {
            "user": "Count the symbols in [[MOLECULAR_INPUT_0]].",
            "reasoning_content": "There are 900 repetitions of four symbols, giving 3600.",
            "answer": conversion._evidence_answer(["900 times four gives 3600."], selected),
        }
        if selected.name == "json":
            completion["answer"] = json.loads(completion["answer"])
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(completion)}}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        await convert_work_batches(
            [WorkBatch(WorkItem(source, str(input_path), 0, 1), 0)], "http://test", client, 1, 1, str(output_root)
        )
    record = pq.read_table(_output_path(source, str(input_path), 0, 0, str(output_root))).to_pylist()[0]
    user = record["messages"][0]["content"][0]["text"]
    assert f"<dna>{sequence}</dna>" in user
    assert "MOLECULAR_INPUT" not in user
    assert "Private reference" not in user
    assert "3600" not in user


@pytest.mark.asyncio
async def test_rejected_chunk_preserves_siblings_and_retries_after_other_batches(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(conversion, "MAX_ATTEMPTS", 1)
    monkeypatch.setattr(conversion, "DEFERRED_RETRY_DELAY", 0)
    source = Source(conversion.BIO_INSTRUCTION, "", 1, 3)
    input_path = tmp_path / "input.parquet"
    rows = [
        {"id": name, "text": f"{name}: Question: count symbols in <dna>ACGT</dna>. Reference: 4."}
        for name in ["bad", "sibling", "other"]
    ]
    pq.write_table(pa.Table.from_pylist(rows), input_path, row_group_size=2)
    work = [WorkBatch(WorkItem(source, str(input_path), 0, 2), 0), WorkBatch(WorkItem(source, str(input_path), 1, 1), 0)]
    output_root = tmp_path / "output"
    recovered = False
    requested = []

    async def response(request: httpx.Request) -> httpx.Response:
        nonlocal recovered
        body = json.loads(request.content)
        prompt = body["messages"][1]["content"]
        row_id = prompt.split("<source_passage>\n", 1)[1].split(":", 1)[0]
        requested.append(row_id)
        if row_id == "other":
            partials = list((output_root / "partials").glob("*.parquet"))
            assert len(partials) == 1
            assert [r["source_id"] for r in pq.read_table(partials[0]).to_pylist()] == [f"{source.name}:sibling:0"]
            rejected = json.loads(next((output_root / "rejections").glob("*.json")).read_text())
            assert [r["source_id"] for r in rejected["rejected_chunks"]] == [f"{source.name}:bad:0"]
            assert not Path(_output_path(source, str(input_path), 0, 0, str(output_root))).exists()
            recovered = True
        selected = next(f for f in conversion.FORMATS if f"({f.name}):" in prompt)
        completion = {
            "user": "Count the symbols in [[MOLECULAR_INPUT_0]].",
            "reasoning_content": "The input has four symbols.",
            "answer": conversion._evidence_answer(["Four symbols."], selected),
        }
        if row_id == "bad" and not recovered:
            completion["user"] = "Count symbols in <dna>ACGTA</dna>."
        if selected.name == "json":
            completion["answer"] = json.loads(completion["answer"])
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(completion)}}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        await convert_work_batches(work, "http://test", client, 2, 1, str(output_root))
    records = [
        row for path in (output_root / "outputs/main").glob("*.parquet") for row in pq.read_table(path).to_pylist()
    ]
    assert {row["source_id"] for row in records} == {f"{source.name}:{name}:0" for name in ["bad", "sibling", "other"]}
    assert requested.count("sibling") == 1
    assert not list((output_root / "partials").glob("*"))
    assert not list((output_root / "rejections").glob("*"))
