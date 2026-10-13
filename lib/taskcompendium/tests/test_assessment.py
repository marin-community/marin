# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from zephyr.context import ZephyrContext
from zephyr.readers import load_parquet

from taskcompendium.pipeline.assessment import (
    AssessmentSampling,
    assessment_inputs,
    assessment_judgments,
    prepare_assessment,
)
from taskcompendium.pipeline.assessment_review import (
    ASSESSMENT_RUBRIC,
    AssessmentModel,
    IncompleteAssessment,
    grade_assessment,
)
from taskcompendium.pipeline.models import FilterPolicy
from taskcompendium.pipeline.review_requests import RequestObservation, RequestOutput
from taskcompendium.pipeline.source_processing import SourcePipelineConfig, SourceProcessingMode, run_source_pipeline
from taskcompendium.pipeline.source_quality import SourceQualityPolicy
from taskcompendium.pipeline.source_verification import SourceVerificationPolicy
from taskcompendium.pipeline.stages import AuditExecution

from .pipeline_stages import convert_svamp, fixture_recipe


@pytest.fixture
def context(tmp_path):
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        yield context


def curated_source(context, root, rows):
    source = root / "raw"
    source.mkdir()
    (source / "source.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    output = root / "curated"
    run_source_pipeline(
        fixture_recipe(convert_svamp, rubric=None),
        context,
        str(source),
        str(output),
        SourcePipelineConfig(
            SourceProcessingMode.FULL,
            SourceQualityPolicy(sample_size=10),
            SourceVerificationPolicy(20, 0, 1, 0.95),
            None,
            AuditExecution(),
            FilterPolicy(),
            2,
            None,
        ),
        mode=SourceProcessingMode.FULL,
        canonical_source="fixture",
    )
    return output


@dataclass(frozen=True)
class AssessmentTransport:
    root: str

    def __call__(self, requests):
        observations = []
        for request in requests:
            task_id = request["custom_id"]
            with (Path(self.root) / "calls.jsonl").open("a") as stream:
                stream.write(json.dumps(request) + "\n")
            if (Path(self.root) / task_id).exists():
                row = {"custom_id": task_id, "error": {"message": "provider unavailable"}}
            else:
                packet = json.loads(request["body"]["messages"][-1]["content"])
                verdict = {
                    "task_id": task_id,
                    "verdict": "good",
                    "explanation": "The supplied numeric reference agrees.",
                    "evidence_ids": ["task" if "task" in packet["evidence"] else "normalize"],
                }
                row = {
                    "custom_id": task_id,
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": {
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": [
                                            {
                                                "id": "assessment",
                                                "type": "function",
                                                "function": {
                                                    "name": "assess_task",
                                                    "arguments": json.dumps(verdict),
                                                },
                                            }
                                        ],
                                    },
                                }
                            ],
                        },
                    },
                }
            observations.append(RequestObservation((task_id,), json.dumps(row)))
        return RequestOutput(tuple(requests), observations)


@dataclass(frozen=True)
class TruncatedAssessmentTransport:
    root: str
    token_limit: int

    def __call__(self, requests):
        result = AssessmentTransport(self.root)(requests)
        observations = []
        for request, observation in zip(requests, result.observations, strict=True):
            if request["body"]["max_tokens"] <= self.token_limit:
                row = json.loads(observation.output)
                row["response"]["body"]["choices"][0]["finish_reason"] = "length"
                row["response"]["body"]["choices"][0]["message"] = {"role": "assistant", "content": ""}
                observation = replace(observation, output=json.dumps(row))
            observations.append(observation)
        return replace(result, observations=observations)


@dataclass(frozen=True)
class MismatchedIdentityTransport:
    root: str

    def __call__(self, requests):
        result = AssessmentTransport(self.root)(requests)
        observations = []
        for request, observation in zip(requests, result.observations, strict=True):
            schema = request["body"]["tools"][0]["function"]["parameters"]
            if "const" not in schema["properties"]["task_id"]:
                row = json.loads(observation.output)
                function = row["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"]
                verdict = json.loads(function["arguments"])
                verdict["task_id"] += "-typo"
                function["arguments"] = json.dumps(verdict)
                observation = replace(observation, output=json.dumps(row))
            observations.append(observation)
        return replace(result, observations=observations)


def test_assessment_recovers_identity_mismatch_without_misattributing_grade(context, tmp_path):
    curated = curated_source(
        context,
        tmp_path,
        [{"Body": f"Ada has {value} apples.", "Question": "How many?", "Answer": str(value)} for value in (2, 3)],
    )
    inputs = tmp_path / "inputs"
    prepare_assessment(
        context,
        curated_root=str(curated),
        sampling=AssessmentSampling(admitted=2),
        output_root=str(inputs),
    )
    judgments = tmp_path / "judgments"
    grade_assessment(
        context,
        curated_root=str(curated),
        inputs_root=str(inputs),
        rubric=ASSESSMENT_RUBRIC,
        model=AssessmentModel("fixture", "revision"),
        submit=MismatchedIdentityTransport(str(tmp_path)),
        cache_root=str(tmp_path / "cache"),
        output_root=str(judgments),
    )
    manifest = assessment_judgments(str(judgments))
    rows = [row for path in manifest.grade_paths for row in load_parquet(path)]
    expected = {row.task_id for row in assessment_inputs(str(inputs)).sample}
    assert {(row["task_id"], row["status"], row["verdict"]) for row in rows} == {
        (task_id, "reviewed", "good") for task_id in expected
    }
    calls = [json.loads(row) for row in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 4
    first, retried = calls[:2], calls[2:]
    assert all(
        "const" not in row["body"]["tools"][0]["function"]["parameters"]["properties"]["task_id"] for row in first
    )
    assert {row["custom_id"] for row in retried} == expected
    for row in retried:
        assert row["body"]["tools"][0]["function"]["parameters"]["properties"]["task_id"]["const"] == row["custom_id"]


def test_assessment_recovers_truncated_judgment_with_larger_retry_budget(context, tmp_path):
    curated = curated_source(
        context, tmp_path, [{"Body": "Ada has two apples.", "Question": "How many?", "Answer": "2"}]
    )
    inputs = tmp_path / "inputs"
    prepare_assessment(
        context,
        curated_root=str(curated),
        sampling=AssessmentSampling(admitted=1),
        output_root=str(inputs),
    )
    judgments = tmp_path / "judgments"
    grade_assessment(
        context,
        curated_root=str(curated),
        inputs_root=str(inputs),
        rubric=ASSESSMENT_RUBRIC,
        model=AssessmentModel("fixture", "revision", max_tokens=100, retry_max_tokens=200),
        submit=TruncatedAssessmentTransport(str(tmp_path), token_limit=100),
        cache_root=str(tmp_path / "cache"),
        output_root=str(judgments),
    )
    rows = [row for path in assessment_judgments(str(judgments)).grade_paths for row in load_parquet(path)]
    assert [(row["status"], row["verdict"]) for row in rows] == [("reviewed", "good")]
    calls = [json.loads(row) for row in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [row["body"]["max_tokens"] for row in calls] == [100, 200]
    assert calls[0]["custom_id"] == calls[1]["custom_id"]


def test_assessment_retains_curation_evidence_and_retries_only_uncached_requests(context, tmp_path):
    curated = curated_source(
        context,
        tmp_path,
        [
            {"Body": f"Ada has {value} apples.", "Question": "How many apples?", "Answer": str(value)}
            for value in range(1, 5)
        ]
        + [{"Answer": "0"}],
    )
    inputs = tmp_path / "inputs"
    sampling = AssessmentSampling(admitted=2, excluded=1)
    prepare_assessment(
        context,
        curated_root=str(curated),
        sampling=sampling,
        output_root=str(inputs),
    )
    manifest = assessment_inputs(str(inputs))
    assert {row.cohort: row.population for row in manifest.populations} == {"admitted": 4, "excluded": 1}
    packets = [row for path in manifest.packet_paths for row in load_parquet(path)]
    assert Counter(row["cohort"] for row in packets) == {"admitted": 2, "excluded": 1}
    admitted = next(row for row in packets if row["cohort"] == "admitted")
    evidence = json.loads(admitted["packet_json"])
    assert evidence["task"]["id"] == admitted["task_id"]
    assert evidence["verify"]["admission"] == "admitted"
    excluded = next(row for row in packets if row["cohort"] == "excluded")
    assert "task" not in json.loads(excluded["packet_json"])
    assert "task: no converted TaskSpec" in excluded["omissions"]

    unavailable = tmp_path / admitted["task_id"]
    unavailable.touch()
    judgments = tmp_path / "judgments"
    options = dict(
        curated_root=str(curated),
        inputs_root=str(inputs),
        rubric=ASSESSMENT_RUBRIC,
        model=AssessmentModel("fixture", "revision"),
        submit=AssessmentTransport(str(tmp_path)),
        cache_root=str(tmp_path / "cache"),
        output_root=str(judgments),
    )
    with pytest.raises(IncompleteAssessment):
        grade_assessment(context, **options)
    partial = assessment_judgments(str(judgments))
    partial_rows = [row for path in partial.grade_paths for row in load_parquet(path)]
    assert Counter(row["status"] for row in partial_rows) == {"reviewed": 2, "unavailable": 1}
    unavailable.unlink()
    grade_assessment(context, **options)
    calls = [json.loads(row)["custom_id"] for row in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert Counter(calls) == {row["task_id"]: 4 if row["task_id"] == admitted["task_id"] else 1 for row in packets}
    completed = [row for path in assessment_judgments(str(judgments)).grade_paths for row in load_parquet(path)]
    assert all(row["status"] == "reviewed" for row in completed)
    assert {row["task_id"] for row in completed} == {row["task_id"] for row in packets}
    assert (judgments / "responses/attempt-01.jsonl").is_file()

    repeated = tmp_path / "repeated"
    prepare_assessment(
        context,
        curated_root=str(curated),
        sampling=sampling,
        output_root=str(repeated),
    )
    assert assessment_inputs(str(repeated)).sample == manifest.sample


def test_assessment_rejects_duplicate_ids_before_small_sampling(context, tmp_path):
    curated = curated_source(
        context,
        tmp_path,
        [
            {"Body": "Ada has two apples.", "Question": "How many apples?", "Answer": "2"},
        ],
    )
    final = next((curated / "final").glob("*.parquet"))
    table = pq.read_table(final)
    pq.write_table(pa.concat_tables([table, table]), final)
    with pytest.raises(Exception, match="Duplicate assessment task ID"):
        prepare_assessment(
            context,
            curated_root=str(curated),
            sampling=AssessmentSampling(admitted=1),
            output_root=str(tmp_path / "inputs"),
        )


def test_empty_release_keeps_excluded_diagnosis(context, tmp_path):
    curated = curated_source(context, tmp_path, [{"Answer": "0"}])
    inputs, judgments = tmp_path / "inputs", tmp_path / "judgments"
    prepare_assessment(
        context,
        curated_root=str(curated),
        sampling=AssessmentSampling(),
        output_root=str(inputs),
    )
    grade_assessment(
        context,
        curated_root=str(curated),
        inputs_root=str(inputs),
        rubric=ASSESSMENT_RUBRIC,
        model=AssessmentModel("fixture", "revision"),
        submit=AssessmentTransport(str(tmp_path)),
        cache_root=str(tmp_path / "cache"),
        output_root=str(judgments),
    )
    rows = [row for path in assessment_judgments(str(judgments)).grade_paths for row in load_parquet(path)]
    assert [(row["cohort"], row["status"]) for row in rows] == [("excluded", "reviewed")]
    assert {row.cohort: row.population for row in assessment_inputs(str(inputs)).populations} == {
        "admitted": 0,
        "excluded": 1,
    }


def test_assessment_locates_evidence_after_empty_physical_row_groups(context, tmp_path):
    curated = curated_source(
        context,
        tmp_path,
        [{"Body": f"Ada has {value} apples.", "Question": "How many apples?", "Answer": str(value)} for value in (1, 2)],
    )
    normalize = next((curated / "normalize").glob("*.parquet"))
    table = pq.read_table(normalize)
    rewritten = normalize.with_suffix(".rewritten")
    with pq.ParquetWriter(rewritten, table.schema) as writer:
        writer.write_table(table.slice(0, 0))
        writer.write_table(table)
    rewritten.replace(normalize)
    metadata = pq.ParquetFile(normalize).metadata
    assert metadata.num_row_groups == 2 and metadata.row_group(0).num_rows == 0
    inputs = tmp_path / "inputs"
    prepare_assessment(
        context,
        curated_root=str(curated),
        sampling=AssessmentSampling(admitted=2),
        output_root=str(inputs),
    )
    manifest = assessment_inputs(str(inputs))
    packets = [row for path in manifest.packet_paths for row in load_parquet(path)]
    assert len(packets) == 2
    assert all(json.loads(row["packet_json"])["task"]["id"] == row["task_id"] for row in packets)


def test_assessment_keeps_task_context_when_audit_views_repeat_large_payloads(context, tmp_path):
    curated = curated_source(
        context,
        tmp_path,
        [
            {"Body": "Ada has two apples.", "Question": "How many apples?", "Answer": "2"},
        ],
    )
    for view in ("normalize", "review", "verify"):
        path = next((curated / view).glob("*.parquet"))
        table = pq.read_table(path)
        rows = table.to_pylist()
        for row in rows:
            row["normalization_changes"] = [
                {
                    "field": "prompt",
                    "reason": "adapt answer delivery",
                    "original": "original " * 20_000,
                    "replacement": "replacement " * 20_000,
                }
            ]
            if "raw_json" in row:
                row["raw_json"] = json.dumps({"duplicated_source_blob": "x" * 200_000})
        replacement = path.with_suffix(".rewritten")
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), replacement)
        replacement.replace(path)
    inputs = tmp_path / "inputs"
    sampling = AssessmentSampling(admitted=1)
    prepare_assessment(
        context,
        curated_root=str(curated),
        sampling=sampling,
        output_root=str(inputs),
    )
    manifest = assessment_inputs(str(inputs))
    (packet,) = [row for path in manifest.packet_paths for row in load_parquet(path)]
    evidence = json.loads(packet["packet_json"])
    assert evidence["task"]["id"] == packet["task_id"]
    assert evidence["submission"]["answer_path"] is None
    assert evidence["normalize"]["normalization_changes"][0]["original"]["truncated"]
    assert "task_json" not in evidence["review"] and "task_json" not in evidence["verify"]
    assert "raw_json" not in evidence["verify"]
    assert len(packet["packet_json"]) <= sampling.max_packet_characters
