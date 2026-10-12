# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from zephyr.context import ZephyrContext
from zephyr.readers import load_parquet

from taskcompendium.pipeline.assessment import (
    AssessmentSampling,
    assessment_summary,
    prepare_assessment,
    read_json,
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
    prepare_assessment(context, curated_root=str(curated), sampling=sampling, output_root=str(inputs))
    manifest = read_json(str(inputs / "manifest.json"))
    assert manifest["admitted_population"] == 4
    assert manifest["excluded_population"] == 1
    packets = [row for path in manifest["packet_paths"] for row in load_parquet(path)]
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
        inputs_root=str(inputs),
        rubric=ASSESSMENT_RUBRIC,
        model=AssessmentModel("fixture", "revision"),
        submit=AssessmentTransport(str(tmp_path)),
        cache_root=str(tmp_path / "cache"),
        output_root=str(judgments),
    )
    with pytest.raises(IncompleteAssessment):
        grade_assessment(context, **options)
    partial = read_json(str(judgments / "manifest.json"))
    assert partial["counts"] == {"reviewed": 2, "unavailable": 1}
    unavailable.unlink()
    grade_assessment(context, **options)
    calls = [json.loads(row)["custom_id"] for row in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert Counter(calls) == {row["task_id"]: 4 if row["task_id"] == admitted["task_id"] else 1 for row in packets}
    report = assessment_summary(str(inputs), str(judgments))
    assert report["assessment"]["admitted"]["grade"] == "good"
    # A model's favorable opinion cannot certify missing task evidence.
    assert report["assessment"]["excluded"]["grade"] == "unknown"
    assert report["curation"]["admission_counts"]["rejected"] == 1
    assert (judgments / "responses/attempt-01.jsonl").is_file()

    repeated = tmp_path / "repeated"
    prepare_assessment(context, curated_root=str(curated), sampling=sampling, output_root=str(repeated))
    assert read_json(str(repeated / "manifest.json"))["sample"] == manifest["sample"]


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
    prepare_assessment(context, curated_root=str(curated), sampling=AssessmentSampling(), output_root=str(inputs))
    grade_assessment(
        context,
        inputs_root=str(inputs),
        rubric=ASSESSMENT_RUBRIC,
        model=AssessmentModel("fixture", "revision"),
        submit=AssessmentTransport(str(tmp_path)),
        cache_root=str(tmp_path / "cache"),
        output_root=str(judgments),
    )
    report = assessment_summary(str(inputs), str(judgments))
    assert report["assessment"]["admitted"]["sample_count"] == 0
    assert report["assessment"]["admitted"]["grade"] == "unknown"
    assert report["assessment"]["excluded"]["sample_count"] == 1


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
        context, curated_root=str(curated), sampling=AssessmentSampling(admitted=2), output_root=str(inputs)
    )
    manifest = read_json(str(inputs / "manifest.json"))
    packets = [row for path in manifest["packet_paths"] for row in load_parquet(path)]
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
    prepare_assessment(context, curated_root=str(curated), sampling=sampling, output_root=str(inputs))
    manifest = read_json(str(inputs / "manifest.json"))
    (packet,) = [row for path in manifest["packet_paths"] for row in load_parquet(path)]
    evidence = json.loads(packet["packet_json"])
    assert evidence["task"]["id"] == packet["task_id"]
    assert evidence["submission"]["answer_path"] is None
    assert evidence["normalize"]["normalization_changes"][0]["original"]["truncated"]
    assert "task_json" not in evidence["review"] and "task_json" not in evidence["verify"]
    assert "raw_json" not in evidence["verify"]
    assert len(packet["packet_json"]) <= sampling.max_packet_characters
