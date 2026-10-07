# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source gates bound conversion while conserving the raw source ledger."""

import gzip
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.plan import compute_plan

from taskcompendium.datasets.numeric_answers import normalize_svamp, svamp_policy
from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    FilterPolicy,
    ImportFailureKind,
    ImportRejection,
    RawRow,
    VerificationReport,
)
from taskcompendium.pipeline.review import BatchReviewer, completion_body
from taskcompendium.pipeline.source_processing import (
    SourcePipelineConfig,
    SourceProcessingMode,
    merge_raw_samples,
    run_source_pipeline,
    sample_raw_rows,
)
from taskcompendium.pipeline.source_quality import SourceQualityPolicy
from taskcompendium.pipeline.source_verification import SourceVerificationPolicy
from taskcompendium.pipeline.stages import AuditExecution, prepare_source

from .pipeline_stages import fixture_recipe, review_config
from .test_pipeline import BatchService, Output


@dataclass(frozen=True)
class RecordingNormalizer:
    directory: str
    unsupported: bool = False

    def __call__(self, row: RawRow) -> TaskSpec | ImportRejection:
        path = Path(self.directory) / row.id
        with path.open("a") as stream:
            stream.write("converted\n")
        if self.unsupported and row.data["Answer"] != "1":
            return ImportRejection(kind=ImportFailureKind.UNSUPPORTED, reason="unsupported_variant", detail="fixture")
        return normalize_svamp(row)


@dataclass(frozen=True)
class RecordingDecoder:
    directory: str

    def __call__(self, row, _root):
        with (Path(self.directory) / row["Answer"]).open("a") as stream:
            stream.write("decoded\n")
        decoded = {key: value for key, value in row.items() if key != "task_binary"}
        return {**decoded, "decode_receipt": "decoded source representation"}


def skipped_goldens(_task: TaskSpec) -> VerificationReport:
    return VerificationReport([CheckResult(check="golden", status=CheckStatus.SKIPPED, detail="No supplied golden")])


def unavailable_goldens(_task: TaskSpec) -> VerificationReport:
    return VerificationReport([CheckResult(check="golden", status=CheckStatus.INFRA_ERROR, detail="Worker unavailable")])


def failed_goldens(_task: TaskSpec) -> VerificationReport:
    return VerificationReport([CheckResult(check="golden", status=CheckStatus.FAIL, detail="Golden failed")])


def parquet_rows(path):
    return [row for file in sorted(Path(path).glob("*.parquet")) for row in pq.read_table(file).to_pylist()]


class PanelPlanContext(ZephyrContext):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.panel_source_shards = []

    def execute(self, dataset, **kwargs):
        if isinstance(dataset.source, list) and dataset.source and isinstance(dataset.source[0], tuple):
            self.panel_source_shards.append(len(compute_plan(dataset).source_items))
        return super().execute(dataset, **kwargs)


def test_raw_sample_is_partition_and_order_independent():
    rows = [{"locator": f"part.jsonl:{i}", "data": {"value": i}} for i in range(400)]
    whole = sample_raw_rows(iter(rows), size=100, seed=11)
    merged = merge_raw_samples(
        (sample_raw_rows(iter(rows[i::7]), size=100, seed=11) for i in range(7)),
        size=100,
        seed=11,
    )
    assert whole == merged == sample_raw_rows(iter(reversed(rows)), size=100, seed=11)
    assert whole.population_count == 400
    assert len(whole.rows) == 100


def test_preparation_bounds_lossless_review_files_with_skewed_duplicate_rows(tmp_path):
    source, prepared = tmp_path / "source", tmp_path / "prepared"
    source.mkdir()
    rows = [
        {
            "Body": "Aya has one apple.",
            "Question": "How many apples does Aya have?",
            "Answer": "1",
            "Equation": "1",
            "padding": "é" * size,
        }
        for size in (0, 256, 4096, 32768, 0, 256, 4096, 0)
    ]
    (source / "source.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    recipe = fixture_recipe(replace(svamp_policy(), check_suite=None))
    byte_limit = 16 * 1024
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        manifest = prepare_source(
            str(source),
            str(prepared),
            recipe,
            recipe.inputs.files,
            None,
            AuditExecution(review_batch_size=2, review_input_bytes=byte_limit),
            context=context,
        )
    assert manifest["review_input_bytes"] == byte_limit
    records = []
    for path in sorted((prepared / "review-inputs").glob("*.jsonl.gz")):
        with gzip.open(path, "rb") as stream:
            line = stream.read()
        batch = json.loads(line)["records"]
        assert len(batch) <= 2
        if len(line) > byte_limit:
            assert len(batch) == 1
            assert batch[0]["raw"]["data"] == rows[3]
        records.extend(batch)
    records.sort(key=lambda record: int(record["source"]["row"].rsplit(":", 1)[1]))
    assert manifest["input_rows"] == len(records) == len(rows)
    assert records[0]["decision"] is None
    for index, (record, original) in enumerate(zip(records, rows, strict=True)):
        assert record["raw"]["data"] == original
        task = TaskSpec.model_validate(record["normalized"])
        assert task == normalize_svamp(RawRow(task.id, task.source, original))
        if index:
            assert record["decision"]["reasons"] == ["exact_semantic_duplicate"]
            assert record["decision"]["duplicate_of"] == records[0]["task_id"]


@pytest.mark.parametrize(
    "mode,quality,expected_conversions",
    [
        (SourceProcessingMode.SAMPLE, "good", 100),
        (SourceProcessingMode.FULL, "good", 125),
        (SourceProcessingMode.FULL, "bad", 100),
        (SourceProcessingMode.FULL, "some_issues", 125),
    ],
)
def test_source_gate_bounds_conversion_and_preserves_joined_ledgers(tmp_path, mode, quality, expected_conversions):
    source = tmp_path / "source"
    conversions = tmp_path / "conversions"
    decodings = tmp_path / "decodings"
    source.mkdir()
    conversions.mkdir()
    decodings.mkdir()
    rows = [
        {
            "Body": f"Aya has {i} apples." + (" Details." * 1024 if i in (1, 125) else ""),
            "Question": "How many apples does Aya have?",
            "Answer": str(i),
            "Equation": str(i),
            "source_notes": "Retain this raw annotation." * (256 if i == 125 else 1),
        }
        for i in range(1, 126)
    ]
    pq.write_table(
        pa.Table.from_pylist([{**row, "task_binary": f"archive-{i}".encode()} for i, row in enumerate(rows)]),
        source / "source.parquet",
    )
    recipe = fixture_recipe(replace(svamp_policy(), normalize=RecordingNormalizer(str(conversions))))
    recipe = replace(
        recipe,
        inputs=replace(
            recipe.inputs,
            files=replace(
                recipe.inputs.files,
                patterns=("source.parquet",),
                format=SourceFormat.PARQUET,
                decoder=RecordingDecoder(str(decodings)),
            ),
        ),
    )
    service = BatchService(quality=quality)
    reviewer = BatchReviewer(service, "fixture", "revision")
    config = SourcePipelineConfig(
        mode,
        SourceQualityPolicy(),
        SourceVerificationPolicy(10, 0, 1, 1),
        review_config(reviewer),
        AuditExecution(reviewer=reviewer),
        FilterPolicy(),
        normalized_shards=2,
    )
    with PanelPlanContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        result = run_source_pipeline(
            recipe,
            context,
            str(source),
            str(tmp_path / "output"),
            recipe.inputs.files,
            config,
            CheckSuite("goldens", "1", {}, skipped_goldens),
            canonical_source=recipe.name,
        )
        # The procedure leaves the caller's pool entered and usable.
        from_list_result = context.execute(Dataset.from_list([1]).count()).results
    assert from_list_result == [1]
    report = json.loads(Path(result.report_path).read_text())
    telemetry = json.loads(Path(report["telemetry"]).read_text())
    assert telemetry["source"] == recipe.name and telemetry["status"] == "completed"
    phases = {phase["phase"]: phase for phase in telemetry["phases"]}
    assert {"raw_sample", "sample_prepare", "quality_review", "audit_review", "filter", "verification"} <= phases.keys()
    for phase_name in ("sample_prepare", "audit_review", "filter", "verification"):
        assert any(execution["operation"] == "manifest_count" for execution in phases[phase_name]["executions"])
    executions = [execution for phase in telemetry["phases"] for execution in phase["executions"]]
    ids = [execution["execution_id"] for execution in executions if execution["execution_id"]]
    assert len(ids) == len(set(ids)) and all(execution["status"] == "completed" for execution in executions)
    assert any("source/decode/seconds" in execution["counters"] for execution in executions)
    assert context.panel_source_shards == [7, 7]
    files = list(conversions.iterdir())
    assert len(files) == expected_conversions
    assert all(file.read_text() == "converted\n" for file in files)
    decoded_files = list(decodings.iterdir())
    assert len(decoded_files) == expected_conversions
    assert all(file.read_text() == "decoded\n" for file in decoded_files)
    raw = parquet_rows(Path(result.hf_path) / "locators")
    analysis = parquet_rows(result.analysis_path)
    normalized = parquet_rows(result.normalized_path)
    assert len(raw) == len(analysis) == 125
    assert len(normalized) == expected_conversions
    raw_by_id = {row["task_id"]: row for row in raw}
    expected_tasks = {}
    for row in normalized:
        task = TaskSpec.model_validate_json(row["task_json"])
        source_record = raw_by_id[task.id]
        expected_source = Source(
            dataset=recipe.source.dataset,
            revision=recipe.source.revision,
            row=f"{recipe.source.config}:{recipe.source.split}:{source_record['source_locator']}",
            importer_revision=recipe.version,
        )
        expected_id = f"{recipe.name}-{canonical_sha256(expected_source.model_dump())}"
        original_row = rows[int(source_record["source_locator"].rsplit(":", 1)[1])]
        expected = normalize_svamp(RawRow(expected_id, expected_source, original_row))
        assert task == expected
        expected_tasks[expected_id] = expected
    for requests in service.batches.values():
        for request in requests:
            assert request["body"] == completion_body(
                expected_tasks[request["custom_id"]], recipe.policy.rubric, reviewer.model, reviewer.max_tokens
            )
    assert all(row["raw_input_sha256"] == raw_by_id[row["task_id"]]["raw_input_sha256"] for row in normalized)
    assert all(row["raw_sha256"] != row["raw_input_sha256"] for row in normalized)
    assert all(row["raw_sha256"] is None for row in raw)
    assert len(list(Path(result.normalized_path).glob("*.parquet"))) == 2
    assert all("raw_json" not in row for row in raw)
    assert json.loads((Path(result.hf_path) / "manifest.json").read_text())["source_input"] == str(source)
    assert not list((tmp_path / "output/work").rglob("*.parquet"))
    assert not list((tmp_path / "output/work").rglob("batch-*.jsonl.gz"))
    assert {row["task_id"] for row in raw} == {row["task_id"] for row in analysis}
    assert all("task_json" not in row and "raw_json" not in row for row in analysis)
    report = json.loads(Path(result.report_path).read_text())
    assert not report["raw_population_census"]
    assert report["quality"]["status"] == {"bad": "reject", "good": "trust", "some_issues": "full_review"}[quality]
    reviewed = sum(len(requests) for requests in service.files.values())
    assert reviewed == (125 if quality == "some_issues" else 100)
    if quality == "bad":
        assert all(row["filter_status"] == "reject" for row in analysis)
        assert not parquet_rows(result.accepted_path)
    elif quality == "good":
        assert all(row["grader_readiness"] == "unverified" for row in parquet_rows(result.verification_path))
    else:
        assert {row["filter_status"] for row in analysis} == {"reject"}


@pytest.mark.parametrize("population_count", [10, 125])
def test_unsupported_raw_panel_never_becomes_a_small_population_census(tmp_path, population_count):
    source, conversions = tmp_path / "source", tmp_path / "conversions"
    source.mkdir()
    conversions.mkdir()
    (source / "source.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "Body": f"Aya has {i} apples.",
                    "Question": "How many?",
                    "Answer": str(i),
                    "Equation": str(i),
                }
            )
            + "\n"
            for i in range(2, population_count + 2)
        )
    )
    recipe = fixture_recipe(replace(svamp_policy(), normalize=RecordingNormalizer(str(conversions), unsupported=True)))
    reviewer = BatchReviewer(BatchService(), "fixture", "revision")
    config = SourcePipelineConfig(
        SourceProcessingMode.FULL,
        SourceQualityPolicy(),
        SourceVerificationPolicy(10, 0, 1, 1),
        review_config(reviewer),
        AuditExecution(reviewer=reviewer),
        FilterPolicy(),
        normalized_shards=2,
    )
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        result = run_source_pipeline(
            recipe,
            context,
            str(source),
            str(tmp_path / "output"),
            recipe.inputs.files,
            config,
            CheckSuite("goldens", "1", {}, skipped_goldens),
            canonical_source=recipe.name,
        )
    report = json.loads(Path(result.report_path).read_text())
    assert report["quality"]["status"] == "incomplete"
    assert report["raw_population_census"] == (population_count <= 100)
    assert not report["full_expansion"]
    assert result.status == "incomplete"
    assert len(list(conversions.iterdir())) == min(100, population_count)
    analysis = parquet_rows(result.analysis_path)
    assert len(analysis) == population_count
    assert {row["filter_status"] for row in analysis} == {"defer"}


def test_completed_source_rerun_reuses_inference_cache_without_scratch(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "source.jsonl").write_text(
        json.dumps(
            {
                "Body": "Aya has 2 apples.",
                "Question": "How many apples?",
                "Answer": "2",
                "Equation": "2",
            }
        )
        + "\n"
    )
    recipe = fixture_recipe(svamp_policy())
    initial = BatchService()
    reviewer = BatchReviewer(initial, "fixture", "revision", query_cache_root=str(tmp_path / "cache"))
    config = SourcePipelineConfig(
        SourceProcessingMode.SAMPLE,
        SourceQualityPolicy(),
        SourceVerificationPolicy(10, 0, 1, 1),
        review_config(reviewer),
        AuditExecution(reviewer=reviewer),
        FilterPolicy(),
        normalized_shards=2,
    )
    suite = CheckSuite("goldens", "1", {}, skipped_goldens)
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        first = run_source_pipeline(
            recipe,
            context,
            str(source),
            str(tmp_path / "output"),
            recipe.inputs.files,
            config,
            suite,
            canonical_source=recipe.name,
        )
        first_analysis = parquet_rows(first.analysis_path)
        assert not list((tmp_path / "output/work").rglob("*.parquet"))
        resumed = BatchService(interrupted=True)
        second_reviewer = replace(reviewer, client=resumed)
        config = replace(config, execution=replace(config.execution, reviewer=second_reviewer))
        second = run_source_pipeline(
            recipe,
            context,
            str(source),
            str(tmp_path / "output"),
            recipe.inputs.files,
            config,
            suite,
            canonical_source=recipe.name,
        )
    assert parquet_rows(second.analysis_path) == first_analysis
    assert not resumed.files and not resumed.batches
    assert list((tmp_path / "output/work/quality/evidence").glob("*/attempt-*/reviews.json"))


@pytest.mark.parametrize("failure", ["quality_panel", "verification"])
def test_incomplete_source_retry_reuses_successful_reviews(tmp_path, failure):
    source = tmp_path / "source"
    source.mkdir()
    population = 1 if failure == "verification" else 125
    (source / "source.jsonl").write_text(
        "".join(
            json.dumps(
                {"Body": f"Aya has {i} apples.", "Question": "How many apples?", "Answer": str(i), "Equation": str(i)}
            )
            + "\n"
            for i in range(1, population + 1)
        )
    )
    recipe = fixture_recipe(svamp_policy())
    service = BatchService(interrupted=failure == "quality_panel")
    reviewer = BatchReviewer(service, "fixture", "revision", max_attempts=1, query_cache_root=str(tmp_path / "cache"))
    config = SourcePipelineConfig(
        SourceProcessingMode.FULL,
        SourceQualityPolicy(),
        SourceVerificationPolicy(10, 0, 1, 1),
        review_config(reviewer),
        AuditExecution(reviewer=reviewer),
        FilterPolicy(),
        normalized_shards=2,
    )
    suite = CheckSuite("goldens", "1", {}, unavailable_goldens if failure == "verification" else skipped_goldens)
    output = str(tmp_path / "output")
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        first = run_source_pipeline(
            recipe, context, str(source), output, recipe.inputs.files, config, suite, canonical_source=recipe.name
        )
        assert first.status == "incomplete"
        assert Path(first.report_path).exists()
        submitted = sum(len(rows) for rows in service.files.values())
        first_files = set(service.files)
        second = run_source_pipeline(
            recipe,
            context,
            str(source),
            output,
            recipe.inputs.files,
            config,
            CheckSuite("goldens", "1", {}, skipped_goldens),
            canonical_source=recipe.name,
        )
    assert second.status == "completed"
    retried_ids = [
        row["custom_id"] for file_id, rows in service.files.items() if file_id not in first_files for row in rows
    ]
    failed_ids = [row["custom_id"] for row in service.batches["batch-0"]] if failure == "quality_panel" else []
    assert sorted(retried_ids) == sorted(failed_ids)
    assert submitted == (100 if failure == "quality_panel" else population)
    assert not list((tmp_path / "output/work").rglob("*.parquet"))
    analysis = parquet_rows(second.analysis_path)
    assert len(analysis) == len({row["task_id"] for row in analysis}) == population
    assert all(row["review_status"] not in {"invalid", "unavailable"} for row in analysis)


class PartiallyUnavailableReview(BatchService):
    def __init__(self, bad_count):
        super().__init__()
        self.bad_count = bad_count
        self.observed = []

    def output(self, batch):
        rows = [json.loads(line) for line in super().output(batch).output.splitlines()]
        for row in rows:
            task_id = row["custom_id"]
            if task_id not in self.observed:
                self.observed.append(task_id)
            index = self.observed.index(task_id)
            if index == 0:
                row["response"] = {"status_code": 503, "body": {"error": "Review unavailable"}}
            elif index <= self.bad_count:
                arguments = row["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"]
                verdict = json.loads(arguments["arguments"])
                arguments["arguments"] = json.dumps({**verdict, "quality": "bad"})
        return Output("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize(
    "mode,population,bad_count,quality,processed,verification",
    [
        (SourceProcessingMode.SAMPLE, 125, 0, "trust", 100, "skipped"),
        (SourceProcessingMode.FULL, 125, 0, "trust", 125, "skipped"),
        (SourceProcessingMode.SAMPLE, 30, 0, "census", 30, "skipped"),
        (SourceProcessingMode.FULL, 125, 20, "full_review", 125, "skipped"),
        (SourceProcessingMode.SAMPLE, 125, 0, "trust", 100, "rejected"),
    ],
)
def test_resolved_source_gate_finishes_with_unavailable_task_deferred(
    tmp_path, mode, population, bad_count, quality, processed, verification
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "source.jsonl").write_text(
        "".join(
            json.dumps({"Body": f"Aya has {i} apples.", "Question": "How many?", "Answer": str(i)}) + "\n"
            for i in range(1, population + 1)
        )
    )
    recipe = fixture_recipe(svamp_policy())
    service = PartiallyUnavailableReview(bad_count)
    reviewer = BatchReviewer(service, "fixture", "revision", max_attempts=1)
    config = SourcePipelineConfig(
        mode,
        SourceQualityPolicy(),
        SourceVerificationPolicy(10, 0, 1, 1),
        review_config(reviewer),
        AuditExecution(reviewer=reviewer),
        FilterPolicy(),
        normalized_shards=2,
    )
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        result = run_source_pipeline(
            recipe,
            context,
            str(source),
            str(tmp_path / "output"),
            recipe.inputs.files,
            config,
            CheckSuite("goldens", "1", {}, failed_goldens if verification == "rejected" else skipped_goldens),
            canonical_source=recipe.name,
        )
    assert result.status == (
        "gated" if verification == "rejected" else "sampled" if processed < population else "completed"
    )
    report = json.loads(Path(result.report_path).read_text())
    assert report["quality"]["status"] == quality
    assert report["incomplete_reviews"] == 1
    assert report["processed_rows"] == processed
    assert report["verification"]["status"] == verification
    analysis = parquet_rows(result.analysis_path)
    deferred = [row for row in analysis if row["review_status"] == "unavailable"]
    assert len(deferred) == 1
    assert deferred[0]["filter_status"] == "defer"
    accepted = parquet_rows(result.accepted_path)
    assert len(accepted) == (0 if verification == "rejected" else processed - bad_count - 1)
    assert deferred[0]["task_id"] not in {row["task_id"] for row in accepted}
    assert list((tmp_path / "output/work/quality/evidence").glob("*/attempt-*/reviews.json"))


@dataclass(frozen=True)
class MixedPanelNormalizer:
    directory: str
    defect_limit: int

    def __call__(self, row: RawRow) -> TaskSpec | ImportRejection:
        task = RecordingNormalizer(self.directory)(row)
        if int(row.data["Answer"]) <= self.defect_limit:
            return ImportRejection(
                kind=ImportFailureKind.SOURCE_DEFECT, reason="invalid_test_contract", detail="Malformed source tests"
            )
        return task


@pytest.mark.parametrize(
    "defect_limit,expected_quality,expected_status", [(2, "trust", "sampled"), (101, "reject", "gated")]
)
def test_raw_panel_with_source_defects_keeps_fixed_draw_and_decisive_quality(
    tmp_path, defect_limit, expected_quality, expected_status
):
    source, conversions = tmp_path / "source", tmp_path / "conversions"
    source.mkdir()
    conversions.mkdir()
    (source / "source.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "Body": f"Aya has {index} apples.",
                    "Question": "How many?",
                    "Answer": str(index),
                    "Equation": str(index),
                }
            )
            + "\n"
            for index in range(1, 102)
        )
    )
    recipe = fixture_recipe(replace(svamp_policy(), normalize=MixedPanelNormalizer(str(conversions), defect_limit)))
    service = BatchService()
    reviewer = BatchReviewer(service, "fixture", "revision")
    config = SourcePipelineConfig(
        SourceProcessingMode.SAMPLE,
        SourceQualityPolicy(),
        SourceVerificationPolicy(10, 0, 1, 1),
        review_config(reviewer),
        AuditExecution(reviewer=reviewer),
        FilterPolicy(),
        normalized_shards=2,
    )
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        result = run_source_pipeline(
            recipe,
            context,
            str(source),
            str(tmp_path / "output"),
            recipe.inputs.files,
            config,
            CheckSuite("goldens", "1", {}, skipped_goldens),
            canonical_source=recipe.name,
        )
    report = json.loads(Path(result.report_path).read_text())
    assert report["raw_sample_count"] == report["quality"]["population"]["input_count"] == 100
    assert report["quality"]["status"] == expected_quality
    if defect_limit == 2:
        assert report["quality"]["assessments"]["defect"] in (1, 2)
        assert report["quality"]["defect_fraction"] in (0.01, 0.02)
    else:
        assert report["quality"]["assessments"]["defect"] == 100
        assert report["quality"]["defect_fraction"] == 1.0
        assert service.files == {}
    assert result.status == expected_status
    assert len(list(conversions.iterdir())) == 100


def test_source_failure_retains_nested_preparation_evidence_without_review_requests(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "source.jsonl").write_text(
        json.dumps({"Body": "Aya has 3 apples.", "Question": "How many?", "Answer": "3"}) + "\n"
    )
    recipe = fixture_recipe(svamp_policy())
    reviewer = BatchReviewer(BatchService(), "fixture", "revision")
    config = SourcePipelineConfig(
        SourceProcessingMode.SAMPLE,
        SourceQualityPolicy(),
        SourceVerificationPolicy(1, 0, 1, 1),
        review_config(reviewer),
        AuditExecution(),
        FilterPolicy(),
        normalized_shards=1,
    )
    with ZephyrContext(max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        with pytest.raises(ValueError, match="requires a reviewer transport"):
            run_source_pipeline(
                recipe,
                context,
                str(source),
                str(tmp_path / "output"),
                recipe.inputs.files,
                config,
                CheckSuite("goldens", "1", {}, skipped_goldens),
                canonical_source="catalog-selection",
            )
    report = json.loads((tmp_path / "output/telemetry.json").read_text())
    assert report["source"] == "catalog-selection"
    assert report["status"] == "failed" and report["error_type"] == "ValueError"
    assert [phase["phase"] for phase in report["phases"]] == [
        "raw_sample",
        "panel_normalize",
        "sample_prepare",
        "quality_review",
    ]
    prepared = report["phases"][2]
    assert prepared["status"] == "completed"
    assert [execution["operation"] for execution in prepared["executions"]] == ["prepare", "manifest_count"]
    assert all(execution["execution_id"] and execution["status"] == "completed" for execution in prepared["executions"])
    assert report["phases"][-1]["executions"] == []
    assert report["phases"][-1]["status"] == "failed"
    assert not list((tmp_path / "output/work/quality").glob("**/reviews.json"))


def unbound_controls(_task: TaskSpec) -> VerificationReport:
    return VerificationReport([CheckResult(check="native_runtime", status=CheckStatus.UNSUPPORTED, detail="Unbound")])


class MixedSourceQualityReview(BatchService):
    def output(self, batch):
        rows = [json.loads(line) for line in super().output(batch).output.splitlines()]
        for row in rows:
            if int(row["custom_id"][-1], 16) < 4:
                function = row["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"]
                verdict = json.loads(function["arguments"])
                verdict["quality"] = "some_issues"
                function["arguments"] = json.dumps(verdict)
        return Output("".join(json.dumps(row) + "\n" for row in rows))


def test_normalize_only_retains_population_and_sample_reviews_without_inference(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = [
        {"Body": f"Aya has {i} apples.", "Question": "How many apples?", "Answer": str(i), "Equation": str(i)}
        for i in range(1, 126)
    ]
    (source / "source.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    recipe = fixture_recipe(svamp_policy())
    service = MixedSourceQualityReview()
    reviewer = BatchReviewer(service, "fixture", "revision")
    config = SourcePipelineConfig(
        SourceProcessingMode.SAMPLE,
        SourceQualityPolicy(),
        SourceVerificationPolicy(100, 0, 1, 1),
        review_config(reviewer),
        AuditExecution(reviewer=reviewer),
        FilterPolicy(),
        normalized_shards=2,
    )
    suite = CheckSuite("unbound", "1", {}, unbound_controls)
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        sample = run_source_pipeline(
            recipe,
            context,
            str(source),
            str(tmp_path / "sample"),
            recipe.inputs.files,
            config,
            suite,
            canonical_source=recipe.name,
        )
        sample_report = json.loads(Path(sample.report_path).read_text())
        assert sample_report["quality"]["status"] == "full_review"
        assert sample_report["verification"]["counts"]["unsupported"] > 0
        assert not parquet_rows(sample.accepted_path)
        sample_rows = {row["task_id"]: row for row in parquet_rows(sample.analysis_path)}
        untouched_service = BatchService()
        full_config = replace(
            config,
            mode=SourceProcessingMode.NORMALIZE_ONLY,
            execution=replace(config.execution, reviewer=replace(reviewer, client=untouched_service)),
        )
        result = run_source_pipeline(
            recipe,
            context,
            str(source),
            str(tmp_path / "full"),
            recipe.inputs.files,
            full_config,
            suite,
            previous_sample_path=str(tmp_path / "sample"),
            previous_verification_report=str(tmp_path / "sample/verification/report.json"),
            canonical_source=recipe.name,
        )
    final = parquet_rows(result.analysis_path)
    assert len(final) == len(parquet_rows(result.normalized_path)) == 125
    assert not untouched_service.batches
    for row in final:
        if row["task_id"] in sample_rows:
            original = sample_rows[row["task_id"]]
            assert (row["review_status"], row["review_quality"], row["review_defects"]) == (
                original["review_status"],
                original["review_quality"],
                original["review_defects"],
            )
        else:
            assert row["filter_status"] == "defer"
            assert row["filter_reasons"] == ["readiness:unbound_controls"]
            assert row["review_quality"] is None
    assert len(parquet_rows(result.accepted_path)) == 0
    report = json.loads(Path(result.report_path).read_text())
    assert report["processed_rows"] == report["raw_population_count"] == 125
    assert report["unprocessed_rows"] == 0
    assert report["quality"] == json.loads(Path(sample.report_path).read_text())["quality"]
