# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source gates bound conversion, conserve the raw source ledger and admit only ready rows to final/."""

import gzip
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from verifyit.spec import JudgeSpec
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.plan import compute_plan
from zephyr.readers import load_jsonl

from taskcompendium.grader import verifyit_package
from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.models import NoGrader, ResourceGroups, Source, TaskSpec
from taskcompendium.pipeline.controls import GradingMachines, answer_reply, reference_reply
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import (
    Controls,
    FilterPolicy,
    ImportFailureKind,
    ImportRejection,
    OracleCommand,
    RawRow,
    SourceRecipe,
)
from taskcompendium.pipeline.review import BatchReviewer, completion_body
from taskcompendium.pipeline.source_processing import (
    SourcePipelineConfig,
    SourcePipelineResult,
    SourceProcessingMode,
    merge_raw_samples,
    run_source_pipeline,
    sample_raw_rows,
)
from taskcompendium.pipeline.source_quality import SourceQualityPolicy
from taskcompendium.pipeline.source_verification import SourceVerificationPolicy
from taskcompendium.pipeline.stages import AuditExecution, prepare_source
from taskcompendium.runtime.resources import inline_resource

from .pipeline_stages import (
    GRADER_ENVIRONMENT,
    SOURCE_FILES,
    FixtureGradingMachines,
    UnavailableImages,
    convert_svamp,
    fixture_recipe,
    review_config,
    script_graded,
    svamp_row_task,
)
from .test_pipeline import BatchService, Output

REFERENCE_CONTROLS = Controls(golden=reference_reply)


@dataclass(frozen=True)
class RecordingConverter:
    directory: str
    unsupported: bool = False

    def __call__(self, row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
        path = Path(self.directory) / row.id
        with path.open("a") as stream:
            stream.write("converted\n")
        if self.unsupported and row.data["Answer"] != "1":
            return ImportRejection(kind=ImportFailureKind.UNSUPPORTED, reason="unsupported_variant", detail="fixture")
        return svamp_row_task(row)


@dataclass(frozen=True)
class RecordingDecoder:
    directory: str

    def __call__(self, row, _context):
        with (Path(self.directory) / row["Answer"]).open("a") as stream:
            stream.write("decoded\n")
        decoded = {key: value for key, value in row.items() if key != "task_binary"}
        return {**decoded, "decode_receipt": "decoded source representation"}


def convert_script_graded(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    """The arithmetic task graded in its image, with the reference answer kept as an oracle file."""
    task = svamp_row_task(row)
    if isinstance(task, ImportRejection):
        return task
    answer = str(row.data["Answer"])
    task = task.model_copy(
        update={"resources": ResourceGroups(oracle=(inline_resource("solution/answer.txt", answer.encode()),))}
    )
    return script_graded(task, f'test "$(cat answer.txt)" = {answer}\n'.encode())


def oracle_answer(_task: TaskSpec) -> OracleCommand:
    return OracleCommand("cp /solution/answer.txt answer.out", answer_file="answer.out")


ORACLE_CONTROLS = Controls(golden=oracle_answer)


def convert_mixed_graders(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    """The arithmetic task with the grader named by the row's ``grader`` field."""
    kind = row.data["grader"]
    if kind == "script":
        return convert_script_graded(row, context)
    task = svamp_row_task(row)
    if isinstance(task, ImportRejection) or kind == "in_process":
        return task
    if kind == "none":
        grader = NoGrader(reason="The source evaluator is unavailable")
    else:
        grader = verifyit_package(JudgeSpec(references=(row.data["Answer"],)), environment=GRADER_ENVIRONMENT).grader
    return TaskSpec.model_validate_json(task.model_copy(update={"grader": grader}).model_dump_json())


def apple_rows(count: int) -> list[dict]:
    return [
        {"Body": f"Aya has {index} apples.", "Question": "How many?", "Answer": str(index)}
        for index in range(1, count + 1)
    ]


def write_jsonl(source: Path, rows: list[dict]) -> None:
    source.mkdir(parents=True, exist_ok=True)
    (source / "source.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def parquet_rows(path):
    return [row for file in sorted(Path(path).glob("*.parquet")) for row in pq.read_table(file).to_pylist()]


def pipeline_config(
    mode: SourceProcessingMode,
    reviewer: BatchReviewer,
    *,
    execution: AuditExecution | None = None,
    verification: SourceVerificationPolicy = SourceVerificationPolicy(10, 0, 1, 1),
    machines: GradingMachines | None = None,
    normalized_shards: int = 2,
) -> SourcePipelineConfig:
    return SourcePipelineConfig(
        mode,
        SourceQualityPolicy(),
        verification,
        review_config(reviewer),
        execution if execution is not None else AuditExecution(reviewer=reviewer),
        FilterPolicy(),
        normalized_shards=normalized_shards,
        machines=machines,
    )


def run_pipeline(
    recipe: SourceRecipe, source: Path, output: Path, config: SourcePipelineConfig, **options
) -> SourcePipelineResult:
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(output.parent / "chunks")) as context:
        return run_source_pipeline(
            recipe, context, str(source), str(output), config, canonical_source=recipe.name, **options
        )


def read_json(path) -> dict:
    return json.loads(Path(path).read_text())


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
    write_jsonl(source, rows)
    recipe = fixture_recipe(convert_svamp)
    byte_limit = 16 * 1024
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        manifest = prepare_source(
            str(source),
            str(prepared),
            recipe,
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
        assert task == svamp_row_task(RawRow(task.id, task.source, original))
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
    recipe = fixture_recipe(
        RecordingConverter(str(conversions)),
        controls=REFERENCE_CONTROLS,
        source=replace(
            SOURCE_FILES,
            patterns=("source.parquet",),
            format=SourceFormat.PARQUET,
            decode=RecordingDecoder(str(decodings)),
        ),
    )
    service = BatchService(quality=quality)
    reviewer = BatchReviewer(service, "fixture", "revision")
    config = pipeline_config(mode, reviewer)
    with PanelPlanContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        result = run_source_pipeline(
            recipe, context, str(source), str(tmp_path / "output"), config, canonical_source=recipe.name
        )
        # The procedure leaves the caller's pool entered and usable.
        from_list_result = context.execute(Dataset.from_list([1]).count()).results
    assert from_list_result == [1]
    report = read_json(result.manifest_path)
    telemetry = read_json(report["telemetry"])
    assert telemetry["source"] == recipe.name and telemetry["status"] == "completed"
    operations = {
        phase["phase"]: [execution["operation"] for execution in phase["executions"]] for phase in telemetry["phases"]
    }
    operations.pop("verification", None)
    # The panel is prepared on the driver, and each later phase is one execution.
    expected_operations = {
        "raw_sample": ["execute"],
        "panel_normalize": ["execute"],
        "sample_prepare": [],
        "quality_review": ["review"],
        "audit_review": ["review"],
        "filter": ["filter"],
        "export": ["export"],
    }
    if expected_conversions > 100:
        expected_operations["full_prepare"] = ["prepare"]
    assert operations == expected_operations
    executions = [execution for phase in telemetry["phases"] for execution in phase["executions"]]
    ids = [execution["execution_id"] for execution in executions if execution["execution_id"]]
    assert len(ids) == len(set(ids)) and all(execution["status"] == "completed" for execution in executions)
    assert any("source/decode/seconds" in execution["counters"] for execution in executions)
    assert context.panel_source_shards == [7]
    files = list(conversions.iterdir())
    assert len(files) == expected_conversions
    assert all(file.read_text() == "converted\n" for file in files)
    decoded_files = list(decodings.iterdir())
    assert len(decoded_files) == expected_conversions
    assert all(file.read_text() == "decoded\n" for file in decoded_files)
    raw = parquet_rows(Path(result.download_path) / "locators")
    review = parquet_rows(result.review_path)
    normalized = parquet_rows(result.normalize_path)
    assert len(raw) == len(review) == 125
    assert len(normalized) == expected_conversions
    raw_by_id = {row["task_id"]: row for row in raw}
    expected_tasks = {}
    for row in normalized:
        task = TaskSpec.model_validate_json(row["task_json"])
        source_record = raw_by_id[task.id]
        expected_source = Source(
            dataset=recipe.source.dataset,
            revision=recipe.source.revision,
            row=source_record["source_locator"],
            importer_revision=recipe.version,
        )
        expected_id = f"{recipe.name}-{canonical_sha256(expected_source.model_dump())}"
        original_row = rows[int(source_record["source_locator"].rsplit(":", 1)[1])]
        expected = svamp_row_task(RawRow(expected_id, expected_source, original_row))
        assert task == expected
        expected_tasks[expected_id] = expected
    assert recipe.rubric is not None
    for requests in service.batches.values():
        for request in requests:
            assert request["body"] == completion_body(
                expected_tasks[request["custom_id"]], recipe.rubric, reviewer.model, reviewer.max_tokens
            )
    assert all(row["raw_input_sha256"] == raw_by_id[row["task_id"]]["raw_input_sha256"] for row in normalized)
    assert all(row["raw_sha256"] != row["raw_input_sha256"] for row in normalized)
    assert all(row["raw_sha256"] is None for row in raw)
    assert len(list(Path(result.normalize_path).glob("*.parquet"))) == 2
    assert all("raw_json" not in row for row in raw)
    assert read_json(Path(result.download_path) / "manifest.json")["source_input"] == str(source)
    assert not list((tmp_path / "output/work").rglob("*.parquet"))
    assert not list((tmp_path / "output/work").rglob("batch-*.jsonl.gz"))
    assert {row["task_id"] for row in raw} == {row["task_id"] for row in review}
    assert all("task_json" not in row and "raw_json" not in row for row in review)
    assert not report["raw_population_census"]
    assert report["quality"]["status"] == {"bad": "reject", "good": "trust", "some_issues": "full_review"}[quality]
    reviewed = sum(len(requests) for requests in service.files.values())
    assert reviewed == (125 if quality == "some_issues" else 100)
    if quality == "bad":
        assert all(row["filter_status"] == "reject" for row in review)
        assert not parquet_rows(result.final_path)
    elif quality == "good":
        verified = parquet_rows(result.verify_path)
        assert {row["admission"] for row in verified} == {"admitted"}
        assert len(parquet_rows(result.final_path)) == len(verified) == expected_conversions
    else:
        assert {row["filter_status"] for row in review} == {"reject"}


@pytest.mark.parametrize("population_count", [10, 125])
def test_unsupported_raw_panel_never_becomes_a_small_population_census(tmp_path, population_count):
    source, conversions = tmp_path / "source", tmp_path / "conversions"
    conversions.mkdir()
    write_jsonl(source, apple_rows(population_count + 1)[1:])
    recipe = fixture_recipe(RecordingConverter(str(conversions), unsupported=True))
    reviewer = BatchReviewer(BatchService(), "fixture", "revision")
    result = run_pipeline(recipe, source, tmp_path / "output", pipeline_config(SourceProcessingMode.FULL, reviewer))
    report = read_json(result.manifest_path)
    assert report["quality"]["status"] == "incomplete"
    assert report["raw_population_census"] == (population_count <= 100)
    assert not report["full_expansion"]
    assert result.status == "incomplete"
    assert len(list(conversions.iterdir())) == min(100, population_count)
    review = parquet_rows(result.review_path)
    assert len(review) == population_count
    assert {row["filter_status"] for row in review} == {"defer"}


@dataclass(frozen=True)
class AlternatingParts:
    """Read a JSONL file in parts that take every ``count``-th row, recording each row they produce."""

    count: int
    produced: str

    def size(self, file, _context: ConversionContext) -> int:
        return sum(1 for _ in load_jsonl(str(file)))

    def __call__(self, file, _context: ConversionContext, part: int, indices: frozenset[int] | None):
        for index, row in enumerate(load_jsonl(str(file))):
            if index % self.count == part and (indices is None or index in indices):
                (Path(self.produced) / str(index)).touch()
                yield index, row


@pytest.mark.parametrize("mode", list(SourceProcessingMode))
def test_source_read_in_parts_publishes_the_views_of_a_whole_read(tmp_path, mode):
    source, produced = tmp_path / "source", tmp_path / "produced"
    produced.mkdir()
    write_jsonl(source, apple_rows(125))
    reviewer = BatchReviewer(BatchService(), "fixture", "revision")
    config = pipeline_config(mode, reviewer)
    whole = run_pipeline(fixture_recipe(convert_svamp), source, tmp_path / "whole", config)
    parts = AlternatingParts(3, str(produced))
    parted_recipe = fixture_recipe(convert_svamp, source=replace(SOURCE_FILES, parts=parts))
    parted = run_pipeline(parted_recipe, source, tmp_path / "parted", config)
    for view in ("normalize", "final"):
        files = sorted(path.name for path in (tmp_path / "whole" / view).glob("*.parquet"))
        assert files == sorted(path.name for path in (tmp_path / "parted" / view).glob("*.parquet"))
        assert all(
            (tmp_path / "whole" / view / name).read_bytes() == (tmp_path / "parted" / view / name).read_bytes()
            for name in files
        )
    # A sample produces only the rows it processes; the ledger lists every row, hashing those produced.
    normalized = parquet_rows(parted.normalize_path)
    assert len(normalized) == (100 if mode == SourceProcessingMode.SAMPLE else 125)
    assert {f"source.jsonl:{path.name}" for path in produced.iterdir()} == {row["source_row"] for row in normalized}
    processed = {row["task_id"] for row in normalized}
    expected_ledger = [
        {**row, "raw_input_sha256": row["raw_input_sha256"] if row["task_id"] in processed else None}
        for row in parquet_rows(Path(whole.download_path) / "locators")
    ]
    assert sorted(parquet_rows(Path(parted.download_path) / "locators"), key=lambda row: row["task_id"]) == sorted(
        expected_ledger, key=lambda row: row["task_id"]
    )
    assert read_json(parted.manifest_path)["raw_population_count"] == 125


def test_completed_source_rerun_reuses_inference_cache_without_scratch(tmp_path):
    source = tmp_path / "source"
    write_jsonl(source, apple_rows(2)[1:])
    recipe = fixture_recipe(convert_svamp)
    initial = BatchService()
    reviewer = BatchReviewer(initial, "fixture", "revision", query_cache_root=str(tmp_path / "cache"))
    config = pipeline_config(SourceProcessingMode.SAMPLE, reviewer)
    first = run_pipeline(recipe, source, tmp_path / "output", config)
    first_review = parquet_rows(first.review_path)
    assert not list((tmp_path / "output/work").rglob("*.parquet"))
    resumed = BatchService(interrupted=True)
    second_reviewer = replace(reviewer, client=resumed)
    config = replace(config, execution=replace(config.execution, reviewer=second_reviewer))
    second = run_pipeline(recipe, source, tmp_path / "output", config)
    assert parquet_rows(second.review_path) == first_review
    assert not resumed.files and not resumed.batches
    assert list((tmp_path / "output/work/quality/evidence").glob("*/attempt-*/reviews.json"))


@pytest.mark.parametrize("failure", ["quality_panel", "verification"])
def test_incomplete_source_retry_reuses_successful_reviews(tmp_path, failure):
    source = tmp_path / "source"
    population = 1 if failure == "verification" else 125
    write_jsonl(source, apple_rows(population))
    if failure == "verification":
        recipe = fixture_recipe(convert_script_graded, controls=ORACLE_CONTROLS)
        unavailable, available = FixtureGradingMachines(UnavailableImages()), FixtureGradingMachines()
    else:
        recipe = fixture_recipe(convert_svamp)
        unavailable = available = None
    service = BatchService(interrupted=failure == "quality_panel")
    reviewer = BatchReviewer(service, "fixture", "revision", max_attempts=1, query_cache_root=str(tmp_path / "cache"))
    config = pipeline_config(SourceProcessingMode.FULL, reviewer, machines=unavailable)
    first = run_pipeline(recipe, source, tmp_path / "output", config)
    assert first.status == "incomplete"
    assert Path(first.manifest_path).exists()
    submitted = sum(len(rows) for rows in service.files.values())
    first_files = set(service.files)
    second = run_pipeline(recipe, source, tmp_path / "output", replace(config, machines=available))
    assert second.status == "completed"
    retried_ids = [
        row["custom_id"] for file_id, rows in service.files.items() if file_id not in first_files for row in rows
    ]
    failed_ids = [row["custom_id"] for row in service.batches["batch-0"]] if failure == "quality_panel" else []
    assert sorted(retried_ids) == sorted(failed_ids)
    assert submitted == (100 if failure == "quality_panel" else population)
    assert not list((tmp_path / "output/work").rglob("*.parquet"))
    review = parquet_rows(second.review_path)
    assert len(review) == len({row["task_id"] for row in review}) == population
    assert all(row["review_status"] not in {"invalid", "unavailable"} for row in review)
    assert len(parquet_rows(second.final_path)) == population


class PartiallyUnavailableReview(BatchService):
    """Fail the first ``unavailable_count`` tasks it sees and judge the next ``bad_count`` bad."""

    def __init__(self, bad_count, unavailable_count=1):
        super().__init__()
        self.bad_count = bad_count
        self.unavailable_count = unavailable_count
        self.observed = []

    def output(self, batch):
        rows = [json.loads(line) for line in super().output(batch).output.splitlines()]
        for row in rows:
            task_id = row["custom_id"]
            if task_id not in self.observed:
                self.observed.append(task_id)
            index = self.observed.index(task_id)
            if index < self.unavailable_count:
                row["response"] = {"status_code": 503, "body": {"error": "Review unavailable"}}
            elif index < self.unavailable_count + self.bad_count:
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
    write_jsonl(source, apple_rows(population))
    # A golden the grader rejects fails every sampled control, so the source is rejected.
    controls = (
        Controls(golden=lambda task: answer_reply(task, "__incorrect_answer__")) if verification == "rejected" else None
    )
    recipe = fixture_recipe(convert_svamp, controls=controls)
    service = PartiallyUnavailableReview(bad_count)
    reviewer = BatchReviewer(service, "fixture", "revision", max_attempts=1)
    result = run_pipeline(recipe, source, tmp_path / "output", pipeline_config(mode, reviewer))
    assert result.status == (
        "gated" if verification == "rejected" else "sampled" if processed < population else "completed"
    )
    report = read_json(result.manifest_path)
    assert report["quality"]["status"] == quality
    assert report["unavailable_reviews"] == 1
    assert report["processed_rows"] == processed
    assert report["verification"]["status"] == verification
    review = parquet_rows(result.review_path)
    deferred = [row for row in review if row["review_status"] == "unavailable"]
    assert len(deferred) == 1
    assert deferred[0]["filter_status"] == "defer"
    final = parquet_rows(result.final_path)
    assert len(final) == (0 if verification == "rejected" else processed - bad_count - 1)
    assert deferred[0]["task_id"] not in {row["task_id"] for row in final}
    assert list((tmp_path / "output/work/quality/evidence").glob("*/attempt-*/reviews.json"))


@pytest.mark.parametrize(
    "unavailable,bad_count,quality,status,admitted",
    [(3, 20, "full_review", "completed", 102), (10, 0, "incomplete", "incomplete", 0)],
    ids=["decision_reached", "decision_open"],
)
def test_unavailable_reviews_leave_a_source_incomplete_only_when_the_gate_cannot_decide(
    tmp_path, unavailable, bad_count, quality, status, admitted
):
    source = tmp_path / "source"
    write_jsonl(source, apple_rows(125))
    reviewer = BatchReviewer(PartiallyUnavailableReview(bad_count, unavailable), "fixture", "revision", max_attempts=1)
    result = run_pipeline(
        fixture_recipe(convert_svamp),
        source,
        tmp_path / "output",
        pipeline_config(SourceProcessingMode.FULL, reviewer),
    )
    report = read_json(result.manifest_path)
    assert report["quality"]["status"] == quality
    assert result.status == report["status"] == status
    assert report["unavailable_reviews"] == unavailable
    review = parquet_rows(result.review_path)
    deferred = [row for row in review if row["review_status"] == "unavailable"]
    assert len(deferred) == unavailable
    assert {row["filter_status"] for row in deferred} == {"defer"}
    assert len(parquet_rows(result.final_path)) == admitted


@dataclass(frozen=True)
class MixedPanelConverter:
    directory: str
    defect_limit: int

    def __call__(self, row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
        task = RecordingConverter(self.directory)(row, context)
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
    conversions.mkdir()
    write_jsonl(source, apple_rows(101))
    recipe = fixture_recipe(MixedPanelConverter(str(conversions), defect_limit))
    service = BatchService()
    reviewer = BatchReviewer(service, "fixture", "revision")
    result = run_pipeline(recipe, source, tmp_path / "output", pipeline_config(SourceProcessingMode.SAMPLE, reviewer))
    report = read_json(result.manifest_path)
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
    write_jsonl(source, apple_rows(3)[2:])
    recipe = fixture_recipe(convert_svamp)
    reviewer = BatchReviewer(BatchService(), "fixture", "revision")
    config = pipeline_config(
        SourceProcessingMode.SAMPLE,
        reviewer,
        execution=AuditExecution(),
        verification=SourceVerificationPolicy(1, 0, 1, 1),
        normalized_shards=1,
    )
    with ZephyrContext(max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        with pytest.raises(ValueError, match="requires a reviewer"):
            run_source_pipeline(
                recipe, context, str(source), str(tmp_path / "output"), config, canonical_source="catalog-selection"
            )
    report = read_json(tmp_path / "output/telemetry.json")
    assert report["source"] == "catalog-selection"
    assert report["status"] == "failed" and report["error_type"] == "ValueError"
    assert [phase["phase"] for phase in report["phases"]] == [
        "raw_sample",
        "panel_normalize",
        "sample_prepare",
        "quality_review",
    ]
    assert report["phases"][2]["status"] == "completed"
    assert read_json(tmp_path / "output/work/sample/manifest.json")["input_rows"] == 1
    assert list((tmp_path / "output/work/sample/review-inputs").glob("batch-*.jsonl.gz"))
    assert report["phases"][-1]["executions"] == []
    assert report["phases"][-1]["status"] == "failed"
    assert not list((tmp_path / "output/work/quality").glob("**/reviews.json"))


@pytest.mark.parametrize(
    "mode,processed,status",
    [(SourceProcessingMode.SAMPLE, 100, "sampled"), (SourceProcessingMode.FULL, 120, "completed")],
)
def test_source_without_rubric_keeps_converted_rows_without_review_requests(tmp_path, mode, processed, status):
    source = tmp_path / "source"
    write_jsonl(source, apple_rows(120))
    service = BatchService()
    reviewer = BatchReviewer(service, "fixture", "revision")
    result = run_pipeline(
        fixture_recipe(convert_svamp, rubric=None), source, tmp_path / "output", pipeline_config(mode, reviewer)
    )
    report = read_json(result.manifest_path)
    assert not service.files and not service.batches
    assert report["quality"]["status"] == "unreviewed"
    assert result.status == status
    assert report["processed_rows"] == processed
    converted = [
        row for row in parquet_rows(result.review_path) if row["filter_reasons"] != ["source_gate:not_expanded"]
    ]
    assert len(converted) == processed
    assert {(row["quality_basis"], row["review_status"], row["filter_status"]) for row in converted} == {
        ("unreviewed", None, "keep")
    }
    assert {row["task_id"] for row in parquet_rows(result.final_path)} == {row["task_id"] for row in converted}
    assert report["admission_counts"] == {"admitted": processed}


def test_admission_admits_only_rows_with_a_ready_grader_and_names_each_output_view(tmp_path):
    source = tmp_path / "source"
    graders = ("in_process", "none", "judge", "script")
    rows = [{**row, "grader": grader} for grader, row in zip(graders, apple_rows(len(graders)), strict=True)] + [
        {"Body": "Aya has apples.", "Question": "How many?", "Answer": "unknown", "grader": "in_process"}
    ]
    write_jsonl(source, rows)
    reviewer = BatchReviewer(BatchService(), "fixture", "revision")
    result = run_pipeline(
        fixture_recipe(convert_mixed_graders, rubric=None),
        source,
        tmp_path / "output",
        pipeline_config(SourceProcessingMode.FULL, reviewer),
    )
    output = tmp_path / "output"
    manifest = read_json(result.manifest_path)
    views = ("download", "normalize", "review", "verify", "final")
    assert manifest["datasets"] == {view: str(output / view) for view in views}
    assert [
        result.download_path,
        result.normalize_path,
        result.review_path,
        result.verify_path,
        result.final_path,
    ] == [str(output / view) for view in views]
    assert result.manifest_path == str(output / "manifest.json")
    assert manifest["telemetry"] == str(output / "telemetry.json") and (output / "telemetry.json").exists()
    assert manifest["admission_counts"] == {
        "admitted": 2,
        "no_grader": 1,
        "unverified": 1,
        "rejected": 1,
    }
    assert manifest["admission"] == "admitted"
    # Without controls, verification is skipped and a sandbox grader stays unverified.
    assert manifest["verification"]["status"] == "skipped"
    assert read_json(output / "verify/manifest.json")["controls"] is False
    by_grader = {}
    for row in parquet_rows(result.verify_path):
        locator = int(row["source_locator"].rsplit(":", 1)[1])
        by_grader[rows[locator]["grader"] if locator < len(graders) else "invalid"] = row["admission"]
    assert by_grader == {
        "in_process": "admitted",
        "none": "no_grader",
        "judge": "admitted",
        "script": "unverified",
        "invalid": "rejected",
    }
    final = parquet_rows(result.final_path)
    assert sorted(int(row["source_locator"].rsplit(":", 1)[1]) for row in final) == [0, 2]


@pytest.mark.parametrize("controls", [None, REFERENCE_CONTROLS], ids=["no_controls", "controls"])
def test_judge_graded_source_skips_verification_and_admits_its_rows(tmp_path, controls):
    # The campaign has no grading machines, so running declared controls on a judge task would raise;
    # the declared-controls case shows that no judge task is sampled.
    source = tmp_path / "source"
    write_jsonl(source, [{**row, "grader": "judge"} for row in apple_rows(3)])
    reviewer = BatchReviewer(BatchService(), "fixture", "revision")
    result = run_pipeline(
        fixture_recipe(convert_mixed_graders, rubric=None, controls=controls),
        source,
        tmp_path / "output",
        pipeline_config(SourceProcessingMode.FULL, reviewer),
    )
    manifest = read_json(result.manifest_path)
    verification = read_json(tmp_path / "output/verify/report.json")
    assert verification["status"] == "skipped"
    assert verification["reason"] == "judge grader; no control path yet"
    assert verification["results"] == []
    assert result.status == "completed"
    assert manifest["admission"] == "admitted"
    assert manifest["admission_counts"] == {"admitted": 3}
    assert len(parquet_rows(result.final_path)) == 3


@pytest.mark.parametrize(
    "machines,admission,status",
    [
        (FixtureGradingMachines(), "admitted", "completed"),
        (FixtureGradingMachines(UnavailableImages()), "deferred", "incomplete"),
    ],
    ids=["verified", "machines_unavailable"],
)
def test_sandbox_rows_reach_final_only_after_source_verification_passes(tmp_path, machines, admission, status):
    source = tmp_path / "source"
    write_jsonl(source, apple_rows(3))
    reviewer = BatchReviewer(BatchService(), "fixture", "revision")
    result = run_pipeline(
        fixture_recipe(convert_script_graded, rubric=None, controls=ORACLE_CONTROLS),
        source,
        tmp_path / "output",
        pipeline_config(SourceProcessingMode.FULL, reviewer, machines=machines),
    )
    manifest = read_json(result.manifest_path)
    assert result.status == status
    assert manifest["admission_counts"] == {admission: 3}
    assert len(parquet_rows(result.final_path)) == (3 if admission == "admitted" else 0)
    checks = {check["check"]: check["status"] for row in parquet_rows(result.verify_path) for check in row["checks"]}
    assert checks == {"golden": "pass" if admission == "admitted" else "infra_error"}
