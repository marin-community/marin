# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tarfile
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from harbor_config.models.task.config import TaskConfig, VerifierEnvironmentMode
from taskcompendium.models import NoGrader
from taskcompendium.pipeline.models import NormalizedTask

from experiments.post_training.task_curation.compare_harbor import compare_harbor
from experiments.post_training.task_curation.datasets.tasktrove import calendar, math, python_tests
from experiments.post_training.task_curation.harbor import TASKS_SCHEMA, UnsupportedHarborTask, harbor_record
from experiments.post_training.task_curation.tests.conversion import convert_row
from experiments.post_training.tasktrove.publish import TASKS_SCHEMA as RELEASE_SCHEMA

FIXTURES = Path(__file__).parent / "fixtures"
GRADER_IMAGE = "example.test/grader@sha256:" + "a" * 64


def archive_files(blob: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as archive:
        files = {}
        for member in archive:
            handle = archive.extractfile(member)
            if handle is not None:
                files[member.name] = handle.read()
        return files


@pytest.fixture(params=[("calendar", calendar), ("math_prism", math), ("stack_pytest", python_tests)])
def normalized_row(request) -> tuple[dict, NormalizedTask]:
    name, module = request.param
    pipeline = next(pipeline for pipeline in module.pipelines() if pipeline.name == f"tasktrove-{name}")
    source_path = f"{name}-original.tar.gz"
    converted = convert_row(pipeline, {"path": source_path, "task_binary": (FIXTURES / f"{name}.tar.gz").read_bytes()})
    assert isinstance(converted, NormalizedTask)
    config = pipeline.atlas_id.removeprefix("Task Trove:")
    return {
        "task_id": converted.task.id,
        "task_json": converted.task.model_dump_json(),
        "original_path": source_path,
        "source_row": f"{config}/tasks.parquet:0",
        "normalization_changes": [change.model_dump() for change in converted.changes],
    }, converted


def test_harbor_lowering_preserves_delivery_and_private_resource_boundaries(normalized_row) -> None:
    row, converted = normalized_row
    record = harbor_record(row, grader_image=GRADER_IMAGE, family="fixture")
    files = archive_files(record.task_binary)
    config = TaskConfig.model_validate_toml(files["task.toml"].decode())
    assert config.metadata["taskcompendium_id"] == converted.task.id
    assert config.metadata["tasktrove_path"] == record.path == row["original_path"]
    assert config.verifier.environment_mode == VerifierEnvironmentMode.SEPARATE
    assert config.verifier.environment.docker_image == GRADER_IMAGE
    assert not any(path.startswith("solution/") for path in files)
    assert not any(path.startswith(("environment/files/tests/", "environment/files/solution/")) for path in files)
    for resource in converted.task.resources.verifier:
        assert "tests/" + resource.path in files
    if converted.task.answer_type == "text":
        assert b"/app/answer.txt" in files["instruction.md"]
        assert [artifact.source for artifact in config.artifacts] == ["/app/answer.txt"]
    else:
        assert {artifact.source for artifact in config.artifacts} == set(converted.task.output_paths)
    oracle = {resource.path for resource in converted.task.resources.oracle if resource.path.startswith("solution/")}
    if oracle:
        assert record.solution_binary is not None
        assert oracle <= archive_files(record.solution_binary).keys()


def test_harbor_comparison_reports_population_difference_without_claiming_runtime_parity(
    normalized_row, tmp_path
) -> None:
    row, _ = normalized_row
    record = harbor_record(row, grader_image=GRADER_IMAGE, family="fixture")
    output = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist([asdict(record)], schema=TASKS_SCHEMA), output)
    assert pq.ParquetFile(output).schema_arrow.equals(RELEASE_SCHEMA, check_metadata=False)
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps({"by_source": {record.source: {"converted": 2, "duplicate": 1}, "missing": {"converted": 3}}})
    )
    report = compare_harbor(output, golden, sources=(record.source, "missing"), golden_revision="release1")
    assert report["rows"] == 1
    populations = {source["source"]: source for source in report["sources"]}
    assert populations[record.source]["delta"] == -1
    assert populations["missing"]["generated"] == 0
    assert populations["missing"]["delta"] == -3
    assert not report["golden_counts_match"]
    assert not report["runtime_verified"]
    assert report["harbor_configs_valid"] and report["oracle_solutions_separate"]


def test_harbor_export_does_not_replace_unavailable_grader_with_passing_stub(normalized_row) -> None:
    row, converted = normalized_row
    unavailable = converted.task.model_copy(update={"grader": NoGrader(reason="Needs source validation")})
    with pytest.raises(UnsupportedHarborTask, match="Unsupported grader: none"):
        harbor_record({**row, "task_json": unavailable.model_dump_json()}, grader_image=GRADER_IMAGE, family="fixture")
