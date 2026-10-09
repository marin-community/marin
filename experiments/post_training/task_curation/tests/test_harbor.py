# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json
import shlex
import subprocess
import tarfile
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from harbor_config.models.task.config import TaskConfig, VerifierEnvironmentMode
from taskcompendium.models import NoGrader, ScriptGrader, VerifyitGrader, verifyit_answer_file, verifyit_spec
from taskcompendium.pipeline.models import NormalizedTask
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.compare_harbor import compare_harbor
from experiments.post_training.task_curation.datasets.tasktrove import (
    calendar,
    instruction_following,
    math,
    puzzles,
    python_tests,
)
from experiments.post_training.task_curation.harbor import TASKS_SCHEMA, harbor_record, main
from experiments.post_training.task_curation.sources import all_sources
from experiments.post_training.task_curation.tests.conversion import convert_row, tasktrove_row
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
    source = next(source for source in module.sources() if source.name == f"tasktrove-{name}")
    pipeline = source.pipeline
    assert pipeline is not None
    source_path = f"{name}-original.tar.gz"
    converted = convert_row(pipeline, {"path": source_path, "task_binary": (FIXTURES / f"{name}.tar.gz").read_bytes()})
    assert isinstance(converted, NormalizedTask)
    config = source.metadata.name
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
    if isinstance(converted.task.grader, VerifyitGrader):
        # Harbor bypasses test.sh when this reserved filename exists. The wrapper
        # must run to load our bundled verifyit and install public grader inputs.
        assert "tests/verifier.toml" not in files
        command = shlex.split(files["tests/test.sh"].decode().splitlines()[-1])
        assert command[:2] == ["exec", "python3"]
        assert command[-1].removeprefix("/") in files
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


def test_harbor_public_staging_preserves_submitted_edits(normalized_row, tmp_path) -> None:
    row, converted = normalized_row
    task = converted.task
    answer_path = task.grader.answer_path if isinstance(task.grader, ScriptGrader) else None
    if isinstance(task.grader, VerifyitGrader) and task.answer_type == "text":
        answer_path = verifyit_answer_file(verifyit_spec(task.grader))
    output = (answer_path or task.output_paths[0]).removeprefix("/")
    resources = task.resources.model_copy(
        update={
            "worker": (
                *task.resources.worker,
                inline_resource(output, b"initial contents"),
                inline_resource("app/staging-input.txt", b"public input"),
            )
        }
    )
    edited = task.model_copy(update={"resources": resources})
    record = harbor_record({**row, "task_json": edited.model_dump_json()}, grader_image=GRADER_IMAGE, family="fixture")
    files = archive_files(record.task_binary)
    workspace, public = tmp_path / "workspace", tmp_path / "public"
    submitted = workspace / output
    submitted.parent.mkdir(parents=True)
    submitted.write_text("agent's edited contents")
    for name, data in files.items():
        if name.startswith("tests/public/"):
            path = public / name.removeprefix("tests/public/")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    # Execute the emitted staging command against an isolated filesystem root.
    command = shlex.split(files["tests/test.sh"].decode().splitlines()[2])
    command[-2:] = [str(public) + "/.", str(workspace)]
    subprocess.run(command, check=True)
    assert submitted.read_text() == "agent's edited contents"
    assert (workspace / "app/staging-input.txt").read_text() == "public input"
    submitted.unlink()
    subprocess.run(command, check=True)
    assert not submitted.exists()


@pytest.mark.parametrize(
    "mode,reference,valid,invalid",
    [
        (
            "exact",
            {"gold": "Defect, Salt, chair", "answer_type": "ordered_list"},
            "Defect, Salt, chair",
            "chair, Salt, Defect",
        ),
        ("math", {"gold": "3", "answer_type": "number"}, r"\boxed{3}", r"\boxed{4}"),
        (
            "json-schema",
            {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            '{"name":"Ada"}',
            '{"name":3}',
        ),
    ],
)
def test_harbor_in_process_contract_runs_bundled_grader(mode, reference, valid, invalid, tmp_path) -> None:
    if mode == "json-schema":
        source = next(source for source in instruction_following.sources() if source.name == "tasktrove-structured")
        prompt = "Produce JSON with a string name. Write your final JSON to `/app/answer.txt`."
        private = {"tests/verifier_data.json": json.dumps({"schema_type": "json", "schema": reference}).encode()}
    else:
        source = puzzles.sources()[0]
        prompt = "Solve the puzzle. Write ONLY your final answer to **`/app/answer.txt`**."
        private = {"tests/gold.json": json.dumps(reference).encode()}
    assert source.pipeline is not None
    converted = convert_row(source.pipeline, tasktrove_row({"instruction.md": prompt.encode(), **private}))
    assert isinstance(converted, NormalizedTask)
    row = {
        "task_json": converted.task.model_dump_json(),
        "source_row": source.metadata.name + "/tasks.parquet:0",
        "original_path": "fixture-task",
        "normalization_changes": [change.model_dump() for change in converted.changes],
    }
    record = harbor_record(row, grader_image=GRADER_IMAGE, family=source.metadata.family)
    files = archive_files(record.task_binary)
    assert files["instruction.md"].decode() == prompt
    assert not any(path.startswith("environment/files/") for path in files)
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    workspace, logs = tmp_path / "app", tmp_path / "logs"
    workspace.mkdir()
    # Relocate only the container's absolute paths; run the archive's wrapper and bundled runtime.
    script = files["tests/test.sh"].decode().replace("/tests/", str(tmp_path / "tests") + "/").rstrip()
    script += f" --workspace {shlex.quote(str(workspace))} --logs-dir {shlex.quote(str(logs))}\n"
    for answer, expected in ((valid, 1.0), (invalid, 0.0)):
        (workspace / "answer.txt").write_text(answer)
        subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True)
        verdict = json.loads((logs / "verdict.json").read_text())
        assert verdict["status"] == "scored"
        assert verdict["reward"] == expected


def test_harbor_cli_joins_registry_metadata_and_accounts_for_unsupported_rows(normalized_row, tmp_path) -> None:
    row, converted = normalized_row
    source = all_sources()["Task Trove:" + row["source_row"].split("/", 1)[0]]
    unsupported = converted.task.model_copy(update={"grader": NoGrader(reason="Needs source validation")})
    input_root, output_root = tmp_path / "input", tmp_path / "output"
    normalized = input_root / "normalize"
    normalized.mkdir(parents=True)
    (input_root / "manifest.json").write_text(json.dumps({"source": source.name}))
    pq.write_table(
        pa.Table.from_pylist([row, {**row, "task_id": "unavailable", "task_json": unsupported.model_dump_json()}]),
        normalized / "part-00000.parquet",
    )
    result = CliRunner().invoke(
        main,
        ["--input-root", str(input_root), "--output-root", str(output_root), "--grader-image", GRADER_IMAGE],
    )
    assert result.exit_code == 0, result.output
    report = json.loads((output_root / "manifest.json").read_text())
    assert (report["input_rows"], report["exported_rows"], report["rejected_rows"]) == (2, 1, 1)
    assert report["rejections"] == [
        {"task_id": "unavailable", "path": row["original_path"], "reason": "Unsupported grader: none"}
    ]
    exported = pq.read_table(output_root / "tasks.parquet").to_pylist()
    assert len(exported) == 1
    assert exported[0]["family"] == source.metadata.family
    assert report["atlas_id"] == source.metadata.id
