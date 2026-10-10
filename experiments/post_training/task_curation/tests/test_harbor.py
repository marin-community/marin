# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
import shlex
import subprocess
import tarfile
import tomllib
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from taskcompendium.convert.answers import answer_task
from experiments.post_training.task_curation.datasets.tasktrove.conversion.verifyit_build import verifyit_build_context
from taskcompendium.grader import verifyit_package
from experiments.post_training.task_curation.tasktrove.harbor_export import UnsupportedHarborTask, harbor_record
from taskcompendium.models import (
    AnswerType,
    ArtifactKind,
    DockerBuildContext,
    EnvironmentRequirements,
    FileReward,
    NoGrader,
    ResourceGroups,
    RewardFile,
    ScriptGrader,
    Source,
    StdoutReward,
    TaskSpec,
    VerifierArtifact,
    VerifyitGrader,
)
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from verifyit.grade import grade, read_output
from verifyit.spec import ExactSpec, JsonSchemaSpec, McqSpec, PytestSpec, ScriptSpec, parse_spec, render_spec

from experiments.post_training.task_curation.datasets.arc import arc
from experiments.post_training.task_curation.datasets.environments import VERIFYIT_PACKAGE
from experiments.post_training.task_curation.datasets.tasktrove import qa
from experiments.post_training.task_curation.images.build import BASE_IMAGE
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.sources import all_sources
from experiments.post_training.task_curation.tasktrove.export import main
from experiments.post_training.task_curation.tests.conversion import converted_task, tasktrove_row

GRADER_IMAGE = "example.test/grader@sha256:" + "a" * 64


def archive_files(blob: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as archive:
        files = {}
        for member in archive:
            handle = archive.extractfile(member)
            if handle is not None:
                files[member.name] = handle.read()
        return files


@pytest.fixture
def normalized_row() -> tuple[dict, TaskSpec]:
    source = Source(
        dataset="open-thoughts/TaskTrove",
        revision="pinned",
        row="laion__all-puzzles-v2/tasks.parquet:0",
        importer_revision="1",
    )
    task = answer_task(RawRow("fixture", source, {}), prompt="Give the answer.", spec=ExactSpec(expected=("gold",)))
    task = task.model_copy(
        update={
            "resources": ResourceGroups(
                worker=(inline_resource("app/input.txt", b"public input"),),
                verifier=(inline_resource("private.txt", b"private reference"),),
                oracle=(
                    inline_resource("environment/Dockerfile", b"FROM python:3.12-slim\nWORKDIR /app\n"),
                    inline_resource("solution/solve.sh", b"echo gold > /app/answer.txt\n"),
                ),
            )
        }
    )
    return {
        "task_id": task.id,
        "task_json": task.model_dump_json(),
        "original_path": "fixture-task",
        "source_row": source.row,
    }, task


def test_harbor_lowering_preserves_delivery_and_private_resource_boundaries(normalized_row) -> None:
    row, _ = normalized_row
    record = harbor_record(
        row,
        fallback_actor_image=BASE_IMAGE,
        verifyit_package_root=VERIFYIT_PACKAGE,
        grader_image=GRADER_IMAGE,
        family="fixture",
    )
    files = archive_files(record.task_binary)
    config = tomllib.loads(files["task.toml"].decode())
    assert files["environment/Dockerfile"].decode().startswith("FROM python:3.12-slim\nWORKDIR /app\n")
    assert config["verifier"]["environment_mode"] == "shared"
    assert "environment" not in config["verifier"]
    assert "tests/Dockerfile" not in files
    assert not any(path.startswith("tests/public/") for path in files)
    assert not any(path.startswith("solution/") for path in files)
    assert not any(path.startswith(("environment/files/tests/", "environment/files/solution/")) for path in files)
    # Harbor's reserved spec filename bypasses our bundled runtime wrapper.
    assert "tests/verifier.toml" not in files
    assert files["tests/private.txt"] == b"private reference"
    assert files["environment/files/app/input.txt"] == b"public input"
    assert b"/app/answer.txt" in files["instruction.md"]
    assert not config["artifacts"]
    assert archive_files(record.solution_binary)["solution/solve.sh"] == b"echo gold > /app/answer.txt\n"


def test_source_recipe_provenance_does_not_override_other_datasets(normalized_row):
    row, converted = normalized_row
    task = converted.model_copy(update={"source": converted.source.model_copy(update={"dataset": "other"})})
    record = harbor_record(
        {**row, "task_json": task.model_dump_json()},
        fallback_actor_image=BASE_IMAGE,
        grader_image=GRADER_IMAGE,
        family="fixture",
    )
    expected_image = task.environment_requirements.docker_image or BASE_IMAGE
    assert archive_files(record.task_binary)["environment/Dockerfile"].decode().startswith(f"FROM {expected_image}\n")


@pytest.mark.parametrize(("letter", "reward"), [("B", 1.0), ("A", 0.0)])
def test_harbor_mcqa_file_format_matches_its_grader(normalized_row, tmp_path, letter, reward):
    row, original = normalized_row
    task = answer_task(
        RawRow(original.id, original.source, {}),
        prompt="Choose a fruit. A. Potato B. Apple\nReturn one option letter from A through B.",
        spec=McqSpec(expected="B", options=2),
    )
    task = task.model_copy(update={"resources": original.resources})
    files = archive_files(
        harbor_record(
            {**row, "task_json": task.model_dump_json()},
            grader_image=None,
            family="qa",
            fallback_actor_image=BASE_IMAGE,
            verifyit_package_root=VERIFYIT_PACKAGE,
        ).task_binary
    )
    (tmp_path / "answer.txt").write_text(letter)
    spec = parse_spec(files["tests/taskcompendium-verifier.toml"].decode())
    assert grade(spec, tmp_path, tmp_path).reward == reward


@pytest.mark.parametrize("role", ["actor", "grader"])
def test_harbor_rejects_unbuilt_context_instead_of_substituting_fallback_image(normalized_row, role):
    row, converted = normalized_row
    converted = converted.model_copy(update={"source": converted.source.model_copy(update={"dataset": "generic"})})
    environment = EnvironmentRequirements(
        docker_build=DockerBuildContext(
            files=(
                inline_resource("Dockerfile", b"FROM source:latest\nCOPY required.bin /required.bin\n"),
                inline_resource("required.bin", b"source environment data"),
            )
        )
    )
    if role == "actor":
        task = converted.model_copy(update={"environment_requirements": environment})
    else:
        task = converted.model_copy(update={"grader": converted.grader.model_copy(update={"environment": environment})})
    with pytest.raises(UnsupportedHarborTask, match="Docker build contexts"):
        harbor_record(
            {**row, "task_json": task.model_dump_json()},
            fallback_actor_image=BASE_IMAGE,
            verifyit_package_root=VERIFYIT_PACKAGE,
            grader_image=GRADER_IMAGE,
            family="fixture",
        )


@pytest.mark.parametrize("answer_type", [AnswerType.TEXT, AnswerType.FILE])
def test_harbor_public_staging_preserves_submitted_edits(normalized_row, tmp_path, answer_type) -> None:
    row, converted = normalized_row
    task = converted.model_copy(update={"source": converted.source.model_copy(update={"dataset": "generic"})})
    output = "app/answer.txt"
    if answer_type == AnswerType.FILE:
        output = "app/solution.py"
        task = task.model_copy(
            update={
                "answer_type": AnswerType.FILE,
                "output_paths": ("/app/solution.py",),
                "grader": ScriptGrader(
                    argv=("bash", "/tests/test.sh"),
                    answer_path=None,
                    environment=EnvironmentRequirements(docker_image=GRADER_IMAGE),
                    reward=FileReward(files=(RewardFile(path="/logs/verifier/reward.txt", format="number"),)),
                ),
                "resources": task.resources.model_copy(
                    update={"verifier": (inline_resource("test.sh", b"#!/bin/bash\nexit 0\n"),)}
                ),
            }
        )
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
    record = harbor_record(
        {**row, "task_json": edited.model_dump_json()},
        fallback_actor_image=BASE_IMAGE,
        verifyit_package_root=VERIFYIT_PACKAGE,
        grader_image=GRADER_IMAGE,
        family="fixture",
    )
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
    "spec,resources,valid,invalid",
    [
        (ExactSpec(expected=("gold",)), (), "gold", r"\boxed{gold}"),
        (
            JsonSchemaSpec(schema="schema.json"),
            (inline_resource("schema.json", b'{"type":"object","required":["name"]}'),),
            '{"name":"Ada"}',
            "{}",
        ),
    ],
)
def test_harbor_emitted_wrapper_grades_text_and_private_resources(
    normalized_row, spec, resources, valid, invalid, tmp_path
):
    _, original = normalized_row
    source = original.source.model_copy(update={"dataset": "generic"})
    task = answer_task(RawRow("fixture", source, {}), prompt="Give the answer.", spec=spec, resources=resources)
    record = harbor_record(
        {"task_json": task.model_dump_json(), "source_row": task.source.row, "original_path": "fixture-task"},
        fallback_actor_image=BASE_IMAGE,
        grader_image=GRADER_IMAGE,
        family="fixture",
    )
    files = archive_files(record.task_binary)
    instruction = files["instruction.md"].decode()
    assert "/app/answer.txt" in instruction
    assert not any(path.startswith("environment/files/") for path in files)
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    workspace, logs = tmp_path / "app", tmp_path / "logs"
    workspace.mkdir()
    # Relocate only the container's absolute paths; run the archive's wrapper and bundled runtime.
    script = (
        files["tests/test.sh"]
        .decode()
        .replace("/tests/", str(tmp_path / "tests") + "/")
        .replace(" --workspace /app ", f" --workspace {shlex.quote(str(workspace))} ")
        .replace(" --logs-dir /logs/verifier ", f" --logs-dir {shlex.quote(str(logs))} ")
    )
    for answer, expected in ((valid, 1.0), (invalid, 0.0)):
        (workspace / "answer.txt").write_text(answer)
        subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True)
        verdict = json.loads((logs / "verdict.json").read_text())
        assert verdict["status"] == "scored"
        assert verdict["reward"] == expected


def test_harbor_cli_joins_registry_metadata_and_accounts_for_unsupported_rows(normalized_row, tmp_path) -> None:
    row, converted = normalized_row
    source = all_sources()["Task Trove:" + row["source_row"].split("/", 1)[0]]
    unsupported = converted.model_copy(update={"grader": NoGrader(reason="Needs source validation")})
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
    table = pq.read_table(output_root / "tasks.parquet")
    # Independent 12-column contract consumed by Harbor's TaskTrove loader.
    assert [(field.name, str(field.type)) for field in table.schema] == [
        (name, "string")
        for name in ("path", "source", "family", "template_id", "converter", "mode", "dockerfile_id", "language")
    ] + [
        ("tags", "list<element: string>"),
        ("has_solution", "bool"),
        ("task_binary", "binary"),
        ("solution_binary", "binary"),
    ]
    assert all(field.nullable for field in table.schema)
    exported = table.to_pylist()
    assert len(exported) == 1
    assert exported[0]["family"] == source.info.family
    files = archive_files(exported[0]["task_binary"])
    assert {"instruction.md", "task.toml", "environment/Dockerfile", "tests/test.sh"} <= files.keys()
    assert not any(path.startswith("solution/") for path in files)
    config = tomllib.loads(files["task.toml"].decode())
    assert config["metadata"]["tasktrove_path"] == exported[0]["path"] == row["original_path"]
    assert config["metadata"]["tasktrove_source"] == exported[0]["source"] == row["source_row"].split("/", 1)[0]
    assert config["metadata"]["taskcompendium_id"] == converted.id
    assert report["source_id"] == source.info.id


def test_harbor_judge_receives_canonical_text_at_declared_path(tmp_path) -> None:
    answer_path = "/app/custom-answer.txt"
    source = next(source for source in qa.sources() if source.name == "knowledge-openqa")
    question = "Which planet is known as the red planet?"
    original = f"Write your concise final answer to `/app/response.txt`.\n\n{question}"
    pipeline = cast(CurationRecipe, source.config)
    task = converted_task(
        pipeline,
        tasktrove_row(
            {
                "instruction.md": original.encode(),
                "tests/test.sh": b"#!/bin/bash\nexit 99\n",
                "tests/sitecustomize.py": b"raise RuntimeError('archived runtime')\n",
                "environment/Dockerfile": b"FROM python:3.12-slim\nWORKDIR /app\n",
                "tests/verifier_data.json": json.dumps({"instruction": question, "expected_answers": ["Mars"]}).encode(),
            }
        ),
    )
    grader = cast(VerifyitGrader, task.grader)
    grader = grader.model_copy(update={"parameters": {**grader.parameters, "output": answer_path}})
    task = task.model_copy(update={"grader": grader})
    row = {
        "task_json": task.model_dump_json(),
        "original_path": "judge-fixture.tar.gz",
        "source_row": cast(HfSource, pipeline.source).files[0] + ":0",
    }
    record = harbor_record(
        row,
        fallback_actor_image=BASE_IMAGE,
        verifyit_package_root=VERIFYIT_PACKAGE,
        grader_image=GRADER_IMAGE,
        family=source.info.family,
    )
    files = archive_files(record.task_binary)
    config = tomllib.loads(files["task.toml"].decode())
    instruction = files["instruction.md"].decode()
    assert "/app/response.txt" not in instruction
    assert files["tests/source/test.sh"] == b"#!/bin/bash\nexit 99\n"
    assert "tests/sitecustomize.py" not in files
    assert not any(name.startswith("environment/files/tests/") for name in files)
    workspace = tmp_path / "app"
    workspace.mkdir()
    candidate = "Mars, with a complete explanation.\n"
    assert not config["artifacts"]
    transferred = workspace / Path(answer_path).relative_to("/app")
    transferred.write_text(candidate)
    # Exercise the judge's file-read boundary without calling any judge model.
    spec = parse_spec(files["tests/taskcompendium-verifier.toml"].decode())
    assert read_output(spec, workspace) == candidate


@pytest.mark.parametrize("mode", ["inductive", "transductive"])
def test_harbor_arc_runs_shipped_scorer_and_preserves_submission_paths(mode, tmp_path):
    source = next(source for source in arc.sources() if source.name == f"tasktrove-arc_{mode}")
    grid = [[0, 1], [2, 9]]
    data = {"test_cases": [{"input": grid, "output": grid}]} if mode == "inductive" else {"expected_output": grid}
    task = converted_task(
        cast(CurationRecipe, source.config),
        tasktrove_row(
            {"instruction.md": b"Solve the grid puzzle.", "tests/verifier_data.json": json.dumps(data).encode()}
        ),
    )
    record = harbor_record(
        {"task_json": task.model_dump_json(), "original_path": "arc.tar.gz", "source_row": "arc/tasks.parquet:0"},
        fallback_actor_image=BASE_IMAGE,
        verifyit_package_root=VERIFYIT_PACKAGE,
        grader_image=GRADER_IMAGE,
        family=source.info.family,
    )
    files = archive_files(record.task_binary)
    config = tomllib.loads(files["task.toml"].decode())
    assert tuple(artifact["source"] for artifact in config["artifacts"]) == task.output_paths
    assert not any(path.startswith("environment/files/tests/") for path in files)
    for path, content in files.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.endswith((".py", ".sh", ".toml")):
            content = (
                content.decode()
                .replace("/tests", str(tmp_path / "tests"))
                .replace("/app/", str(tmp_path / "app") + "/")
                .replace("/logs/", str(tmp_path / "logs") + "/")
                .encode()
            )
        target.write_bytes(content)
    (tmp_path / "app").mkdir()
    answer_path = tmp_path / "app" / ("solution.py" if mode == "inductive" else "answer.txt")
    positive = arc.literal_transform(grid) if mode == "inductive" else arc.grid_text(grid)
    negative = "def transform(grid): return [[8]]" if mode == "inductive" else "8"
    for candidate, expected in [(positive, 1), (negative, 0)]:
        # Inductive grading removes hidden config before executing the candidate.
        if mode == "inductive":
            (tmp_path / "tests/config.json").write_bytes(files["tests/config.json"])
        answer_path.write_text(candidate)
        result = subprocess.run(["bash", str(tmp_path / "tests/test.sh")], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert float((tmp_path / "logs/verifier/reward.txt").read_text()) == expected


@pytest.mark.parametrize(
    "script", ["print('')", "print('log only')", "print('nan')", "print(1); raise RuntimeError('grader failed')"]
)
def test_harbor_stdout_failures_do_not_emit_a_reward(script, tmp_path):
    source = next(source for source in arc.sources() if source.name == "tasktrove-arc_transductive")
    task = converted_task(
        cast(CurationRecipe, source.config),
        tasktrove_row({"instruction.md": b"Solve.", "tests/verifier_data.json": b'{"expected_output":[[1]]}'}),
    )
    task = task.model_copy(
        update={
            "resources": task.resources.model_copy(update={"verifier": (inline_resource("grade.py", script.encode()),)}),
            "grader": ScriptGrader(
                argv=("python3", "/tests/grade.py"),
                answer_path=None,
                environment=EnvironmentRequirements(docker_image=GRADER_IMAGE),
                reward=StdoutReward(),
            ),
        }
    )
    files = archive_files(
        harbor_record(
            {"task_json": task.model_dump_json(), "original_path": "arc.tar.gz", "source_row": "arc/tasks.parquet:0"},
            fallback_actor_image=BASE_IMAGE,
            verifyit_package_root=VERIFYIT_PACKAGE,
            grader_image=GRADER_IMAGE,
            family=source.info.family,
        ).task_binary
    )
    for path, content in files.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    script = (
        files["tests/test.sh"]
        .decode()
        .replace("/tests/", str(tmp_path / "tests") + "/")
        .replace("/logs/", str(tmp_path / "logs") + "/")
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode != 0
    assert not (tmp_path / "logs/verifier/reward.txt").exists()


@pytest.fixture
def repository_task(normalized_row) -> TaskSpec:
    _, task = normalized_row
    environment = EnvironmentRequirements(
        capabilities=("git_repository",),
        docker_build=verifyit_build_context("FROM python:3.12-slim\nWORKDIR /testbed\n", (), package=VERIFYIT_PACKAGE),
    )
    grader = ScriptGrader(
        argv=("bash", "/tests/test.sh"),
        cwd="/testbed",
        environment=environment,
        answer_path=None,
        artifacts=(VerifierArtifact(source="/testbed", target="/testbed", kind=ArtifactKind.DIRECTORY),),
        reward=FileReward(files=(RewardFile(path="/logs/verifier/reward.txt", format="number"),)),
    )
    return task.model_copy(
        update={
            "answer_type": AnswerType.WORKSPACE_STATE,
            "environment_requirements": environment,
            "grader": grader,
            "tags": ("swe-repo",),
            "resources": ResourceGroups(
                verifier=(
                    inline_resource("test.sh", b"#!/bin/bash\nexit 0\n"),
                    inline_resource(
                        "taskcompendium-verifier.toml",
                        render_spec(PytestSpec(paths=("test_product.py",), workspace="/testbed")).encode(),
                    ),
                )
            ),
        }
    )


def test_harbor_repository_uses_shared_actor_state(repository_task, tmp_path):
    workspace = tmp_path / "testbed"
    workspace.mkdir()
    product = workspace / "product.py"
    product.write_text("candidate repair")
    dependencies = tmp_path / "installed-dependencies"
    dependencies.write_text("actor-installed package")
    probe = f"#!/bin/bash\ncat product.py; cat {shlex.quote(str(dependencies))}\n".encode()
    task = repository_task.model_copy(
        update={
            "grader": repository_task.grader.model_copy(update={"cwd": str(workspace)}),
            "resources": repository_task.resources.model_copy(
                update={
                    "verifier": tuple(
                        inline_resource(resource.path, probe) if resource.path == "test.sh" else resource
                        for resource in repository_task.resources.verifier
                    )
                }
            ),
        }
    )
    row = {
        "task_json": task.model_dump_json(),
        "original_path": "swesmith-fixture",
        "source_row": "swesmith/tasks.parquet:0",
    }
    files = archive_files(
        harbor_record(
            row, fallback_actor_image=BASE_IMAGE, verifyit_package_root=VERIFYIT_PACKAGE, grader_image=None, family="swe"
        ).task_binary
    )
    config = tomllib.loads(files["task.toml"].decode())
    assert config["verifier"]["environment_mode"] == "shared"
    assert "environment" not in config["verifier"]
    assert not config["artifacts"]
    assert "tests/Dockerfile" not in files and "tests/verifier.toml" not in files
    assert not any(path.startswith("tests/public/") for path in files)
    context = cast(DockerBuildContext, task.environment_requirements.docker_build)
    for resource in context.files:
        content = files["environment/" + resource.path]
        if resource.path == "Dockerfile":
            assert content.startswith(resource_bytes(resource))
        else:
            assert content == resource_bytes(resource)
    # Execute the complete emitted wrapper in the existing actor tree.
    script = files["tests/test.sh"].decode()
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "candidate repairactor-installed package"
    assert product.read_text() == "candidate repair"


@pytest.mark.parametrize("line", ["RUN pip install rewardkit", "COPY tests/missing.py /tests/missing.py"])
def test_harbor_repository_retains_old_environment_exclusions(repository_task, line):
    context = cast(DockerBuildContext, repository_task.environment_requirements.docker_build)
    files = tuple(
        (
            inline_resource(resource.path, resource_bytes(resource) + (line + "\n").encode())
            if resource.path == "Dockerfile"
            else resource
        )
        for resource in context.files
    )
    environment = repository_task.environment_requirements.model_copy(
        update={"docker_build": context.model_copy(update={"files": files})}
    )
    task = repository_task.model_copy(update={"environment_requirements": environment})
    row = {
        "task_json": task.model_dump_json(),
        "original_path": "swesmith-fixture",
        "source_row": "swesmith/tasks.parquet:0",
    }
    with pytest.raises(UnsupportedHarborTask):
        harbor_record(
            row, fallback_actor_image=BASE_IMAGE, verifyit_package_root=VERIFYIT_PACKAGE, grader_image=None, family="swe"
        )


def test_openqa_keeps_conversion_but_filters_disclosed_reference():
    source = next(source for source in qa.sources() if source.name == "science-openqa")
    reference = "alpha beta gamma"
    task = converted_task(
        cast(CurationRecipe, source.config),
        tasktrove_row(
            {
                "instruction.md": f"Prove the answer is {reference}.".encode(),
                "tests/verifier_data.json": (
                    json.dumps({"instruction": "Prove the identity.", "reference_answer": reference}).encode()
                ),
            }
        ),
    )
    with pytest.raises(UnsupportedHarborTask, match="gold_leak"):
        harbor_record(
            {"task_json": task.model_dump_json(), "source_row": task.source.row, "original_path": "leaked-reference"},
            grader_image=GRADER_IMAGE,
            family=source.info.family,
            fallback_actor_image=BASE_IMAGE,
        )


def test_harbor_nonrepository_shared_wrapper_reads_actor_edits_and_dependencies(normalized_row, tmp_path):
    row, original = normalized_row
    workspace = tmp_path / "actor"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("edited answer")
    dependencies = tmp_path / "installed"
    dependencies.mkdir()
    (dependencies / "actor_dependency.py").write_text('expected = "edited answer"\n')
    checker = b"""import os
from pathlib import Path
import actor_dependency
answer = (Path(os.environ["VERIFYIT_WORKSPACE"]) / "answer.txt").read_text()
print(float(answer == actor_dependency.expected))
"""
    package = verifyit_package(
        ScriptSpec(path="check.py", workspace=str(workspace)),
        (inline_resource("check.py", checker),),
        environment=EnvironmentRequirements(docker_image=GRADER_IMAGE),
    )
    task = original.model_copy(
        update={
            "grader": package.grader,
            "resources": original.resources.model_copy(update={"verifier": package.resources}),
        }
    )
    files = archive_files(
        harbor_record(
            {**row, "task_json": task.model_dump_json()},
            fallback_actor_image=BASE_IMAGE,
            verifyit_package_root=VERIFYIT_PACKAGE,
            grader_image=None,
            family="fixture",
        ).task_binary
    )
    config = tomllib.loads(files["task.toml"].decode())
    assert config["verifier"]["environment_mode"] == "shared"
    assert not config["artifacts"]
    tests = tmp_path / "tests"
    for name, content in files.items():
        if name.startswith("tests/"):
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    logs = tmp_path / "logs"
    script = files["tests/test.sh"].decode().replace("/tests/", str(tests) + "/").rstrip()
    script += " --logs-dir " + shlex.quote(str(logs))
    environment = {**os.environ, "PYTHONPATH": str(dependencies)}
    for candidate, expected in [("edited answer", 1.0), ("initial answer", 0.0)]:
        (workspace / "answer.txt").write_text(candidate)
        result = subprocess.run(["bash", "-c", script], env=environment, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert float((logs / "reward.txt").read_text()) == expected
