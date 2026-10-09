# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from click.testing import CliRunner
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import ArtifactStep, StepContext, run
from taskcompendium.models import NoGrader
from upath import UPath

from experiments.post_training.task_curation.datasets.tasktrove import nl2bash
from experiments.post_training.task_curation.harbor import archive_bytes
from experiments.post_training.task_curation.harbor_export import harbor_export_step
from experiments.post_training.task_curation.harbor_export_contract import VerifierPayloadIdentity
from experiments.post_training.task_curation.rl_smoke import main, smoke_step
from experiments.post_training.task_curation.tests.conversion import converted_task, tasktrove_row

GRADER_IMAGE = "example.test/grader@sha256:" + "a" * 64


@pytest.fixture
def normalized_rows():
    source = nl2bash.sources()[0]
    assert source.pipeline is not None
    task = converted_task(
        source.pipeline,
        tasktrove_row(
            {
                "instruction.md": b"List /workspace and save output to /output/command_capture.txt.",
                "environment/Dockerfile": b"FROM debian:bookworm\nWORKDIR /workspace\n",
                "tests/test.sh": b"#!/bin/bash\nexit 1\n",
                "tests/verifier_data.json": b'{"expected_output": "notes.txt"}',
            }
        ),
    )
    rejected = task.model_copy(update={"grader": NoGrader(reason="fixture missing grader")})
    rows = [
        {
            "task_id": value.id,
            "task_json": value.model_dump_json(),
            "source_row": nl2bash.CONFIG + "/tasks.parquet:" + str(index),
            "original_path": str(index),
        }
        for index, value in enumerate((task, rejected))
    ]
    return source.name, rows


@pytest.mark.parametrize("protocol", ["file", "memory"])
def test_export_artifact_resolves_into_smoke_launch_document(normalized_rows, tmp_path, protocol, monkeypatch):
    root = UPath(str(tmp_path)) if protocol == "file" else UPath("memory://harbor-export" + str(tmp_path))
    input_root = root / "normalized"
    (input_root / "normalize").mkdir(parents=True)
    source_name, rows = normalized_rows
    (input_root / "manifest.json").write_text(json.dumps({"source": source_name}))
    with (input_root / "normalize/part-00000.parquet").open("wb") as output:
        pq.write_table(pa.Table.from_pylist(rows), output)
    normalized = ArtifactStep.adopt("tests/normalized", "2026.10.09", str(input_root))
    export = harbor_export_step(normalized, name="tests/export", version="2026.10.09", grader_image=GRADER_IMAGE)
    output_root = export.path(str(root))
    monkeypatch.setenv("MARIN_PREFIX", str(root))
    result = run(export, max_concurrent=1)[0]
    assert (result.exported_rows, result.rejected_rows) == (1, 1)
    manifest = json.loads((UPath(output_root) / "manifest.json").read_text())
    with (UPath(output_root) / "tasks.parquet").open("rb") as exported_file:
        exported = pq.read_table(exported_file).to_pylist()
    identity = VerifierPayloadIdentity()
    identity.add(exported[0]["task_binary"])
    assert manifest["verify_tool_ref"] == identity.ref == result.verify_tool_ref
    assert not manifest["runtime_verified"]

    with build_context(BuildContext(versions=VersionCodex(default="2026.10.09"))):
        step = smoke_step(export)
    launch = yaml.safe_load(
        step.build_config(
            StepContext.for_run(
                output_path=str(root / "rl"), prefix=str(root), runtime_args=step.runtime_args, deps=step.deps
            )
        ).launch_config_yaml
    )
    source = json.loads(json.dumps(launch))["inputs"]["train_data"][0]
    assert source["uri"] == str(UPath(output_root) / "tasks.parquet")
    assert source["relative_path"] == "tasks.parquet"
    assert source["verifier_ref"] == identity.ref
    assert source["kind"] == "tasktrove_parquet"
    selection = source["selection"]
    assert exported[0]["source"] in selection["sources"]
    assert exported[0]["mode"] in selection["modes"]
    assert set(selection["tags"]) <= set(exported[0]["tags"])


@pytest.mark.parametrize(
    "path,content,mode",
    [
        ("tests/test.sh", b"new wrapper", "755"),
        ("tests/Dockerfile", b"FROM different:tag", "644"),
        ("tests/helper.py", b"private checker", "755"),
        ("task.toml", b"new dispatch", "644"),
    ],
)
def test_verifier_identity_tracks_payload_recipe_and_modes(path, content, mode):
    files = {
        "tests/test.sh": b"original wrapper",
        "tests/Dockerfile": b"FROM source:tag",
        "tests/helper.py": b"private checker",
        "task.toml": b"dispatch",
        "instruction.md": b"public prompt",
    }
    original = VerifierPayloadIdentity()
    original.add(archive_bytes(files, {}))
    changed = VerifierPayloadIdentity()
    changed.add(archive_bytes({**files, path: content}, {path: mode}))
    assert changed.ref != original.ref
    equivalent = VerifierPayloadIdentity()
    equivalent.add(archive_bytes({**files, "instruction.md": b"different public prompt"}, {}))
    equivalent.add(archive_bytes(files, {}))
    assert equivalent.ref == original.ref


def test_smoke_plan_uses_export_dependency_without_executing_input(tmp_path: Path):
    missing = tmp_path / "not-materialized"
    result = CliRunner().invoke(
        main,
        [
            "--version",
            "2026.10.09",
            "--normalized-source",
            str(missing),
            "--normalized-name",
            "tests/normalized",
            "--normalized-version",
            "2026.10.09",
            "--grader-image",
            GRADER_IMAGE,
        ],
    )
    assert result.exit_code == 0, result.output
    assert not missing.exists()
    assert "data/rl/nl2bash-harbor" in result.output
    assert "tests/normalized" in result.output
