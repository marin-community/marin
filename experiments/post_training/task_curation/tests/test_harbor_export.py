# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import verifyit
import yaml
from marin.execution.artifact import FingerprintMismatchError
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import ArtifactStep, StepContext, run
from taskcompendium.harbor import export as harbor
from taskcompendium.harbor.export import VerifierPayloadIdentity
from taskcompendium.models import NoGrader
from upath import UPath

from experiments.post_training.task_curation.datasets.environments import VERIFYIT_PACKAGE
from experiments.post_training.task_curation.datasets.tasktrove import nl2bash
from experiments.post_training.task_curation.pipeline import CurationRecipe
from experiments.post_training.task_curation.rl_smoke import smoke_step
from experiments.post_training.task_curation.tasktrove.export import harbor_export_step
from experiments.post_training.task_curation.tests.conversion import converted_task, tasktrove_row

GRADER_IMAGE = "example.test/grader@sha256:" + "a" * 64


@pytest.fixture
def normalized_rows():
    source = nl2bash.sources()[0]
    task = converted_task(
        cast(CurationRecipe, source.config),
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
    return source, rows


@pytest.mark.parametrize("protocol", ["file", "memory"])
def test_export_artifact_resolves_into_smoke_launch_document(normalized_rows, tmp_path, protocol, monkeypatch):
    root = UPath(str(tmp_path)) if protocol == "file" else UPath("memory://harbor-export" + str(tmp_path))
    input_root = root / "normalized"
    (input_root / "normalize").mkdir(parents=True)
    declaration, rows = normalized_rows
    (input_root / "manifest.json").write_text(json.dumps({"source": declaration.name}))
    with (input_root / "normalize/part-00000.parquet").open("wb") as output:
        pq.write_table(pa.Table.from_pylist(rows), output)
    normalized = ArtifactStep.adopt("tests/normalized", "2026.10.09", str(input_root))
    bound = replace(declaration, info=replace(declaration.info, id="fixture-atlas-id", family="fixture-family"))
    export = harbor_export_step(
        normalized, source=bound, name="tests/export", version="2026.10.09", grader_image=GRADER_IMAGE
    )
    output_root = export.path(str(root))
    monkeypatch.setenv("MARIN_PREFIX", str(root))
    result = run(export, max_concurrent=1)[0]
    assert (result.exported_rows, result.rejected_rows) == (1, 1)
    manifest = json.loads((UPath(output_root) / "manifest.json").read_text())
    with (UPath(output_root) / "tasks.parquet").open("rb") as exported_file:
        exported = pq.read_table(exported_file).to_pylist()
    assert manifest["source_id"] == "fixture-atlas-id"
    assert exported[0]["family"] == "fixture-family"
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
    source = launch["inputs"]["train_data"][0]
    assert source["uri"] == str(UPath(output_root) / "tasks.parquet")
    assert source["relative_path"] == "tasks.parquet"
    assert source["verifier_ref"] == identity.ref
    assert source["kind"] == "tasktrove_parquet"


@pytest.mark.parametrize(
    "changed_file",
    [
        Path(verifyit.__file__).with_name("candidate_file.py"),
        Path(harbor.__file__).with_name("tasktrove.py"),
        VERIFYIT_PACKAGE / "pyproject.toml",
    ],
)
def test_export_dependency_edit_invalidates_fingerprint_pin(changed_file, tmp_path, monkeypatch):
    normalized = ArtifactStep.adopt("tests/normalized", "2026.10.09", str(tmp_path / "not-materialized"))
    original_read = Path.read_bytes
    harbor.verifier_runtime.cache_clear()
    before = harbor_export_step(
        normalized, source=nl2bash.sources()[0], name="tests/export", version="2026.10.09", grader_image=GRADER_IMAGE
    )

    # Change one filesystem input, without editing the checkout or materializing the source.
    def edited_read(path):
        content = original_read(path)
        return content + b"\n# changed export input\n" if path == changed_file else content

    monkeypatch.setattr(Path, "read_bytes", edited_read)
    harbor.verifier_runtime.cache_clear()
    after = harbor_export_step(
        normalized, source=nl2bash.sources()[0], name="tests/export", version="2026.10.09", grader_image=GRADER_IMAGE
    )
    harbor.verifier_runtime.cache_clear()
    assert after.fingerprint() != before.fingerprint()
    with pytest.raises(FingerprintMismatchError):
        replace(after, expected_fingerprint=before.fingerprint()).lower()
    assert not (tmp_path / "not-materialized").exists()


@pytest.mark.parametrize("field", ["id", "family"])
def test_source_metadata_changes_export_fingerprint(field, tmp_path):
    normalized = ArtifactStep.adopt("tests/normalized", "2026.10.09", str(tmp_path / "not-materialized"))
    source = nl2bash.sources()[0]
    changed = replace(source, info=replace(source.info, **{field: "changed"}))
    before = harbor_export_step(
        normalized, source=source, name="tests/export", version="2026.10.09", grader_image=GRADER_IMAGE
    )
    after = harbor_export_step(
        normalized, source=changed, name="tests/export", version="2026.10.09", grader_image=GRADER_IMAGE
    )
    assert after.fingerprint() != before.fingerprint()
