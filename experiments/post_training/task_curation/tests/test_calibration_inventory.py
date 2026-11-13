# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Calibration inventories distinguish executable native controls from repair candidates."""

import base64
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import NoGrader, Source, TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow

from experiments.post_training.task_curation.calibration_inventory import inventory_source, task_blockers
from experiments.post_training.task_curation.datasets.native_harbor import native_task
from experiments.post_training.task_curation.full_qc_artifacts import FullQCInputs
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import write_text_population
from experiments.post_training.task_curation.remediation_publish import ReleasePlan
from experiments.post_training.task_curation.retention_plan import RetentionPlan
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task


def test_native_inventory_preserves_pinned_multistage_environment_and_detects_fragment_oracle():
    base = "ubuntu@sha256:" + "a" * 64
    files = {
        "instruction.md": b"Complete the code.",
        "task.toml": b'[environment]\n[verifier]\nenvironment_mode="shared"\n',
        "environment/Dockerfile": f"FROM {base} AS builder\nRUN true\nFROM builder\n".encode(),
        "tests/test.sh": b"true\n",
        "solution/completion.txt": b"return 42\n",
    }
    data = {
        "files": {path: base64.b64encode(content).decode() for path, content in files.items()},
        "file_metadata": {path: {"mode": "0644", "mtime_ns": 0} for path in files},
        "archive_links": {},
        "archive_sha256": hashlib.sha256(b"native").hexdigest(),
    }
    task = native_task(
        RawRow("native", Source(dataset="fixture", revision="pin", row="0", importer_revision="1"), data),
        ConversionContext({}, None),
    )
    assert isinstance(task, TaskSpec)
    before = task.model_dump_json()
    assert task_blockers(task) == {"oracle_without_solve_sh"}
    assert task.model_dump_json() == before


def test_mutable_native_environment_is_a_repair_even_with_executable_oracle():
    files = {
        "instruction.md": b"Repair the repository.",
        "task.toml": b'[environment]\n[verifier]\nenvironment_mode="shared"\n',
        "environment/Dockerfile": b"FROM python:3.12\nRUN pip install pytest\n",
        "tests/test.sh": b"true\n",
        "solution/solve.sh": b"true\n",
    }
    data = {
        "files": {path: base64.b64encode(content).decode() for path, content in files.items()},
        "file_metadata": {path: {"mode": "0755", "mtime_ns": 0} for path in files},
        "archive_links": {},
        "archive_sha256": hashlib.sha256(b"native").hexdigest(),
    }
    task = native_task(
        RawRow("native", Source(dataset="fixture", revision="pin", row="0", importer_revision="1"), data),
        ConversionContext({}, None),
    )
    assert isinstance(task, TaskSpec)
    assert task_blockers(task) == {"mutable_docker_base"}


def test_inventory_accounts_for_frozen_drops_and_rejects_changed_source_bytes(tmp_path):
    documents = [json.loads(prebuilt_task(name)["lowered_task_spec"])["task"] for name in ("keep", "drop")]
    for task in documents:
        task["context"]["events"][0]["content"] = task["id"]
    root = str(tmp_path / "population")
    files = write_text_population(((task, {}) for task in documents), root)
    population = SourceTextPopulation(
        path=root,
        source_id="TaskTrove:fixture",
        dataset_id="fixture",
        dataset_revision="pin",
        source_version="1",
        converter_revision="converter",
        text_renderer_version="1",
        task_count=files.task_count,
        text_sha256=files.text_sha256,
        task_specs_sha256=files.task_specs_sha256,
        main_output_dir=files.main_output_dir,
        specs_output_dir=files.specs_output_dir,
        converter_rejections={},
        admission_rejections={},
    )
    write_record(
        ArtifactRecord(
            output_path=root, result_type=result_type_name(SourceTextPopulation), result=population.result_payload()
        )
    )
    drops = tmp_path / "drops.parquet"
    pq.write_table(pa.table({"task_id": ["drop"]}), drops)
    plan = ReleasePlan(
        source_id="TaskTrove:fixture",
        source_dataset_id="fixture",
        source_revision="pin",
        converter_revision="converter",
        converter_version="1",
        input_artifact=root,
        task_specs_sha256=files.task_specs_sha256,
        raw_count=2,
        unique_count=2,
        within_source_duplicates=0,
        raw_lexical_count=1,
        unique_lexical_count=1,
        combined_drop_count=1,
        retained_count=1,
        multiplicities={"1": 2},
        drop_manifest_uri=str(drops),
        combined_drop_manifest_uri=str(drops),
        evidence_url="fixture",
        lexical_use="train",
        census_evidence_url="fixture",
    )
    index_path = tmp_path / "contracts.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": "keep",
                    "task_sha256": canonical_sha256(TaskSpec.model_validate(documents[0]).model_dump(mode="json")),
                    "runtime_sha256": canonical_sha256({}),
                }
            ]
        ),
        index_path,
    )
    inputs = FullQCInputs(
        path=str(tmp_path / "inputs"),
        source_id=plan.source_id,
        population=population,
        retention=RetentionPlan.model_validate(plan.model_dump()),
        drop_manifest_sha256=file_sha256(StoragePath(str(drops))),
        index_path=str(index_path),
        index_sha256=file_sha256(StoragePath(str(index_path))),
        retained_count=1,
        task_bytes=1,
        oracle_tasks=0,
        missing_execution_requirements=0,
        grading_environments={},
    )
    write_record(
        ArtifactRecord(
            output_path=inputs.path, result_type=result_type_name(FullQCInputs), result=inputs.result_payload()
        )
    )
    summary = inventory_source(plan, str(tmp_path / "inventory"), inputs.path)
    assert summary["retained_count"] == 1
    assert summary["oracle_tasks"] == 0
    assert summary["qualification_status"] == "pending"
    assert pq.read_table(summary["ledger_path"]).column("task_id").to_pylist() == ["keep"]

    shard = next(Path(files.specs_output_dir).glob("*.parquet"))
    rows = pq.read_table(shard).to_pylist()
    index = next(i for i, row in enumerate(rows) if row["id"] == "keep")
    document = json.loads(rows[index]["task_spec"])
    document["context"]["events"][0]["content"] = "Changed instruction"
    rows[index]["task_spec"] = json.dumps(document)
    pq.write_table(pa.Table.from_pylist(rows), shard)
    with pytest.raises(ValueError, match="Task or runtime differs from prepared contract"):
        inventory_source(plan, str(tmp_path / "changed-inventory"), inputs.path)


def test_missing_grader_is_unavailable_without_claiming_a_boundary_mismatch():
    task = TaskSpec.model_validate(json.loads(prebuilt_task("preference")["lowered_task_spec"])["task"])
    task = task.model_copy(update={"grader": NoGrader(reason="Preference pair has no executable grader")})
    blockers = task_blockers(task)
    assert "grading_environment_unavailable" in blockers
    assert "grading_environment_mismatch" not in blockers
