# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The retained reader consumes only the published bytes it can verify."""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import DatasetRelease
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    GraderMachine,
    PlainText,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import IntendedUse
from taskcompendium.pipeline.source_processing import SourceProcessingMode

import experiments.post_training.task_curation.retained_release as retained_release
from experiments.post_training.task_curation.campaign import CampaignArtifact
from experiments.post_training.task_curation.retained_release import (
    RetainedCohort,
    RetainedImport,
    run_retained_import,
)
from experiments.post_training.task_curation.source import SourceReference

FIXTURE_IMAGE = "ubuntu:24.04@sha256:" + "a" * 64


@pytest.fixture
def published_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A one-task retained release served from a local file instead of Hugging Face."""
    spec = TaskSpec(
        id="retained-task",
        context=ConversationInput(
            events=(TextMessage(role="user", content="grade this"),),
        ),
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_image=FIXTURE_IMAGE,
            working_directory="/app",
            setup_commands=("true",),
        ),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=ScriptGrader(
            argv=("true",),
            placement=GraderMachine(
                environment=EnvironmentRequirements(
                    command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=FIXTURE_IMAGE
                )
            ),
        ),
        source=Source(
            dataset="upstream/fixture",
            revision="b" * 40,
            row="retained-row",
            importer_revision="2026.11.13.1",
        ),
    )
    encoded = spec.model_dump_json()
    local = tmp_path / "published" / "tasks.parquet"
    local.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "task_spec": [encoded],
                "task_sha256": [hashlib.sha256(encoded.encode()).hexdigest()],
                "runtime": ["{}"],
            }
        ),
        local,
    )
    monkeypatch.setattr(retained_release, "hf_hub_download", lambda *args, **kwargs: str(local), raising=True)
    return local


def _cohort(tasks: Path, tmp_path: Path, *, expected_rows: int) -> RetainedCohort:
    return RetainedCohort(
        name="fixture-retained",
        version="2026.11.13.1",
        release=DatasetRelease(
            repo="open-athena/fixture-retained",
            revision="a" * 40,
            url="https://huggingface.co/datasets/open-athena/fixture-retained/tree/" + "a" * 40,
        ),
        original_dataset=SourceReference(
            "upstream/fixture",
            "b" * 40,
            "https://huggingface.co/datasets/upstream/fixture/tree/" + "b" * 40,
        ),
        expected_rows=expected_rows,
        tasks_parquet_sha256=file_sha256(StoragePath(str(tasks))),
        intended_use=IntendedUse.EVAL,
        evidence_url="s3://example/fixture/result.json",
    )


def test_quick_import_writes_the_assessment_contract(tmp_path: Path, published_tasks: Path) -> None:
    """The manifest carries the datasets views and source the assessment reads."""
    invocation = RetainedImport(
        _cohort(published_tasks, tmp_path, expected_rows=1),
        str(tmp_path / "imported"),
        SourceProcessingMode.QUICK,
        None,
        None,
        None,
    )
    result = run_retained_import(invocation)
    manifest = json.loads((tmp_path / "imported/manifest.json").read_text())
    assert manifest["source"] == "fixture-retained"
    assert manifest["mode"] == SourceProcessingMode.QUICK.value
    views = manifest["datasets"]
    assert {"normalize", "review", "verify", "final"} <= set(views)
    for view in ("review", "verify", "final"):
        assert (Path(views[view]) / "part-000000.parquet").exists()
    final_rows = pq.read_table(Path(views["final"]) / "part-000000.parquet").num_rows
    assert final_rows == 0  # quick admits nothing
    assert manifest["admitted_rows"] == 0
    assert manifest["processed_rows"] == 1 and manifest["unprocessed_rows"] == 0
    assert isinstance(result, CampaignArtifact)


def test_admitted_import_populates_the_final_view_with_normalized_rows(
    tmp_path: Path, published_tasks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An admitted run's final view carries the normalized rows the assessment selects.

    The lexical qualification itself is exercised at bind time (the pin-table
    validators); this test stubs it to focus on the evidence views.
    """
    monkeypatch.setattr(retained_release, "require_lexical_qualification", lambda *args, **kwargs: None)
    invocation = RetainedImport(
        _cohort(published_tasks, tmp_path, expected_rows=1),
        str(tmp_path / "imported"),
        SourceProcessingMode.FULL,
        None,
        None,
        None,
    )
    run_retained_import(invocation)
    manifest = json.loads((tmp_path / "imported/manifest.json").read_text())
    views = manifest["datasets"]
    final = pq.read_table(Path(views["final"]) / "part-000000.parquet")
    assert final.num_rows == 1
    assert final.column("task_id").to_pylist() == ["retained-task"]
    normalized = pq.read_table(Path(views["normalize"]) / "part-000000.parquet", columns=["task_id"])
    assert normalized.column("task_id").to_pylist() == ["retained-task"]
    for view in ("review", "verify"):
        table = pq.read_table(Path(views[view]) / "part-000000.parquet")
        assert table.num_rows == 0
        assert {"task_id", "admission", "filter_reasons"} <= set(table.column_names)


def test_a_row_count_or_digest_drift_fails_closed(tmp_path: Path, published_tasks: Path) -> None:
    """A published release that differs from its receipt refuses to import."""
    cohort = _cohort(published_tasks, tmp_path, expected_rows=2)
    invocation = RetainedImport(cohort, str(tmp_path / "imported"), SourceProcessingMode.QUICK, None, None, None)
    with pytest.raises(ValueError, match="count differs"):
        run_retained_import(invocation)
