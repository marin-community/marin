# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The retained reader consumes only the published bytes it can verify."""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import Cohort, DatasetRelease
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.integrity import DeliveredVerifierFile, RootDeliveryContract, SharedDependencyState
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    ExecutionRequirements,
    GraderMachine,
    PlainText,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.assessment import AssessmentSampling, assessment_inputs, prepare_assessment
from taskcompendium.pipeline.models import IntendedUse
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from zephyr.context import ZephyrContext
from zephyr.readers import load_parquet

import experiments.post_training.task_curation.retained_release as retained_release
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
        execution_requirements=ExecutionRequirements(
            cpus=2, memory_mb=2048, storage_mb=4096, agent_timeout_sec=120, verifier_timeout_sec=60
        ),
        integrity_contract=RootDeliveryContract(
            dependency_state=SharedDependencyState.AGENT_MUTABLE,
            verifier_files=(DeliveredVerifierFile(name="test.sh", mode=0o755, sha256="b" * 64),),
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


def _cohort(tasks: Path, *, expected_rows: int) -> RetainedCohort:
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


def test_full_retained_import_prepares_an_admitted_assessment(
    tmp_path: Path, published_tasks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assessment selects the imported task and writes its evidence packet."""
    # Lexical qualification is checked by the pin-table validator; this fixture
    # tests the assessment boundary without a publication receipt.
    monkeypatch.setattr(retained_release, "require_lexical_qualification", lambda *args, **kwargs: None)
    imported = tmp_path / "imported"
    invocation = RetainedImport(
        _cohort(published_tasks, expected_rows=1),
        str(imported),
        SourceProcessingMode.FULL,
        None,
        None,
        None,
    )
    run_retained_import(invocation)
    inputs = tmp_path / "assessment"
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        prepare_assessment(
            context,
            curated_root=str(imported),
            sampling=AssessmentSampling(admitted=1, excluded=0),
            output_root=str(inputs),
        )
    assessment = assessment_inputs(str(inputs))
    assert [(row.task_id, row.cohort) for row in assessment.sample] == [("retained-task", Cohort.ADMITTED)]
    packets = [row for path in assessment.packet_paths for row in load_parquet(path)]
    task = json.loads(packets[0]["packet_json"])["task"]
    assert task["id"] == "retained-task"
    assert task["execution_requirements"]["memory_mb"] == 2048
    assert task["integrity_contract"]["verifier_files"][0]["mode"] == 0o755
    run_retained_import(
        RetainedImport(
            _cohort(published_tasks, expected_rows=1),
            str(tmp_path / "imported-quick"),
            SourceProcessingMode.QUICK,
            None,
            None,
            None,
        )
    )
    quick_manifest = json.loads((tmp_path / "imported-quick/manifest.json").read_text())
    assert quick_manifest["source"] == "fixture-retained"
    assert quick_manifest["admitted_rows"] == 0
    assert pq.read_table(Path(quick_manifest["datasets"]["final"]) / "part-000000.parquet").num_rows == 0


def test_row_count_drift_fails_closed(tmp_path: Path, published_tasks: Path) -> None:
    """A published release that differs from its receipt refuses to import."""
    cohort = _cohort(published_tasks, expected_rows=2)
    invocation = RetainedImport(cohort, str(tmp_path / "imported"), SourceProcessingMode.QUICK, None, None, None)
    with pytest.raises(ValueError):
        run_retained_import(invocation)
