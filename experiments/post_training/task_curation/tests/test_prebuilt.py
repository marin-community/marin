# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The prebuilt importer admits a cohort only as published and only when valid."""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import DaytonaSafety, OversightEvidence, OversightLevel
from marin.execution.lazy import run
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    FileReward,
    NoGrader,
    PlainText,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation import prebuilt
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.driver import _write_corpus
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, TaskFormat

IMAGE = "ubuntu:24.04@sha256:" + "a" * 64


def _task(task_id: str, *, image: str | None = IMAGE, capabilities: tuple[str, ...] = ()) -> dict:
    requirements = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_image=image,
        working_directory="/app",
        setup_commands=("true",),
        capabilities=capabilities,
    )
    grader = ScriptGrader(
        argv=("true",),
        environment=EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=IMAGE),
        reward=FileReward(files=(RewardFile(path="/logs/verifier/reward.txt", format=RewardFileFormat.NUMBER),)),
    )
    spec = TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content="Do the task."),)),
        environment_requirements=requirements,
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=grader,
        source=Source(dataset="d", revision="r", row=task_id, importer_revision="1"),
        resources=ResourceGroups(),
    )
    return {
        "path": task_id,
        "lowered_task_spec": json.dumps({"task": spec.model_dump(mode="json"), "runtime": {"task_machine": {}}}),
    }


def _write(tmp_path, rows: list[dict]):
    path = tmp_path / "tasks.parquet"
    table = pa.Table.from_pylist(rows, schema=pa.schema([("path", pa.string()), ("lowered_task_spec", pa.string())]))
    pq.write_table(table, path)
    return path


def _cohort(rows: int):
    return prebuilt.PrebuiltCohort(name="c", version="2026.11.12.1", repo="org/repo", revision="rev", expected_rows=rows)


def test_a_published_cohort_is_downloaded_and_validated(tmp_path, monkeypatch):
    path = _write(tmp_path, [_task("a"), _task("b")])
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **_kwargs: str(path))
    assert prebuilt.import_cohort(_cohort(2)) == str(path)


def test_a_cohort_smaller_than_recorded_is_refused(tmp_path, monkeypatch):
    path = _write(tmp_path, [_task("a")])
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **_kwargs: str(path))
    with pytest.raises(ValueError, match="has 1 rows, expected 2"):
        prebuilt.import_cohort(_cohort(2))


def test_an_envelope_without_machine_shapes_is_refused(tmp_path, monkeypatch):
    row = _task("a")
    row["lowered_task_spec"] = json.dumps({"task": json.loads(row["lowered_task_spec"])["task"]})
    path = _write(tmp_path, [row])
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **_kwargs: str(path))
    with pytest.raises(ValueError, match="no runtime machines"):
        prebuilt.import_cohort(_cohort(1))


@pytest.mark.parametrize("mode", [SourceProcessingMode.SAMPLE, SourceProcessingMode.FULL])
def test_prebuilt_import_publishes_readable_admitted_and_converted_views(tmp_path, monkeypatch, mode):
    published = _write(tmp_path, [_task("a"), _task("b")])
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **_kwargs: str(published))
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.setattr("rigging.filesystem.cluster_config.region_from_metadata", lambda: None)
    source = RlDataSource(
        config=_cohort(2),
        pipeline=prebuilt.import_prebuilt,
        info=SourceInfo(id="fixture", title="fixture", origin="fixture"),
    )
    artifact = run(prebuilt.import_prebuilt(source, PipelineOptions(mode, CampaignRuntime())))[0]
    destination = Path(artifact.path)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert (manifest["mode"], manifest["intended_use"]) == (mode.value, "train")
    assert (destination / "tasks.parquet").read_bytes() == published.read_bytes()
    for view in ("normalize", "final"):
        rows = pq.read_table(next((destination / view).glob("*.parquet"))).to_pylist()
        assert [TaskSpec.model_validate_json(row["task_json"]).id for row in rows] == ["a", "b"]

    _write_corpus({"fixture": str(destination)}, [], str(tmp_path / "report.json"), "converted")
    corpus = json.loads((tmp_path / "corpus.json").read_text())
    assert corpus["sources"][0]["converted_rows"] == 2


@pytest.mark.parametrize("mode", [SourceProcessingMode.SAMPLE, SourceProcessingMode.FULL])
@pytest.mark.parametrize("change", ["none", "package", "infrastructure_zero"])
def test_native_harbor_import_preserves_controlled_package(tmp_path, monkeypatch, change, mode):
    task = TaskSpec.model_validate(json.loads(_task("native")["lowered_task_spec"])["task"])
    task = task.model_copy(
        update={
            "tags": ("harbor",),
            "answer_type": AnswerType.WORKSPACE_STATE,
            "environment_requirements": EnvironmentRequirements(),
            "grader": NoGrader(reason="Harbor grades the completed trial"),
            "resources": ResourceGroups(
                verifier=tuple(
                    inline_resource(path, content)
                    for path, content in {
                        "task.toml": b'[environment]\ndocker_image = "custom-task-image"\n',
                        "instruction.md": b"Fix the repository.",
                        "tests/test.sh": b"#!/bin/bash\nexit 0\n",
                    }.items()
                )
            ),
        }
    )
    serialized = task.model_dump_json()
    receipt = {
        "task_id": "native",
        "task_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "status": "admitted",
        "controls": {
            "bug": {"status": "scored", "reward": 0},
            "reference": {"status": "scored", "reward": 1},
        },
    }
    if change == "package":
        serialized = task.model_copy(update={"labels": {"changed": "after-controls"}}).model_dump_json()
    elif change == "infrastructure_zero":
        receipt["controls"]["bug"]["status"] = "infra_error"
    path = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist([{"task_spec": serialized}]), path)
    (tmp_path / "controls.jsonl").write_text(json.dumps(receipt) + "\n")
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **kwargs: str(tmp_path / kwargs["filename"]))
    if change == "none":
        assert prebuilt.import_harbor_cohort(_cohort(1)) == str(path)
        monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
        monkeypatch.setattr("rigging.filesystem.cluster_config.region_from_metadata", lambda: None)
        source = RlDataSource(
            config=_cohort(1),
            pipeline=prebuilt.import_harbor_prebuilt,
            info=SourceInfo(id="native", title="native", origin="fixture", task_format=TaskFormat.HARBOR),
        )
        artifact = run(prebuilt.import_harbor_prebuilt(source, PipelineOptions(mode, CampaignRuntime())))[0]
        destination = Path(artifact.path)
        assert (destination / "tasks.parquet").read_bytes() == path.read_bytes()
        for view in ("normalize", "final"):
            rows = pq.read_table(next((destination / view).glob("*.parquet"))).to_pylist()
            assert rows[0]["task_json"] == serialized
            assert hashlib.sha256(rows[0]["task_json"].encode()).hexdigest() == receipt["task_sha256"]
    else:
        with pytest.raises(ValueError, match=r"differs from its control|controls do not admit"):
            prebuilt.import_harbor_cohort(_cohort(1))


def test_prebuilt_workspace_cohort_rejects_missing_environment_without_capabilities(tmp_path, monkeypatch):
    answer_type = AnswerType.WORKSPACE_STATE
    row = _task("missing", image=None)
    envelope = json.loads(row["lowered_task_spec"])
    envelope["task"]["answer_type"] = answer_type.value
    if answer_type is AnswerType.WORKSPACE_STATE:
        envelope["task"]["grader"]["answer_path"] = None
    row["lowered_task_spec"] = json.dumps(envelope)
    path = _write(tmp_path, [row])
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **_kwargs: str(path))
    with pytest.raises(ValueError, match="declares no environment"):
        prebuilt.import_cohort(_cohort(1))


@pytest.mark.parametrize("qualified", [False, True])
def test_direct_prebuilt_admission_requires_oversight_but_allows_unsafe_tasks(tmp_path, monkeypatch, qualified):
    path = _write(tmp_path, [_task("a")])
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **_kwargs: str(path))
    config = prebuilt.PrebuiltCohort(
        name="unsafe-cohort", version="2026.11.12.1", repo="org/repo", revision="rev", expected_rows=1
    )
    source = RlDataSource(
        config=config,
        pipeline=prebuilt.import_prebuilt,
        info=SourceInfo(
            id="fixture:unsafe",
            title="Unsafe",
            origin="fixture",
            tags=("agentic",),
            task_format=TaskFormat.TASKCOMPENDIUM,
            daytona_snapshot_count=11,
            daytona_count_basis="fixture production identity census",
            daytona_execution_failures=0,
            oversight=(
                OversightEvidence(level=OversightLevel.SAMPLE, verification="fixture sampled controls")
                if qualified
                else None
            ),
        ),
    )
    options = PipelineOptions(mode=SourceProcessingMode.SAMPLE, runtime=CampaignRuntime())
    if not qualified:
        with pytest.raises(ValueError, match="cannot admit tasks"):
            prebuilt.import_prebuilt(source, options)
        return
    # Snapshot-unsafe sources with oversight must still be admitted as data.
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.setattr("rigging.filesystem.cluster_config.region_from_metadata", lambda: None)
    artifact = prebuilt.import_prebuilt(source, options)
    result = run(artifact)[0]
    manifest = json.loads((Path(result.path) / "manifest.json").read_text())
    assert result.status == "completed" and manifest["row_count"] == 1
    assert source.declaration.daytona_safety is DaytonaSafety.UNSAFE


def test_reply_only_rewrite_needs_no_agent_machine(tmp_path, monkeypatch):
    row = _task("reply")
    envelope = json.loads(row["lowered_task_spec"])
    envelope["task"]["environment_requirements"] = {}
    row["lowered_task_spec"] = json.dumps(envelope)
    path = _write(tmp_path, [row])
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **_kwargs: str(path))
    assert prebuilt.import_cohort(_cohort(1)) == str(path)
