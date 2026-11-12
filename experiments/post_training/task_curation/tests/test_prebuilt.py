# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The prebuilt importer admits a cohort only as published and only when valid."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import DaytonaSafety, OversightEvidence, OversightLevel
from marin.execution.lazy import StepContext, run
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.grader import verifyit_package
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import (
    AnswerType,
    EnvironmentRequirements,
    NoGrader,
    ResourceGroups,
    TaskSpec,
)
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation import prebuilt
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.driver import _write_corpus
from experiments.post_training.task_curation.judges import JUDGE_TIERS, JudgeTier
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, TaskFormat
from lib.taskcompendium.tests.decontamination_fixtures import decontamination_evidence
from lib.taskcompendium.tests.oracle_filter_fixtures import oracle_receipt

from .prebuilt_fixtures import prebuilt_task

IMAGE = "ubuntu:24.04@sha256:" + "a" * 64


def _write(tmp_path, rows: list[dict]):
    path = tmp_path / "tasks.parquet"
    table = pa.Table.from_pylist(rows, schema=pa.schema([("path", pa.string()), ("lowered_task_spec", pa.string())]))
    pq.write_table(table, path)
    return path


def _cohort(rows: int):
    return prebuilt.PrebuiltCohort(
        name="c", version="2026.11.12.1", repo="open-athena/repo", revision="a" * 40, expected_rows=rows
    )


def test_a_published_cohort_is_downloaded_and_validated(tmp_path, monkeypatch):
    path = _write(tmp_path, [prebuilt_task("a"), prebuilt_task("b")])
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **_kwargs: str(path))
    assert prebuilt.import_cohort(_cohort(2)) == str(path)


def test_a_cohort_smaller_than_recorded_is_refused(tmp_path, monkeypatch):
    path = _write(tmp_path, [prebuilt_task("a")])
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **_kwargs: str(path))
    with pytest.raises(ValueError, match="has 1 rows, expected 2"):
        prebuilt.import_cohort(_cohort(2))


def test_an_envelope_without_machine_shapes_is_refused(tmp_path, monkeypatch):
    row = prebuilt_task("a")
    row["lowered_task_spec"] = json.dumps({"task": json.loads(row["lowered_task_spec"])["task"]})
    path = _write(tmp_path, [row])
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **_kwargs: str(path))
    with pytest.raises(ValueError, match="no runtime machines"):
        prebuilt.import_cohort(_cohort(1))


@pytest.mark.parametrize("mode", [SourceProcessingMode.SAMPLE, SourceProcessingMode.FULL])
@pytest.mark.parametrize("storage", ["local", "memory"])
def test_prebuilt_import_publishes_readable_admitted_and_converted_views(tmp_path, monkeypatch, mode, storage):
    published = _write(tmp_path, [prebuilt_task("a"), prebuilt_task("b")])
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **_kwargs: str(published))
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.setattr("rigging.filesystem.cluster_config.region_from_metadata", lambda: None)
    source = RlDataSource(
        config=_cohort(2),
        pipeline=prebuilt.import_prebuilt,
        info=SourceInfo(
            id="fixture",
            title="fixture",
            origin="fixture",
            lexical_contamination=decontamination_evidence("open-athena/repo", "a" * 40, 2),
        ),
    )
    step = prebuilt.import_prebuilt(source, PipelineOptions(mode, CampaignRuntime()))
    if storage == "local":
        artifact = run(step)[0]
        destination = StoragePath(artifact.path)
    else:
        destination = StoragePath(f"memory://prebuilt/{tmp_path.name}/{mode.value}")
        step.run(step.build_config(StepContext.for_run(str(destination), str(tmp_path))))
    manifest = json.loads((destination / "manifest.json").read_text())
    assert (manifest["mode"], manifest["intended_use"]) == (mode.value, "train")
    assert (destination / "tasks.parquet").read_bytes() == published.read_bytes()
    for view in ("normalize", "final"):
        with next(iter((destination / view / "*.parquet").glob())).open("rb") as handle:
            rows = pq.read_table(handle).to_pylist()
        assert [TaskSpec.model_validate_json(row["task_json"]).id for row in rows] == ["a", "b"]

    _write_corpus({"fixture": str(destination)}, [], str(tmp_path / "report.json"), "converted")
    corpus = json.loads((tmp_path / "corpus.json").read_text())
    assert corpus["sources"][0]["converted_rows"] == 2


@pytest.mark.parametrize("mode", [SourceProcessingMode.SAMPLE, SourceProcessingMode.FULL])
@pytest.mark.parametrize("change", ["none", "package", "infrastructure_zero"])
def test_native_harbor_import_preserves_controlled_package(tmp_path, monkeypatch, change, mode):
    task = TaskSpec.model_validate(json.loads(prebuilt_task("native")["lowered_task_spec"])["task"])
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
            info=SourceInfo(
                id="native",
                title="native",
                origin="fixture",
                lexical_contamination=decontamination_evidence("open-athena/repo", "a" * 40, 1),
                task_format=TaskFormat.HARBOR,
            ),
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
    row = prebuilt_task("missing", image=None)
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
    path = _write(tmp_path, [prebuilt_task("a")])
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **_kwargs: str(path))
    config = prebuilt.PrebuiltCohort(
        name="unsafe-cohort", version="2026.11.12.1", repo="open-athena/repo", revision="a" * 40, expected_rows=1
    )
    source = RlDataSource(
        config=config,
        pipeline=prebuilt.import_prebuilt,
        info=SourceInfo(
            lexical_contamination=decontamination_evidence("open-athena/repo", "a" * 40, 1),
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
    row = prebuilt_task("reply")
    envelope = json.loads(row["lowered_task_spec"])
    envelope["task"]["environment_requirements"] = {}
    row["lowered_task_spec"] = json.dumps(envelope)
    path = _write(tmp_path, [row])
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **_kwargs: str(path))
    assert prebuilt.import_cohort(_cohort(1)) == str(path)


@pytest.mark.parametrize("change", ["none", "hash", "duplicate", "missing_environment"])
def test_canonical_judged_release_preserves_tasks_without_a_runtime_envelope(tmp_path, monkeypatch, change):
    task = TaskSpec.model_validate(json.loads(prebuilt_task("reply")["lowered_task_spec"])["task"])
    if change != "missing_environment":
        task = task.model_copy(
            update={
                "environment_requirements": EnvironmentRequirements(),
                "grader": (
                    verifyit_package(
                        replace(
                            JUDGE_TIERS[JudgeTier.MEDIUM].spec, question="Do the task.", references=("Correct answer.",)
                        )
                    ).grader
                ),
            }
        )
    else:
        task = task.model_copy(update={"environment_requirements": EnvironmentRequirements()})
        task = task.model_copy(
            update={"grader": task.grader.model_copy(update={"environment": EnvironmentRequirements()})}
        )
    serialized = task.model_dump_json()
    row = {"task_spec": serialized, "task_sha256": hashlib.sha256(serialized.encode()).hexdigest()}
    rows = [row, row] if change == "duplicate" else [row]
    if change == "hash":
        row["task_spec"] = task.model_copy(update={"labels": {"changed": "after-publication"}}).model_dump_json()
    path = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **_kwargs: str(path))
    if change != "none":
        with pytest.raises(ValueError, match=r"recorded hash|Duplicate published task|grading environment requires"):
            prebuilt.import_taskcompendium_cohort(_cohort(len(rows)))
        return
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.setattr("rigging.filesystem.cluster_config.region_from_metadata", lambda: None)
    source = RlDataSource(
        config=_cohort(1),
        pipeline=prebuilt.import_taskcompendium_prebuilt,
        info=SourceInfo(
            id="reply",
            title="reply",
            origin="fixture",
            lexical_contamination=decontamination_evidence("open-athena/repo", "a" * 40, 1),
            task_format=TaskFormat.TASKCOMPENDIUM,
        ),
    )
    artifact = run(
        prebuilt.import_taskcompendium_prebuilt(source, PipelineOptions(SourceProcessingMode.SAMPLE, CampaignRuntime()))
    )[0]
    destination = Path(artifact.path)
    assert (destination / "tasks.parquet").read_bytes() == path.read_bytes()
    for view in ("normalize", "final"):
        admitted = pq.read_table(next((destination / view).glob("*.parquet"))).to_pylist()
        assert [r["task_json"] for r in admitted] == [serialized]


@pytest.mark.parametrize("mode", [SourceProcessingMode.SAMPLE, SourceProcessingMode.FULL])
@pytest.mark.parametrize("state", ["missing", "failed", "passed"])
@pytest.mark.parametrize("encoding", list(prebuilt.PrebuiltEncoding))
def test_prebuilt_admission_requires_exact_successful_oracle_filter(tmp_path, monkeypatch, mode, state, encoding):
    row = prebuilt_task("oracle")
    envelope = json.loads(row["lowered_task_spec"])
    spec = TaskSpec.model_validate(envelope["task"])
    spec = spec.model_copy(
        update={
            "resources": spec.resources.model_copy(update={"oracle": (inline_resource("solution/solve.sh", b"true\n"),)})
        }
    )
    envelope["task"] = spec.model_dump(mode="json")
    row["lowered_task_spec"] = json.dumps(envelope)
    cohort = _write(tmp_path, [row])
    if encoding is prebuilt.PrebuiltEncoding.CANONICAL:
        serialized = spec.model_dump_json()
        pq.write_table(
            pa.Table.from_pylist(
                [{"task_spec": serialized, "task_sha256": hashlib.sha256(serialized.encode()).hexdigest()}]
            ),
            cohort,
        )
    receipt = oracle_receipt([spec], failed=frozenset({spec.id}) if state == "failed" else frozenset())
    receipt.tasks[spec.id] = receipt.tasks[spec.id].model_copy(
        update={
            "runtime_sha256": (
                canonical_sha256(envelope["runtime"]) if encoding is prebuilt.PrebuiltEncoding.LOWERED else None
            )
        }
    )
    receipt_path = tmp_path / prebuilt.ORACLE_FILTER_FILENAME
    receipt_path.write_text(receipt.model_dump_json())

    def download(**kwargs):
        if kwargs["filename"] == prebuilt.TASKS_FILENAME:
            return str(cohort)
        if state == "missing":
            raise FileNotFoundError("oracle-filter.json")
        return str(receipt_path)

    monkeypatch.setattr(prebuilt, "hf_hub_download", download)
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.setattr("rigging.filesystem.cluster_config.region_from_metadata", lambda: None)
    pipeline = (
        prebuilt.import_prebuilt
        if encoding is prebuilt.PrebuiltEncoding.LOWERED
        else prebuilt.import_taskcompendium_prebuilt
    )
    source = RlDataSource(
        config=_cohort(1),
        pipeline=pipeline,
        info=SourceInfo(
            id="fixture:oracle",
            title="oracle",
            origin="fixture",
            lexical_contamination=decontamination_evidence("open-athena/repo", "a" * 40, 1),
            task_format=TaskFormat.TASKCOMPENDIUM,
        ),
    )
    step = pipeline(source, PipelineOptions(mode, CampaignRuntime()))
    destination = tmp_path / "admitted"
    invocation = step.build_config(StepContext.for_run(str(destination), str(tmp_path)))
    if state == "missing":
        with pytest.raises(FileNotFoundError, match=r"oracle-filter\.json"):
            step.run(invocation)
    elif state == "failed":
        with pytest.raises(ValueError, match="failed oracle task"):
            step.run(invocation)
    else:
        artifact = step.run(invocation)
        assert artifact.status == "completed"
        manifest = json.loads((destination / "manifest.json").read_text())
        assert manifest["admitted_rows"] == 1
        assert (destination / prebuilt.ORACLE_FILTER_FILENAME).exists()


@pytest.mark.parametrize("policy", ["public", "other_namespace", "raw_format"])
def test_atlas_private_release_policy_blocks_prebuilt_download(tmp_path, monkeypatch, policy):
    config = _cohort(1)
    if policy == "other_namespace":
        config = prebuilt.PrebuiltCohort(
            name=config.name, version=config.version, repo="other/private", revision=config.revision, expected_rows=1
        )
    document = decontamination_evidence(config.repo, config.revision, 1).model_dump(mode="json")
    if policy == "public":
        document["receipt"]["private"] = False
    elif policy == "raw_format":
        document["receipt"]["format"] = "raw"
    evidence = type(decontamination_evidence(config.repo, config.revision, 1)).model_validate(document)
    source = RlDataSource(
        config=config,
        pipeline=prebuilt.import_prebuilt,
        info=SourceInfo(id="fixture:policy", title="policy", origin="fixture", lexical_contamination=evidence),
    )
    downloaded = []
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **kwargs: downloaded.append(kwargs))
    with pytest.raises(ValueError, match="private open-athena TaskCompendium"):
        prebuilt.import_prebuilt(source, PipelineOptions(SourceProcessingMode.SAMPLE, CampaignRuntime()))
    assert downloaded == []


def test_quick_pass_writes_the_validated_cohort_as_converted_tasks(tmp_path, monkeypatch):
    path = _write(tmp_path, [prebuilt_task("a"), prebuilt_task("b")])
    monkeypatch.setattr(prebuilt, "hf_hub_download", lambda **_kwargs: str(path))
    output = tmp_path / "out"
    source = RlDataSource(
        config=_cohort(2),
        pipeline=prebuilt.import_prebuilt,
        info=SourceInfo(id="fixture:quick", title="quick", origin="fixture"),
    )
    step = prebuilt.import_prebuilt(source, PipelineOptions(SourceProcessingMode.QUICK, CampaignRuntime()))
    artifact = step.run(step.build_config(StepContext.for_run(str(output), str(tmp_path))))

    manifest = json.loads((output / "manifest.json").read_text())
    tasks = [
        TaskSpec.model_validate_json(item)
        for item in pq.read_table(output / "normalize").column("task_json").to_pylist()
    ]
    assert artifact.status == "completed"
    assert [task.id for task in tasks] == ["a", "b"]
    assert manifest["input_rows"] == manifest["converted_rows"] == 2
    assert manifest["rejections"] == {} and manifest["intended_use"] == "train"
    assert manifest["mode"] == "quick"
    assert not (output / "final").exists()
    assert "admitted_rows" not in manifest
