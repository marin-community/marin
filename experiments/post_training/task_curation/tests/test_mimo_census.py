# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The repaired MiMo census: maintained censuses and survey orchestrated over one release."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from experiments.post_training.task_curation.datasets.mimo import census as mimo_census
from experiments.post_training.task_curation.datasets.mimo.census import EXPECTED_TASKS_PARQUET_SHA256, run_census
from experiments.post_training.task_curation.lexical_artifacts import ActorCensus, SourceTextPopulation

SOURCE_ID = "FWeindel:validated-tasks/mimo_v1"
RELEASE_REPO = "open-athena/fweindel-validated-tasks-mimo-v1-pinned-images-20261124-a-v1"
RELEASE_REVISION = "729925160824efba40d672c10ee4311169dc1a8b"
CONVERTER_VERSION = "2026.11.23.pinned-images.1"
CONVERTER_REVISION = "831fb168588948a3a9c28ab1d6063d255555f0f4"
TASKS = 2501
TASK_SPECS_SHA256 = "c0f2f8ea38ad306f658e6022f66460c74598c10087ca4c5401cf6012b4f29a89"


@dataclass
class FakeSurvey:
    sources: list
    calls: list = field(default_factory=list)

    def model_dump_json(self, indent=None):
        self.calls.append("dump")
        return json.dumps({"sources": [row.model_dump(mode="json") for row in self.sources]})


@dataclass
class FakeSurveyRow:
    backend: str
    status: str

    def model_dump(self, mode=None):
        return {"backend": self.backend, "status": self.status}


@pytest.fixture
def population(tmp_path, monkeypatch):
    record = SourceTextPopulation(
        path=str(tmp_path / "population"),
        source_id=SOURCE_ID,
        dataset_id=RELEASE_REPO,
        dataset_revision=RELEASE_REVISION,
        source_version=CONVERTER_VERSION,
        converter_revision=CONVERTER_REVISION,
        text_renderer_version="fixture",
        task_count=TASKS,
        text_sha256="a" * 64,
        task_specs_sha256=TASK_SPECS_SHA256,
        main_output_dir=str(tmp_path / "population"),
        specs_output_dir=str(tmp_path / "population"),
        converter_rejections={},
        admission_rejections={},
    )

    class FrozenPopulation:
        raw_load = classmethod(lambda cls, path: record)

    monkeypatch.setattr(mimo_census, "SourceTextPopulation", FrozenPopulation)
    return record


@pytest.fixture
def workdir(tmp_path):
    directory = tmp_path / "work"
    directory.mkdir()
    return directory


def test_run_census_binds_every_receipt_to_the_release_pin(population, workdir, tmp_path, monkeypatch):
    calls = []

    def fake_measure_population(config):
        calls.append(("actor", config.output_path))
        return ActorCensus(
            path=config.output_path,
            source_id=SOURCE_ID,
            dataset_id=RELEASE_REPO,
            dataset_revision=RELEASE_REVISION,
            converter_version=CONVERTER_VERSION,
            converter_revision=CONVERTER_REVISION,
            input_artifact=population.path,
            task_specs_sha256=TASK_SPECS_SHA256,
            task_count=TASKS,
            daytona_safety="unsafe",
            daytona_snapshot_count=TASKS,
            daytona_machine_slots=TASKS,
            daytona_execution_failures=0,
            daytona_count_basis="fixture",
            daytona_unsafe_reasons=["too_many_snapshots"],
            daytona_unsafe_details={"too_many_snapshots": f"{TASKS} snapshot identities"},
            daytona_unsafe_counts={"too_many_snapshots": TASKS},
            daytona_roles={"agent": {"daytona": {"machine_slots": TASKS, "failures": {}}}, "grader": {}},
            actor_environment_counts={"docker_image": TASKS},
            empty_declared_environment_tasks={},
            snapshot_sizes={f"snap-{index}": 1 for index in range(3)},
            execution_failures={},
            mapping_failures={},
            integrity_contracts={
                "schema_version": 2,
                "groups": [
                    {
                        "count": TASKS,
                        "dependency_state": None,
                        "harness_policy": None,
                        "id": None,
                        "native_family": None,
                        "present": False,
                        "version": None,
                    }
                ],
                "distinct_groups": 1,
                "absent_contract_tasks": TASKS,
                "required_shared_tasks": TASKS,
                "shared_missing_contract_tasks": TASKS,
                "example": None,
                "evidence_url": str(config.output_path),
                "dataset_id": RELEASE_REPO,
                "dataset_revision": RELEASE_REVISION,
                "task_specs_sha256": TASK_SPECS_SHA256,
            },
        )

    def fake_grading(config, output_root):
        calls.append(("grading", output_root, config.input.evidence_url))
        return FakeGradingReceipt(FakeGradingEvidence())

    def fake_derived(config, output_root):
        calls.append(("derived", output_root, config.input.evidence_url))
        return FakeDerivedReceipt(FakeDerivedSummary())

    def fake_static(cohort, context, cohort_evidence, output):
        calls.append(("static", context.input_cohort_sha256, cohort_evidence.sha256))
        output.mkdir(parents=True)
        evidence = {
            "source_id": SOURCE_ID,
            "retained_count": TASKS,
            "static_retained_count": TASKS,
            "exclusion_counts": {},
        }
        (output / "static-retention-evidence.json").write_text(json.dumps(evidence))
        for name in ("oracle-filter.json", "static-exclusions.json", "static-retention.json", "tasks.parquet"):
            (output / name).write_bytes(b"fixture")

    def fake_survey(manifest, checkers, revision):
        calls.append(("survey", revision, manifest["inputs"][0]["expected_tasks"]))
        assert manifest["source_ids"] == [SOURCE_ID]
        assert manifest["catalog_root"] == mimo_census.MANIFEST_BOUND_CATALOG_ROOT
        assert manifest["snapshot_id"] == EXPECTED_TASKS_PARQUET_SHA256
        assert manifest["grading_environments"][SOURCE_ID]["required_counts"] == {"shared": TASKS}
        return FakeSurvey([FakeSurveyRow("daytona", "verify")])

    monkeypatch.setattr(mimo_census, "measure_population", fake_measure_population)
    monkeypatch.setattr(mimo_census, "grading_census_source", fake_grading)
    monkeypatch.setattr(mimo_census, "derived_census_source", fake_derived)
    monkeypatch.setattr(mimo_census, "collect_static_retention", fake_static)
    monkeypatch.setattr(mimo_census, "build_survey", fake_survey)
    monkeypatch.setattr(mimo_census, "production_checkers", lambda **kwargs: {"daytona": lambda task: None})
    monkeypatch.setattr(mimo_census, "check_receipt_evidence", lambda receipt: calls.append(("checked",)))

    tasks = workdir / "release-tasks.parquet"
    tasks.write_bytes(b"fixture-rows")
    monkeypatch.setattr(mimo_census, "download_release", lambda workdir: tasks)

    root = tmp_path / "census-root"
    summary = run_census(str(root), workdir, "fixture-revision")

    kinds = [row[0] for row in calls]
    assert kinds == ["actor", "checked", "grading", "derived", "static", "survey"]
    assert Path(calls[0][1]).name == "actor" and str(root) in calls[0][1]
    grading_call, derived_call = calls[2], calls[3]
    assert grading_call[1] == str(root / "grading") and derived_call[1] == str(root / "derived")
    assert grading_call[2] == derived_call[2] and f"{root}/actor/.artifact.json" in grading_call[2]
    assert calls[4][1] == EXPECTED_TASKS_PARQUET_SHA256 and calls[4][2] == EXPECTED_TASKS_PARQUET_SHA256
    assert calls[5][1] == "fixture-revision" and calls[5][2] == TASKS
    assert summary["tasks"] == TASKS
    assert summary["actor"]["daytona_safety"] == "unsafe"
    assert summary["static_retention"]["static_retained_count"] == TASKS
    assert summary["survey"]["rows"] == [{"backend": "daytona", "status": "verify"}]
    assert (root / "census-summary.json").exists()
    assert (root / "actor-receipts" / "part-000.json").exists()
    for name in mimo_census.STATIC_RETENTION_FILES:
        assert (root / "static-retention" / name).exists()
    receipt = json.loads((root / "actor-receipts" / "part-000.json").read_text())[SOURCE_ID]
    assert receipt["measurement"]["daytona_snapshot_count"] == TASKS
    assert receipt["input"]["evidence_url"] == str(root / "actor" / ".artifact.json")


@dataclass(frozen=True)
class FakeGradingEvidence:
    assessed_tasks: int = TASKS

    @property
    def required_counts(self) -> dict:
        return {"shared": TASKS}

    @property
    def declared_counts(self) -> dict:
        return {"shared": TASKS}

    @property
    def status_counts(self) -> dict:
        return {"matched": TASKS}

    def model_dump(self, mode=None):
        return {
            "assessed_tasks": TASKS,
            "required_counts": {"shared": TASKS},
            "declared_counts": {"shared": TASKS},
            "status_counts": {"matched": TASKS},
            "reason_counts": {},
            "missing_artifact_counts": {},
        }


@dataclass(frozen=True)
class FakeGradingReceipt:
    evidence: FakeGradingEvidence
    task_evidence_uri: str = "fixture://grading-tasks"
    unassessable_reasons: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FakeDerivedSummary:
    task_count: int = TASKS
    grader_type_counts: dict = field(default_factory=lambda: {"script": TASKS})
    agentic_count: int = TASKS
    single_turn_count: int = TASKS
    reply_only_count: int = TASKS
    workspace_graded_count: int = 0
    judge_rubric_counts: dict = field(default_factory=dict)
    judge_reference_count: int = 0
    judge_no_reference_count: int = 0

    def model_dump(self, mode=None):
        return {
            "task_count": TASKS,
            "grader_type_counts": {"script": TASKS},
            "agentic_count": TASKS,
            "single_turn_count": TASKS,
            "reply_only_count": TASKS,
            "workspace_graded_count": 0,
            "judge_rubric_counts": {},
            "judge_reference_count": 0,
            "judge_no_reference_count": 0,
        }


@dataclass(frozen=True)
class FakeDerivedReceipt:
    summary: FakeDerivedSummary = field(default_factory=FakeDerivedSummary)

    def model_dump(self, mode=None):
        return {"summary": self.summary.model_dump(mode="json")}
