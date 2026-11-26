# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.difficulty import DifficultySample
from finestore.rl.status_taxonomy import DataRestriction
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.reviewer_spec import JudgeTier
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation import driver, inventory, publication_gate, sources
from experiments.post_training.task_curation.difficulty_samples import DifficultySource
from experiments.post_training.task_curation.full_qc_inputs import InputConfig
from experiments.post_training.task_curation.judges import JUDGE_TIERS, HostedJudgeEndpoint
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import write_text_population
from experiments.post_training.task_curation.release_stages import alerts as release_stage_alerts
from experiments.post_training.task_curation.retention_plan import RetentionPlan
from experiments.post_training.task_curation.sample_qc_artifacts import SamplePanel
from experiments.post_training.task_curation.tests.admission_fixtures import (
    isolated_admission_census as isolated_admission_census,
)
from experiments.post_training.task_curation.tests.license_fixtures import isolated_license_bans as isolated_license_bans
from experiments.post_training.task_curation.tests.license_fixtures import (
    isolated_license_registry as isolated_license_registry,
)
from experiments.post_training.task_curation.tests.local_grader import (
    LEAN_KERNEL_GRADER_IMAGE,
    LocalGraderMachines,
    local_grader_machines,
)
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task
from experiments.post_training.task_curation.tests.review_server import running_review_server


@pytest.fixture(autouse=True)
def offline_registry(request, monkeypatch):
    if request.node.get_closest_marker("data_integration") is not None:
        return

    request.getfixturevalue("isolated_admission_census")
    monkeypatch.setattr(driver, "all_sources", lambda: {})
    # The defining module's binding is what a local import reads at call time:
    # the declaration audit's replacement-link check imports all_sources inside
    # the function, so patching only the driver's import-time binding left the
    # real registry read live under the download-boundary guard.
    monkeypatch.setattr(sources, "all_sources", lambda: {})
    # Selection restriction tiers come from the configured estate, which offline
    # unit tests cannot assemble; synthetic selections declare no private source.
    # Patch every importing module's binding, not only the defining one.
    open_tier = lambda source_id: DataRestriction.OPEN  # noqa: E731
    monkeypatch.setattr(publication_gate, "configured_restriction", open_tier)
    monkeypatch.setattr(inventory, "configured_restriction", open_tier)


@pytest.fixture(scope="session")
def machines() -> LocalGraderMachines:
    """Docker machines running the local grader image; requesting tests skip without Docker or the image."""
    return local_grader_machines()


@pytest.fixture(scope="session")
def lean_kernel_machines() -> LocalGraderMachines:
    """Docker machines running the Lean grader image; requesting tests skip without it."""
    return local_grader_machines(LEAN_KERNEL_GRADER_IMAGE)


@pytest.fixture
def review_server():
    with running_review_server() as server:
        yield server


@pytest.fixture
def qa_reviewer(review_server, monkeypatch):
    definition = JUDGE_TIERS[JudgeTier.HEAVY]
    assert definition.qa_reviewer is not None
    spec = replace(
        definition.qa_reviewer,
        endpoint=HostedJudgeEndpoint("glm-5.3", f"http://127.0.0.1:{review_server.server_port}/v1", "fixture"),
    )
    review_server.model_ids.append(spec.model)
    monkeypatch.setattr(driver, "JUDGE_TIERS", {**JUDGE_TIERS, JudgeTier.HEAVY: replace(definition, qa_reviewer=spec)})
    return spec


def fixture_qc_inputs(tmp_path, extra_documents=()):
    """A frozen retained population with one dropped task; extra documents widen it."""
    documents = [
        json.loads(prebuilt_task(name)["lowered_task_spec"])["task"] for name in ("kept", "dropped", "unshaped")
    ]
    oracle = TaskSpec.model_validate(documents[0])
    documents[0] = oracle.model_copy(
        update={
            "resources": oracle.resources.model_copy(update={"oracle": (inline_resource("solution/solve.sh", b"true"),)})
        }
    ).model_dump(mode="json")
    documents[2]["execution_requirements"] = None
    documents.extend(extra_documents)
    files = write_text_population(((task, {"deployment": 1}) for task in documents), str(tmp_path / "population"))
    population = SourceTextPopulation(
        path=str(tmp_path / "population"),
        source_id="TaskTrove:fixture",
        dataset_id="open-athena/fixture",
        dataset_revision="pin",
        source_version="1",
        converter_revision="converter-pin",
        text_renderer_version="fixture",
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
            output_path=population.path,
            result_type=result_type_name(type(population)),
            result=population.result_payload(),
        )
    )
    drop_path = tmp_path / "drops.parquet"
    pq.write_table(pa.table({"task_id": ["dropped"]}), drop_path)
    plan = RetentionPlan(
        source_id=population.source_id,
        source_dataset_id=population.dataset_id,
        source_revision=population.dataset_revision,
        converter_revision=population.converter_revision,
        converter_version=population.source_version,
        input_artifact=population.path,
        task_specs_sha256=population.task_specs_sha256,
        raw_count=len(documents),
        combined_drop_count=1,
        retained_count=len(documents) - 1,
        combined_drop_manifest_uri=str(drop_path),
        lexical_use="train",
    )
    return (
        InputConfig(population.path, plan, file_sha256(StoragePath(str(drop_path))), str(tmp_path / "prepared")),
        documents,
    )


@pytest.fixture
def qc_inputs(tmp_path):
    return fixture_qc_inputs(tmp_path)


@pytest.fixture
def difficulty_source(tmp_path):
    declaration = number_source(tmp_path / "declaration").declaration

    def build(task: TaskSpec) -> DifficultySource:
        task_sha = canonical_sha256(task.model_dump(mode="json"))
        panel = SamplePanel(
            source_id=declaration.id,
            dataset_id=declaration.current_release.repo,
            dataset_revision=declaration.current_release.revision,
            converter_version=declaration.pipeline.version,
            input_artifact="fixture:population",
            task_specs_sha256=task_sha,
            scanned_task_specs_sha256=task_sha,
            retained_tasks=1,
            sample_seed=0,
            sample_size=1,
            panel_path="fixture:panel",
            panel_sha256="a" * 64,
            evidence_url="fixture:inventory",
        )
        sample = DifficultySample(
            source_id=declaration.id,
            release=declaration.current_release,
            converter_version=declaration.pipeline.version,
            converter_revision="b" * 40,
            population_count=1,
            task_specs_sha256=task_sha,
            drop_manifest={"uri": "fixture:drops", "sha256": "c" * 64},
            seed=0,
            pre_oracle_filter=False,
            resample_after_oracle_filter=False,
            tasks=[{"task_id": task.id, "task_sha256": task_sha}],
        )
        source = DifficultySource(
            declaration=declaration,
            panel=panel,
            sample=sample,
            sample_receipt={"uri": "fixture:sample", "sha256": "d" * 64},
            backend="local",
        )
        return source

    return build


@pytest.fixture(autouse=True)
def _stage_alert_sink(tmp_path, monkeypatch):
    """Repoint the release-stage alert sink for every test.

    The coordinator ruled (2026-11-23) that tests and fixture dry-runs must never
    emit to the production ops feed: a stage run that forgets its ``feeds``
    argument lands in this per-test file instead of the job-watchdog feed.
    """
    monkeypatch.setattr(release_stage_alerts, "WATCHDOG_FEED", tmp_path / "stage-alerts.jsonl")
