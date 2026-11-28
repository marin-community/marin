# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The census: the stage measures the complete converted population without provider calls."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import DAYTONA_SAFE_SNAPSHOT_LIMIT
from zephyr.context import ZephyrContext

from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.pipeline.census import (
    EnvelopeMeasurement,
    RoleOutcome,
    actor_failure_reason,
    measure_population,
    population_digest,
    unsafe_counts,
    unsafe_reasons,
)
from taskcompendium.pipeline.review import BatchReviewer
from taskcompendium.pipeline.source_processing import (
    SourcePipelineConfig,
    SourceProcessingMode,
    run_source_pipeline,
)
from taskcompendium.pipeline.source_quality import SourceQualityPolicy
from taskcompendium.pipeline.source_verification import SourceVerificationPolicy
from taskcompendium.pipeline.stages import AuditExecution, FilterPolicy

from .pipeline_stages import convert_svamp, fixture_recipe, svamp_task
from .test_pipeline import BatchService

PINNED_IMAGE = "registry/image@sha256:" + "a" * 64


class FixtureCensusEngine:
    """Counts one agent machine slot for every task that declares a native environment."""

    def describe(self) -> str:
        return "fixture engine: one agent machine slot per task that declares an environment"

    def measure(self, envelopes) -> EnvelopeMeasurement:
        converted = 0
        slots = 0
        for envelope in envelopes:
            converted += 1
            declared = envelope.task.get("environment_requirements") or {}
            if any(declared.get(name) is not None for name in ("docker_image", "docker_build", "package_spec")):
                slots += 1
        roles = {"agent": {"daytona": RoleOutcome(machine_slots=slots)} if slots else {}, "grader": {}}
        return EnvelopeMeasurement(converted=converted, machine_slots=slots, roles=roles)


class UndercountingEngine(FixtureCensusEngine):
    """A broken engine whose measurement loses rows; the stage must refuse it."""

    def measure(self, envelopes) -> EnvelopeMeasurement:
        measurement = super().measure(envelopes)
        return EnvelopeMeasurement(
            converted=max(measurement.converted - 1, 0),
            machine_slots=measurement.machine_slots,
            roles=measurement.roles,
        )


def source_rows(count: int, broken: tuple[int, ...]) -> list[dict[str, Any]]:
    rows = [
        {"Body": f"Aya has {index} apples.", "Question": "How many?", "Answer": str(index)}
        for index in range(1, count + 1)
    ]
    for index in broken:
        rows[index - 1].pop("Question")
    return rows


def write_jsonl(source: Path, rows: list[dict[str, Any]]) -> None:
    source.mkdir(parents=True, exist_ok=True)
    (source / "source.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def pipeline_config(mode: SourceProcessingMode, census: Any) -> SourcePipelineConfig:
    reviewer = BatchReviewer(BatchService(), "fixture", "revision")
    return SourcePipelineConfig(
        mode,
        SourceQualityPolicy(),
        SourceVerificationPolicy(10, 0, 1, 1),
        None,
        AuditExecution(reviewer=reviewer),
        FilterPolicy(),
        normalized_shards=2,
        machines=None,
        census=census,
    )


def run_pipeline(recipe, source: Path, output: Path, config: SourcePipelineConfig):
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(output.parent / "chunks")) as context:
        return run_source_pipeline(
            recipe, context, str(source), str(output), config, mode=config.mode, canonical_source=recipe.name
        )


def parquet_rows(path: Path) -> list[dict[str, Any]]:
    files = [path] if path.is_file() else sorted(Path(path).glob("*.parquet"))
    return [row for file in files for row in pq.read_table(file).to_pylist()]


def test_full_source_measures_the_complete_population_and_records_rejections(tmp_path):
    source = tmp_path / "source"
    write_jsonl(source, source_rows(5, broken=(3,)))
    output = tmp_path / "output"
    recipe = fixture_recipe(convert_svamp, rubric=None)
    result = run_pipeline(recipe, source, output, pipeline_config(SourceProcessingMode.FULL, FixtureCensusEngine()))

    manifest = json.loads(Path(result.manifest_path).read_text())
    census = manifest["census"]
    assert census["input"]["measured_tasks"] == 4
    assert census["rejections"] == {"source_defect:missing_prompt": 1}
    assert census["measurement"]["dataset_id"] == "fixture/tasks"
    assert census["measurement"]["dataset_revision"] == "1"
    assert census["measurement"]["daytona_machine_slots"] == 0
    assert census["measurement"]["daytona_snapshot_count"] == 0
    assert census["measurement"]["daytona_roles"] == {"agent": {}, "grader": {}}
    assert census["grading_environment"]["assessed_tasks"] == 4
    assert census["grading_environment"]["required_counts"] == {"separate": 4}
    assert census["integrity_contracts"]["distinct_groups"] == 1
    assert census["integrity_contracts"]["groups"][0]["present"] is False
    assert census["integrity_contracts"]["absent_contract_tasks"] == 4
    assert census["integrity_contracts"]["required_shared_tasks"] == 0

    written = json.loads((output / "census" / "census.json").read_text())
    assert written == census
    normalize_ids = {row["task_id"] for row in parquet_rows(Path(result.normalize_path)) if row["task_json"]}
    assert census["input"]["measured_tasks"] == len(normalize_ids)
    assert result.census_path == str(output / "census")


def test_source_without_an_engine_skips_the_census(tmp_path):
    source = tmp_path / "source"
    write_jsonl(source, source_rows(3, broken=()))
    output = tmp_path / "output"
    recipe = fixture_recipe(convert_svamp, rubric=None)
    result = run_pipeline(recipe, source, output, pipeline_config(SourceProcessingMode.FULL, None))

    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["census"] is None
    assert result.census_path is None
    assert "census" not in manifest["datasets"]


def test_sample_mode_measuring_only_the_panel_records_no_census(tmp_path):
    """A panel-only run cannot bind its counts to the dataset pin, so it records no census."""
    source = tmp_path / "source"
    write_jsonl(source, source_rows(40, broken=()))
    output = tmp_path / "output"
    recipe = fixture_recipe(convert_svamp, rubric=None)
    config = replace(
        pipeline_config(SourceProcessingMode.SAMPLE, FixtureCensusEngine()),
        quality_policy=SourceQualityPolicy(sample_size=8),
    )
    result = run_pipeline(recipe, source, output, config)

    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["census"] is None
    assert result.census_path is None
    assert "census" not in manifest["datasets"]
    assert not (output / "census").exists()


def test_declared_agent_environments_count_machine_slots():
    tasks = [
        svamp_task(f"task-{index}", {"Body": f"Aya has {index} apples.", "Question": "How many?", "Answer": "1"})
        .model_copy(
            update={
                "environment_requirements": EnvironmentRequirements(
                    docker_image=PINNED_IMAGE, command_semantics=CommandSemantics.LINUX_PROCESS
                )
            }
        )
        .model_dump(mode="json")
        for index in range(3)
    ]
    rows = [
        {
            "task_id": task["id"],
            "task_json": json.dumps(task),
            "normalization_kind": None,
            "normalization_reason": None,
        }
        for task in tasks
    ]
    evidence = measure_population(
        iter(rows),
        source_name="fixture",
        dataset_id="fixture",
        dataset_revision="1",
        converter_revision="1",
        evidence_url="memory://census.json",
        engine=FixtureCensusEngine(),
    )
    assert evidence.measurement.daytona_machine_slots == 3
    assert evidence.measurement.daytona_roles["agent"]["daytona"].machine_slots == 3
    assert evidence.measurement.daytona_snapshot_count == 0
    assert evidence.measurement.daytona_execution_failures == 0
    assert evidence.input.measured_tasks == 3
    assert evidence.grading_environment.assessed_tasks == 3


def test_stage_refuses_an_engine_measurement_that_loses_rows(tmp_path):
    source = tmp_path / "source"
    write_jsonl(source, source_rows(3, broken=()))
    output = tmp_path / "output"
    recipe = fixture_recipe(convert_svamp, rubric=None)
    with pytest.raises(ValueError, match="measured 2 of 3"):
        run_pipeline(recipe, source, output, pipeline_config(SourceProcessingMode.FULL, UndercountingEngine()))


def test_actor_failures_classify_by_the_environment_source_that_caused_them():
    measurement = EnvelopeMeasurement(
        converted=3,
        machine_slots=3,
        actor_sources={"docker_build": 2, "packages_lock": 1},
        execution_failures={
            "task-a: Unresolved Docker build context": 1,
            "task-b: machine requires a registry image or Docker build context": 1,
            "task-c: cpus 9999 exceeds the backend limit": 1,
        },
        unsafe_task_counts={"unbuilt_image": 1, "host_image_required": 1, "execution_requirement": 1},
    )
    reasons = unsafe_reasons(measurement)
    assert set(reasons) == {"unbuilt_image", "host_image_required", "execution_requirement"}
    counts = unsafe_counts(measurement)
    assert counts["unbuilt_image"] == 1
    assert counts["host_image_required"] == 1
    assert counts["execution_requirement"] == 1
    assert actor_failure_reason("Unresolved Docker build context", {"docker_build": 1}) == "unbuilt_image"


def test_snapshot_budget_refusal_counts_every_actor_slot_once():
    slots = DAYTONA_SAFE_SNAPSHOT_LIMIT + 1
    measurement = EnvelopeMeasurement(
        converted=slots,
        machine_slots=slots,
        snapshot_sizes={f"snapshot-{index}": 1 for index in range(slots)},
    )
    reasons = unsafe_reasons(measurement)
    assert set(reasons) == {"too_many_snapshots"}
    assert "11 actor snapshot identities" in reasons["too_many_snapshots"]
    assert unsafe_counts(measurement)["too_many_snapshots"] == slots


def test_unsafe_reasons_require_measured_details_for_every_reason():
    measurement = EnvelopeMeasurement(
        converted=2,
        machine_slots=2,
        snapshot_sizes={"snapshot-a": 2},
        execution_failures={"task-a: refused": 1},
        unsafe_task_counts={"execution_requirement": 1},
    )
    reasons = unsafe_reasons(measurement)
    counts = unsafe_counts(measurement)
    assert set(reasons) == {"execution_requirement"}
    assert reasons["execution_requirement"] == "task-a: refused"
    assert counts == {"execution_requirement": 1}


def test_population_digest_orders_rows_by_task_id():
    rows = [("task-b", "{}"), ("task-a", "{}")]
    assert population_digest(rows) == population_digest(list(reversed(rows)))
    assert population_digest(rows) != population_digest([("task-a", '{"id":"task-a"}'), ("task-b", "{}")])
