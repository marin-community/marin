# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Missing panels freeze automatically and become current-population attempt inputs."""

import json
from dataclasses import replace

from finestore.rl.curation import ActorCensusInput
from finestore.rl.difficulty import Solver
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from taskcompendium.models import ConversationInput, TaskSpec, TextMessage

from experiments.post_training.task_curation.difficulty import catalog
from experiments.post_training.task_curation.difficulty.executor import EscalationLog, execute_plan
from experiments.post_training.task_curation.difficulty.ledger import append_rows, read_rows
from experiments.post_training.task_curation.difficulty.planner import AttemptBatch, FreezeSamplePanel, action_id, plan
from experiments.post_training.task_curation.difficulty.remediate import repair_action
from experiments.post_training.task_curation.difficulty.state import CoveragePolicy, Outcome, cell_state
from experiments.post_training.task_curation.difficulty_contracts import (
    CONTRACT_VINTAGE_ABSENT_OMITTED,
    difficulty_task_contract,
)
from experiments.post_training.task_curation.difficulty_samples import load_queue_sources
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import write_text_population
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task
from experiments.post_training.task_curation.tests.test_difficulty_executor import JobStore
from experiments.post_training.task_curation.tests.test_difficulty_reconciler import row
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence


def test_missing_panel_freezes_one_shared_queue_then_plans_original_tasks(tmp_path, monkeypatch, difficulty_source):
    task = TaskSpec.model_validate(json.loads(prebuilt_task("panel")["lowered_task_spec"])["task"])
    tasks = [
        task.model_copy(
            update={"id": name, "context": ConversationInput(events=(TextMessage(role="user", content=name),))}
        )
        for name in ("a", "b")
    ]
    declaration = difficulty_source(tasks[0]).declaration
    files = write_text_population(((task.model_dump(mode="json"), {}) for task in tasks), str(tmp_path / "population"))
    population = SourceTextPopulation(
        path=str(tmp_path / "population"),
        source_id=declaration.id,
        dataset_id=declaration.dataset_id,
        dataset_revision=declaration.dataset_revision,
        source_version=declaration.pipeline.version,
        converter_revision="fixture",
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
    declaration = declaration.model_copy(
        update={
            "task_count": 2,
            "actor_census_input": ActorCensusInput(
                dataset_id=population.dataset_id,
                dataset_revision=population.dataset_revision,
                converter_version=population.source_version,
                converter_revision=population.converter_revision,
                measured_tasks=2,
                task_specs_sha256=population.task_specs_sha256,
                evidence_url=str(tmp_path / "population" / "summary.json"),
            ),
            "within_source_dedup": (
                deduplication_evidence(
                    population.dataset_id, population.dataset_revision, population.source_version, 2
                ).model_copy(update={"drop_manifest_uri": files.drop_manifest_uri})
            ),
        }
    )
    served = {"id": declaration.id, "load_id": "frozen-load", "declaration": declaration.model_dump(mode="json")}

    def query(_sql, _params):
        return {"rows": [served]}

    monkeypatch.setattr(catalog, "applet_query", lambda *args, **kwargs: query)
    ledger = str(tmp_path / "ledger")
    (before,) = catalog.served_catalog("fixture", queues=(), ledger=ledger)
    solvers = (Solver.SMALL, Solver.LARGE, Solver.SNOWBALL, Solver.HOSTED)
    cells = [cell_state([], before, solver, CoveragePolicy()) for solver in solvers]
    (action,) = plan(cells, [])
    assert isinstance(action, FreezeSamplePanel) and len(action.cells) == 4
    old = replace(
        row("old-panel", outcome=Outcome.FINISHED),
        source_id=before.source_id,
        registry_revision=before.registry_revision,
        task_id=None,
        task_sha256=None,
        action_id=action_id(replace(action, inventory_revision=None)),
    )
    append_rows(ledger, [old])
    assert plan(cells, read_rows(ledger)) == [action]
    backend = JobStore(str(tmp_path / "jobs"))
    execute_plan([action], [], [before], ledger, backend, EscalationLog(ledger))
    result = repair_action(action, tmp_path / "repair", ledger)
    backend.complete_repair(action, result, ledger)
    (after,) = catalog.served_catalog("fixture", queues=(), ledger=ledger)
    assert after.registry_revision == before.registry_revision
    expected = {
        (task.id, difficulty_task_contract(task, {}, CONTRACT_VINTAGE_ABSENT_OMITTED)["task_sha256"]) for task in tasks
    }
    assert set(after.tasks) == expected
    (frozen,) = load_queue_sources(after.queue_uri, after.queue_sha256, (after.source_id,))
    assert frozen.sample.population_count == 2 and frozen.sample.seed == 0
    rows = read_rows(ledger)
    next_actions = plan([cell_state(rows, after, solver, CoveragePolicy()) for solver in solvers], rows)
    assert len(next_actions) == 4 and all(isinstance(item, AttemptBatch) for item in next_actions)
    assert not (tmp_path / "ledger" / "ESCALATIONS.md").exists()
    assert catalog.served_catalog("fixture", queues=(), ledger=ledger) == [after]
    served["declaration"] = declaration.model_copy(
        update={"current_release": declaration.current_release.model_copy(update={"revision": "b" * 40})}
    ).model_dump(mode="json")
    (changed,) = catalog.served_catalog("fixture", queues=(), ledger=ledger)
    assert changed.registry_revision != before.registry_revision and not changed.tasks


def test_catalog_batches_large_declarations_at_the_original_load(tmp_path, monkeypatch):
    identities = [
        {
            "id": f"fixture:{i}",
            "load_id": "original",
            "declaration": {"current_release": None, "pipeline": {}, "actor_census_input": None},
        }
        for i in range(9)
    ]
    calls = []

    def query(sql, params):
        if not params:
            return {"rows": identities}
        assert params["load_id"] == "original"
        selected = [value for name, value in params.items() if name.startswith("source_")]
        if len(selected) > 4:
            raise AssertionError("Response would exceed the catalog byte limit")
        calls.append(selected)
        return {
            "rows": [
                {"id": source_id, "declaration": {**identities[0]["declaration"], "payload": "large"}}
                for source_id in selected
            ]
        }

    monkeypatch.setattr(catalog, "applet_query", lambda *args, **kwargs: query)
    result = catalog.served_catalog("fixture", queues=(), ledger=str(tmp_path))
    assert [source.source_id for source in result] == [row["id"] for row in identities]
    assert len(calls) == 3
    assert all(json.loads(source.declaration_json)["payload"] == "large" for source in result)
