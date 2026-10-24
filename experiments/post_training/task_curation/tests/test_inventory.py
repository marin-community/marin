# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Selection over admitted TaskSpecs and the complete historical heldout panel."""

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from marin.execution.lazy import ArtifactStep, run
from taskcompendium.models import (
    AnswerType,
    Boxed,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import IntendedUse
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.driver import main
from experiments.post_training.task_curation.inventory import (
    FULL_VALIDATION_FILENAME,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    SelectionRecipe,
    task_selection_step,
    validation_task,
)

FAMILIES = ("calendar", "competitive_coding", "reasoning_gym", "stem", "structural")


@pytest.fixture(autouse=True)
def local_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.setattr("rigging.filesystem.cluster_config.region_from_metadata", lambda: None)


def source_task(identifier: str) -> TaskSpec:
    return TaskSpec(
        id=identifier,
        source=Source(dataset="fixture", revision="release", row=identifier, importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content=f"Solve {identifier}."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=ScriptGrader(
            argv=("python", "/tests/grade.py"),
            env={"MODE": "fixture"},
            environment=EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="python@sha256:" + "a" * 64
            ),
        ),
        resources=ResourceGroups(verifier=(inline_resource("private.txt", identifier.encode()),)),
    )


def heldout_rows() -> list[dict]:
    rows = []
    for family in FAMILIES:
        for index in range(21):
            identifier = f"heldout-{family}-{index}"
            task: dict[str, Any] = {**source_task(identifier).model_dump(mode="json"), "schema_version": "0.27"}
            if family == "structural":
                task["source"]["dataset"] = "structure-closure"
                task["grader"] = {"kind": "session"}
                config = inline_resource(
                    "config.json",
                    json.dumps({"extras": {"prefix": '{"a":[1', "expected": '{"a":[1,2]}', "format": "json"}}).encode(),
                )
                task["resources"]["verifier"].append(config.model_dump(mode="json"))
            else:
                dataset = {
                    "calendar": "ultra_sft_step3200_calendar_v2",
                    "competitive_coding": "ultra_sft_step3200_comp_coding",
                    "reasoning_gym": "ultra_sft_step3200_reasoning_gym",
                    "stem": "ultra_sft_step3200_stem_mcqa",
                }[family]
                task["resources"]["verifier"].append(
                    inline_resource("config.json", json.dumps({"contract": {"dataset": dataset}}).encode()).model_dump(
                        mode="json"
                    )
                )
            rows.append(
                {
                    "lowered_task_spec": json.dumps(
                        {
                            "task": task,
                            "runtime": {"task_machine": None, "verifier_machine": None},
                            "session": {
                                "task_session": "structure_closure" if family == "structural" else "shellbox",
                                "max_turns": 1,
                            },
                        }
                    ),
                }
            )
    return rows


def recipe(tmp_path: Path) -> SelectionRecipe:
    rows = heldout_rows()
    heldout = tmp_path / "heldout.parquet"
    pq.write_table(pa.Table.from_pylist(rows), heldout)
    return SelectionRecipe(
        name="fixture",
        version="2026.10.23.1",
        sources={"fixture:first": "calendar", "fixture:second": "calendar"},
        family_limits={"calendar": 2},
        heldout={"uri": str(heldout), "version": "2026.10.15.1", "rows": len(rows)},
        validation_per_family=20,
        seed=20261023,
    )


def admitted_source(root: Path, tasks: list[TaskSpec], intended_use: str = "train") -> ArtifactStep:
    (root / "final").mkdir(parents=True)
    (root / "normalize").mkdir()
    (root / "manifest.json").write_text(json.dumps({"intended_use": intended_use, "mode": "full"}))
    pq.write_table(
        pa.Table.from_pylist([{"task_json": task.model_dump_json()} for task in tasks]),
        root / "final/part-00000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([{"task_json": source_task("unadmitted").model_dump_json()}]),
        root / "normalize/part-00000.parquet",
    )
    return ArtifactStep.adopt(f"curated/{root.name}", "2026.10.23.1", str(root))


def read_tasks(path: Path) -> list[TaskSpec]:
    return [TaskSpec.model_validate_json(row["task_spec"]) for row in pq.read_table(path).to_pylist()]


def test_selection_filters_full_heldout_and_duplicates_before_sampling(tmp_path):
    settings = recipe(tmp_path)
    leaked = [
        validation_task(row).model_copy(update={"id": f"changed-id-{index}", "answer_format": Boxed()})
        for index, row in enumerate(heldout_rows()[:21])
    ]
    original = source_task("training-a")
    duplicate = original.model_copy(update={"id": "alias"})
    first = admitted_source(tmp_path / "first", [original, source_task("training-b"), *leaked])
    second = admitted_source(tmp_path / "second", [duplicate, source_task("training-c")])
    curated = {"fixture:first": first, "fixture:second": second}
    artifact = run(task_selection_step(curated, settings))[0]
    output = Path(artifact.path)
    tasks = read_tasks(output / TRAIN_FILENAME)
    assert len(tasks) == 2
    assert {task.id for task in tasks} <= {"training-a", "training-b", "training-c"}
    for task in tasks:
        assert task == source_task(task.id)
        assert resource_bytes(task.resources.verifier[0]) == task.id.encode()
    full = read_tasks(output / FULL_VALIDATION_FILENAME)
    sample = read_tasks(output / VALIDATION_FILENAME)
    families = {f"heldout-{family}-{index}": family for family in FAMILIES for index in range(21)}
    assert [task.id for task in full] == list(families)
    assert Counter(families[task.id] for task in sample) == dict.fromkeys(FAMILIES, 20)
    structural = next(task for task in full if families[task.id] == "structural")
    assert structural.context.events[-1] == TextMessage(role="assistant", content='{"a":[1')
    assert structural.grader.mode == "structure_closure"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["source_counts"] == {
        "fixture:first": {
            "input": 23,
            "heldout_overlap": 21,
            "duplicate": 0,
            "eligible": 2,
            "selected": sum(task.id != "training-c" for task in tasks),
        },
        "fixture:second": {
            "input": 2,
            "heldout_overlap": 0,
            "duplicate": 1,
            "eligible": 1,
            "selected": sum(task.id == "training-c" for task in tasks),
        },
    }
    # A new artifact version and reversed dependency insertion preserve both seeded selections.
    repeated = run(
        task_selection_step(dict(reversed(curated.items())), settings.model_copy(update={"version": "2026.10.23.2"}))
    )[0]
    assert tasks == read_tasks(Path(repeated.path) / TRAIN_FILENAME)
    assert sample == read_tasks(Path(repeated.path) / VALIDATION_FILENAME)


@pytest.mark.parametrize("shared_id", ["shared-unsampled", "shared-alias"])
def test_source_quotas_can_select_an_earlier_sources_unsampled_duplicate(tmp_path, shared_id):
    settings = recipe(tmp_path).model_copy(
        update={"family_limits": {"calendar": 3}, "source_limits": {"fixture:first": 1, "fixture:second": 2}}
    )
    selected = source_task("first-selected")
    shared = source_task("shared-unsampled")
    second = source_task("second-only")
    curated = {
        "fixture:first": admitted_source(tmp_path / "first", [selected, shared]),
        "fixture:second": admitted_source(
            tmp_path / "second",
            [
                selected.model_copy(update={"id": "selected-alias"}),
                shared.model_copy(update={"id": shared_id}),
                second,
                second.model_copy(update={"id": "second-copy"}),
            ],
        ),
    }
    output = run(task_selection_step(curated, settings))[0]
    tasks = read_tasks(Path(output.path) / TRAIN_FILENAME)
    assert {task.id for task in tasks} == {"first-selected", shared_id, "second-only"}


def test_source_shortfall_cannot_be_filled_from_another_source(tmp_path):
    settings = recipe(tmp_path).model_copy(
        update={"family_limits": {"calendar": 3}, "source_limits": {"fixture:first": 1, "fixture:second": 2}}
    )
    curated = {
        "fixture:first": admitted_source(tmp_path / "first", [source_task(f"first-{i}") for i in range(5)]),
        "fixture:second": admitted_source(tmp_path / "second", [source_task("second-only")]),
    }
    selection = task_selection_step(curated, settings)
    with pytest.raises(RuntimeError):
        run(selection)
    assert not (Path(selection.path()) / TRAIN_FILENAME).exists()


@pytest.mark.parametrize("failure", ["shortfall", "evaluation-source"])
def test_selection_does_not_publish_an_incomplete_or_evaluation_training_pool(tmp_path, failure):
    settings = recipe(tmp_path).model_copy(update={"sources": {"fixture:first": "calendar"}})
    source = admitted_source(
        tmp_path / "source", [source_task("training-a")], "eval" if failure == "evaluation-source" else "train"
    )
    step = task_selection_step({"fixture:first": source}, settings)
    with pytest.raises(RuntimeError):
        run(step)
    assert not (Path(step.path()) / TRAIN_FILENAME).exists()


def test_full_cli_appends_selection_and_saved_corpus_can_reselect(tmp_path, monkeypatch):
    declared = next(source for source in skyrl_math.sources() if source.name == "math500")
    assert declared.config is not None
    source = replace(
        declared, config=replace(declared.config, rubric=None, controls=None, intended_use=IntendedUse.TRAIN)
    )
    monkeypatch.setattr("experiments.post_training.task_curation.driver.runnable_sources", lambda: {source.name: source})
    staged = tmp_path / "input"
    staged.mkdir()
    input_file = staged / "test.jsonl"
    input_file.write_text('{"problem": "Two plus two?", "answer": "4"}\n')
    settings = recipe(tmp_path).model_copy(update={"sources": {source.info.id: "math"}, "family_limits": {"math": 1}})
    selection_file = tmp_path / "selection.json"
    selection_file.write_text(settings.model_dump_json())
    cache = tmp_path / "cache"
    options = [
        "--mode",
        "full",
        "--skip-assessment",
        "--cache-dir",
        str(cache),
        "--source",
        source.name,
        "--input-root",
        str(staged),
        "--max-workers",
        "1",
        "--normalized-shards",
        "1",
        "--selection",
        str(selection_file),
    ]
    runner = CliRunner()
    planned = runner.invoke(main, options)
    assert planned.exit_code == 0, planned.output
    plan = json.loads(planned.output)
    assert plan["sources"][0]["name"] in plan["task_selection"]["dependencies"]
    output = Path(plan["task_selection"]["path"])
    assert not output.exists()
    executed = runner.invoke(main, [*options, "--run"])
    assert executed.exit_code == 0, executed.output
    tasks = read_tasks(output / TRAIN_FILENAME)
    assert len(tasks) == 1
    assert tasks[0].context.events[-1].content == "Two plus two?"
    input_file.unlink()
    selection_file.write_text(settings.model_copy(update={"version": "2026.10.23.2"}).model_dump_json())
    resumed = runner.invoke(
        main,
        [
            "--corpus",
            str(cache / "campaigns/corpus.json"),
            "--selection",
            str(selection_file),
            "--cache-dir",
            str(cache),
            "--run",
        ],
    )
    assert resumed.exit_code == 0, resumed.output
    assert tasks == read_tasks(cache / "rl-data/task-selection/fixture/2026.10.23.2" / TRAIN_FILENAME)
