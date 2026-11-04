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
from fray.local_backend import LocalClient
from marin.execution.lazy import ArtifactStep, run
from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.grader import verifyit_package
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
from taskcompendium.pipeline.conversion import convert_row
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import IntendedUse, NormalizedTask, RawRow
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from verifyit.spec import JudgeSpec
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.datasets.evalchemy.code import sources as code_sources
from experiments.post_training.task_curation.datasets.evalchemy.reasoning import sources as reasoning_sources
from experiments.post_training.task_curation.datasets.nemo_tool_calls import convert as convert_tool_call
from experiments.post_training.task_curation.datasets.nemotron_ultra.components import BLENDS
from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.driver import main
from experiments.post_training.task_curation.inventory import (
    AAII_CATEGORIES,
    FULL_VALIDATION_FILENAME,
    HELDOUT_FAMILIES,
    TASK_MANIFEST_FILENAME,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    SelectionRecipe,
    task_selection_step,
    validation_task,
)
from experiments.post_training.task_curation.inventory import (
    TestPanelRecipe as PanelRecipe,
)
from experiments.post_training.task_curation.inventory import (
    test_panel_step as panel_step,
)
from experiments.post_training.task_curation.pipeline import source_recipe

FAMILIES = (*HELDOUT_FAMILIES.values(), "structural")


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
                task["grader"]["environment"] = {
                    "command_semantics": "linux_process",
                    "packages_lock": "numpy==2.3.5\n",
                }
                dataset = next(dataset for dataset, expected in HELDOUT_FAMILIES.items() if expected == family)
                task["tags"] = ["agentic"] if index == 0 else []
                contract = {"dataset": dataset, "pass_rate_passed": 5, "pass_rate_total": 8, "pass_rate": 0.625}
                task["resources"]["verifier"].append(
                    inline_resource("config.json", json.dumps({"contract": contract}).encode()).model_dump(mode="json")
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
        heldout=({"uri": str(heldout), "version": "2026.10.15.1", "rows": len(rows), "format": "historical"},),
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
        validation_task(row, {}).model_copy(update={"id": f"changed-id-{index}", "answer_format": Boxed()})
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


def test_selection_imports_image_graders_from_admitted_components(tmp_path):
    rows = heldout_rows()
    components = HELDOUT_FAMILIES
    environments = {
        family: EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock=f"numpy==2.3.{index}\n"
        )
        for index, family in enumerate(components.values())
    }
    settings = recipe(tmp_path).model_copy(
        update={
            "sources": {BLENDS["rlvr1"][dataset].id: family for dataset, family in components.items()},
            "family_limits": {family: 1 for family in components.values()},
        }
    )
    curated = {}
    for component, family in settings.sources.items():
        task = source_task(f"training-{family}")
        task = task.model_copy(update={"grader": task.grader.model_copy(update={"environment": environments[family]})})
        curated[component] = admitted_source(tmp_path / family, [task])
    originals = {}
    for row in rows:
        document = json.loads(row["lowered_task_spec"])
        task = document["task"]
        if task["source"]["dataset"] != "structure-closure":
            task["grader"]["environment"] = source_task("image").grader.environment.model_dump(mode="json")
            task["grader"]["argv"][0] = "python3"
            task["grader"]["env"] = {"MODE": "historical"}
            task["grader"]["timeout"] = 330
        originals[task["id"]] = task
        row["lowered_task_spec"] = json.dumps(document)
    pq.write_table(pa.Table.from_pylist(rows), settings.heldout[0].uri)
    output = Path(run(task_selection_step(curated, settings))[0].path)
    full = read_tasks(output / FULL_VALIDATION_FILENAME)
    manifest = pq.read_table(output / TASK_MANIFEST_FILENAME).to_pylist()
    assert {row["task_id"] for row in manifest} == {task.id for task in full}
    by_id = {task.id: task for task in full}
    for row in manifest:
        task = by_id[row["task_id"]]
        original = originals[task.id]
        family = next(family for family in FAMILIES if task.id.startswith(f"heldout-{family}-"))
        assert row == {
            "task_id": task.id,
            "family": family,
            "aaii_category": AAII_CATEGORIES[family],
            "agentic": "agentic" in original["tags"],
            "grader_kind": "in_process" if family == "structural" else "package",
            "upstream_pass_rate": None if family == "structural" else 0.625,
        }
        if family == "structural":
            continue
        assert task.grader.environment == environments[family]
        assert task.resources.model_dump(mode="json") == original["resources"]
        assert task.grader.model_dump(mode="json", exclude={"environment"}) == {
            key: value for key, value in original["grader"].items() if key != "environment"
        }
    summary = json.loads((output / "summary.json").read_text())
    assert summary["grader_kind_counts"] == {
        "train": {"package": 4, "image": 0, "remote_judge": 0, "in_process": 0},
        "validation": {"package": 80, "image": 0, "remote_judge": 0, "in_process": 20},
        "full_validation": {"package": 84, "image": 0, "remote_judge": 0, "in_process": 21},
    }


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


def panel_recipe(tmp_path: Path, benchmark: str = "mmlu") -> PanelRecipe:
    declarations = reasoning_sources() if benchmark == "mmlu" else code_sources()
    name = "evalchemy-mmlu-pro" if benchmark == "mmlu" else "evalchemy-mbppplus"
    dataset = next(source.config for source in declarations if source.name == name)
    assert dataset is not None
    source_root, fewshot = tmp_path / "questions", tmp_path / "examples"
    question = {
        "question": "Panel question",
        "options": ["1", "2"],
        "answer": "B",
        "category": "math",
        "cot_content": "A: Let's think step by step. The answer is (B).",
    }
    inputs = {}
    files = [
        (
            source_root,
            dataset.recipe.files[0],
            [
                (
                    {**question, "question": f"Panel question {i}"}
                    if benchmark == "mmlu"
                    else {
                        "prompt": f"Write function number_{i} that returns {i}.",
                        "test_list": [f"assert number_{i}() == {i}"],
                        "test": f"assert number_{i}() == {i}",
                        "code": f"EXAMPLE_SENTINEL_{i} = 1",
                    }
                )
                for i in range(6)
            ],
        )
    ]
    if benchmark == "mmlu":
        inputs["fewshot"] = str(fewshot)
        files.append(
            (
                fewshot,
                dataset.recipe.inputs["fewshot"].files[0],
                [{**question, "question": f"EXAMPLE_SENTINEL_{i}"} for i in range(6)],
            )
        )
    for root, filename, records in files:
        path = root / filename
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(records), path)
    corpus = tmp_path / "general"
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            publish_evaluation_corpus(
                source_recipe(
                    dataset.recipe,
                    inputs,
                    EnvironmentRequirements(
                        command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock="tomli-w==1.2.0\n"
                    ),
                ),
                dataset.protocol.model_copy(update={"expected_population": 6}),
                dataset.item_builder,
                context,
                str(source_root),
                str(corpus),
            )
    finally:
        client.shutdown()
    agents = tmp_path / "agents"
    (agents / "normalize").mkdir(parents=True)
    (agents / "manifest.json").write_text(json.dumps({"intended_use": "eval"}))
    fixture = Path(__file__).parents[4] / "lib/taskcompendium/tests/fixtures/nemo/predicted-action.json"
    tasks = []
    for index in range(4):
        data = json.loads(fixture.read_text())
        data["responses_create_params"]["input"][-1]["content"] = f"Native tool request {index}"
        raw = RawRow(f"agent-{index}", source_task(f"agent-{index}").source, data)
        converted = convert_row(raw, convert_tool_call, ConversionContext({}, None))
        assert isinstance(converted, NormalizedTask)
        tasks.append(converted.task)
    pq.write_table(
        pa.Table.from_pylist([*[{"task_json": task.model_dump_json()} for task in tasks], {"task_json": None}]),
        agents / "normalize/part-00000.parquet",
    )
    exclusions = {}
    for name in ("training", "development"):
        path = tmp_path / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist([{"task_spec": source_task(name).model_dump_json()}]), path)
        exclusions[name] = {"uri": str(path), "version": "2026.11.04", "rows": 1, "format": "task_spec"}
    family = "general" if benchmark == "mmlu" else "coding"
    return PanelRecipe(
        name="test-panel",
        version="2026.11.04",
        seed=20261104,
        sources=(
            {
                "input": {"uri": str(corpus), "version": "2026.11.04", "rows": 6, "format": "evaluation"},
                "family": family,
                "aaii_category": family,
                "license": "fixture-general",
                "count": 5,
            },
            {
                "input": {"uri": str(agents), "version": "2026.11.04", "rows": 4, "format": "normalized"},
                "family": "agents",
                "aaii_category": "agents",
                "license": "fixture-agents",
                "count": 3,
            },
        ),
        fast_counts={family: 2, "agents": 2},
        **exclusions,
    )


@pytest.mark.parametrize("agent_passed", [0, 6])
@pytest.mark.parametrize("benchmark,family", [("mmlu", "general"), ("mbpp", "coding")])
def test_panel_publication_preserves_untouched_strata_and_joins_initializer_evidence(
    tmp_path, agent_passed, benchmark, family
):
    settings = panel_recipe(tmp_path, benchmark)
    original = [case.task for case in evaluation_cases(settings.sources[0].input.uri)]
    assert all("EXAMPLE_SENTINEL" in task.context.events[0].content for task in original)
    provisional = Path(run(panel_step(settings))[0].path)
    full, fast = read_tasks(provisional / FULL_VALIDATION_FILENAME), read_tasks(provisional / VALIDATION_FILENAME)
    assert len(full) == 8 and len(fast) == 4
    assert {task.id for task in fast} < {task.id for task in full}
    assert all("EXAMPLE_SENTINEL" not in task.context.events[0].content for task in full)
    annotations = pq.read_table(provisional / TASK_MANIFEST_FILENAME).to_pylist()
    by_id = {row["task_id"]: row for row in annotations}
    assert Counter(row["family"] for row in annotations) == {family: 5, "agents": 3}
    assert Counter(by_id[task.id]["family"] for task in fast) == {family: 2, "agents": 2}
    assert {row["license"] for row in annotations} == {"fixture-general", "fixture-agents"}
    assert all(row["source"] and row["grader_kind"] in {"package", "in_process"} for row in annotations)
    assert all(row["difficulty"] is None for row in annotations)
    measurements = tmp_path / "pass-rates.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": task.id,
                    "model_revision": "fixture-model",
                    "attempted": 8,
                    "samples": 8,
                    "passed": agent_passed if by_id[task.id]["family"] == "agents" else 6,
                    "mean_score": agent_passed / 8 if by_id[task.id]["family"] == "agents" else 0.75,
                }
                for task in full
            ]
        ),
        measurements,
    )
    final_recipe = PanelRecipe.model_validate(
        {
            **settings.model_dump(),
            "version": "2026.11.04.1",
            "pass_rates": {
                "uri": str(measurements),
                "version": "2026.11.04",
                "rows": 8,
                "model_revision": "fixture-model",
                "run_output": str(tmp_path / "saved-answer-run"),
            },
            "provisional": {
                "uri": str(provisional / FULL_VALIDATION_FILENAME),
                "version": "2026.11.04",
                "rows": 8,
                "format": "task_spec",
            },
        }
    )
    final = Path(run(panel_step(final_recipe))[0].path)
    final_full, final_fast = read_tasks(final / FULL_VALIDATION_FILENAME), read_tasks(final / VALIDATION_FILENAME)
    if agent_passed:
        assert final_full == full and final_fast == fast
    else:
        assert final_full == [task for task in full if by_id[task.id]["family"] == family]
        assert len(final_fast) == 4
    measured = pq.read_table(final / TASK_MANIFEST_FILENAME).to_pylist()
    assert all(row["difficulty"] == 0.25 and row["attempted"] == row["scored"] == 8 for row in measured)
    assert all(row["initializer_pass_rate"] == row["initializer_mean_score"] == 0.75 for row in measured)


@pytest.mark.parametrize(
    "failure",
    [
        "judge",
        "image",
        "training-id",
        "training-context",
        "development-id",
        "development-context",
        "missing-rate",
        "duplicate-rate",
    ],
)
def test_panel_publication_refuses_remote_graders_and_training_or_development_overlap(tmp_path, failure):
    settings = panel_recipe(tmp_path)
    if failure.endswith("rate"):
        provisional = Path(run(panel_step(settings))[0].path)
        tasks = read_tasks(provisional / FULL_VALIDATION_FILENAME)
        rows = [
            {
                "task_id": task.id,
                "model_revision": "fixture-model",
                "attempted": 8,
                "samples": 8,
                "passed": 6,
                "mean_score": 0.75,
            }
            for task in tasks
        ]
        rows = rows[:-1] if failure == "missing-rate" else [*rows, rows[0]]
        measurements = tmp_path / "invalid-pass-rates.parquet"
        pq.write_table(pa.Table.from_pylist(rows), measurements)
        settings = PanelRecipe.model_validate(
            {
                **settings.model_dump(),
                "version": "2026.11.04.1",
                "pass_rates": {
                    "uri": str(measurements),
                    "version": "2026.11.04",
                    "rows": 8,
                    "model_revision": "fixture-model",
                    "run_output": str(tmp_path / "saved-answer-run"),
                },
                "provisional": {
                    "uri": str(provisional / FULL_VALIDATION_FILENAME),
                    "version": "2026.11.04",
                    "rows": 8,
                    "format": "task_spec",
                },
            }
        )
        step = panel_step(settings)
        with pytest.raises(RuntimeError):
            run(step)
        assert not (Path(step.path()) / FULL_VALIDATION_FILENAME).exists()
        assert not (Path(step.path()) / TASK_MANIFEST_FILENAME).exists()
        return
    source = settings.sources[1].input
    path = Path(source.uri) / "normalize/part-00000.parquet"
    rows = pq.read_table(path).to_pylist()
    task = TaskSpec.model_validate_json(rows[0]["task_json"])
    if failure == "judge":
        task = source_task("judge").model_copy(
            update={
                "grader": verifyit_package(JudgeSpec(references=("target",))).grader,
                "tags": ("eval",),
            }
        )
    elif failure == "image":
        task = source_task("image").model_copy(update={"tags": ("eval",)})
    else:
        name, kind = failure.split("-")
        excluded = read_tasks(Path(getattr(settings, name).uri))[0]
        task = task.model_copy(
            update=(
                {"id": excluded.id}
                if kind == "id"
                else {
                    "context": excluded.context,
                    "tags": (*task.tags, "alias"),
                }
            )
        )
    rows[0] = {"task_json": task.model_dump_json()}
    pq.write_table(pa.Table.from_pylist(rows), path)
    step = panel_step(settings)
    with pytest.raises(RuntimeError):
        run(step)
    assert not (Path(step.path()) / FULL_VALIDATION_FILENAME).exists()
    assert not (Path(step.path()) / TASK_MANIFEST_FILENAME).exists()
