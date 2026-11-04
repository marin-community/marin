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
from taskcompendium.convert.answers import source_evidence
from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.grader import verifyit_package
from taskcompendium.identity import canonical_sha256
from taskcompendium.importers.nemo_predicted_action import import_row
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

from experiments.post_training.task_curation.campaign import error_chain
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
    input_file.write_text('{"problem": "Two plus two?", "answer": "4"}\n{"problem": "Two plus three?", "answer": "5"}\n')
    settings = recipe(tmp_path).model_copy(update={"sources": {source.info.id: "math"}, "family_limits": {"math": 1}})
    excluded = source_task("independent-exclusion").model_copy(
        update={
            "context": ConversationInput(events=(TextMessage(role="user", content="Two plus two?"),)),
        }
    )
    excluded_file = tmp_path / "exclude.parquet"
    pq.write_table(pa.Table.from_pylist([{"task_spec": excluded.model_dump_json()}]), excluded_file)
    settings = SelectionRecipe.model_validate(
        {
            **settings.model_dump(),
            "exclusions": [
                {
                    "uri": str(excluded_file),
                    "version": "2026.11.04",
                    "rows": 1,
                    "format": "task_spec",
                }
            ],
        }
    )
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
    assert tasks[0].context.events[-1].content == "Two plus three?"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["source_counts"][source.info.id]["heldout_overlap"] == 1
    expected_ids = {
        json.loads(row["lowered_task_spec"])["task"]["id"] for row in pq.read_table(settings.heldout[0].uri).to_pylist()
    }
    assert {task.id for task in read_tasks(output / FULL_VALIDATION_FILENAME)} == expected_ids
    assert {task.id for task in read_tasks(output / VALIDATION_FILENAME)} <= expected_ids
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
    questions = [
        (
            {**question, "question": f"Panel question {i}"}
            if benchmark == "mmlu"
            else {
                "prompt": f"Write function number_{i} that returns {i}.",
                "test_list": [f"assert number_{i}() == {i}"],
                "test": f"assert number_{i}() == {i}\nassert isinstance(number_{i}(), int)",
                "code": f"def number_{i}():\n    # EXAMPLE_SENTINEL_{i}\n    return {i}",
            }
        )
        for i in range(6)
    ]
    files = [(source_root, dataset.recipe.files[0], questions)]
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
    if benchmark == "mmlu":
        duplicate_case = list(evaluation_cases(str(corpus)))[1]
        tasks.append(
            duplicate_case.task.model_copy(
                update={
                    "id": "cross-part-alias",
                    "context": duplicate_case.item.operation.variants[0].context,
                    "tags": (*duplicate_case.task.tags, "tool-call-proxy"),
                }
            )
        )
    pq.write_table(
        pa.Table.from_pylist([*[{"task_json": task.model_dump_json()} for task in tasks], {"task_json": None}]),
        agents / "normalize/part-00000.parquet",
    )
    general_tasks = [case.task for case in evaluation_cases(str(corpus))]
    exclusions = {}
    for name in ("training", "training-second", "development"):
        path = tmp_path / f"{name}.parquet"
        task = source_task(name)
        if name == "training":
            task = task.model_copy(update={"id": general_tasks[0].id})
        elif name == "training-second":
            task = task.model_copy(
                update={
                    "context": ConversationInput(
                        events=(
                            TextMessage(role="system", content="Historical request wrapper"),
                            TextMessage(role="user", content="  Native\n tool  request\t0  "),
                        )
                    )
                }
            )
        pq.write_table(pa.Table.from_pylist([{"task_spec": task.model_dump_json()}]), path)
        exclusions[name] = {"uri": str(path), "version": "2026.11.04", "rows": 1, "format": "task_spec"}
    corpus_file = tmp_path / "corpus.json"
    corpus_file.write_text(
        json.dumps(
            {
                "population": "converted",
                "inventory": [],
                "sources": [
                    {
                        "source_id": source_id,
                        "root": str(root),
                        "source_key": canonical_sha256(json.loads((root / "manifest.json").read_text())),
                    }
                    for source_id, root in (("Evalchemy:questions", corpus), ("NeMo:predicted-action", agents))
                ],
            }
        )
    )
    family = "general" if benchmark == "mmlu" else "competitive_coding"
    return PanelRecipe(
        name="test-panel",
        version="2026.11.04",
        seed=20261104,
        sources=(
            {
                "name": "questions",
                "source_id": "Evalchemy:questions",
                "rows": 6,
                "format": "evaluation",
                "required_tags": ["eval"],
                "family": family,
                "aaii_category": "general" if benchmark == "mmlu" else "coding",
                "license": "fixture-general",
                "count": 5,
            },
            {
                "name": "tool-calls",
                "source_id": "NeMo:predicted-action",
                "rows": len(tasks),
                "format": "normalized",
                "required_tags": ["eval", "tool-call-proxy"],
                "family": "agents",
                "aaii_category": "agents",
                "license": "fixture-agents",
                "count": 3,
            },
        ),
        fast_counts={family: 3, "agents": 1},
        training=(exclusions["training"], exclusions["training-second"]),
        development=exclusions["development"],
    )


def publish_panel(settings: PanelRecipe, tmp_path: Path, cache: str = "artifacts"):
    recipe_file = tmp_path / f"panel-{settings.version}.json"
    recipe_file.write_text(settings.model_dump_json())
    result = CliRunner().invoke(
        main,
        [
            "--corpus",
            str(tmp_path / "corpus.json"),
            "--test-panel",
            str(recipe_file),
            "--cache-dir",
            str(tmp_path / cache),
            "--run",
        ],
    )
    return result, tmp_path / cache / "rl-data/task-selection" / settings.name / settings.version


def measured_recipe(settings: PanelRecipe, tmp_path: Path, rows: list[dict]) -> PanelRecipe:
    measurements = tmp_path / "pass-rates.parquet"
    pq.write_table(pa.Table.from_pylist(list(reversed(rows))), measurements)
    return PanelRecipe.model_validate(
        {
            **settings.model_dump(),
            "version": "2026.11.04.1",
            "pass_rates": {
                "uri": str(measurements),
                "version": "2026.11.04",
                "rows": len(rows),
                "model_revision": "fixture-model",
            },
        }
    )


def panel_rates(tasks: list[TaskSpec]) -> list[dict]:
    return [
        {
            "task_id": task.id,
            "task_digest": canonical_sha256(task.model_dump(mode="json")),
            "model_revision": "fixture-model",
            "attempted": 8,
            "samples": 8,
            "passed": index + 1,
            "mean_score": (index + 1) / 8,
        }
        for index, task in enumerate(tasks)
    ]


@pytest.mark.parametrize(
    "benchmark,family,category", [("mmlu", "general", "general"), ("mbpp", "competitive_coding", "coding")]
)
def test_panel_publication_preserves_untouched_strata_and_joins_initializer_evidence(
    tmp_path, benchmark, family, category
):
    settings = panel_recipe(tmp_path, benchmark)
    original_tasks = [case.task for case in evaluation_cases(str(tmp_path / "general"))]
    assert all("EXAMPLE_SENTINEL" in task.context.events[0].content for task in original_tasks)
    result, provisional = publish_panel(settings, tmp_path)
    assert result.exit_code == 0, result.output
    full, fast = read_tasks(provisional / FULL_VALIDATION_FILENAME), read_tasks(provisional / VALIDATION_FILENAME)
    assert len(full) == 8 and len(fast) == 4
    assert {task.id for task in fast} < {task.id for task in full}
    assert all("EXAMPLE_SENTINEL" not in task.context.events[0].content for task in full)
    originals = {task.id: task for task in original_tasks}
    agents = {
        task.id: task
        for task in (
            TaskSpec.model_validate_json(row["task_json"])
            for row in pq.read_table(tmp_path / "agents/normalize/part-00000.parquet").to_pylist()
            if row["task_json"] is not None
        )
    }
    for task in full:
        if task.id in agents:
            assert task == agents[task.id]
            continue
        original = originals[task.id]
        assert task.model_dump(exclude={"context"}) == original.model_dump(exclude={"context"})
        assert len(task.context.events) == 1 and task.context.events[0].role == "user"
        prompt = task.context.events[0].content
        if benchmark == "mmlu":
            target = original.context.events[0].content.rsplit("Question:\n", 1)[1]
            assert "Question:\n" + target in prompt
            assert "Panel question" in prompt and "A. 1" in prompt and "B. 2" in prompt
        else:
            evidence = source_evidence(original)
            assert evidence["prompt"] in prompt
            assert all(public_test in prompt for public_test in evidence["test_list"])
            assert "isinstance" not in prompt
    annotations = pq.read_table(provisional / TASK_MANIFEST_FILENAME).to_pylist()
    result, replayed = publish_panel(settings, tmp_path, "replayed-artifacts")
    assert result.exit_code == 0, result.output
    assert read_tasks(replayed / FULL_VALIDATION_FILENAME) == full
    assert read_tasks(replayed / VALIDATION_FILENAME) == fast
    assert pq.read_table(replayed / TASK_MANIFEST_FILENAME).to_pylist() == annotations
    by_id = {row["task_id"]: row for row in annotations}
    assert by_id.keys() == {task.id for task in full}
    assert len(by_id) == len(annotations) == len(full)
    for task in full:
        agent = task.id in agents
        native = agents[task.id] if agent else originals[task.id]
        row = by_id[task.id]
        assert row["part"] == ("tool-calls" if agent else "questions")
        assert row["family"] == ("agents" if agent else family)
        assert row["aaii_category"] == ("agents" if agent else category)
        assert row["license"] == ("fixture-agents" if agent else "fixture-general")
        assert row["source"] == native.source.model_dump_json()
        assert row["agentic"] is agent
        assert row["grader_kind"] == ("in_process" if agent else "package")
    assert Counter(row["family"] for row in annotations) == {family: 5, "agents": 3}
    assert Counter(row["part"] for row in annotations) == {"questions": 5, "tool-calls": 3}
    assert Counter(by_id[task.id]["family"] for task in fast) == {family: 3, "agents": 1}
    assert all(row["difficulty"] is None for row in annotations)
    summary = json.loads((provisional / "summary.json").read_text())
    assert summary["source_counts"]["questions"]["overlap"] == 1
    assert summary["source_counts"]["tool-calls"]["overlap"] == 1
    assert original_tasks[0].id not in by_id and "agent-0" not in by_id
    if benchmark == "mmlu":
        assert summary["source_counts"]["tool-calls"]["duplicate"] == 1
    native_data = json.loads(
        (Path(__file__).parents[4] / "lib/taskcompendium/tests/fixtures/nemo/predicted-action.json").read_text()
    )
    native_data["responses_create_params"]["input"][-1]["content"] = "Native tool request 1"
    native = import_row(native_data, canonical_sha256(native_data))
    actual = next(task for task in full if task.id == "agent-1")
    assert actual.model_dump(exclude={"id", "source", "tags"}) == native.model_dump(exclude={"id", "source", "tags"})
    rates = panel_rates(full)
    expected = {row["task_id"]: row for row in rates}
    final_recipe = measured_recipe(settings, tmp_path, rates)
    result, final = publish_panel(final_recipe, tmp_path)
    assert result.exit_code == 0, result.output
    assert read_tasks(final / FULL_VALIDATION_FILENAME) == full
    assert read_tasks(final / VALIDATION_FILENAME) == fast
    for row in pq.read_table(final / TASK_MANIFEST_FILENAME).to_pylist():
        assert row["difficulty"] == 1 - expected[row["task_id"]]["mean_score"]
    subset = final_recipe.model_copy(
        update={
            "version": "2026.11.04.2",
            "sources": (final_recipe.sources[0],),
            "fast_counts": {family: 4},
        }
    )
    result, subset_output = publish_panel(subset, tmp_path)
    assert result.exit_code == 0, result.output
    assert read_tasks(subset_output / FULL_VALIDATION_FILENAME) == [task for task in full if task.id in originals]
    assert len(read_tasks(subset_output / VALIDATION_FILENAME)) == 4
    path = tmp_path / "agents/normalize/part-00000.parquet"
    rows = pq.read_table(path).to_pylist()
    measured_agent = next(task for task in full if task.id in agents)
    calls = json.loads(json.dumps(measured_agent.grader.parameters["expected_calls"]))
    calls[0]["arguments"]["user_id"] = "DIFFERENT_USER"
    changed = measured_agent.model_copy(
        update={
            "grader": measured_agent.grader.model_copy(
                update={
                    "parameters": {**measured_agent.grader.parameters, "expected_calls": calls},
                }
            )
        }
    )
    for row in rows:
        if row["task_json"] is not None and json.loads(row["task_json"])["id"] == changed.id:
            row["task_json"] = changed.model_dump_json()
    pq.write_table(pa.Table.from_pylist(rows), path)
    stale = final_recipe.model_copy(update={"version": "2026.11.04.3"})
    result, stale_output = publish_panel(stale, tmp_path)
    assert result.exit_code != 0
    assert "Pass-rate evidence measures different task content" in error_chain(result.exception)
    assert not (stale_output / FULL_VALIDATION_FILENAME).exists()


@pytest.mark.parametrize(
    "failure,cause",
    [
        ("judge", "Test panel requires a local grader"),
        ("image", "Test panel requires a local grader"),
        ("training-id", "Test panel part shortfall: tool-calls"),
        ("training-context", "Test panel part shortfall: tool-calls"),
        ("training-whitespace", "Test panel part shortfall: tool-calls"),
        ("development-id", "Test panel part shortfall: tool-calls"),
        ("development-context", "Test panel part shortfall: tool-calls"),
        ("train-manifest", "Panel source is not reserved for evaluation"),
        ("eval-tag", "Test panel source must reserve its tasks for evaluation"),
        ("source-count", "Panel source row count changed"),
        ("part-shortfall", "Test panel part shortfall: tool-calls"),
        ("fast-shortfall", "Fast-panel source shortfall for general"),
        ("fast-coverage", "Fast-panel quotas must cover every selected family"),
        ("missing-rate", "Pass-rate evidence must cover every selected panel task"),
        ("duplicate-rate", "Panel pass-rate rows must have unique task IDs"),
        ("count-rate", "Panel pass-rate rows must have unique task IDs"),
        ("attempted-rate", "Panel pass-rate evidence is incomplete"),
        ("model-rate", "Panel pass-rate evidence is incomplete"),
        ("digest-rate", "Pass-rate evidence measures different task content"),
    ],
)
def test_panel_publication_refuses_remote_graders_and_training_or_development_overlap(tmp_path, failure, cause):
    settings = panel_recipe(tmp_path)
    if failure.endswith("rate"):
        result, provisional = publish_panel(settings, tmp_path)
        assert result.exit_code == 0, result.output
        rows = panel_rates(read_tasks(provisional / FULL_VALIDATION_FILENAME))
        field, value = {
            "missing-rate": ("task_id", "unknown"),
            "duplicate-rate": ("task_id", rows[0]["task_id"]),
            "count-rate": ("samples", 8),
            "attempted-rate": ("attempted", 7),
            "model-rate": ("model_revision", "another-model"),
            "digest-rate": ("task_digest", "0" * 64),
        }[failure]
        rows[-1] = {**rows[-1], field: value}
        settings = measured_recipe(settings, tmp_path, rows)
        if failure == "count-rate":
            settings = settings.model_copy(
                update={"pass_rates": settings.pass_rates.model_copy(update={"rows": len(rows) + 1})}
            )
    elif failure == "source-count":
        settings = settings.model_copy(
            update={
                "sources": (
                    settings.sources[0],
                    settings.sources[1].model_copy(update={"rows": settings.sources[1].rows + 1}),
                )
            }
        )
    elif failure in ("part-shortfall", "fast-shortfall", "fast-coverage"):
        if failure == "part-shortfall":
            settings = settings.model_copy(
                update={"sources": (settings.sources[0], settings.sources[1].model_copy(update={"count": 4}))}
            )
        else:
            settings = settings.model_copy(
                update={"fast_counts": {"general": 6, "agents": 1} if failure == "fast-shortfall" else {"general": 3}}
            )
    elif failure == "train-manifest":
        manifest_file = tmp_path / "agents/manifest.json"
        manifest = json.loads(manifest_file.read_text())
        manifest["intended_use"] = "train"
        manifest_file.write_text(json.dumps(manifest))
        corpus_file = tmp_path / "corpus.json"
        corpus = json.loads(corpus_file.read_text())
        corpus["sources"][1]["source_key"] = canonical_sha256(manifest)
        corpus_file.write_text(json.dumps(corpus))
    else:
        path = tmp_path / "agents/normalize/part-00000.parquet"
        rows = pq.read_table(path).to_pylist()
        task = TaskSpec.model_validate_json(rows[1]["task_json"])
        if failure == "judge":
            task = source_task("judge").model_copy(
                update={
                    "grader": verifyit_package(JudgeSpec(references=("target",))).grader,
                    "tags": ("eval", "tool-call-proxy"),
                }
            )
        elif failure == "image":
            task = source_task("image").model_copy(update={"tags": ("eval", "tool-call-proxy")})
        elif failure == "eval-tag":
            task = task.model_copy(update={"tags": tuple(tag for tag in task.tags if tag != "eval")})
            settings = settings.model_copy(
                update={
                    "sources": (
                        settings.sources[0],
                        settings.sources[1].model_copy(update={"required_tags": ("tool-call-proxy",)}),
                    )
                }
            )
        elif failure == "training-whitespace":
            task = task.model_copy(
                update={
                    "context": ConversationInput(
                        events=(
                            TextMessage(role="system", content="Another request wrapper"),
                            TextMessage(role="user", content=" Native    tool\nrequest  0 "),
                        )
                    )
                }
            )
        else:
            name, kind = failure.split("-")
            exclusion = settings.training[0] if name == "training" else settings.development
            excluded = read_tasks(Path(exclusion.uri))[0]
            if kind == "context":
                excluded = excluded.model_copy(
                    update={
                        "context": ConversationInput(
                            events=(TextMessage(role="system", content="Complete the supplied tool action."),)
                        )
                    }
                )
                pq.write_table(pa.Table.from_pylist([{"task_spec": excluded.model_dump_json()}]), exclusion.uri)
            task = task.model_copy(update={"id": excluded.id} if kind == "id" else {"context": excluded.context})
        rows[1] = {"task_json": task.model_dump_json()}
        pq.write_table(pa.Table.from_pylist(rows), path)
    result, output = publish_panel(settings, tmp_path)
    assert result.exit_code != 0
    assert cause in error_chain(result.exception)
    assert not (output / FULL_VALIDATION_FILENAME).exists()
    assert not (output / TASK_MANIFEST_FILENAME).exists()
