# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from finestore.rl.strata import DifficultyBands, MembershipState, SelectionParent, TaskStrata, read_selection_manifest
from fray.local_backend import LocalClient
from marin.execution.lazy import ArtifactStep, run
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.answers import source_evidence
from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import Outcome
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import (
    AnswerType,
    Boxed,
    CommandSemantics,
    ConversationInput,
    DockerBuildContext,
    EnvironmentRequirements,
    GraderMachine,
    NoGrader,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
    grader_environment,
)
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from taskcompendium.pipeline.models import IntendedUse
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import JudgeSpec
from zephyr.context import ZephyrContext

from experiments.evaluation.benchmarks.code import sources as code_sources
from experiments.evaluation.benchmarks.reasoning import sources as reasoning_sources
from experiments.post_training.rl_compose import sampled_tasks_step
from experiments.post_training.task_curation import selection as selection_module
from experiments.post_training.task_curation.campaign import error_chain
from experiments.post_training.task_curation.datasets.nemo_tool_calls import sources as tool_call_sources
from experiments.post_training.task_curation.datasets.nemotron_ultra.components import BLENDS
from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.driver import main
from experiments.post_training.task_curation.pipeline import source_recipe
from experiments.post_training.task_curation.selection import (
    AAII_CATEGORIES,
    FULL_VALIDATION_FILENAME,
    HELDOUT_BLEND,
    HELDOUT_FAMILIES,
    LOCAL_GRADERS,
    PASS_RATE_ATTEMPTS,
    STRUCTURAL_FAMILY,
    TASK_MANIFEST_FILENAME,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    EvalLiteRecipe,
    GraderKind,
    HeldoutSource,
    PanelFormat,
    ReleaseAdmission,
    ReleasedSource,
    SelectionRecipe,
    eval_lite_step,
    task_selection_step,
    validation_task,
)
from experiments.post_training.task_curation.swe_verified_release import convert_release
from experiments.post_training.task_curation.tests.conversion import converted_task

FAMILIES = (*HELDOUT_FAMILIES.values(), STRUCTURAL_FAMILY)


@pytest.fixture(autouse=True)
def local_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.setattr("rigging.filesystem.cluster_config.region_from_metadata", lambda: None)


def source_task(identifier: str, *, dataset: str = "fixture", image: bool = False) -> TaskSpec:
    return TaskSpec(
        id=identifier,
        source=Source(dataset=dataset, revision="release", row=identifier, importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content=f"Solve {identifier}."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=ScriptGrader(
            argv=("python", "/tests/grade.py"),
            env={"MODE": "fixture"},
            placement=GraderMachine(
                environment=EnvironmentRequirements(
                    command_semantics=CommandSemantics.LINUX_PROCESS,
                    **({"docker_image": "python@sha256:" + "a" * 64} if image else {"packages_lock": "numpy==2.3.5\n"}),
                )
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
            if family == STRUCTURAL_FAMILY:
                task["source"]["dataset"] = "structure-closure"
                task["grader"] = {"kind": "session"}
                config = inline_resource(
                    "config.json",
                    json.dumps({"extras": {"prefix": '{"a":[1', "expected": '{"a":[1,2]}', "format": "json"}}).encode(),
                )
                task["resources"]["verifier"].append(config.model_dump(mode="json"))
            else:
                # The pinned heldout specs keep the 0.30 grader shape.
                del task["grader"]["placement"]
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
                                "task_session": "structure_closure" if family == STRUCTURAL_FAMILY else "shellbox",
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


def admitted_source(root: Path, tasks: list[TaskSpec], intended_use: str = "train", mode: str = "full") -> ArtifactStep:
    (root / "final").mkdir(parents=True)
    (root / "normalize").mkdir()
    (root / "manifest.json").write_text(json.dumps({"intended_use": intended_use, "mode": mode}))
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


@pytest.mark.parametrize("mode", ["sample", "full"])
def test_selection_filters_full_heldout_and_duplicates_before_sampling(tmp_path, mode):
    settings = recipe(tmp_path)
    leaked = [
        validation_task(row, {}).model_copy(update={"id": f"changed-id-{index}", "answer_format": Boxed()})
        for index, row in enumerate(heldout_rows()[:21])
    ]
    leaked.append(
        leaked[0].model_copy(
            update={
                "id": "changed-wrapper",
                "context": chat(
                    ("system", "Another wrapper"), ("user", "  " + leaked[0].context.events[-1].content + "\n")
                ),
            }
        )
    )
    training = {
        identifier: source_task(identifier).model_copy(update={"tags": ("agentic",)})
        for identifier in ("training-a", "training-b", "training-c")
    }
    original = training["training-a"]
    duplicate = original.model_copy(update={"id": "alias"})
    first = admitted_source(tmp_path / "first", [original, training["training-b"], *leaked], mode=mode)
    second = admitted_source(tmp_path / "second", [duplicate, training["training-c"]], mode=mode)
    curated = {"fixture:first": first, "fixture:second": second}
    selection_step = task_selection_step(curated, settings)
    artifact = run(selection_step)[0]
    output = Path(artifact.path)
    tasks = read_tasks(output / TRAIN_FILENAME)
    assert len(tasks) == 2
    assert {task.id for task in tasks} <= {"training-a", "training-b", "training-c"}
    for task in tasks:
        assert task == training[task.id]
    panel = read_selection_manifest(str(output))
    assert panel is not None
    carried = pq.read_table(output / TRAIN_FILENAME).to_pylist()
    assert {row["family"] for row in carried} == {"calendar"}
    assert {row["aai_category"] for row in carried} == {AAII_CATEGORIES["calendar"]}
    assert all(row["agentic"] is True for row in carried)
    assert all(
        row["source"] == ("fixture:second" if json.loads(row["task_spec"])["id"] == "training-c" else "fixture:first")
        and row["topic"] == row["difficulty"] == "untagged"
        for row in carried
    )
    assert all(
        TaskStrata.model_validate({key: row[key] for key in TaskStrata.model_fields})
        == panel.identity(json.loads(row["task_spec"])["id"]).strata
        for row in carried
    )

    derived = Path(
        run(sampled_tasks_step(selection_step, "train", 1, 1, revision="2026.11.17.1", namespace="fixture"))[0].path
    )
    sampled = read_tasks(derived / TRAIN_FILENAME)
    assert len(sampled) == 1 and sampled[0] in tasks
    subset = read_selection_manifest(str(derived))
    assert subset is not None
    assert set(subset.tasks) == {sampled[0].id}
    assert subset.identity(sampled[0].id) == panel.identity(sampled[0].id)

    full = read_tasks(output / FULL_VALIDATION_FILENAME)
    for row in pq.read_table(output / FULL_VALIDATION_FILENAME).to_pylist():
        task = json.loads(row["task_spec"])
        assert row["agentic"] == ("agentic" in task["tags"]) == panel.identity(task["id"]).strata.agentic
    sample = read_tasks(output / VALIDATION_FILENAME)
    families = {f"heldout-{family}-{index}": family for family in FAMILIES for index in range(21)}
    assert [task.id for task in full] == list(families)
    assert Counter(families[task.id] for task in sample) == dict.fromkeys(FAMILIES, 20)
    structural = next(task for task in full if families[task.id] == STRUCTURAL_FAMILY)
    assert structural.context.events[-1] == TextMessage(role="assistant", content='{"a":[1')
    assert structural.grader.mode == "structure_closure"
    assert panel.identity(structural.id).strata.source == "SnowballV01:structure-closure/validation"
    assert (
        panel.identity("heldout-calendar-0").strata.source
        == "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_calendar_v2"
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["source_counts"] == {
        "fixture:first": {
            "input": 24,
            "heldout_overlap": 22,
            "excluded": 0,
            "duplicate": 0,
            "eligible": 2,
            "selected": sum(task.id != "training-c" for task in tasks),
        },
        "fixture:second": {
            "input": 2,
            "heldout_overlap": 0,
            "excluded": 0,
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
    environments = {
        family: EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock=f"numpy==2.3.{index}\n"
        )
        for index, family in enumerate(HELDOUT_FAMILIES.values())
    }
    settings = recipe(tmp_path).model_copy(
        update={
            "sources": {
                **{BLENDS[HELDOUT_BLEND][dataset].id: family for dataset, family in HELDOUT_FAMILIES.items()},
                "fixture:image": "agents",
            },
            "family_limits": {family: 1 for family in (*HELDOUT_FAMILIES.values(), "agents")},
            "admitted_graders": LOCAL_GRADERS | {GraderKind.IMAGE},
        }
    )
    curated = {}
    for component, family in settings.sources.items():
        task = source_task(f"training-{family}", image=family == "agents")
        if family != "agents":
            placement = GraderMachine(environment=environments[family])
            task = task.model_copy(update={"grader": task.grader.model_copy(update={"placement": placement})})
        curated[component] = admitted_source(tmp_path / family, [task])
    originals = {}
    for row in rows:
        document = json.loads(row["lowered_task_spec"])
        task = document["task"]
        if task["source"]["dataset"] != "structure-closure":
            image_environment = grader_environment(source_task("image", image=True).grader)
            assert image_environment is not None
            task["grader"]["environment"] = image_environment.model_dump(mode="json")
            task["grader"]["argv"][0] = "python3"
            task["grader"]["env"] = {"MODE": "historical"}
            task["grader"]["timeout"] = 330
        originals[task["id"]] = task
        row["lowered_task_spec"] = json.dumps(document)
    pq.write_table(pa.Table.from_pylist(rows), settings.heldout[0].uri)
    output = Path(run(task_selection_step(curated, settings))[0].path)
    full = read_tasks(output / FULL_VALIDATION_FILENAME)
    manifest = pq.read_table(output / TASK_MANIFEST_FILENAME).to_pylist()
    assert {row["task_id"] for row in manifest} == {task.id for task in (*full, *read_tasks(output / TRAIN_FILENAME))}
    by_id = {task.id: task for task in full}
    for row in manifest:
        if row["task_id"] not in by_id:
            continue
        task = by_id[row["task_id"]]
        original = originals[task.id]
        family = next(family for family in FAMILIES if task.id.startswith(f"heldout-{family}-"))
        assert {
            key: row[key]
            for key in ("task_id", "family", "aai_category", "agentic", "grader_kind", "measured_difficulty")
        } == {
            "task_id": task.id,
            "family": family,
            "aai_category": AAII_CATEGORIES[family],
            "agentic": "agentic" in original["tags"],
            "grader_kind": "in_process" if family == STRUCTURAL_FAMILY else "package",
            "measured_difficulty": None if family == STRUCTURAL_FAMILY else 0.375,
        }
        if family == STRUCTURAL_FAMILY:
            continue
        assert grader_environment(task.grader) == environments[family]
        assert task.resources.model_dump(mode="json") == original["resources"]
        assert task.grader.model_dump(mode="json", exclude={"placement"}) == {
            key: value for key, value in original["grader"].items() if key not in ("environment", "collect", "artifacts")
        }
    summary = json.loads((output / "summary.json").read_text())
    assert summary["grader_kind_counts"] == {
        "train": {"package": 4, "image": 1, "remote_judge": 0, "in_process": 0, "harness": 0},
        "validation": {"package": 80, "image": 0, "remote_judge": 0, "in_process": 20, "harness": 0},
        "full_validation": {"package": 84, "image": 0, "remote_judge": 0, "in_process": 21, "harness": 0},
    }


def released_training(path: Path, *tasks: TaskSpec) -> ReleasedSource:
    source = task_file(path, list(tasks))
    return ReleasedSource(
        **source.model_dump(),
        family="agentic",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        agentic=True,
        license="Apache-2.0",
    )


def released_probe(root, release, tasks, *, graded, pin=True):
    name, version = "probe", "2026.11.18.1"
    selection = root / selection_module.SELECTION_PREFIX / name / version
    selection.mkdir(parents=True)
    (selection / "summary.json").write_text(
        json.dumps({"recipe": {"released": [{"sha256": release.sha256}] if pin else []}})
    )
    rows = []
    for index, task in enumerate(tasks):
        tags = task.tags if "agentic" in task.tags else (*task.tags, "agentic")
        task = task.model_copy(update={"tags": tags})
        rows.append(
            {
                "selection": f"{name}@{version}",
                "task_id": task.id,
                "task_digest": canonical_sha256(task.model_dump(mode="json")),
                "outcome": Outcome.GRADED if index < graded else Outcome.INFRA_ERROR,
            }
        )
    attempts = root / "attempts.parquet"
    pq.write_table(pa.Table.from_pylist(rows), attempts)
    return ReleaseAdmission(uri=str(attempts), sha256=hashlib.sha256(attempts.read_bytes()).hexdigest())


def harness_task(tags=("harbor",)) -> TaskSpec:
    return source_task("agent-task", dataset="long-loop-smartie-swesmith").model_copy(
        update={"grader": NoGrader(reason="The harness grades the completed trial"), "tags": tags}
    )


def released_recipe(release: ReleasedSource, **options) -> SelectionRecipe:
    return SelectionRecipe(
        name="mixed",
        version="2026.11.19.1",
        sources={},
        family_limits={"agentic": 1},
        heldout=(),
        validation_per_family=1,
        seed=20261023,
        released=(release,),
    ).model_copy(update=options)


@pytest.mark.parametrize(
    ("setup_suffix", "existing_commands", "error"),
    [
        pytest.param("", (), None, id="published"),
        pytest.param(None, (), "no declared Harbor bootstrap command", id="missing-bootstrap"),
        pytest.param("", ("echo configured",), "already declares setup_commands", id="existing-setup"),
        pytest.param('user="root"\n', (), "unsupported Harbor setup keys: user", id="unknown-key"),
    ],
)
def test_sealed_swe_conversion_preserves_rows_and_records_hashes(tmp_path, setup_suffix, existing_commands, error):
    commands = ("python fixture_setup.py", "python other_setup.py")
    tasks = []
    for index, command in enumerate(commands):
        config = '[environment]\ndocker_image="fixture/image:latest"\n'
        if index == 0 or setup_suffix is not None:
            config += f'[environment.setup]\ncommand="{command}"\ntimeout_sec=1800\n'
            if index == 1:
                config += setup_suffix
        resources = ResourceGroups(verifier=(inline_resource("task.toml", config.encode()),))
        requirements = EnvironmentRequirements(setup_commands=existing_commands if index == 1 else ())
        tasks.append(
            source_task(f"agent-{index}").model_copy(
                update={"resources": resources, "environment_requirements": requirements}
            )
        )
    original = tmp_path / "sealed.parquet"
    serialized = [task.model_dump_json() for task in tasks]
    table = pa.table({"task_spec": serialized, "family": ["agentic", "agentic"]})
    pq.write_table(table, original)
    source_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    output = tmp_path / "derived"

    def convert():
        return convert_release(StoragePath(str(original)), source_sha, 2, StoragePath(str(output)), "test-commit")

    if error is not None:
        with pytest.raises(ValueError, match=f"agent-1 .*{error}"):
            convert()
        assert not output.exists()
        return

    manifest = convert()
    parquet = output / "agentic-train.parquet"
    result = pq.read_table(parquet)
    assert result.schema == table.schema and result["family"] == table["family"]
    for task, command, before, after, receipt in zip(
        tasks, commands, serialized, result["task_spec"].to_pylist(), manifest["tasks"], strict=True
    ):
        expected = json.loads(before)
        expected["environment_requirements"]["setup_commands"] = [command]
        assert json.loads(after) == expected
        assert receipt == {
            "task_id": task.id,
            "input_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "output_sha256": hashlib.sha256(after.encode()).hexdigest(),
        }
    assert manifest["input_uri"] == str(original) and manifest["input_sha256"] == source_sha
    assert manifest["code_commit"] == "test-commit" and manifest["rows"] == 2
    assert manifest["output_sha256"] == hashlib.sha256(parquet.read_bytes()).hexdigest()
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert convert() == manifest


def test_selection_mixes_released_and_curated_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(selection_module, "CANONICAL_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(selection_module, "MIN_PROBED_TASKS", 3)
    repeated = source_task("evaluation-overlap")
    one_turn_task = source_task("one-turn")
    alias = one_turn_task.model_copy(update={"id": "cross-source-alias"})
    repaired = harness_task()
    old_task = repaired.model_copy(
        update={"environment_requirements": EnvironmentRequirements(setup_commands=("false",))}
    )
    skipped = source_task("skip-exact-task")
    tasks = (repaired, repeated, alias, skipped)
    release = released_training(tmp_path / "release.parquet", *tasks)
    release = release.model_copy(update={"admission": released_probe(tmp_path, release, tasks, graded=len(tasks))})
    skip_list = task_file(
        tmp_path / "skip.parquet", [selection_module.with_agentic_tag(task) for task in (old_task, skipped)]
    )
    heldout = source_task("validation", dataset="structure-closure")
    one_turn = "fixture:first"
    settings = released_recipe(
        release,
        sources={one_turn: "nonagentic"},
        family_limits={"nonagentic": 1, "agentic": 1},
        source_limits={one_turn: 1, release.uri: 1},
        heldout=(task_file(tmp_path / "heldout.parquet", [heldout]),),
        exclusions=(
            task_file(tmp_path / "evallite.parquet", [repeated]),
            skip_list.model_copy(update={"format": PanelFormat.TASK_DIGEST}),
        ),
        admitted_graders=LOCAL_GRADERS | {GraderKind.HARNESS},
    )
    curated = {one_turn: admitted_source(tmp_path / "curated", [one_turn_task])}
    step = task_selection_step(curated, settings)
    changed = settings.model_copy(update={"released": (release.model_copy(update={"sha256": "0" * 64}),)})
    assert step.fingerprint() != task_selection_step(curated, changed).fingerprint()

    replacement = task_file(tmp_path / "replacement.parquet", [repeated, alias, skipped, source_task("replaced")])
    open_path = StoragePath.open
    release_read = False

    def open_release(path, mode="rb", **kwargs):
        nonlocal release_read
        if str(path) == release.uri and mode == "rb":
            if release_read:
                path = StoragePath(replacement.uri)
            release_read = True
        return open_path(path, mode, **kwargs)

    monkeypatch.setattr(StoragePath, "open", open_release)
    root = Path(run(step)[0].path)
    summary = json.loads((root / "summary.json").read_text())
    assert summary["harness_graded_datasets"] == ["long-loop-smartie-swesmith"]
    assert summary["grader_kind_counts"]["train"]["harness"] == 1
    assert summary["source_counts"][release.uri] == {
        "input": 4,
        "selected": 1,
        "eligible": 1,
        "heldout_overlap": 0,
        "excluded": 2,
        "duplicate": 1,
    }
    rows = pq.read_table(root / TRAIN_FILENAME).to_pylist()
    labels = {}
    for row in rows:
        task = TaskSpec.model_validate_json(row["task_spec"])
        labels[task.id] = (row["family"], row["aai_category"], row["agentic"], row["difficulty"], task.tags)
    assert labels == {
        "agent-task": ("agentic", "coding", True, "unrated", ("harbor", "agentic")),
        "one-turn": ("nonagentic", "untagged", False, "untagged", ()),
    }


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("digest", "SHA-256 mismatch"),
        ("count", "row count changed"),
        ("schema", "Unsupported TaskSpec schema"),
        ("tag", "contradicts the source"),
        ("harness", "does not admit harness graders"),
        ("harness-without-tag", "harness grader requires the harbor tag"),
    ],
)
def test_selection_refuses_invalid_released_tasks(tmp_path, monkeypatch, fault, message):
    monkeypatch.setattr(selection_module, "CANONICAL_ARTIFACT_ROOT", str(tmp_path))
    task = source_task("agent-task")
    if fault == "schema":
        task = task.model_copy(update={"schema_version": "0.27"})
    if fault == "tag":
        task = task.model_copy(update={"tags": ("agentic",)})
    if fault in ("harness", "harness-without-tag"):
        task = harness_task(("harbor",) if fault == "harness" else ())
    release = released_training(tmp_path / "release.parquet", task)
    update = {"digest": {"sha256": "0" * 64}, "count": {"rows": 2}, "tag": {"agentic": False}}
    release = release.model_copy(update=update.get(fault, {}))
    release = release.model_copy(update={"admission": released_probe(tmp_path, release, (task,), graded=1)})
    settings = released_recipe(
        release,
        admitted_graders=LOCAL_GRADERS | {GraderKind.HARNESS} if fault == "harness-without-tag" else LOCAL_GRADERS,
    )
    step = task_selection_step({}, settings)
    with pytest.raises(RuntimeError) as error:
        run(step)
    assert message in error_chain(error.value)
    assert not (Path(step.path()) / TRAIN_FILENAME).exists()


@pytest.mark.parametrize(
    ("fault", "error"),
    [
        ("missing", "requires admission evidence"),
        ("hash", "Admission record SHA-256 mismatch"),
        ("pin", "does not pin released source"),
        ("size", "Found 49 of 49 graded"),
        ("coverage", "Found 44 of 50 graded"),
        ("repeated-task", "Found 1 of 1 graded"),
        ("different-content", "Found 0 of 0 graded"),
    ],
)
def test_selection_requires_matching_released_task_admission(tmp_path, monkeypatch, fault, error):
    monkeypatch.setattr(selection_module, "CANONICAL_ARTIFACT_ROOT", str(tmp_path))
    tasks = tuple(source_task(f"probed-{index}") for index in range(50))
    release = released_training(tmp_path / "release.parquet", *tasks)
    if fault != "missing":
        admitted = tasks[:49] if fault == "size" else tasks
        if fault == "repeated-task":
            admitted = (tasks[0],) * 50
        if fault == "different-content":
            admitted = tuple(task.model_copy(update={"tags": ("different",)}) for task in tasks)
        graded = 44 if fault == "coverage" else len(admitted)
        probe = released_probe(tmp_path, release, admitted, graded=graded, pin=fault != "pin")
        if fault == "hash":
            probe = probe.model_copy(update={"sha256": "0" * 64})
        release = release.model_copy(update={"admission": probe})
    step = task_selection_step({}, released_recipe(release))
    with pytest.raises(RuntimeError) as raised:
        run(step)
    assert error in error_chain(raised.value)
    assert not (Path(step.path()) / TRAIN_FILENAME).exists()


def test_grader_admission_preserves_published_selection_fingerprints():
    recipes = Path(__file__).parents[1] / "recipes"
    training = SelectionRecipe.model_validate_json((recipes / "snowball-100k.json").read_text())
    agentic = SelectionRecipe.model_validate_json((recipes / "snowball-100k-agentic.json").read_text())
    evaluation = EvalLiteRecipe.model_validate_json((recipes / "eval-lite.json").read_text())
    for recipe in recipes.glob("*.json"):
        for source in json.loads(recipe.read_text()).get("released", ()):
            assert ReleasedSource.model_validate(source).admission is not None, recipe.name

    def curated(sources):
        return {
            name: ArtifactStep.adopt(f"inputs/source-{index}", "2026.11.19.1", f"/fixtures/corpus/source-{index}")
            for index, name in enumerate(sorted(set(sources)))
        }

    training_sources = curated(training.sources)
    assert "released" not in training.model_dump(mode="json")
    assert task_selection_step(training_sources, training).fingerprint() == "7fb51836"
    explicit = training.model_copy(update={"admitted_graders": LOCAL_GRADERS | {GraderKind.IMAGE}})
    assert task_selection_step(training_sources, explicit).fingerprint() != "7fb51836"
    evaluation_sources = curated(part.source_id for part in evaluation.sources)
    assert eval_lite_step(evaluation_sources, evaluation).fingerprint() == "4fd3f893"
    task_selection_step(curated(agentic.sources), agentic)


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


@pytest.mark.parametrize(
    "failure,cause",
    [
        ("evaluation-source", "not a training population"),
        ("development-image", "Development panel requires local graders"),
        ("training-image", "Training selection does not admit image graders"),
        ("training-judge", "Training selection does not admit remote_judge graders"),
    ],
)
def test_selection_does_not_publish_for_invalid_source_or_grader(tmp_path, failure, cause):
    settings = recipe(tmp_path).model_copy(update={"sources": {"fixture:first": "calendar"}})
    training = [source_task("training-a"), source_task("training-b")]
    if failure.startswith("development-"):
        heldout = [validation_task(row, {}) for row in heldout_rows()]
        heldout[0] = heldout[0].model_copy(update={"grader": source_task("image", image=True).grader})
        settings = settings.model_copy(update={"heldout": (task_file(tmp_path / "development.parquet", heldout),)})
    elif failure.startswith("training-"):
        grader = (
            source_task("image", image=True).grader
            if failure == "training-image"
            else verifyit_package(JudgeSpec(references=("target",))).grader
        )
        training[0] = training[0].model_copy(update={"grader": grader})
    source = admitted_source(tmp_path / "source", training, "eval" if failure == "evaluation-source" else "train")
    step = task_selection_step({"fixture:first": source}, settings)
    with pytest.raises(RuntimeError) as error:
        run(step)
    assert cause in error_chain(error.value)
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
    excluded = source_task("independent-exclusion").model_copy(
        update={"context": chat(("system", "Historical wrapper"), ("user", "  Two plus\n two? "))}
    )
    settings = recipe(tmp_path).model_copy(
        update={
            "sources": {source.info.id: "math"},
            "family_limits": {"math": 1},
            "exclusions": (task_file(tmp_path / "exclude.parquet", [excluded]),),
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
    assert summary["source_counts"][source.info.id]["excluded"] == 1
    heldout_ids = {json.loads(row["lowered_task_spec"])["task"]["id"] for row in heldout_rows()}
    assert {task.id for task in read_tasks(output / FULL_VALIDATION_FILENAME)} == heldout_ids
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


NEMO_FIXTURE = Path(__file__).parents[4] / "lib/taskcompendium/tests/fixtures/nemo/predicted-action.json"
TOOL_CALLS = "agents/normalize/part-00000.parquet"


class PanelRefused(Exception):
    pass


def chat(*turns: tuple[str, str]) -> ConversationInput:
    return ConversationInput(events=tuple(TextMessage(role=role, content=content) for role, content in turns))


def task_file(path: Path, tasks: list[TaskSpec]) -> HeldoutSource:
    pq.write_table(pa.Table.from_pylist([{"task_spec": task.model_dump_json()} for task in tasks]), path)
    return HeldoutSource(uri=str(path), version="2026.11.04", rows=len(tasks), format="task_spec")


def nemo_row(request: str) -> dict:
    data = json.loads(NEMO_FIXTURE.read_text())
    data["responses_create_params"]["input"][-1]["content"] = request
    return data


def write_corpus(tmp_path: Path) -> None:
    roots = {"Evalchemy:questions": tmp_path / "general", "NeMo:predicted-action": tmp_path / "agents"}
    sources = [
        {
            "source_id": source,
            "root": str(root),
            "source_key": canonical_sha256(json.loads((root / "manifest.json").read_text())),
        }
        for source, root in roots.items()
    ]
    (tmp_path / "corpus.json").write_text(json.dumps({"population": "converted", "inventory": [], "sources": sources}))


def tool_calls(tmp_path: Path) -> list[TaskSpec]:
    rows = pq.read_table(tmp_path / TOOL_CALLS).to_pylist()
    return [TaskSpec.model_validate_json(row["task_json"]) for row in rows if row["task_json"] is not None]


def edit_tool_call(tmp_path: Path, edit: Callable[[TaskSpec], TaskSpec]) -> None:
    """Replace agent-1, which every fixture panel selects."""
    rows = pq.read_table(tmp_path / TOOL_CALLS).to_pylist()
    rows[1] = {"task_json": edit(TaskSpec.model_validate_json(rows[1]["task_json"])).model_dump_json()}
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / TOOL_CALLS)


def eval_lite_recipe(tmp_path: Path, benchmark: str = "mmlu") -> EvalLiteRecipe:
    """Six Evalchemy questions and NeMo agent rows, with two training files and a development panel."""
    mmlu = benchmark == "mmlu"
    name = "evalchemy-mmlu-pro" if mmlu else "evalchemy-mbppplus"
    dataset = next(
        source.config
        for source in (*reasoning_sources(), *code_sources())
        if source.name == name and source.config is not None
    )
    question = {
        "options": ["1", "2"],
        "answer": "B",
        "category": "math",
        "cot_content": "A: Let's think step by step. The answer is (B).",
    }
    questions = [
        (
            {**question, "question": f"Panel question {i}"}
            if mmlu
            else {
                "prompt": f"Write function number_{i} that returns {i}.",
                "test_list": [f"assert number_{i}() == {i}"],
                "test": f"assert number_{i}() == {i}\nassert isinstance(number_{i}(), int)",
                "code": f"def number_{i}():\n    # EXAMPLE_SENTINEL_{i}\n    return {i}",
            }
        )
        for i in range(6)
    ]
    files = {tmp_path / "questions" / dataset.recipe.files[0]: questions}
    if mmlu:
        examples = [{**question, "question": f"EXAMPLE_SENTINEL_{i}"} for i in range(6)]
        files[tmp_path / "examples" / dataset.recipe.inputs["fewshot"].files[0]] = examples
    for path, records in files.items():
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(records), path)
    corpus, client = tmp_path / "general", LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            publish_evaluation_corpus(
                source_recipe(
                    dataset.recipe,
                    {"fewshot": str(tmp_path / "examples")} if mmlu else {},
                    EnvironmentRequirements(
                        command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock="tomli-w==1.2.0\n"
                    ),
                ),
                dataset.protocol.model_copy(update={"expected_population": 6}),
                dataset.item_builder,
                context,
                str(tmp_path / "questions"),
                str(corpus),
            )
    finally:
        client.shutdown()
    agents = tmp_path / "agents"
    (agents / "normalize").mkdir(parents=True)
    (agents / "manifest.json").write_text(json.dumps({"intended_use": "eval"}))
    tool_call = tool_call_sources()[0].config
    tasks = [
        converted_task(tool_call, nemo_row(f"Native tool request {i}")).model_copy(update={"id": f"agent-{i}"})
        for i in range(4)
    ]
    tasks.append(tasks[2].model_copy(update={"id": "agent-2-copy"}))
    cases = list(evaluation_cases(str(corpus)))
    if mmlu:
        task, alias = cases[1].task, {"id": "cross-part-alias", "context": cases[1].item.operation.variants[0].context}
        tasks.append(task.model_copy(update={**alias, "tags": (*task.tags, "tool-call-proxy")}))
    rows = [*({"task_json": task.model_dump_json()} for task in tasks), {"task_json": None}]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / TOOL_CALLS)
    write_corpus(tmp_path)
    excluded = {
        "training": {"id": cases[0].task.id, "context": chat(("system", "Complete the training action."))},
        "training-second": {
            "context": chat(("system", "Historical request wrapper"), ("user", "  Native\n tool  request\t0  "))
        },
        "development": {"context": chat(("system", "Complete the development action."))},
    }
    exclusions = {
        key: task_file(tmp_path / f"{key}.parquet", [source_task(key).model_copy(update=update)])
        for key, update in excluded.items()
    }
    family = "general" if mmlu else "competitive_coding"
    return EvalLiteRecipe(
        name="eval-lite",
        version="2026.11.04",
        seed=20261104,
        sources=(
            {
                "source_id": "Evalchemy:questions",
                "rows": 6,
                "format": "evaluation",
                "family": family,
                "aaii_category": "general" if mmlu else "coding",
                "license": "fixture-general",
                "count": 5,
            },
            {
                "source_id": "NeMo:predicted-action",
                "rows": len(tasks),
                "format": "normalized",
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


def publish_eval_lite(settings: EvalLiteRecipe, tmp_path: Path, cache: str = "artifacts") -> Path:
    """Publish through the driver; a refusal raises its cause chain after checking nothing was published."""
    recipe_file = tmp_path / f"panel-{settings.version}.json"
    recipe_file.write_text(settings.model_dump_json())
    output = tmp_path / cache / "rl-data/task-selection" / settings.name / settings.version
    arguments = ["--corpus", str(tmp_path / "corpus.json"), "--eval-lite", str(recipe_file), "--run"]
    result = CliRunner().invoke(main, [*arguments, "--cache-dir", str(tmp_path / cache)])
    if result.exit_code != 0:
        assert not (output / FULL_VALIDATION_FILENAME).exists()
        raise PanelRefused(error_chain(result.exception))
    return output


def measured_eval_lite_recipe(
    settings: EvalLiteRecipe, tmp_path: Path, tasks: list[TaskSpec], **last: Any
) -> EvalLiteRecipe:
    """Eight-answer rates for ``tasks``, with ``last`` overriding the final row, written in reverse order."""
    rows = [
        {
            "task_id": task.id,
            "task_digest": canonical_sha256(task.model_dump(mode="json")),
            "model_revision": "fixture-model",
            "attempted": PASS_RATE_ATTEMPTS,
            "samples": PASS_RATE_ATTEMPTS,
            "passed": index + 1,
            "mean_score": (index + 1) / PASS_RATE_ATTEMPTS,
        }
        for index, task in enumerate(tasks)
    ]
    rows[-1].update(last)
    path = tmp_path / "pass-rates.parquet"
    pq.write_table(pa.Table.from_pylist(rows[::-1]), path)
    return EvalLiteRecipe.model_validate(
        {
            **settings.model_dump(),
            "version": "2026.11.04.1",
            "pass_rates": {
                "uri": str(path),
                "version": "2026.11.04",
                "rows": len(rows),
                "model_revision": "fixture-model",
            },
        }
    )


@pytest.mark.parametrize("benchmark", ["mmlu", "mbpp"])
def test_eval_lite_publication_preserves_untouched_strata_and_joins_initializer_evidence(tmp_path, benchmark):
    settings = eval_lite_recipe(tmp_path, benchmark)
    questions = settings.sources[0]
    originals = {case.task.id: case.task for case in evaluation_cases(str(tmp_path / "general"))}
    provisional = publish_eval_lite(settings, tmp_path)
    full, fast = read_tasks(provisional / FULL_VALIDATION_FILENAME), read_tasks(provisional / VALIDATION_FILENAME)
    agents = {task.id: task for task in tool_calls(tmp_path)}
    for task in full:
        if task.id in agents:
            assert task == agents[task.id]
            continue
        original = originals[task.id]
        assert task.model_dump(exclude={"context"}) == original.model_dump(exclude={"context"})
        assert len(task.context.events) == 1 and task.context.events[0].role == "user"
        prompt = task.context.events[0].content
        assert "EXAMPLE_SENTINEL" not in prompt
        if benchmark == "mmlu":
            target = original.context.events[0].content.rsplit("Question:\n", 1)[1]
            assert "Question:\n" + target in prompt
        else:
            evidence = source_evidence(original)
            assert evidence["prompt"] in prompt
            assert all(public_test in prompt for public_test in evidence["test_list"])
            assert "isinstance" not in prompt
    panel = read_selection_manifest(str(provisional))
    assert panel is not None
    annotations = panel.tasks
    assert list(annotations) == [task.id for task in full]
    for filename in (FULL_VALIDATION_FILENAME, VALIDATION_FILENAME):
        for row in pq.read_table(provisional / filename).to_pylist():
            assert row["agentic"] == (json.loads(row["task_spec"])["id"] in agents)
    for task in full:
        row = annotations[task.id]
        agent = task.id in agents
        assert (row.family, row.aai_category, row.grader_kind, row.source, row.raw_source, row.license) == (
            "agents" if agent else questions.family,
            "agents" if agent else questions.aaii_category,
            "in_process" if agent else "package",
            "NeMo:predicted-action" if agent else questions.source_id,
            task.source.model_dump_json(),
            "fixture-agents" if agent else "fixture-general",
        )
        assert row.agentic == agent and row.topic == row.difficulty == "untagged"
    families = {task_id: row.family for task_id, row in annotations.items()}
    assert Counter(families[task.id] for task in fast) == {questions.family: 3, "agents": 1}
    counts = json.loads((provisional / "summary.json").read_text())["source_counts"]
    assert counts["Evalchemy:questions"]["overlap"] == counts["NeMo:predicted-action"]["overlap"] == 1
    assert counts["NeMo:predicted-action"].get("duplicate", 0) == 1 + (benchmark == "mmlu")
    assert {next(iter(originals)), "agent-0", "agent-2-copy"}.isdisjoint(families)
    saturated = fast[-1]
    measured_tasks = [task for task in full if task.id != saturated.id] + [saturated]
    measured = measured_eval_lite_recipe(settings, tmp_path, measured_tasks)
    final = publish_eval_lite(measured, tmp_path)
    assert read_tasks(final / FULL_VALIDATION_FILENAME) == full
    final_panel = read_selection_manifest(str(final))
    assert final_panel is not None
    for index, task in enumerate(measured_tasks):
        row = final_panel.tasks[task.id]
        assert row.measured_difficulty == 1 - (index + 1) / PASS_RATE_ATTEMPTS
        assert [
            (ref.task_digest, ref.model_revision, ref.attempted, ref.scored, ref.passed, ref.mean_score)
            for ref in row.references
        ] == [(canonical_sha256(task.model_dump(mode="json")), "fixture-model", 8, 8, index + 1, (index + 1) / 8)]
    assert final_panel.tasks[saturated.id].state == MembershipState.RETIRED
    assert saturated.id not in {task.id for task in read_tasks(final / VALIDATION_FILENAME)}
    parent_table = pq.read_table(final / FULL_VALIDATION_FILENAME)
    parent_rows = parent_table.to_pylist()
    parent_rows[0]["task_spec"] = json.dumps(json.loads(parent_rows[0]["task_spec"]), indent=2, sort_keys=True)
    pq.write_table(pa.Table.from_pylist(parent_rows, schema=parent_table.schema), final / FULL_VALIDATION_FILENAME)
    template = next(task for task in full if task.id in originals)
    additions, source_parts, inputs = [], [], {}
    for group, mean in (("a", 0.0), ("b", 1.0), ("c", 0.5)):
        tasks = [
            template.model_copy(
                update={
                    "id": f"addition-{group}-{i}",
                    "context": chat(("user", f"An additional untouched {group} question {i}.")),
                    "source": template.source.model_copy(update={"dataset": f"fixture:{group}", "row": str(i)}),
                }
            )
            for i in range(2)
        ]
        additions.extend((task, mean) for task in tasks)
        if group == "a":
            tasks.append(template.model_copy(update={"id": "parent-alias"}))
        source_id = f"fixture:{group}"
        source = task_file(tmp_path / f"addition-{group}.parquet", tasks)
        inputs[source_id] = ArtifactStep.adopt(f"fixture/addition-{group}", source.version, source.uri)
        source_parts.append(
            questions.model_copy(
                update={
                    "source_id": source_id,
                    "format": PanelFormat.TASK_SPEC,
                    "rows": len(tasks),
                    "count": 2,
                    "topic": f"topic-{group}",
                }
            )
        )
    reference_path = tmp_path / "second-reference.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": task.id,
                    "task_digest": canonical_sha256(task.model_dump(mode="json")),
                    "model_revision": "second-model",
                    "attempted": 8,
                    "samples": 8,
                    "passed": int(mean * 8),
                    "mean_score": mean,
                }
                for task, mean in additions
            ]
        ),
        reference_path,
    )
    successor = measured.model_copy(
        update={
            "version": "2026.11.04.2",
            "seed": 3,
            "parent": SelectionParent(root=str(final), name=measured.name, version=measured.version),
            "pass_rates": measured.pass_rates.model_copy(
                update={"uri": str(reference_path), "rows": 6, "model_revision": "second-model"}
            ),
            "difficulty_bands": DifficultyBands(hard_below=0.1, easy_above=1.0),
            "sources": tuple(source_parts),
        }
    )
    successor_output = Path(run(eval_lite_step(inputs, successor))[0].path)
    successor_panel = read_selection_manifest(str(successor_output))
    assert successor_panel is not None and successor_panel.definition.parent == successor.parent

    def retained_tasks(path):
        return {
            json.loads(value)["id"]: value
            for value in pq.read_table(path / FULL_VALIDATION_FILENAME)["task_spec"].to_pylist()
        }

    parent_tasks, successor_tasks = retained_tasks(final), retained_tasks(successor_output)
    assert successor_tasks.keys() == parent_tasks.keys() | {task.id for task, _ in additions}
    assert all(successor_tasks[task_id] == value for task_id, value in parent_tasks.items())
    assert successor_panel.tasks[saturated.id].state == MembershipState.RETIRED
    assert all(successor_panel.tasks[task_id].references == row.references for task_id, row in final_panel.tasks.items())
    assert all(successor_panel.tasks[task.id].state != MembershipState.RETIRED for task, _ in additions)
    active = [row for row in successor_panel.tasks.values() if row.state == MembershipState.ACTIVE]
    assert Counter(row.family for row in active) == settings.fast_counts
    cells = Counter((row.source, row.topic, row.difficulty) for row in active if row.family == questions.family)
    parent_source = questions.source_id
    assert cells == {
        (parent_source, "untagged", "learnable"): 1,
        ("fixture:a", "topic-a", "hard"): 1,
        ("fixture:c", "topic-c", "learnable"): 1,
    }
    successor_summary = json.loads((successor_output / "summary.json").read_text())
    assert successor_summary["source_counts"]["fixture:a"]["duplicate"] == 1

    def other_user(task: TaskSpec) -> TaskSpec:
        calls = json.loads(json.dumps(task.grader.parameters["expected_calls"]))
        calls[0]["arguments"]["user_id"] = "DIFFERENT_USER"
        parameters = {**task.grader.parameters, "expected_calls": calls}
        return task.model_copy(update={"grader": task.grader.model_copy(update={"parameters": parameters})})

    edit_tool_call(tmp_path, other_user)
    with pytest.raises(PanelRefused, match="Pass-rate evidence measures different task content"):
        publish_eval_lite(measured.model_copy(update={"version": "2026.11.04.3"}), tmp_path)


LOCAL_GRADER, AGENT_SHORTFALL = "EvalLite requires a local grader", "EvalLite part shortfall: NeMo:predicted-action"
PROXY = ("eval", "tool-call-proxy")
TASK_EDITS: dict[str, Callable[[TaskSpec], TaskSpec]] = {
    "judge": (
        lambda task: source_task("judge").model_copy(
            update={"grader": verifyit_package(JudgeSpec(references=("target",))).grader, "tags": PROXY}
        )
    ),
    "docker-build": lambda task: task.model_copy(
        update={
            "tags": PROXY,
            "grader": (
                source_task("docker-build").grader.model_copy(
                    update={
                        "placement": GraderMachine(
                            environment=EnvironmentRequirements(
                                command_semantics=CommandSemantics.LINUX_PROCESS,
                                docker_build=DockerBuildContext(
                                    files=(inline_resource("Dockerfile", b"FROM python:3.12"),)
                                ),
                            )
                        )
                    }
                )
            ),
        }
    ),
    "eval-tag": lambda task: task.model_copy(update={"tags": tuple(tag for tag in task.tags if tag != "eval")}),
}


@pytest.mark.parametrize(
    "failure,cause",
    [
        ("judge", LOCAL_GRADER),
        ("docker-build", LOCAL_GRADER),
        ("development-id", AGENT_SHORTFALL),
        ("train-manifest", "Panel source is not reserved for evaluation"),
        ("eval-tag", "EvalLite source must reserve its tasks for evaluation"),
        ("missing-rate", "Pass-rate evidence must cover every selected panel task"),
    ],
)
def test_eval_lite_publication_refuses_bad_sources_overlaps_shortfalls_and_rate_evidence(tmp_path, failure, cause):
    settings = eval_lite_recipe(tmp_path)
    if failure.endswith("rate"):
        tasks = read_tasks(publish_eval_lite(settings, tmp_path) / FULL_VALIDATION_FILENAME)
        settings = measured_eval_lite_recipe(settings, tmp_path, tasks, task_id="unknown")
    elif failure == "train-manifest":
        (tmp_path / "agents/manifest.json").write_text(json.dumps({"intended_use": "train"}))
        write_corpus(tmp_path)
    elif failure in TASK_EDITS:
        edit_tool_call(tmp_path, TASK_EDITS[failure])
    elif failure == "development-id":
        excluded = read_tasks(Path(settings.development.uri))[0]
        edit_tool_call(tmp_path, lambda task: task.model_copy(update={"id": excluded.id}))
    with pytest.raises(PanelRefused, match=cause):
        publish_eval_lite(settings, tmp_path)
