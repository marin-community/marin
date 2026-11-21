# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import json

import pyarrow.parquet as pq
import pytest
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PackageSpec,
    PlainText,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)

from experiments.post_training.task_curation import forge_release
from experiments.post_training.task_curation.forge_release import Choice
from experiments.post_training.task_curation.lowered_envelope import LOWERED_TASK_SPEC_COLUMN

PASS = {"status": "graded", "reward": 1.0, "passed": None}
FAIL = {"status": "graded", "reward": 0.0, "passed": None}


SHELL_SIMULATOR = EnvironmentRequirements(
    capabilities=("shell",), command_semantics=CommandSemantics.SHELL_SIMULATOR, working_directory="/app"
)
GRADER_BASE = EnvironmentRequirements(
    capabilities=("shell",),
    command_semantics=CommandSemantics.LINUX_PROCESS,
    docker_image=forge_release.GRADER_BASE_IMAGE,
    working_directory="/app",
)
# How forge's local sandbox records a grader-base machine: no image, an empty package spec.
HOSTED = GRADER_BASE.model_copy(update={"docker_image": None, "package_spec": PackageSpec(pypi_packages=())})
ACTORLESS = EnvironmentRequirements()


def _task(identity: str, requirements: EnvironmentRequirements = SHELL_SIMULATOR, resources: dict | None = None) -> dict:
    task = TaskSpec(
        id=identity,
        context=ConversationInput(events=(TextMessage(role="user", content="Answer the question."),)),
        environment_requirements=requirements,
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="fixture"),
        source=Source(dataset="capability", revision="r", row=identity, importer_revision="1"),
    )
    if resources is not None:
        task = task.model_copy(update={"resources": ResourceGroups.model_validate(resources)})
    return task.model_dump(mode="json")


def _resource(path: str) -> dict:
    return {"path": path, "source": {"content_base64": base64.b64encode(b"payload").decode()}}


def _record(
    identity: str,
    passes: int,
    *,
    band: str = "in_band",
    notes: tuple = (),
    revert: bool = False,
    authored: EnvironmentRequirements = SHELL_SIMULATOR,
    hosted: EnvironmentRequirements = SHELL_SIMULATOR,
    resources: dict | None = None,
) -> dict:
    capability = identity.rsplit("/", 1)[0]
    trials = [{"trial": i, "grade": PASS if i < passes else FAIL, "cause": None} for i in range(8)]
    decisions = [{"kind": "accept", "band": band, "revert": revert}]
    return {
        "schema": forge_release.RECORD_SCHEMA,
        "id": identity,
        "capability": capability,
        "arm": "b4",
        "workpiece": {
            "terminal": "accepted",
            "notes": list(notes),
            "missing_evidence": [],
            "proposal": {"header": {"verification": "code", "environment": "shellsim"}},
            "submission": {
                "controls": (
                    [{}] * 4
                    + [
                        {
                            "kind": "positive",
                            "candidate": {"reply": "done", "files": []},
                            "expected": {"status": "graded", "reward_min": 1.0, "reward_max": 1.0},
                        }
                    ]
                ),
                "verification": None,
                "task": _task(identity, authored),
            },
            "evidence": [{"confirmation": True, "solver": trials}],
            "decisions": decisions,
        },
        "rollout_spec": {
            "task": _task(identity, hosted, resources),
            "task_machine": {"cpus": None, "memory_mb": 8192, "storage_mb": None, "gpus": 0},
            "verifier_machine": None,
        },
    }


def test_selection_covers_every_capability_before_a_second_task():
    records = [_record("d01.a/01", 4), _record("d01.a/02", 4), _record("d02.b/01", 0, band="too_hard")]
    selection = forge_release.select([forge_release.task_summary(r) for r in records], 2)
    chosen = sorted(item.summary.capability for item in selection if item.choice is Choice.SELECTED)
    # The too-hard task is its capability's only one, so it outranks the first capability's second in-band task.
    assert chosen == ["d01.a", "d02.b"]


def test_tasks_with_notes_are_never_selected():
    records = [_record("d01.a/01", 4, notes=({"part": "grader"},)), _record("d01.a/02", 8, band="too_easy")]
    selection = forge_release.select([forge_release.task_summary(r) for r in records], 1)
    by_id = {item.summary.id: item.choice for item in selection}
    assert by_id == {"d01.a/01": Choice.NOT_CLEAN, "d01.a/02": Choice.SELECTED}


def test_selection_prefers_in_band_tasks_within_a_capability():
    records = [_record("d01.a/01", 4), _record("d01.a/02", 8, band="too_easy")]
    selection = forge_release.select([forge_release.task_summary(r) for r in records], 1)
    assert {item.summary.id for item in selection if item.choice is Choice.SELECTED} == {"d01.a/01"}


def test_release_writes_selected_rows_and_selection_evidence(tmp_path):
    records = [_record("d01.a/01", 4), _record("d02.b/01", 3), _record("d02.b/02", 2)]
    manifest = forge_release.write_release(records, 2, tmp_path, "memory://records")
    rows = pq.read_table(tmp_path / forge_release.LOWERED_FILENAME).to_pylist()
    assert manifest["selected"] == len(rows) == 2
    assert {row["path"] for row in rows} == {"d01.a/01", "d02.b/01"}
    lines = (tmp_path / forge_release.SELECTION_FILENAME).read_text().splitlines()
    assert {json.loads(line)["choice"] for line in lines} == {"selected", "below_coverage_tier"}


def test_release_ids_and_provenance_come_from_the_forge_task_id(tmp_path):
    records = [_record("d01.a/02", 4), _record("d01.a/03", 3)]
    # Authors wrote their own spec ids; two tasks in one capability can claim the same one.
    for record in records:
        record["rollout_spec"]["task"]["id"] = "d01.a/3"
    forge_release.write_release(records, 2, tmp_path, "memory://records")
    rows = pq.read_table(tmp_path / forge_release.LOWERED_FILENAME).to_pylist()
    tasks = {row["path"]: json.loads(row[LOWERED_TASK_SPEC_COLUMN])["task"] for row in rows}
    assert {path: (task["id"], task["source"]["row"]) for path, task in tasks.items()} == {
        "d01.a/02": ("d01.a/02", "d01.a/02"),
        "d01.a/03": ("d01.a/03", "d01.a/03"),
    }
    assert {task["source"]["dataset"] for task in tasks.values()} == {forge_release.SOURCE_DATASET}
    # Unset newer fields are omitted, so rows load with TaskCompendium versions that predate them.
    assert all("integrity_contract" not in task and TaskSpec.model_validate(task) for task in tasks.values())


def test_a_locally_sandboxed_machine_is_restored_to_the_image_its_author_named():
    record = _record("d01.a/01", 4, authored=GRADER_BASE, hosted=HOSTED)
    lowered = forge_release.lowered_spec(record)
    requirements = lowered.task.environment_requirements
    assert (requirements.docker_image, requirements.package_spec) == (forge_release.GRADER_BASE_IMAGE, None)
    assert forge_release.task_summary(record).environment == "linux"


def test_a_restored_machine_that_disagrees_with_its_author_is_refused():
    record = _record("d01.a/01", 4, authored=SHELL_SIMULATOR, hosted=HOSTED)
    with pytest.raises(ValueError, match="restored environment_requirements"):
        forge_release.lowered_spec(record)


WORKSPACE_RESOURCES = {
    "worker": [
        {"path": "data.csv", "source": {"content_base64": "cGF5bG9hZA=="}},
        {"path": "app/data.csv", "source": {"content_base64": "cGF5bG9hZA=="}},
    ],
    "verifier": [{"path": "grade.py", "source": {"content_base64": "cHJpbnQoJ2hpJyk="}}],
}


def test_workspace_resources_root_at_the_declared_working_directory():
    record = _record("d01.a/01", 4, resources=WORKSPACE_RESOURCES)
    task = forge_release.lowered_spec(record).task
    workspace = {resource.path for resource in (*task.resources.worker, *task.resources.all)}
    # A root install of each rooted path lands the file under /app, where the author declared it.
    assert workspace == {"app/data.csv", "app/app/data.csv"}
    # Verifier resources stay authored: the grading machine installs those under /tests.
    assert [resource.path for resource in task.resources.verifier] == ["grade.py"]


def test_workspace_resource_rooting_preserves_each_file_declaration():
    record = _record("d01.a/01", 4)
    record["rollout_spec"]["task"]["resources"] = {
        "worker": [
            {"path": "run.sh", "source": {"content_base64": "c2g="}, "mode": "0755", "mtime_ns": 17},
        ]
    }
    task = forge_release.lowered_spec(record).task
    (resource,) = task.resources.worker
    assert (resource.path, resource.mode, resource.mtime_ns) == ("app/run.sh", "0755", 17)


def test_workspace_resources_without_a_working_directory_are_refused():
    record = _record("d01.a/01", 4)
    record["rollout_spec"]["task"]["resources"] = {"worker": [_resource("data.csv")]}
    record["rollout_spec"]["task"]["environment_requirements"]["working_directory"] = None
    with pytest.raises(ValueError, match="workspace resources require a declared working directory"):
        forge_release.lowered_spec(record)


def test_a_root_working_directory_leaves_workspace_paths_authored():
    record = _record("d01.a/01", 4)
    rooted = SHELL_SIMULATOR.model_copy(update={"working_directory": "/"}).model_dump(mode="json")
    record["workpiece"]["submission"]["task"]["environment_requirements"] = rooted
    record["rollout_spec"]["task"]["environment_requirements"] = rooted
    record["rollout_spec"]["task"]["resources"] = {"worker": [_resource("data.csv")]}
    task = forge_release.lowered_spec(record).task
    assert [resource.path for resource in task.resources.worker] == ["data.csv"]


def test_release_counts_the_workspace_repair_in_its_manifest(tmp_path):
    records = [
        _record("d01.a/01", 4, resources=WORKSPACE_RESOURCES),
        _record("d02.b/01", 3),
    ]
    manifest = forge_release.write_release(records, 2, tmp_path, "memory://records")
    assert manifest["resource_repair"] == {"tasks": 1, "files": 2}
    rows = pq.read_table(tmp_path / forge_release.LOWERED_FILENAME).to_pylist()
    tasks = {row["path"]: json.loads(row[LOWERED_TASK_SPEC_COLUMN])["task"] for row in rows}
    assert tasks["d01.a/01"]["resources"]["worker"][0]["path"] == "app/data.csv"
    # The golden control rides in the private oracle group, not the workspace groups.
    assert all(tasks["d02.b/01"]["resources"][group] == [] for group in ("all", "worker"))
    assert {row["path"] for row in tasks["d02.b/01"]["resources"]["oracle"]} == {
        "solution/solve.sh",
        "solution/answer.txt",
    }


def _positive_control(files: dict[str, str], reply: str) -> dict:
    return {
        "kind": "positive",
        "candidate": {
            "reply": reply,
            "files": [
                {
                    "path": path,
                    "source": {"kind": "inline_file", "content_base64": base64.b64encode(text.encode()).decode()},
                }
                for path, text in files.items()
            ],
        },
        "expected": {"status": "graded", "reward_min": 1.0, "reward_max": 1.0},
    }


def _with_control(record: dict, control: dict) -> dict:
    """Replace the fixture's default positive control with the given one."""
    submission = record["workpiece"]["submission"]
    submission["controls"] = [*submission["controls"][:-1], control]
    return record


def test_a_machine_task_publishes_its_golden_control_as_oracle_resources():
    record = _record("d01.a/01", 4)
    _with_control(record, _positive_control({"notes.txt": "n"}, "the answer is 7"))
    task = forge_release.lowered_spec(record).task
    oracle = {resource.path: resource for resource in task.resources.oracle}
    assert set(oracle) == {"solution/solve.sh", "solution/answer.txt"}
    script = base64.b64decode(oracle["solution/solve.sh"].source.content_base64).decode()
    # The golden file is rewritten at its authored working-directory location.
    assert "cat > '/app/notes.txt' <<'FORGE_GOLDEN_" in script and "\nn\n" in script
    assert oracle["solution/solve.sh"].mode == "0755"
    reply = base64.b64decode(oracle["solution/answer.txt"].source.content_base64).decode()
    assert reply == "the answer is 7"


def test_a_file_answer_publishes_only_the_solve_script():
    record = _record("d01.a/01", 4)
    record["rollout_spec"]["task"]["answer_type"] = "file"
    _with_control(record, _positive_control({"matrix.tsv": "a\tb"}, "wrote the matrix"))
    task = forge_release.lowered_spec(record).task
    assert [resource.path for resource in task.resources.oracle] == ["solution/solve.sh"]


def test_an_actorless_task_publishes_no_golden_resources():
    record = _record("d01.a/01", 4, authored=ACTORLESS, hosted=ACTORLESS)
    _with_control(record, _positive_control({}, "42"))
    task = forge_release.lowered_spec(record).task
    assert task.resources.oracle == ()


def test_an_actorless_task_with_preserved_solution_material_publishes_the_answer():
    record = _record("d01.a/01", 4, authored=ACTORLESS, hosted=ACTORLESS)
    record["rollout_spec"]["task"]["resources"] = {
        "oracle": [{"path": "solution/reference.json", "source": {"content_base64": base64.b64encode(b"{}").decode()}}]
    }
    _with_control(record, _positive_control({}, "the key"))
    task = forge_release.lowered_spec(record).task
    assert [resource.path for resource in task.resources.oracle] == ["solution/reference.json", "solution/answer.txt"]


def test_a_task_without_a_positive_control_is_refused():
    record = _record("d01.a/01", 4)
    record["workpiece"]["submission"]["controls"] = [{}] * 4
    with pytest.raises(ValueError, match="no positive control"):
        forge_release.lowered_spec(record)


def test_the_release_manifest_counts_the_golden_solution(tmp_path):
    records = [_record("d01.a/01", 4), _record("d02.b/01", 3)]
    manifest = forge_release.write_release(records, 2, tmp_path, "memory://records")
    # Both fixture tasks run on a machine with a text answer: solve script plus reply.
    assert manifest["golden_solution"] == {"tasks": 2, "files": 4}


def test_a_non_utf8_golden_file_round_trips_through_base64():
    record = _record("d01.a/01", 4)
    payload = b"caf\xe9 menu \x99 byte-preserving"
    control = {
        "kind": "positive",
        "candidate": {
            "reply": "done",
            "files": [
                {
                    "path": "menu.txt",
                    "source": {"kind": "inline_file", "content_base64": base64.b64encode(payload).decode()},
                }
            ],
        },
        "expected": {"status": "graded", "reward_min": 1.0, "reward_max": 1.0},
    }
    _with_control(record, control)
    task = forge_release.lowered_spec(record).task
    (script_resource,) = (r for r in task.resources.oracle if r.path == "solution/solve.sh")
    script = base64.b64decode(script_resource.source.content_base64).decode()
    line = next(line for line in script.splitlines() if "menu.txt" in line)
    assert "base64 -d > '/app/menu.txt'" in line
    embedded = line.split("printf '%s' '")[1].split("'")[0]
    assert base64.b64decode(embedded) == payload
