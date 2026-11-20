# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.rl.skyrl_records import RESOLVED_LAUNCH_FILENAME
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.grading_result import GradingFailure, Outcome
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.post_training.task_curation.selection import (
    SELECTION_PREFIX,
    TASK_SPEC_COLUMN,
    TRAIN_FILENAME,
    HeldoutSource,
    PanelFormat,
    _panel_source_tasks,
)
from experiments.post_training.task_curation.task_health import (
    ATTEMPT_SCHEMA,
    SKIP_FILENAME,
    TAIL_BYTES,
    Phase,
    extract_attempts,
    publish_skip_list,
)

STARTED = "2026-11-20T10:00:00Z"
SETUP = "pip install -e ."
SHARED_IMAGE = "swe/shared@sha256:" + "b" * 64
MISSING_MODULE = "ModuleNotFoundError: No module named 'pytest'"
SETUP_EXIT = "Declared task setup command 0 exited 1"
ALL_PHASES = ("environment_setup", "agent_setup", "agent_execution", "verifier")


def harbor_task(identifier: str, image: str | None = None, setup: str = SETUP) -> TaskSpec:
    """A Harbor trial task whose package, including task.toml, is archived in its verifier resources."""
    config = f'[environment]\ndocker_image = "{image or "swe/" + identifier}"\n'
    return TaskSpec(
        id=identifier,
        source=Source(dataset="fixture-swe", revision="release", row=identifier, importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content=f"Fix {identifier}."),)),
        environment_requirements=EnvironmentRequirements(setup_commands=(setup,)),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=NoGrader(reason="Harbor grades the completed trial"),
        resources=ResourceGroups(verifier=(inline_resource("task.toml", config.encode()),)),
        tags=("harbor", "agentic"),
    )


def selection_digest(task: TaskSpec) -> str:
    """The digest a selection's task manifest records for a training row."""
    return canonical_sha256(task.model_dump(mode="json"))


def write_tasks(path: Path, tasks: list[TaskSpec]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([{TASK_SPEC_COLUMN: task.model_dump_json()} for task in tasks]), path)
    return path


def trial_result(*started: str, rewards: dict | None = None, exception: tuple[str, str] | None = None) -> dict:
    """Harbor's result.json fields that extraction reads."""
    timing = {"started_at": STARTED, "finished_at": STARTED}
    error = None
    if exception is not None:
        error = {
            "exception_type": exception[0],
            "exception_message": exception[1],
            "exception_traceback": "",
            "occurred_at": STARTED,
        }
    return {
        "task_checksum": "c" * 64,
        "verifier_result": None if rewards is None else {"rewards": rewards},
        "exception_info": error,
        **{phase: timing if phase in started else None for phase in ALL_PHASES},
    }


def setup_receipt(completed: bool) -> dict:
    failure = {} if completed else {"exception_type": "RuntimeError"}
    command = hashlib.sha256(SETUP.encode()).hexdigest()
    return {"started_at": STARTED, "command_sha256": [command], "completed": completed, **failure}


def write_attempt(
    store: Path, attempt_id: str, task: TaskSpec, result: dict, receipt: dict, outputs: dict[str, str]
) -> None:
    """Lay out one attempt as the installed-agent runtime publishes it."""
    files = {
        "identity.json": json.dumps({"session_id": "session", "attempt_id": attempt_id, "task_id": task.id}),
        "task/task.toml": resource_bytes(task.resources.verifier[0]).decode(),
        f"trials/{attempt_id}/result.json": json.dumps(result),
        "engine/task-setup/receipt.json": json.dumps(receipt),
        **outputs,
    }
    for name, text in files.items():
        path = store / attempt_id / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def test_extract_records_each_attempts_grade_phase_and_output_tails(tmp_path):
    task = harbor_task("swe:1")
    selection = write_tasks(tmp_path / "selection" / TRAIN_FILENAME, [task]).parent
    store = tmp_path / "attempts/trajectories/installed-agents"
    run_output = tmp_path / "run"
    run_output.mkdir()
    source = {
        "uri": str(selection),
        "identity": f"{SELECTION_PREFIX}/fixture@2026.11.20.1:fingerprint",
        "local_path": "/data/fixture",
        "relative_path": TRAIN_FILENAME,
    }
    launch = {
        "inputs": {"train_data": [source]},
        "skyrl": {"terminal_bench_config": {"installed_agent": {"output_path": str(store)}}},
    }
    (run_output / RESOLVED_LAUNCH_FILENAME).write_text(json.dumps({"config": launch}))
    write_attempt(store, "graded", task, trial_result(*ALL_PHASES, rewards={"reward": 0.0}), setup_receipt(True), {})
    test_output = "préparing tests\n" * 400 + MISSING_MODULE + "\n"
    write_attempt(
        store,
        "no-reward",
        task,
        trial_result(*ALL_PHASES, exception=("RewardFileNotFoundError", "No reward file found")),
        setup_receipt(True),
        {"trials/no-reward/verifier/test-stdout.txt": test_output},
    )
    write_attempt(
        store,
        "setup-exit",
        task,
        trial_result("environment_setup", exception=("RuntimeError", SETUP_EXIT)),
        setup_receipt(False),
        {
            "engine/task-setup/0.stdout": "Preparing the fixture package\n",
            "engine/task-setup/0.stderr": "E: Unable to locate package libfixture\n",
        },
    )
    output = StoragePath(str(tmp_path / "health"))

    path = extract_attempts(StoragePath(str(run_output)), "fixture-run@2026.11.20.1", output)

    rows = {row["attempt_id"]: row for row in pq.read_table(str(path)).to_pylist()}
    assert {attempt: (row["outcome"], row["grading_failure"], row["phase"]) for attempt, row in rows.items()} == {
        "graded": (Outcome.GRADED, None, Phase.VERIFIER),
        "no-reward": (Outcome.INFRA_ERROR, GradingFailure.MISSING_REWARD, Phase.VERIFIER),
        "setup-exit": (Outcome.INFRA_ERROR, GradingFailure.MISSING_REWARD, Phase.TASK_SETUP),
    }
    assert rows["graded"]["verifier_stdout"] is None and rows["graded"]["address"] is None
    assert rows["no-reward"]["verifier_stdout"] == test_output.encode()[-TAIL_BYTES:].decode(errors="ignore")
    assert rows["setup-exit"]["setup_stdout"] == "Preparing the fixture package\n"
    assert rows["setup-exit"]["setup_stderr"] == "E: Unable to locate package libfixture\n"
    assert (rows["setup-exit"]["error"], rows["setup-exit"]["setup_completed"]) == (SETUP_EXIT, False)
    assert rows["setup-exit"]["address"] == str(store / "setup-exit")
    assert {(row["selection"], row["task_digest"]) for row in rows.values()} == {
        ("fixture@2026.11.20.1", selection_digest(task))
    }
    assert extract_attempts(StoragePath(str(run_output)), "fixture-run@2026.11.20.1", output) == path

    write_attempt(store, "later", task, trial_result(), setup_receipt(True), {})
    with pytest.raises(FileExistsError):
        extract_attempts(StoragePath(str(run_output)), "fixture-run@2026.11.20.1", output)


def test_skip_lists_tasks_and_environments_that_fail_setup_or_verification(tmp_path):
    environments = {
        "repeated": ("swe/repeated", SETUP),
        "repeated-neighbor": ("swe/repeated", SETUP),
        "recovered": ("swe/recovered", SETUP),
        "infrastructure": ("swe/infrastructure", SETUP),
        "once": ("swe/once", SETUP),
        "incomplete": ("swe/incomplete", SETUP),
        "shared-1": (SHARED_IMAGE, SETUP),
        "shared-2": (SHARED_IMAGE, SETUP),
        "shared-untried": (SHARED_IMAGE, SETUP),
        "shared-other-setup": (SHARED_IMAGE, "make setup"),
    }
    tasks = {name: harbor_task(name, image, setup) for name, (image, setup) in environments.items()}
    selection = write_tasks(tmp_path / "selection" / TRAIN_FILENAME, list(tasks.values())).parent

    def attempt(name: str, phase: Phase, error: str | None, outcome: Outcome | None = Outcome.INFRA_ERROR) -> dict:
        image, setup = environments[name]
        return {
            "task_id": name,
            "task_digest": selection_digest(tasks[name]),
            "outcome": outcome,
            "phase": phase,
            "error": error,
            "docker_image": image,
            "setup_sha256": hashlib.sha256(setup.encode()).hexdigest(),
            "address": f"s3://fixture/installed-agents/{name}-{phase}",
        }

    first_run = [
        attempt("repeated", Phase.VERIFIER, f"{MISSING_MODULE}\nTraceback (most recent call last):"),
        attempt("repeated", Phase.TASK_SETUP, SETUP_EXIT),
        attempt("recovered", Phase.VERIFIER, MISSING_MODULE),
        attempt("recovered", Phase.VERIFIER, MISSING_MODULE),
        attempt("infrastructure", Phase.ENVIRONMENT, "Sandbox creation failed"),
        attempt("infrastructure", Phase.AGENT, "Agent timed out"),
        attempt("once", Phase.TASK_SETUP, SETUP_EXIT),
        attempt("incomplete", Phase.TASK_SETUP, None, None),
        attempt("incomplete", Phase.TASK_SETUP, None, None),
        attempt("shared-1", Phase.TASK_SETUP, SETUP_EXIT),
        attempt("shared-2", Phase.VERIFIER, MISSING_MODULE),
    ]
    second_run = [
        attempt("repeated", Phase.VERIFIER, MISSING_MODULE),
        attempt("recovered", Phase.VERIFIER, None, Outcome.GRADED),
        attempt("infrastructure", Phase.ENVIRONMENT, "Sandbox creation failed"),
        attempt("infrastructure", Phase.AGENT, "Agent timed out"),
        attempt("shared-1", Phase.TASK_SETUP, SETUP_EXIT),
        attempt("shared-2", Phase.VERIFIER, MISSING_MODULE),
    ]
    inputs = []
    for index, rows in enumerate((first_run, second_run)):
        inputs.append(tmp_path / f"run-{index}" / "attempts.parquet")
        inputs[-1].parent.mkdir()
        records = [
            {**row, "run": f"fixture-{index}@2026.11.20.1", "attempt_id": f"attempt-{ordinal}"}
            for ordinal, row in enumerate(rows)
        ]
        pq.write_table(pa.Table.from_pylist(records, schema=ATTEMPT_SCHEMA), inputs[-1])
    output = tmp_path / "skip"

    def publish(destination: Path, release: Path | None = None) -> dict:
        return publish_skip_list(
            StoragePath(str(selection)),
            [StoragePath(str(path)) for path in inputs],
            StoragePath(str(destination)),
            "test-commit",
            StoragePath(str(release)) if release else None,
        )

    manifest = publish(output)

    listed = {row["task_id"]: row for row in pq.read_table(output / SKIP_FILENAME).to_pylist()}
    assert {task_id: row["unit"] for task_id, row in listed.items()} == {
        "repeated": "task",
        "shared-1": "task",
        "shared-2": "task",
        "shared-untried": "environment",
    }
    assert (listed["repeated"]["failed_attempts"], listed["repeated"]["error"]) == (3, MISSING_MODULE)
    exclusion = HeldoutSource(
        uri=str(output / SKIP_FILENAME), version="2026.11.20.1", rows=len(listed), format=PanelFormat.TASK_SPEC
    )
    assert {task.id: task for task in _panel_source_tasks(exclusion, {})} == {name: tasks[name] for name in listed}
    assert manifest["inputs"] == {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (selection / TRAIN_FILENAME, *inputs)
    }

    fixed = harbor_task("shared-1", SHARED_IMAGE, "pip install -e . pytest")
    release = write_tasks(tmp_path / "release.parquet", [fixed, tasks["repeated"]])
    publish(tmp_path / "skip-after-release", release)
    after_release = pq.read_table(tmp_path / "skip-after-release" / SKIP_FILENAME)
    assert set(after_release["task_id"].to_pylist()) == {"repeated", "shared-2", "shared-untried"}

    empty = StoragePath(str(tmp_path / "empty-skip"))
    one_attempt = inputs[0].parent / "one-attempt.parquet"
    pq.write_table(pq.read_table(inputs[0]).slice(0, 1), one_attempt)
    duplicate = one_attempt.with_name("duplicate.parquet")
    duplicate.write_bytes(one_attempt.read_bytes())
    with pytest.raises(ValueError, match="No tasks meet the skip criteria"):
        publish_skip_list(
            StoragePath(str(selection)),
            [StoragePath(str(one_attempt)), StoragePath(str(duplicate))],
            empty,
            "test-commit",
        )
    assert not empty.exists()
