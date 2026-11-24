# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.task_health import ATTEMPT_RECEIPTS_DIRECTORY, AttemptReceipt
from marin.execution.lazy import ArtifactStep, run
from marin.rl.skyrl_records import RESOLVED_LAUNCH_FILENAME, TERMINAL_RECEIPT_FILENAME
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

from experiments.post_training.task_curation import task_health
from experiments.post_training.task_curation.selection import (
    SELECTION_PREFIX,
    TASK_SPEC_COLUMN,
    TRAIN_FILENAME,
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


def test_native_receipts_drive_task_health_without_the_harbor_trial_store(tmp_path):
    task = harbor_task("example", SHARED_IMAGE)
    export = tmp_path / "export"
    receipts = export / ATTEMPT_RECEIPTS_DIRECTORY
    receipts.mkdir(parents=True)
    receipt = AttemptReceipt(
        attempt_id="trial-1",
        task_id=task.id,
        task_digest=selection_digest(task),
        outcome=Outcome.INFRA_ERROR,
        grading_failure=None,
        exception_type="VerifierRuntimeError",
        error="Task setup command exited 1",
        phase=Phase.TASK_SETUP,
        docker_image=SHARED_IMAGE,
        setup_sha256=hashlib.sha256(SETUP.encode()).hexdigest(),
        setup_completed=False,
        setup_stdout=None,
        setup_stderr=MISSING_MODULE,
        verifier_stdout=None,
        verifier_stderr=None,
    )
    first = receipts / "first.jsonl"
    first.write_text(json.dumps(asdict(receipt)) + "\n")
    second = receipts / "second.jsonl"
    second.write_text(json.dumps(asdict(replace(receipt, attempt_id="trial-2"))) + "\n")
    config = {
        "skyrl": {"trainer": {"export_path": str(export)}},
        "inputs": {"train_data": [{"identity": f"{SELECTION_PREFIX}/fixture-pi:1"}]},
    }
    snapshot = task_health._extract_attempts(config, "fixture@1", StoragePath(str(tmp_path / "snapshot")))
    rows = pq.read_table(str(snapshot)).to_pylist()

    assert [row["selection"] for row in rows] == ["fixture-pi", "fixture-pi"]
    assert [row["address"] for row in rows] == [str(first), str(second)]
    assert {row["task_digest"] for row in rows} == {receipt.task_digest}
    training = write_tasks(tmp_path / TRAIN_FILENAME, [task])
    skip = StoragePath(str(tmp_path / "skip"))
    publish_skip_list(StoragePath(str(training.parent)), [snapshot], skip, "fixture")
    assert pq.read_table(str(skip / SKIP_FILENAME))["task_id"].to_pylist() == [task.id]


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


def test_extract_records_each_attempts_grade_phase_and_output_tails(tmp_path, monkeypatch):
    task = harbor_task("swe:1")
    selection = write_tasks(tmp_path / "selection" / TRAIN_FILENAME, [task]).parent
    store = tmp_path / "attempts/trajectories/installed-agents"
    run_output = tmp_path / "run-artifacts/fixture-run/2026.11.20.1"
    run_output.mkdir(parents=True)
    source = {
        "uri": str(selection),
        "identity": f"{SELECTION_PREFIX}/fixture@2026.11.20.1:fingerprint",
        "local_path": "/data/fixture",
        "relative_path": TRAIN_FILENAME,
    }
    launch = {
        "run": {"id": "fixture-native-run", "attempt_id": "fixture-native-attempt"},
        "artifacts": {"export_root": str(run_output / "exports")},
        "inputs": {"train_data": [source]},
        "skyrl": {"terminal_bench_config": {"installed_agent": {"output_path": str(store)}}},
    }
    (run_output / RESOLVED_LAUNCH_FILENAME).write_text(json.dumps({"config": launch}))
    (run_output / TERMINAL_RECEIPT_FILENAME).write_text(json.dumps({"config": launch, "result": {"state": "succeeded"}}))
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
    (store / "partial").mkdir()
    (store / "partial" / "upload-incomplete").touch()
    health_root = tmp_path / "health"
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "run-artifacts"))
    monkeypatch.setattr(task_health, "TASK_HEALTH_ROOT", str(health_root))

    artifact = ArtifactStep.adopt("fixture-run", "2026.11.20.1", str(run_output))
    (health,) = run(task_health.task_health_step(artifact))
    snapshot = json.loads((StoragePath(health.path) / "attempts.json").read_text())
    path = StoragePath(snapshot["attempts_uri"])
    output = path.parent

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
    launch["run"]["attempt_id"] = "fixture-retry"
    (run_output / RESOLVED_LAUNCH_FILENAME).write_text(json.dumps({"config": launch}))
    (run_output / TERMINAL_RECEIPT_FILENAME).write_text(json.dumps({"config": launch, "result": {"state": "failed"}}))
    monkeypatch.setattr(task_health, "configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(
        sys, "argv", ["task_health", "extract", "--cluster", "fixture", "--run", "fixture-run@2026.11.20.1"]
    )
    task_health.main()
    version_root = health_root / "attempts/fixture-run/2026.11.20.1"
    (retry_snapshot,) = set(version_root.glob(f"*/{task_health.ATTEMPTS_FILENAME}")) - {Path(str(path))}
    assert set(pq.read_table(retry_snapshot)["attempt_id"].to_pylist()) == {*rows, "later"}
    assert set(pq.read_table(str(path))["attempt_id"].to_pylist()) == rows.keys()


def test_skip_lists_tasks_and_environments_that_fail_setup_or_verification(tmp_path, monkeypatch, capsys):
    environments = {
        "repeated": ("swe/repeated", SETUP),
        "repeated-neighbor": ("swe/repeated", SETUP),
        "recovered": ("swe/recovered", SETUP),
        "infrastructure": ("swe/infrastructure", SETUP),
        "infrastructure-neighbor": ("swe/infrastructure", SETUP),
        "changed-task": ("swe/changed", SETUP),
        "changed-neighbor": ("swe/changed", SETUP),
        "once": ("swe/once", SETUP),
        "incomplete": ("swe/incomplete", SETUP),
        "shared-1": (SHARED_IMAGE, SETUP),
        "shared-2": (SHARED_IMAGE, SETUP),
        "shared-untried": (SHARED_IMAGE, SETUP),
        "shared-other-setup": (SHARED_IMAGE, "make setup"),
    }
    tasks = {name: harbor_task(name, image, setup) for name, (image, setup) in environments.items()}
    selection = write_tasks(tmp_path / "selections/fixture/2026.11.20.1" / TRAIN_FILENAME, list(tasks.values())).parent

    def attempt(name: str, phase: Phase, error: str | None, outcome: Outcome | None = Outcome.INFRA_ERROR) -> dict:
        image, setup = environments[name]
        return {
            "task_id": name,
            "task_digest": selection_digest(tasks[name]),
            "outcome": outcome,
            "phase": phase,
            "error": error,
            "exception_type": "VerifierRuntimeError" if phase == Phase.VERIFIER and error else None,
            "setup_completed": False if phase == Phase.TASK_SETUP and error else None,
            "verifier_stdout": error if phase == Phase.VERIFIER else None,
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
        {**attempt("infrastructure", Phase.VERIFIER, "Service failed"), "exception_type": "AddTestsDirError"},
        {**attempt("infrastructure", Phase.VERIFIER, "Partial test output"), "exception_type": "TrialTimeoutError"},
        {**attempt("infrastructure", Phase.VERIFIER, "Partial test output"), "exception_type": "SandboxDiskFullError"},
        {**attempt("infrastructure", Phase.TASK_SETUP, "Service failed"), "exception_type": "RuntimeServiceError"},
        {**attempt("infrastructure", Phase.TASK_SETUP, "Daytona output lost"), "exception_type": "DaytonaCommandError"},
        attempt("infrastructure", Phase.TASK_SETUP, "Setup process failed to start"),
        attempt("infrastructure", Phase.TASK_SETUP, "Setup process failed to start"),
        {**attempt("infrastructure", Phase.VERIFIER, "Sandbox went away"), "verifier_stdout": None},
        {**attempt("infrastructure", Phase.VERIFIER, "Sandbox went away"), "verifier_stdout": None},
        attempt("once", Phase.TASK_SETUP, SETUP_EXIT),
        attempt("incomplete", Phase.TASK_SETUP, None, None),
        attempt("incomplete", Phase.TASK_SETUP, None, None),
        attempt("shared-1", Phase.TASK_SETUP, SETUP_EXIT),
        attempt("shared-2", Phase.VERIFIER, MISSING_MODULE),
    ]
    old_task_failure = {**attempt("changed-task", Phase.TASK_SETUP, SETUP_EXIT), "task_digest": "1" * 64}
    first_run.extend([old_task_failure, attempt("changed-task", Phase.TASK_SETUP, SETUP_EXIT)])
    second_run = [
        attempt("repeated", Phase.VERIFIER, MISSING_MODULE),
        attempt("recovered", Phase.VERIFIER, None, Outcome.GRADED),
        attempt("infrastructure", Phase.ENVIRONMENT, "Sandbox creation failed"),
        attempt("infrastructure", Phase.AGENT, "Agent timed out"),
        {**attempt("infrastructure", Phase.VERIFIER, "Partial test output"), "exception_type": "TrialTimeoutError"},
        {**attempt("infrastructure", Phase.VERIFIER, "Partial test output"), "exception_type": "SandboxDiskFullError"},
        attempt("shared-1", Phase.TASK_SETUP, SETUP_EXIT),
        attempt("shared-2", Phase.VERIFIER, MISSING_MODULE),
        old_task_failure,
        attempt("changed-task", Phase.TASK_SETUP, SETUP_EXIT),
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
    output = tmp_path / "skip/fixture/2026.11.20.2"

    def publish(destination: Path, release: Path | None = None) -> dict:
        return publish_skip_list(
            StoragePath(str(selection)),
            [StoragePath(str(path)) for path in inputs],
            StoragePath(str(destination)),
            "test-commit",
            StoragePath(str(release)) if release else None,
        )

    monkeypatch.setattr(task_health, "TASK_HEALTH_ROOT", str(tmp_path))
    monkeypatch.setattr(task_health, "SELECTION_ROOT", str(tmp_path / "selections"))
    monkeypatch.setattr(task_health, "configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(
        task_health.Provenance,
        "capture",
        lambda: task_health.Provenance("test-tree", "test-commit", False, "test-branch", "test"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "task_health",
            "skip",
            "--selection",
            "fixture@2026.11.20.1",
            "--attempts",
            *map(str, inputs),
            "--version",
            "2026.11.20.2",
        ],
    )
    task_health.main()
    manifest = json.loads(capsys.readouterr().out)

    listed = {row["task_id"]: row for row in pq.read_table(output / SKIP_FILENAME).to_pylist()}
    assert {task_id: row["unit"] for task_id, row in listed.items()} == {
        "repeated": "task",
        "changed-task": "task",
        "shared-1": "task",
        "shared-2": "task",
        "shared-untried": "environment",
    }
    assert (listed["repeated"]["failed_attempts"], listed["repeated"]["error"]) == (3, MISSING_MODULE)
    assert {name: TaskSpec.model_validate_json(row[TASK_SPEC_COLUMN]) for name, row in listed.items()} == {
        name: tasks[name] for name in listed
    }
    assert manifest["inputs"] == {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (selection / TRAIN_FILENAME, *inputs)
    }

    fixed = harbor_task("shared-1", SHARED_IMAGE, "pip install -e . pytest")
    untagged = tasks["repeated"].model_copy(update={"tags": ("harbor",)})
    release = write_tasks(tmp_path / "release.parquet", [fixed, untagged])
    publish(tmp_path / "skip-after-release", release)
    after_release = pq.read_table(tmp_path / "skip-after-release" / SKIP_FILENAME)
    assert set(after_release["task_id"].to_pylist()) == {"repeated", "shared-2", "shared-untried", "changed-task"}

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
