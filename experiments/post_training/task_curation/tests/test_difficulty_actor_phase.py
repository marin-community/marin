# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actor-phase failure typing and the attempt batch's registry credential."""

import hashlib
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from finestore.rl.difficulty import Solver

from experiments.post_training.task_curation.difficulty.importer import receipt_row
from experiments.post_training.task_curation.difficulty.planner import AttemptBatch, SourceAttempts
from experiments.post_training.task_curation.difficulty.run_action import REGISTRY_AUTH_SECRET, attempt_command
from experiments.post_training.task_curation.difficulty.state import GapReason, Outcome
from experiments.post_training.task_curation.difficulty_prepared import PreparedTask
from experiments.post_training.task_curation.difficulty_trials import (
    DEFECTIVE_ACTOR_IMAGES,
    actor_phase_gap_reason,
    pinned_registry_reference,
)

FULL_INDEX_PIN = (
    "ghcr.io/marin-community/rl-data-openresearcher@sha256:"
    "ff448d3a683247068454eaa67ee7e25e75b5357ee74e80696b5df18d15e016fa"
)


def prepared_task(pin: str, tmp_path) -> PreparedTask:
    directory = tmp_path / "task"
    directory.mkdir()
    (directory / "task-spec.json").write_text(json.dumps({"environment_requirements": {"docker_image": pin}}))
    return PreparedTask(
        task_id="openresearcher-task",
        runtime_sha256="a" * 64,
        path=str(directory),
        workload="task_spec",
        backend="local",
        reason=None,
        judge_model=None,
    )


def trial_result(*, exception: bool, env_finished: bool, verifier_started: bool):
    return SimpleNamespace(
        exception_info=SimpleNamespace(exception_type="RuntimeError") if exception else None,
        verifier=SimpleNamespace(started_at="2026-11-22T00:00:00+00:00") if verifier_started else None,
        environment_setup=SimpleNamespace(
            started_at="2026-11-22T00:00:00+00:00",
            finished_at="2026-11-22T00:01:00+00:00" if env_finished else None,
        ),
    )


def test_broken_actor_pin_stays_the_source_image_defect(tmp_path):
    task = prepared_task(next(iter(DEFECTIVE_ACTOR_IMAGES)), tmp_path)
    result = trial_result(exception=True, env_finished=False, verifier_started=False)
    reason = actor_phase_gap_reason(task, result)
    assert reason is not None and reason.startswith("environment_artifact_missing:")


def test_unfinished_environment_start_types_the_image_ref(tmp_path):
    task = prepared_task(FULL_INDEX_PIN, tmp_path)
    result = trial_result(exception=True, env_finished=False, verifier_started=False)
    assert actor_phase_gap_reason(task, result) == f"environment_start_failure:{FULL_INDEX_PIN}"


def test_started_environment_never_types_a_start_failure(tmp_path):
    task = prepared_task(FULL_INDEX_PIN, tmp_path)
    crashed_after_start = trial_result(exception=True, env_finished=True, verifier_started=False)
    assert actor_phase_gap_reason(task, crashed_after_start) is None
    graded = trial_result(exception=True, env_finished=True, verifier_started=True)
    assert actor_phase_gap_reason(task, graded) is None
    clean = trial_result(exception=False, env_finished=True, verifier_started=False)
    assert actor_phase_gap_reason(task, clean) is None


def test_start_failure_receipt_types_the_image_reference(tmp_path):
    """A failed-pull attempt receipt lands as the typed start failure, never coverage."""
    receipt = tmp_path / "attempt.json"
    receipt.write_text(
        json.dumps(
            {
                "task_id": "openresearcher-task",
                "status": "infrastructure",
                "reward": None,
                "reason": f"environment_start_failure:{FULL_INDEX_PIN}",
            }
        )
        + "\n"
    )
    sample = {
        "source_id": "OpenResearcher:OpenResearcher-Dataset",
        "release": {"repo": "open-athena/openresearcher", "revision": "8d5a48bf4994f861c903cc90f34e1530c25ea052"},
        "converter_version": "2026.11.13.1",
        "converter_revision": "847c3c52650c45bc2cb8dacde9e4a4150c8ecd9a",
        "task_specs_sha256": "d" * 64,
        "tasks": [{"task_id": "openresearcher-task", "task_sha256": "b" * 64}],
    }
    row = receipt_row(
        {"uri": str(receipt), "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()},
        sample,
        Solver.HOSTED,
        "run",
        "2026-11-22T00:00:00+00:00",
    )
    assert row.outcome == Outcome.UNSCORED
    assert row.reason == GapReason.ENVIRONMENT_START_FAILURE
    assert row.artifact_uri == FULL_INDEX_PIN


def test_attempt_batch_carries_its_registry_credential():
    source = SourceAttempts(
        "OpenResearcher:OpenResearcher-Dataset",
        "population",
        (("openresearcher-task", "b" * 64, 1),),
        "s3://queue/sources.json",
        "c" * 64,
        (),
    )
    action = AttemptBatch(Solver.HOSTED, (source,), 1)
    with_auth = attempt_command(action, "manifest.json", "ledger", "output", "/tmp/local", registry_auth=True)
    assert with_auth[-2:] == ["--registry-auth-secret", REGISTRY_AUTH_SECRET]
    without_auth = attempt_command(action, "manifest.json", "ledger", "output", "/tmp/local", registry_auth=False)
    assert "--registry-auth-secret" not in without_auth


def test_registry_reference_pins_its_digest(monkeypatch):
    pinned_registry_reference.cache_clear()
    pinned = "ghcr.io/marin-community/rollout-images:task-curation-env-0427f1c8ec6612e4@sha256:" + "e" * 64
    assert pinned_registry_reference(pinned) == pinned

    calls = []

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        return SimpleNamespace(stdout="sha256:" + "e" * 64 + "\n", returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)
    reference = "ghcr.io/marin-community/rollout-images:task-curation-env-0427f1c8ec6612e4"
    resolved = pinned_registry_reference(reference, authfile="/run/auth.json")
    assert resolved == reference + "@sha256:" + "e" * 64
    assert calls[0][0:2] == ["skopeo", "inspect"] and calls[0][-1] == reference
    assert calls[0][calls[0].index("--src-authfile") + 1] == "/run/auth.json"
    # One registry lookup per (reference, authfile) pair.
    pinned_registry_reference(reference, authfile="/run/auth.json")
    assert len(calls) == 1
