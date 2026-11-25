# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actor-phase failure typing and the attempt batch's registry credential."""

import base64
import hashlib
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from finestore.rl.difficulty import Solver

from experiments.post_training.task_curation.difficulty import run_action
from experiments.post_training.task_curation.difficulty.importer import receipt_row
from experiments.post_training.task_curation.difficulty.planner import AttemptBatch, SourceAttempts
from experiments.post_training.task_curation.difficulty.run_action import REGISTRY_AUTH_SECRET, attempt_command
from experiments.post_training.task_curation.difficulty.state import GapReason, Outcome
from experiments.post_training.task_curation.difficulty_preparation import grader_tier_backend
from experiments.post_training.task_curation.difficulty_prepared import (
    GRADER_PRIVILEGE_DROP_UID,
    PreparedTask,
    grader_requires_multi_uid,
)
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


@pytest.mark.parametrize("regional", [False, True])
def test_native_action_entrypoint_bounds_only_regional_host_concurrency(monkeypatch, tmp_path, regional):
    """A TPU host cap changes scheduling, preserving the frozen runtime command."""
    source = SourceAttempts("source", "population", (("task", "b" * 64, 1),), "queue", "c" * 64, ())
    action = AttemptBatch(Solver.SMALL, (source,), 1)
    ledger = str(tmp_path / "ledger")
    manifest, serving_pool = "manifest.json", "regional-pool"
    monkeypatch.setattr(
        "sys.argv", ["run_action", "--manifest", manifest, "--ledger", ledger, "--serving-pool", serving_pool]
    )
    monkeypatch.setattr(run_action, "configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(run_action, "load_action", lambda _: action)
    monkeypatch.setattr(run_action, "campaign_record", lambda _: {"campaign_id": "frozen"} if regional else None)
    monkeypatch.setattr(run_action, "marin_prefix", lambda: str(tmp_path))
    monkeypatch.setattr(run_action, "get_secret", lambda _: "fixture")
    calls = []
    monkeypatch.setattr(run_action, "subprocess", SimpleNamespace(run=lambda command, **kwargs: calls.append(command)))
    run_action.main()
    assert len(calls) == 1
    actual = calls[0]
    assert actual[actual.index("--workers") + 1] == ("4" if regional else "16")
    baseline = attempt_command(
        action,
        manifest,
        ledger,
        actual[actual.index("--output-root") + 1],
        actual[actual.index("--local-root") + 1],
        registry_auth=True,
        serving_pool=serving_pool,
    )
    actual = list(actual)
    actual[actual.index("--workers") + 1] = baseline[baseline.index("--workers") + 1]
    assert actual == baseline


def test_registry_reference_pins_its_digest(monkeypatch):
    pinned_registry_reference.cache_clear()
    pinned = "ghcr.io/marin-community/rollout-images:task-curation-env-0427f1c8ec6612e4@sha256:" + "e" * 64
    assert pinned_registry_reference(pinned) == pinned

    calls = []

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        return SimpleNamespace(stdout="sha256:" + "e" * 64 + "\n", returncode=0, stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    reference = "ghcr.io/marin-community/rollout-images:task-curation-env-0427f1c8ec6612e4"
    resolved = pinned_registry_reference(reference, authfile="/run/auth.json")
    assert resolved == reference + "@sha256:" + "e" * 64
    assert calls[0][0:2] == ["skopeo", "inspect"]
    assert calls[0][calls[0].index("--authfile") + 1] == "/run/auth.json"
    assert calls[0][-1] == f"docker://{reference}"
    # One registry lookup per (reference, authfile) pair.
    pinned_registry_reference(reference, authfile="/run/auth.json")
    assert len(calls) == 1


def test_registry_reference_reports_the_skopeo_failure(monkeypatch):
    pinned_registry_reference.cache_clear()

    def fake_run(command, check, capture_output, text):
        return SimpleNamespace(stdout="", returncode=1, stderr='time="..." level=fatal msg="boom"')

    monkeypatch.setattr("subprocess.run", fake_run)
    reference = "ghcr.io/marin-community/rollout-images:task-curation-env-0427f1c8ec6612e4"
    try:
        pinned_registry_reference(reference, authfile="/run/auth.json")
        raise AssertionError("expected the digest lookup to raise")
    except ValueError as error:
        assert "rc=1" in str(error) and "boom" in str(error)


def _spec_with_verifier_resources(*contents: str) -> dict:
    return {
        "resources": {
            "verifier": [
                {
                    "path": f"file-{index}.py",
                    "source": {
                        "kind": "inline_file",
                        "content_base64": base64.b64encode(content.encode()).decode(),
                    },
                }
                for index, content in enumerate(contents)
            ]
        }
    }


def test_grader_requires_multi_uid_detects_the_privilege_drop():
    dropping = _spec_with_verifier_resources(
        "import json\n", f"sandbox = LocalLeanSandbox({GRADER_PRIVILEGE_DROP_UID})\n"
    )
    assert grader_requires_multi_uid(dropping) is True
    plain = _spec_with_verifier_resources("import json\nrun_tests()\n")
    assert grader_requires_multi_uid(plain) is False
    assert grader_requires_multi_uid({"resources": {}}) is False


def test_grader_tier_backend_routes_a_privilege_dropping_grader_to_daytona():
    assert grader_tier_backend("local", native=False, requires_multi_uid=True) == "daytona"
    assert grader_tier_backend("local", native=False, requires_multi_uid=False) == "local"
    assert grader_tier_backend("daytona", native=False, requires_multi_uid=True) == "daytona"
    assert grader_tier_backend("local", native=True, requires_multi_uid=True) == "local"
