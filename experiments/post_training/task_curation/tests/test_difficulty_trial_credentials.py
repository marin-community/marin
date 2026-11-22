# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provider credentials reach the verifier env from the wave's resolved bundle, never free text."""

import json

import pytest

pytest.importorskip("harbor")

from finestore.rl.difficulty import DifficultyGap, Solver, difficulty_gap_reason

from experiments.post_training.task_curation import difficulty_trials
from experiments.post_training.task_curation.datasets.environments import LEAN_GRADER_PACKAGES
from experiments.post_training.task_curation.difficulty.importer import receipt_row
from experiments.post_training.task_curation.difficulty.state import GapReason
from experiments.post_training.task_curation.difficulty_prepared import PreparedTask
from experiments.post_training.task_curation.difficulty_trials import (
    grader_infrastructure_reason,
    resolve_provider_secrets,
    task_guest_environment,
    task_required_provider_secrets,
)
from experiments.post_training.task_curation.images.build import (
    DEFAULT_REPOSITORY,
    identity_digest,
    image_tag,
)


@pytest.fixture
def judged_bundle(tmp_path):
    """A native harbor bundle whose verifier resolves the Together credential."""
    directory = tmp_path / "packet"
    directory.mkdir()
    (directory / "task.toml").write_text(
        "[verifier]\n" "timeout_sec = 600.0\n" 'env = { TOGETHER_API_KEY = "${TOGETHER_API_KEY}" }\n'
    )
    (directory / "instruction.md").write_text("Original task")
    (directory / "tests").mkdir()
    (directory / "tests/test.sh").write_text("exit 0\n")
    return directory


def test_bundle_credential_templates_are_detected(judged_bundle):
    assert task_required_provider_secrets((judged_bundle,)) == frozenset({"TOGETHER_API_KEY"})


def test_bundle_without_credential_templates_requires_nothing(tmp_path):
    directory = tmp_path / "plain"
    directory.mkdir()
    (directory / "task.toml").write_text('[environment]\nworkdir = "/app"\n')
    (directory / "instruction.md").write_text("Original task")
    (directory / "tests").mkdir()
    (directory / "tests/test.sh").write_text("exit 0\n")
    assert task_required_provider_secrets((directory,)) == frozenset()


def test_ungrantable_provider_credentials_fail_the_wave_closed():
    with pytest.raises(ValueError, match="cannot grant"):
        resolve_provider_secrets(("OPENAI_API_KEY",))


def test_unresolved_credentials_fail_the_task_before_any_attempt(judged_bundle):
    task = PreparedTask(
        task_id="task",
        runtime_sha256="a" * 64,
        path=str(judged_bundle),
        workload="native",
        backend="local",
        reason=None,
        judge_model=None,
    )
    with pytest.raises(ValueError, match="did not resolve"):
        task_guest_environment(task, {})


def test_resolved_credentials_grant_the_guest_environment(judged_bundle, monkeypatch):
    monkeypatch.setattr(difficulty_trials, "require_secret", lambda name: "resolved-" + name.lower())
    resolved = resolve_provider_secrets(("TOGETHER_API_KEY",))
    task = PreparedTask(
        task_id="task",
        runtime_sha256="a" * 64,
        path=str(judged_bundle),
        workload="native",
        backend="local",
        reason=None,
        judge_model=None,
    )
    assert task_guest_environment(task, resolved) == {"TOGETHER_API_KEY": "resolved-together_api_key"}


def _fake_storage_path(payloads: dict[str, str]):
    class _FakePath:
        def __init__(self, url: str):
            self._url = url

        def read_text(self) -> str:
            return payloads[self._url]

    return _FakePath


def test_normalized_lock_drops_only_the_staging_comment():
    staged = (
        "pandas==2.2.2 \\\n    --hash=sha256:aa\n"
        "    # via -r /var/folders/x/tmpABC/requirements.in\n\nskyrl-gym==0.1.0\n"
    )
    plain = "pandas==2.2.2 \\\n    --hash=sha256:aa\n\nskyrl-gym==0.1.0\n"
    assert difficulty_trials._normalized_lock(staged) == difficulty_trials._normalized_lock(plain)


def test_frozen_lock_resolves_to_the_declared_toolchain_image(monkeypatch):

    lean_lock = "# via -r /var/folders/x/tmpABC/requirements.in\npandas==2.2.2\n"
    monkeypatch.setattr(difficulty_trials, "StoragePath", _fake_storage_path({"s3://fixture/lock": lean_lock}))
    monkeypatch.setattr(difficulty_trials, "_declared_lock", lambda environment: "pandas==2.2.2\n")
    difficulty_trials.declared_grader_machine_image.cache_clear()
    image = difficulty_trials.declared_grader_machine_image("s3://fixture/lock")
    assert image == image_tag(DEFAULT_REPOSITORY, identity_digest(LEAN_GRADER_PACKAGES))


def test_unmatched_lock_keeps_the_existing_lock_built_machine(monkeypatch):
    monkeypatch.setattr(difficulty_trials, "StoragePath", _fake_storage_path({"s3://fixture/lock": "pandas==2.2.2\n"}))
    monkeypatch.setattr(difficulty_trials, "_declared_lock", lambda environment: "sympy==1.13.3\n")
    difficulty_trials.declared_grader_machine_image.cache_clear()
    assert difficulty_trials.declared_grader_machine_image("s3://fixture/lock") is None


def test_task_spec_grader_image_resolution(tmp_path, monkeypatch):
    lock_url = "s3://fixture/lock"
    bundle = tmp_path / "packet"
    bundle.mkdir()
    (bundle / "task-spec.json").write_text(json.dumps({"grader": {"environment": {"packages_lock": lock_url}}}))
    task = PreparedTask(
        task_id="task",
        runtime_sha256="a" * 64,
        path=str(bundle),
        workload="task_spec",
        backend="local",
        reason=None,
        judge_model=None,
    )
    monkeypatch.setattr(difficulty_trials, "declared_grader_machine_image", lambda url: "ghcr.io/x:tag")
    assert difficulty_trials.task_spec_grader_machine_image(task) == "ghcr.io/x:tag"
    native = task.model_copy(update={"workload": "native"})
    assert difficulty_trials.task_spec_grader_machine_image(native) is None
    (bundle / "task-spec.json").unlink()
    assert difficulty_trials.task_spec_grader_machine_image(task) is None


def test_verifier_phase_failures_classify_as_typed_grader_infrastructure():
    assert grader_infrastructure_reason(verifier_reached=False, verdict=None) is None
    assert grader_infrastructure_reason(verifier_reached=True, verdict=None) == (
        DifficultyGap.GRADER_INFRASTRUCTURE_FAILURE.value
    )


def test_infra_error_verdicts_carry_their_typed_transport_identity():
    verdict = {
        "status": "infra_error",
        "reward": 0.0,
        "detail": {
            "transport_attempts": (
                {"attempt": 1, "outcome": "failed", "status": 401, "reason": "http_401"},
                {"attempt": 2, "outcome": "succeeded"},
            ),
        },
    }
    assert grader_infrastructure_reason(verifier_reached=True, verdict=verdict) == (
        difficulty_gap_reason(DifficultyGap.GRADER_INFRASTRUCTURE_FAILURE, "http_401")
    )
    graded = {"status": "scored", "reward": 1.0}
    assert grader_infrastructure_reason(verifier_reached=True, verdict=graded) is None
    invalid = {"status": "invalid_task", "reward": 0.0}
    assert grader_infrastructure_reason(verifier_reached=True, verdict=invalid) is None


def test_imported_receipts_carry_the_error_identity(tmp_path):
    attempt = {
        "task_id": "task",
        "status": "infrastructure",
        "reward": None,
        "reason": "grader_infrastructure_failure:http_401",
    }
    path = tmp_path / "attempt.json"
    path.write_text(json.dumps(attempt))
    sample = {
        "source_id": "source",
        "release": {"repo": "fixture", "revision": "pin", "url": "https://fixture.invalid/pin"},
        "converter_version": "fixture",
        "converter_revision": "a" * 40,
        "task_specs_sha256": "b" * 64,
        "tasks": [{"task_id": "task", "task_sha256": "a" * 64}],
    }
    row = receipt_row(
        {"uri": str(path), "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()},
        sample,
        Solver.HOSTED,
        "run",
        "2026-11-21T00:00:00+00:00",
    )
    assert row.error_identity == "http_401"
    assert row.reason == GapReason.GRADER_INFRASTRUCTURE_FAILURE
