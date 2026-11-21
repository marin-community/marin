# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provider credentials reach the verifier env from the wave's resolved bundle, never free text."""

import pytest

pytest.importorskip("harbor")

from finestore.rl.difficulty import DifficultyGap, difficulty_gap_reason  # noqa: E402

from experiments.post_training.task_curation import difficulty_trials  # noqa: E402
from experiments.post_training.task_curation.difficulty_prepared import PreparedTask  # noqa: E402
from experiments.post_training.task_curation.difficulty_trials import (  # noqa: E402
    grader_infrastructure_reason,
    resolve_provider_secrets,
    task_guest_environment,
    task_required_provider_secrets,
)


@pytest.fixture
def judged_bundle(tmp_path):
    """A native harbor bundle whose verifier resolves the Together credential."""
    directory = tmp_path / "packet"
    directory.mkdir()
    (directory / "task.toml").write_text(
        "[verifier]\n"
        'timeout_sec = 600.0\n'
        'env = { TOGETHER_API_KEY = "${TOGETHER_API_KEY}" }\n'
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


def test_ungrantable_provider_credentials_fail_the_wave_closed(judged_bundle):
    with pytest.raises(ValueError, match="cannot grant"):
        resolve_provider_secrets(("OPENAI_API_KEY",))


def test_unresolved_credentials_fail_the_task_before_any_attempt(judged_bundle):
    task = PreparedTask(
        task_id="task", runtime_sha256="a" * 64, path=str(judged_bundle),
        workload="native", backend="local", reason=None, judge_model=None,
    )
    with pytest.raises(ValueError, match="did not resolve"):
        task_guest_environment(task, {})


def test_resolved_credentials_grant_the_guest_environment(judged_bundle, monkeypatch):
    monkeypatch.setattr(difficulty_trials, "require_secret", lambda name: "resolved-" + name.lower())
    resolved = resolve_provider_secrets(("TOGETHER_API_KEY",))
    task = PreparedTask(
        task_id="task", runtime_sha256="a" * 64, path=str(judged_bundle),
        workload="native", backend="local", reason=None, judge_model=None,
    )
    assert task_guest_environment(task, resolved) == {"TOGETHER_API_KEY": "resolved-together_api_key"}


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
