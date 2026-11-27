# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provider credential templates resolve host-side; guests get dummy values, never a real provider credential."""

import asyncio
import json

import pytest

pytest.importorskip("harbor")

from finestore.rl.difficulty import DifficultyGap, Solver, difficulty_gap_reason
from harbor.utils.env import validate_task_environment

from experiments.post_training.task_curation import difficulty_trials, toolchain_binding
from experiments.post_training.task_curation.datasets.environments import LEAN_GRADER_PACKAGES
from experiments.post_training.task_curation.difficulty.importer import receipt_row
from experiments.post_training.task_curation.difficulty.ledger_worker import source_has_judge_traffic
from experiments.post_training.task_curation.difficulty.state import GapReason
from experiments.post_training.task_curation.difficulty_config import SUBSTITUTE_JUDGE_MODEL
from experiments.post_training.task_curation.difficulty_prepared import (
    PreparedTask,
    attempt_judge_model,
    bundle_judge_template_var,
    task_declared_judge_model,
)
from experiments.post_training.task_curation.difficulty_trials import (
    grader_infrastructure_reason,
    resolve_provider_secrets,
    task_guest_environment,
    task_required_provider_secrets,
    trial_environment_options,
    trial_persistent_environment,
    worker_host_capabilities,
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


@pytest.fixture
def declared_judge_bundle(tmp_path):
    """A native bundle whose verifier names its own Together judge and credential."""
    directory = tmp_path / "rewardkit"
    directory.mkdir()
    (directory / "task.toml").write_text(
        "[verifier]\n"
        "timeout_sec = 600.0\n"
        'env = { REWARDKIT_JUDGE = "together_ai/Qwen/Qwen3.5-9B", TOGETHER_API_KEY = "${TOGETHER_API_KEY}" }\n'
    )
    (directory / "instruction.md").write_text("Original task")
    (directory / "tests").mkdir()
    (directory / "tests/test.sh").write_text("exit 0\n")
    return directory


def test_bundle_credential_templates_are_detected(judged_bundle):
    assert task_required_provider_secrets((judged_bundle,)) == frozenset({"TOGETHER_API_KEY"})


def test_trial_keeps_daytona_authority_during_another_trials_sdk_upload(tmp_path, monkeypatch):
    sdk = pytest.importorskip("daytona._utils.environment")
    environment = {
        "DAYTONA_DATA_API_KEY": "fixture-data",
        "DAYTONA_API_KEY": "fixture-other-account",
        "DAYTONA_API_URL": "https://app.daytona.io/api",
        "DAYTONA_TARGET": "us",
    }
    monkeypatch.setattr(difficulty_trials.os, "environ", environment.copy())
    capabilities = worker_host_capabilities({}, environment)
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "instruction.md").write_text("Original task")
    (packet / "task.toml").write_text('[environment]\ndocker_image = "alpine:3.22"\n')
    (packet / "tests").mkdir()
    (packet / "tests" / "test.sh").write_text("exit 0\n")
    config = difficulty_trials.TrialConfig(
        task=difficulty_trials.TaskConfig(path=packet),
        trials_dir=tmp_path / "trials",
        trial_name="candidate",
        agent=difficulty_trials.AgentConfig(name="terminus-2", model_name="openai/fixture"),
        environment=difficulty_trials.EnvironmentConfig(type="daytona"),
    )

    async def run_without_provider_io(trial):
        connection = trial.host_capabilities.daytona
        assert connection.api_key == "fixture-data"
        assert connection.api_url == environment["DAYTONA_API_URL"]
        assert connection.target == "us"
        assert trial.host_capabilities.model_authentication == {"http://relay/v1": "EMPTY"}
        return "constructed"

    monkeypatch.setattr(difficulty_trials.Trial, "run", run_without_provider_io)
    # The SDK clears the process environment while another trial builds/uploads.
    # Keep the real Trial.create path: it validates Daytona's client factory.
    with sdk.isolated_env({"AWS_ACCESS_KEY_ID": "fixture-storage"}):
        trial, result = asyncio.run(
            difficulty_trials.execute_trial(config, {"http://relay/v1": "EMPTY"}, {}, capabilities)
        )
    assert result == "constructed"
    assert trial.host_capabilities.daytona is capabilities.daytona


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


def test_resolved_credentials_grant_harbor_dummy_credentials(judged_bundle, monkeypatch):
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
    granted = task_guest_environment(task, resolved)
    assert granted == {"TOGETHER_API_KEY": "EMPTY"}


def test_daytona_options_keep_the_composed_grader_machine_image():
    """A tier-routed task's grader must still run from its digest-pinned composed image.

    48a1d5796a routes privilege-dropping graders up to Daytona after planning;
    the Daytona options swap must not discard the grader machine image binding
    (machine.py:652 refuses a bare host image), and worker-local options must
    not leak into the Daytona environment.
    """
    routed = PreparedTask(
        task_id="task",
        runtime_sha256="a" * 64,
        path=None,
        workload="task_spec",
        backend="daytona",
        reason=None,
        judge_model=None,
    )
    pinned = "ghcr.io/marin-community/rollout-images@sha256:" + "f" * 64
    options = trial_environment_options(routed, {"registry_authfile": "auth.json"}, "snapshot", pinned)
    assert options == {
        "provisioning": "snapshot",
        "trial_attempt_timeout_sec": difficulty_trials.ATTEMPT_TIMEOUT,
        "create_admission_command": list(difficulty_trials.DATA_ADMISSION_COMMAND),
        "grader_machine_image": pinned,
    }
    local = routed.model_copy(update={"backend": "local"})
    assert trial_environment_options(local, {"registry_authfile": "auth.json"}, "snapshot", pinned) == {
        "registry_authfile": "auth.json",
        "grader_machine_image": pinned,
    }


def test_declared_judge_model_parses_the_frozen_bundle(declared_judge_bundle):
    task = PreparedTask(
        task_id="task",
        runtime_sha256="a" * 64,
        path=str(declared_judge_bundle),
        workload="native",
        backend="daytona",
        reason=None,
        judge_model=None,
    )
    assert task_declared_judge_model(task) == "Qwen/Qwen3.5-9B"
    assert attempt_judge_model(task) == "Qwen/Qwen3.5-9B"


def test_env_template_judge_resolves_its_default_and_fails_closed(tmp_path):
    """A TaskTrove-style `${VAR:-default}` judge resolves to the model the verifier calls."""
    directory = tmp_path / "rewardkit"
    directory.mkdir()
    (directory / "task.toml").write_text(
        "[verifier]\n"
        'env = { OPENAI_API_KEY = "${OPENAI_API_KEY}", JUDGE_MODEL = "${JUDGE_MODEL:-}", '
        'REWARDKIT_JUDGE = "${JUDGE_MODEL:-openai/gpt-4o-mini}" }\n'
    )
    (directory / "instruction.md").write_text("Original task")
    (directory / "tests").mkdir()
    (directory / "tests/test.sh").write_text("exit 0\n")
    task = PreparedTask(
        task_id="task",
        runtime_sha256="a" * 64,
        path=str(directory),
        workload="native",
        backend="local",
        reason=None,
        judge_model=None,
    )
    with pytest.raises(ValueError, match="openai/gpt-4o-mini"):
        task_declared_judge_model(task)


def test_repaired_judge_template_rides_the_bundle_template_to_the_substitute(tmp_path):
    """The re-declared judge pins the bundle's own template variable to the served substitute."""
    directory = tmp_path / "rewardkit"
    directory.mkdir()
    (directory / "task.toml").write_text(
        "[verifier]\n"
        'env = { OPENAI_API_KEY = "${OPENAI_API_KEY}", JUDGE_MODEL = "${JUDGE_MODEL:-}", '
        'REWARDKIT_JUDGE = "${JUDGE_MODEL:-openai/gpt-4o-mini}" }\n'
    )
    (directory / "instruction.md").write_text("Original task")
    (directory / "tests").mkdir()
    (directory / "tests/test.sh").write_text("exit 0\n")
    task = PreparedTask(
        task_id="task",
        runtime_sha256="a" * 64,
        path=str(directory),
        workload="native",
        backend="daytona",
        reason=None,
        judge_model=SUBSTITUTE_JUDGE_MODEL,
    )
    assert bundle_judge_template_var(task) == "JUDGE_MODEL"
    assert trial_persistent_environment(task) == {
        "TOGETHER_AI_API_BASE": "http://127.0.0.1:8787/v1",
        "JUDGE_MODEL": "together_ai/zai-org/GLM-5.3-Flash",
    }


def test_repaired_judge_covers_the_bundle_credential_templates_with_the_guest_dummy(tmp_path):
    """The substitute judge rides the wave's Together grant; the bundle's own templates get dummies."""
    directory = tmp_path / "rewardkit"
    directory.mkdir()
    (directory / "task.toml").write_text(
        "[verifier]\n"
        'env = { OPENAI_API_KEY = "${OPENAI_API_KEY}", JUDGE_MODEL = "${JUDGE_MODEL:-}", '
        'REWARDKIT_JUDGE = "${JUDGE_MODEL:-openai/gpt-4o-mini}" }\n'
    )
    (directory / "instruction.md").write_text("Original task")
    (directory / "tests").mkdir()
    (directory / "tests/test.sh").write_text("exit 0\n")
    task = PreparedTask(
        task_id="task",
        runtime_sha256="a" * 64,
        path=str(directory),
        workload="native",
        backend="daytona",
        reason=None,
        judge_model=SUBSTITUTE_JUDGE_MODEL,
    )
    assert difficulty_trials.substitute_judge_credentials([task]) == frozenset({"OPENAI_API_KEY"})
    granted = task_guest_environment(task, {})
    assert granted == {"OPENAI_API_KEY": "EMPTY"}
    validate_task_environment(granted, protected_values=frozenset({"resolved-openai_api_key"}))


def test_declared_judge_outside_the_together_route_fails_closed(tmp_path):
    directory = tmp_path / "rewardkit"
    directory.mkdir()
    (directory / "task.toml").write_text('[verifier]\nenv = { REWARDKIT_JUDGE = "openai/gpt-5" }\n')
    (directory / "instruction.md").write_text("Original task")
    (directory / "tests").mkdir()
    (directory / "tests/test.sh").write_text("exit 0\n")
    task = PreparedTask(
        task_id="task",
        runtime_sha256="a" * 64,
        path=str(directory),
        workload="native",
        backend="local",
        reason=None,
        judge_model=None,
    )
    with pytest.raises(ValueError, match="outside the"):
        attempt_judge_model(task)


def test_unqualified_declared_judge_fails_closed(declared_judge_bundle):
    spec = (declared_judge_bundle / "task.toml").read_text().replace("Qwen/Qwen3.5-9B", "Qwen/Unqualified-9B")
    (declared_judge_bundle / "task.toml").write_text(spec)
    task = PreparedTask(
        task_id="task",
        runtime_sha256="a" * 64,
        path=str(declared_judge_bundle),
        workload="native",
        backend="local",
        reason=None,
        judge_model=None,
    )
    with pytest.raises(ValueError, match="has not qualified"):
        attempt_judge_model(task)


def test_task_spec_workload_declares_no_judge(declared_judge_bundle):
    task = PreparedTask(
        task_id="task",
        runtime_sha256="a" * 64,
        path=str(declared_judge_bundle),
        workload="task_spec",
        backend="local",
        reason=None,
        judge_model=None,
    )
    assert task_declared_judge_model(task) is None


def test_prepared_judge_takes_precedence_over_the_declared_one(declared_judge_bundle):
    task = PreparedTask(
        task_id="task",
        runtime_sha256="a" * 64,
        path=str(declared_judge_bundle),
        workload="native",
        backend="local",
        reason=None,
        judge_model="zai-org/GLM-5.3",
    )
    assert attempt_judge_model(task) == "zai-org/GLM-5.3"


def test_declared_judge_persistent_environment_reaches_the_sandbox(declared_judge_bundle):
    """The declared judge's base-URL override rides the environment config's persistent env.

    Harbor resolves only declared templates from the guest environment, so the
    projection must flow through EnvironmentConfig.env (persistent_env) to reach
    the verifier's LiteLLM client - and the sandbox guard must accept it.
    """
    task = PreparedTask(
        task_id="task",
        runtime_sha256="a" * 64,
        path=str(declared_judge_bundle),
        workload="native",
        backend="daytona",
        reason=None,
        judge_model=None,
    )
    persistent = trial_persistent_environment(task)
    assert persistent == {"TOGETHER_AI_API_BASE": "http://127.0.0.1:8787/v1"}
    validate_task_environment(persistent, protected_values=frozenset({"resolved-together_api_key"}))


def test_plain_task_carries_no_persistent_environment():
    task = PreparedTask(
        task_id="task",
        runtime_sha256="a" * 64,
        path=None,
        workload="native",
        backend="local",
        reason=None,
        judge_model=None,
    )
    assert trial_persistent_environment(task) == {}


def test_source_has_judge_traffic_includes_declared_judges(declared_judge_bundle):
    declared = PreparedTask(
        task_id="declared",
        runtime_sha256="a" * 64,
        path=str(declared_judge_bundle),
        workload="native",
        backend="daytona",
        reason=None,
        judge_model=None,
    )
    plain = PreparedTask(
        task_id="plain",
        runtime_sha256="a" * 64,
        path=None,
        workload="native",
        backend="local",
        reason=None,
        judge_model=None,
    )
    prepared_judge = plain.model_copy(update={"judge_model": "zai-org/GLM-5.3"})
    assert source_has_judge_traffic([declared]) is True
    assert source_has_judge_traffic([plain]) is False
    assert source_has_judge_traffic([prepared_judge]) is True


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


@pytest.fixture
def shared_toolchain_lookup(monkeypatch):
    lookups = (difficulty_trials.declared_grader_machine_image, toolchain_binding.resolved_toolchain_reference)
    for lookup in lookups:
        lookup.cache_clear()
    monkeypatch.setattr(toolchain_binding, "built_image", lambda environment: None)
    yield toolchain_binding
    for lookup in lookups:
        lookup.cache_clear()


def test_frozen_lock_resolves_to_the_declared_toolchain_image(monkeypatch, shared_toolchain_lookup):

    lean_lock = "# via -r /var/folders/x/tmpABC/requirements.in\npandas==2.2.2\n"
    monkeypatch.setattr(shared_toolchain_lookup, "StoragePath", _fake_storage_path({"s3://fixture/lock": lean_lock}))
    monkeypatch.setattr(
        shared_toolchain_lookup,
        "declared_lock",
        lambda environment: "pandas==2.2.2\n" if environment == LEAN_GRADER_PACKAGES else None,
    )
    image = difficulty_trials.declared_grader_machine_image("s3://fixture/lock")
    assert image == image_tag(DEFAULT_REPOSITORY, identity_digest(LEAN_GRADER_PACKAGES))


def test_unmatched_lock_keeps_the_existing_lock_built_machine(monkeypatch, shared_toolchain_lookup):
    monkeypatch.setattr(
        shared_toolchain_lookup, "StoragePath", _fake_storage_path({"s3://fixture/lock": "pandas==2.2.2\n"})
    )
    monkeypatch.setattr(shared_toolchain_lookup, "declared_lock", lambda environment: "sympy==1.13.3\n")
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
