"""Regression tests for the trial verifier-result invariant.

A single-step trial with verification enabled must never finalize and emit a
``result.json`` in a *silently unscored* state: ``verifier_result is None`` with
NO ``exception_info`` recorded at all (invisible in ``exception_stats`` while
also dropped from the scored denominator, since ``n_trials`` keys off
``verifier_result``, not the exception type).

Crucially, when TWO errors occur, the operative (later, terminal) error must
win — ``_record_exception`` is LAST-error-wins (commit ``7b470fd2``). When the
agent times out first (a benign ``AgentTimeoutError``, EXPECTED and meant to
fall through to scoring) and the verifier is then reached but raises a real
infra failure before producing a result (``AddTestsDirError`` — e.g. a Daytona
401 the retry couldn't ride out), that LATER real error SUPERSEDES the benign
timeout. This is the flawed_summ tb2 lesson: first-wins masked the real infra
failure as a benign timeout, so the trial was mis-classified benign and the
resume ``--filter-error-type`` re-run list never re-ran it, stalling legs at
~81% valid-complete. ``AddTestsDirError`` is a NON-benign infra error (re-run);
``AgentTimeoutError`` is benign — so the leaf MUST report ``AddTestsDirError``
to be counted and re-run. Do NOT re-flip this to preserve the earlier timeout.

The trial is still excluded from the scored denominator (verifier_result is
None) and still visible in ``exception_stats`` (under the terminal error's
type). Only when ``exception_info`` is empty does the guard stamp a marker:
``VerificationNotCompletedError`` if the verifier was reached, else the
``TrialNotScoredError`` base.

These tests drive a real :class:`~harbor.trial.trial.Trial` with a stubbed agent
that times out and a stubbed environment whose ``upload_dir`` fails (so the
verifier is invoked but raises before producing a result).
"""

import asyncio
import json
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.agent.context import AgentContext
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.trial.trial import Trial


class TimeoutAgent(BaseAgent):
    """Agent that sleeps past the (very small) agent timeout, then is cancelled.

    Mirrors the production failure: ``_execute_agent`` wraps ``run`` in
    ``asyncio.wait_for`` and raises ``AgentTimeoutError`` on timeout.
    """

    @staticmethod
    def name() -> str:
        return "timeout-agent"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        await asyncio.sleep(3600)


class UploadFailsEnvironment(BaseEnvironment):
    """Mounted env whose ``upload_dir`` fails, so the verifier raises.

    ``Verifier.verify`` uploads the tests dir before running and wraps any
    failure in ``AddTestsDirError``. This reproduces the timed-out-sandbox
    state where verification is *invoked* but cannot produce a result.
    """

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.DOCKER

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(mounted=True)

    def _validate_definition(self):
        pass

    async def start(self, force_build: bool) -> None:
        pass

    async def stop(self, delete: bool):
        pass

    async def prepare_logs_for_host(self) -> None:
        pass

    async def upload_file(self, source_path, target_path):
        pass

    async def upload_dir(self, source_dir, target_dir):
        raise RuntimeError("simulated sandbox upload failure after agent timeout")

    async def download_file(self, source_path, target_path):
        pass

    async def download_dir(self, source_dir, target_dir):
        pass

    async def exec(self, command, cwd=None, env=None, timeout_sec=None):
        pass


def _create_task_dir(root: Path) -> Path:
    """Create a minimal valid task directory with a real test.sh."""
    task_dir = root / "test-task"
    task_dir.mkdir()

    # Tiny agent timeout so TimeoutAgent.run() is cut off by wait_for.
    (task_dir / "task.toml").write_text(
        "[agent]\ntimeout_sec = 0.05\n[verifier]\ntimeout_sec = 10.0\n[environment]\n"
    )
    (task_dir / "instruction.md").write_text("Do nothing.")

    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")

    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )

    return task_dir


async def _make_trial(tmp_path: Path) -> Trial:
    """Trial with TimeoutAgent + UploadFailsEnvironment and verification ON."""
    task_dir = _create_task_dir(tmp_path)
    trials_dir = tmp_path / "trials"
    trials_dir.mkdir()

    config = TrialConfig(
        task=TaskConfig(path=task_dir),
        trials_dir=trials_dir,
        agent=AgentConfig(
            import_path="tests.unit.test_trial_verifier_invariant:TimeoutAgent"
        ),
        environment=EnvironmentConfig(
            import_path=(
                "tests.unit.test_trial_verifier_invariant:UploadFailsEnvironment"
            ),
            delete=True,
        ),
        # Verification ENABLED: this is the path that must never silently drop.
        verifier=VerifierConfig(disable=False),
    )
    trial = await Trial.create(config)
    assert isinstance(trial.agent, TimeoutAgent)
    assert isinstance(trial.agent_environment, UploadFailsEnvironment)
    return trial


class TestVerifierResultInvariant:
    """A finalized single-step trial must never be silently unscored."""

    async def test_later_infra_error_supersedes_benign_timeout(self, tmp_path: Path):
        """Agent times out first (benign ``AgentTimeoutError``), then the verifier
        is reached but raises a real infra failure (``AddTestsDirError``) before
        producing a result. Under LAST-error-wins (``7b470fd2``) the later real
        error SUPERSEDES the benign timeout, so the leaf is truthfully typed as
        the infra error — the flawed_summ fix that stops a benign timeout from
        masking the real cause and hiding it from the resume re-run filter. The
        trial is still unscored (``verifier_result is None``) and so still
        excluded from the scored denominator, but its ``exception_stats`` bucket
        carries the TERMINAL (non-benign) cause.
        """
        trial = await _make_trial(tmp_path)

        result = await trial.run()

        # The verifier never produced a result (the bug's necessary condition).
        if result.verifier_result is not None:
            raise AssertionError(
                "expected verifier_result to be None for this failure mode"
            )

        # The never-scored state must be visible (exception_info present)...
        if result.exception_info is None:
            raise AssertionError(
                "trial finalized unscored with no exception_info recorded "
                "(silently-dropped masked state)"
            )
        # ...and the later real infra error must win over the benign timeout,
        # NOT be masked by it (and NOT flattened into a VerificationNotCompleted).
        if result.exception_info.exception_type != "AddTestsDirError":
            raise AssertionError(
                "expected the later AddTestsDirError to supersede the benign "
                f"AgentTimeoutError, got {result.exception_info.exception_type}"
            )

    async def test_emitted_result_json_records_terminal_infra_error(
        self, tmp_path: Path
    ):
        """The persisted ``result.json`` (what downstream harvests) must carry
        the terminal infra error (not the masked benign timeout), and the trial
        must be finalized.
        """
        trial = await _make_trial(tmp_path)

        await trial.run()

        result_path = trial.paths.result_path
        if not result_path.exists():
            raise AssertionError("result.json was not emitted")

        emitted = json.loads(result_path.read_text())

        # Finalized (finished_at set) — the trial did complete its lifecycle.
        if emitted.get("finished_at") is None:
            raise AssertionError("emitted result.json has no finished_at")

        if emitted.get("verifier_result") is not None:
            raise AssertionError("expected verifier_result null in emitted result")

        exc = emitted.get("exception_info")
        if exc is None:
            raise AssertionError(
                "emitted result.json is silently unscored: verifier_result null "
                "with no exception_info"
            )
        if exc.get("exception_type") != "AddTestsDirError":
            raise AssertionError(
                "emitted result.json should record the terminal AddTestsDirError, "
                f"not the masked benign timeout, got {exc.get('exception_type')}"
            )
