"""Run original Harbor trials through the engine's exact model-request contract."""

import asyncio
import hashlib
import json
import math
import shlex
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.trial.result import TrialResult
from harbor.trial.hooks import TrialEvent, TrialHookEvent
from harbor.trial.trial import Trial
from rolloutengine.harness.core import HarnessContext, ModelSession
from rolloutengine.harness.external import AgentCompletion, complete_agent_prompt
from rolloutengine.models import ContextRecovery, LimitReached, ModelTurn, Prompt, RolloutEnded, RolloutLimits
from taskcompendium.models import TaskSpec
from skyrl_gym.task_records import StepReward
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome

from skyrl_train.rollouts.agent_backend import AgentGenerationStopped, AgentInferenceBackend
from skyrl_train.rollouts.agent_trace import recorded_completion
from skyrl_train.rollouts.harbor_settings import HARBOR_EXCEPTION
from skyrl_train.rollouts.training_record import TrainingRollout


def trial_grade(result: TrialResult, reward_key: str) -> GradeResult:
    """Preserve available scores and original failures for the training error policy."""
    diagnostics = {"harbor_trial_uri": result.trial_uri}
    if result.exception_info is not None:
        diagnostics[HARBOR_EXCEPTION] = result.exception_info.exception_type
    verifier = result.verifier_result
    if verifier is None or verifier.rewards is None or reward_key not in verifier.rewards:
        return GradeResult(
            Outcome.INFRA_ERROR,
            None,
            "Harbor trial has no verifier score",
            diagnostics=diagnostics,
            failure=GradingFailure.MISSING_REWARD,
        )
    score = float(verifier.rewards[reward_key])
    if not math.isfinite(score) or not 0 <= score <= 1:
        return GradeResult(
            Outcome.INFRA_ERROR,
            None,
            "Harbor verifier score is not finite and normalized",
            diagnostics=diagnostics,
            failure=GradingFailure.INVALID_REWARD,
        )
    return GradeResult(Outcome.GRADED, score, passed=score > 0, diagnostics=diagnostics)


async def run_task_setup(
    environment: BaseEnvironment,
    commands: tuple[str, ...],
    *,
    cwd: str | None,
    timeout: float,
    evidence_dir: Path,
) -> None:
    """Run explicitly declared setup as trusted root before model access."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command_sha256": [hashlib.sha256(command.encode()).hexdigest() for command in commands],
        "completed": False,
    }
    try:
        async with asyncio.timeout(timeout):
            for index, command in enumerate(commands):
                result = await environment.exec(
                    f"exec /bin/sh -c {shlex.quote(command)}", cwd=cwd, user="root", timeout_sec=math.ceil(timeout)
                )
                (evidence_dir / f"{index}.stdout").write_text(result.stdout or "")
                (evidence_dir / f"{index}.stderr").write_text(result.stderr or "")
                if result.return_code != 0:
                    raise RuntimeError(f"Declared task setup command {index} exited {result.return_code}")
        receipt["completed"] = True
    except BaseException as error:
        receipt["exception_type"] = type(error).__name__
        raise
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        (evidence_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


async def run_harbor_trial(
    trial: Trial,
    backend: AgentInferenceBackend,
    complete: AgentCompletion,
    evidence_dir: Path,
    *,
    setup_commands: tuple[str, ...],
    setup_cwd: str | None,
    setup_timeout: float,
) -> TrialResult:
    """Let Harbor own setup, verification and cleanup while the engine serves tokens."""
    if not isinstance(trial.agent, BaseInstalledAgent):
        raise ValueError("Harbor training requires an installed agent with rollout capture")
    correlation_id = trial.agent.rollout_correlation_id
    if correlation_id is None:
        raise ValueError("Harbor training requires rollout capture to be enabled")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    if setup_commands:

        async def prepare_task(_event: TrialHookEvent) -> None:
            await run_task_setup(
                trial.agent_environment,
                setup_commands,
                cwd=setup_cwd,
                timeout=setup_timeout,
                evidence_dir=evidence_dir / "task-setup",
            )

        trial.add_hook(TrialEvent.ENVIRONMENT_READY, prepare_task)
    trace_path = evidence_dir / "turns.jsonl"
    with backend.session(correlation_id, recorded_completion(complete, trace_path)):
        return await trial.run()


class HarborTrialHarness:
    """Let Harbor finish verification after the model reaches its rollout limit."""

    context_recovery = ContextRecovery.AFTER_TURN

    def __init__(self, run_trial: Callable[[AgentCompletion], Awaitable[TrialResult]], reward_key: str) -> None:
        self.run_trial = run_trial
        self.reward_key = reward_key
        self._grade: GradeResult | None = None
        self._model_error: RolloutEnded | None = None

    def with_verifier_result(self, rollout: TrainingRollout) -> TrainingRollout:
        """Keep the completed trial's verdict even when it produced no model turn."""
        if self._grade is None or self._model_error is not None:
            return rollout
        # A completed Harbor trial can time out before any response. The generic
        # engine's early-harness-return failure is not a verifier failure then.
        failure = rollout.failure if rollout.response_token_ids else None
        # Zero model actions have an empty exact logprob vector, not missing
        # behavior evidence. Real model failures remain on the path above.
        logprobs = rollout.logprobs if rollout.response_token_ids else ()
        return replace(rollout, grade=self._grade, failure=failure, logprobs=logprobs)

    def check(self, task: TaskSpec, limits: RolloutLimits) -> None:
        pass

    async def run(self, context: HarnessContext, session: ModelSession) -> None:
        self._grade = None
        self._model_error = None
        last: tuple[Prompt, ModelTurn] | None = None
        stopped: str | None = None
        lock = asyncio.Lock()

        async def complete(prompt: Prompt) -> ModelTurn:
            nonlocal last, stopped
            async with lock:
                if stopped is not None:
                    raise AgentGenerationStopped(stopped)
                try:
                    turn = await complete_agent_prompt(
                        session, prompt, last, repeat_stall_batches=context.limits.repeat_stall_batches
                    )
                except LimitReached as error:
                    stopped = error.stop_reason
                    raise AgentGenerationStopped(stopped) from error
                except RolloutEnded as error:
                    stopped = "error"
                    self._model_error = error
                    raise AgentGenerationStopped(stopped) from error
                last = (prompt, turn)
                return turn

        result = await self.run_trial(complete)
        self._grade = trial_grade(result, self.reward_key)
        if last is not None:
            session.record(data=StepReward(grade=self._grade))
        if self._model_error is not None:
            raise self._model_error
