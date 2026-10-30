"""Run original Harbor trials through the engine's exact model-request contract."""

import asyncio
import math
from collections.abc import Awaitable, Callable
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from rolloutengine.harness.core import HarnessContext, ModelSession
from rolloutengine.harness.external import AgentCompletion
from rolloutengine.models import LimitReached, ModelTurn, Prompt, RolloutEnded, RolloutLimits
from taskcompendium.models import TaskSpec
from skyrl_gym.task_records import StepReward
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome

from skyrl_train.rollouts.agent_backend import AgentGenerationStopped, AgentInferenceBackend
from skyrl_train.rollouts.agent_trace import recorded_completion
from skyrl_train.rollouts.harbor_settings import HARBOR_EXCEPTION


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


async def run_harbor_trial(
    trial: Trial,
    backend: AgentInferenceBackend,
    complete: AgentCompletion,
    evidence_dir: Path,
) -> TrialResult:
    """Let Harbor own setup, verification and cleanup while the engine serves tokens."""
    if not isinstance(trial.agent, BaseInstalledAgent):
        raise ValueError("Harbor training requires an installed agent with rollout capture")
    correlation_id = trial.agent.rollout_correlation_id
    if correlation_id is None:
        raise ValueError("Harbor training requires rollout capture to be enabled")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    trace_path = evidence_dir / "turns.jsonl"
    with backend.session(correlation_id, recorded_completion(complete, trace_path)):
        return await trial.run()


class HarborTrialHarness:
    """Let Harbor finish verification after the model reaches its rollout limit."""

    def __init__(self, run_trial: Callable[[AgentCompletion], Awaitable[TrialResult]], reward_key: str) -> None:
        self.run_trial = run_trial
        self.reward_key = reward_key

    def check(self, task: TaskSpec, limits: RolloutLimits) -> None:
        pass

    async def run(self, context: HarnessContext, session: ModelSession) -> None:
        last: tuple[Prompt, ModelTurn] | None = None
        stopped: str | None = None
        model_error: RolloutEnded | None = None
        lock = asyncio.Lock()

        async def complete(prompt: Prompt) -> ModelTurn:
            nonlocal last, stopped, model_error
            async with lock:
                if stopped is not None:
                    raise AgentGenerationStopped(stopped)
                if last is not None and prompt == last[0]:
                    return last[1]
                history = () if last is None else (*last[0].messages, last[1].message)
                try:
                    if (
                        last is not None
                        and prompt.options == last[0].options
                        and prompt.messages[: len(history)] == history
                    ):
                        turn = await session.reply(prompt.messages[len(history) :], token_ids=prompt.token_ids)
                    else:
                        turn = await session.start(prompt)
                except LimitReached as error:
                    stopped = error.stop_reason
                    raise AgentGenerationStopped(stopped) from error
                except RolloutEnded as error:
                    stopped = "error"
                    model_error = error
                    raise AgentGenerationStopped(stopped) from error
                last = (prompt, turn)
                return turn

        result = await self.run_trial(complete)
        if last is not None:
            session.record(data=StepReward(grade=trial_grade(result, self.reward_key)))
        if model_error is not None:
            raise model_error
