"""Run original Harbor trials through the engine's exact model-request contract."""

import asyncio
import math
from collections.abc import Awaitable, Callable
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from rolloutengine.agent_session import AgentCompletion, AgentTaskSession
from rolloutengine.contracts import ModelTurn, SessionStart
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


class HarborTrialSession(AgentTaskSession):
    """Expose a Harbor trial's model requests and final verifier result to the engine."""

    def __init__(self, run_trial: Callable[[AgentCompletion], Awaitable[TrialResult]], reward_key: str) -> None:
        result: TrialResult | None = None
        generation_stopped = asyncio.Event()
        generation_stop_reason = ""

        async def run(complete: AgentCompletion) -> None:
            nonlocal result

            async def bounded_complete(request: SessionStart) -> ModelTurn:
                if generation_stopped.is_set():
                    raise AgentGenerationStopped(generation_stop_reason)
                response = asyncio.create_task(complete(request))
                stopped = asyncio.create_task(generation_stopped.wait())
                try:
                    done, _ = await asyncio.wait((response, stopped), return_when=asyncio.FIRST_COMPLETED)
                    if response in done:
                        return response.result()
                    raise AgentGenerationStopped(generation_stop_reason)
                finally:
                    response.cancel()
                    stopped.cancel()
                    await asyncio.gather(response, stopped, return_exceptions=True)

            result = await run_trial(bounded_complete)

        async def grade(messages, *, stop_reason, outputs) -> GradeResult:
            nonlocal generation_stop_reason
            if result is None and self.agent is not None and not self.agent.done():
                generation_stop_reason = stop_reason
                generation_stopped.set()
                # The engine's verifier deadline bounds this wait. Harbor owns
                # agent exit, verification, and environment teardown.
                await asyncio.shield(self.agent)
            if result is None:
                return GradeResult(Outcome.UNAVAILABLE, None, "Harbor trial did not return a result")
            return trial_grade(result, reward_key)

        super().__init__(run, grade)
