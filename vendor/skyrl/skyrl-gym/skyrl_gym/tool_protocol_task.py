"""One-turn protocol diagnostics from exact, retained model-token prefixes."""

from dataclasses import replace
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field
from rolloutengine.contracts import ModelTurn, SessionStart, Transition
from rolloutengine.spec import LoweredTaskSpec
from rolloutengine.task_session import session_start
from verifyit.tool_call_markup import TOOL_CLOSE, ToolCallStatus, classify_tool_call_continuation
from shellbox.machine import Machine
from taskcompendium.grading_result import GradeResult, Outcome

from skyrl_gym.source_task import session_parameters


class ReplayPrompt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_token_ids: tuple[Annotated[int, Field(strict=True, ge=0)], ...] = Field(min_length=1)


class ToolProtocolReplaySession:
    """Score a completed tool call without executing it or scoring the BFCL answer."""

    def __init__(self, lowered: LoweredTaskSpec, _machine: Machine | None):
        self.task = lowered.task
        self.prompt = ReplayPrompt.model_validate(session_parameters(self.task).extras)
        self.result = GradeResult(Outcome.UNAVAILABLE, None, "No model continuation was completed")

    async def prepare(self) -> SessionStart:
        start = session_start(self.task, shell_tool=None, provider_tools=())
        return replace(
            start,
            prompt_token_ids=self.prompt.prompt_token_ids,
            options={"stop": [TOOL_CLOSE], "include_stop_str_in_output": True, "skip_special_tokens": False},
        )

    async def advance(self, turn: ModelTurn) -> Transition:
        status = classify_tool_call_continuation(turn.text)
        passed = status is ToolCallStatus.VALID
        self.result = GradeResult(
            Outcome.GRADED,
            float(passed),
            passed=passed,
            diagnostics={"protocol_status": status.value, "stop_reason": turn.stop_reason},
        )
        return Transition(
            done=True, reward=float(passed), grade=self.result, metrics={f"tool_protocol/{status.value}": 1.0}
        )

    async def grade(
        self, messages: tuple[dict[str, Any], ...], *, stop_reason: str, outputs: dict[str, bytes]
    ) -> GradeResult:
        return self.result

    async def close(self) -> None:
        pass
