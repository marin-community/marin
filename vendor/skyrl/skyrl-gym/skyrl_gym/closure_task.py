"""One-turn structure completion from exact model-token prefixes."""

from dataclasses import asdict, replace
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from verifyit.closure_scoring import (
    CLOSURE_REGION_SCORER,
    SNOWBALL_TURN_END_MARKERS,
    ClosureFormat,
    closure_response_content,
    closure_state,
    score_closure,
    score_closure_region,
)
from rolloutengine.models import ModelTurn, Prompt
from rolloutengine.models import TaskRolloutSpec
from verifyit.structure_closure import PrefixStatus
from shellbox.machine import Machine
from taskcompendium.grading_result import GradeResult, Outcome

from skyrl_gym.task_records import TaskTransition, episode_prompt
from skyrl_gym.source_task import session_parameters

STRUCTURE_CLOSURE_SESSION = "structure_closure"
class ClosurePrompt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_token_ids: tuple[Annotated[int, Field(strict=True, ge=0)], ...] = Field(min_length=1)
    format: ClosureFormat
    family: str = Field(min_length=1)
    control: str = Field(min_length=1)
    prefix: str
    expected: str

    @model_validator(mode="after")
    def valid_reference(self) -> "ClosurePrompt":
        if not self.expected.startswith(self.prefix):
            raise ValueError("Closure reference must extend the supplied structure prefix")
        if closure_state(self.prefix, self.format).status != PrefixStatus.EXTENDABLE:
            raise ValueError("Closure task requires an unfinished, extendable prefix")
        if not score_closure(self.expected, self.expected, self.format).success:
            raise ValueError("Closure reference must be complete and valid")
        return self


class ClosureReplaySession:
    """Grade syntax and content separately, without executing the described tools."""

    def __init__(self, lowered: TaskRolloutSpec, _machine: Machine | None):
        self.task = lowered.task
        self.prompt = ClosurePrompt.model_validate(session_parameters(self.task).extras)
        self.result = GradeResult(Outcome.UNAVAILABLE, None, "No model continuation was completed")

    async def prepare(self) -> Prompt:
        return replace(
            episode_prompt(self.task),
            token_ids=self.prompt.prompt_token_ids,
            options={"skip_special_tokens": False},
        )

    async def advance(self, turn: ModelTurn) -> TaskTransition:
        content = closure_response_content(
            turn.text, stop_reason=turn.stop_reason, terminal_markers=SNOWBALL_TURN_END_MARKERS
        )
        score = score_closure_region(self.prompt.prefix + content, self.prompt.expected, self.prompt.format)
        self.result = GradeResult(
            Outcome.GRADED,
            float(score.success),
            passed=score.success,
            diagnostics={
                "closure": self.prompt.model_dump(mode="json", exclude={"prompt_token_ids"}),
                "scorer": CLOSURE_REGION_SCORER,
                **asdict(score),
                "stop_reason": turn.stop_reason,
            },
        )
        metric = f"{CLOSURE_REGION_SCORER}/{self.prompt.family}/{self.prompt.control}"
        return TaskTransition(
            done=True,
            reward=float(score.success),
            grade=self.result,
            metrics={f"{metric}/{name}": float(value) for name, value in asdict(score).items()},
        )

    async def grade(
        self, messages: tuple[dict[str, Any], ...], *, stop_reason: str, outputs: dict[str, bytes]
    ) -> GradeResult:
        return self.result

    async def close(self) -> None:
        pass
