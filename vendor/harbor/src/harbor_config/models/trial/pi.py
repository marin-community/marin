"""Validated pre-action plan for one Pi DAgger trial.

The training orchestrator samples the plan before Harbor allocates the trial.
Pi consumes one decision for each model response, including final answers; a
response's entire multi-tool batch has the same executor.
"""

from dataclasses import dataclass
from typing import Any, Literal, Mapping
from pydantic import BaseModel, ConfigDict, StrictInt, field_validator, model_validator
from urllib.parse import urlparse


@dataclass(frozen=True)
class PiDaggerPlan:
    protocol: str
    teacher_model: str
    teacher_api_base: str
    executors: tuple[str, ...]
    trial_id: str
    policy_step: int

    def __post_init__(self) -> None:
        if self.protocol not in ("turn_mixture", "prefix_takeover"):
            raise ValueError("unsupported Pi DAgger protocol")
        if not all(
            isinstance(item, str) and item.strip()
            for item in (self.teacher_model, self.teacher_api_base, self.trial_id)
        ):
            raise ValueError(
                "Pi DAgger teacher model, endpoint and trial ID must be nonempty"
            )
        parsed = urlparse(self.teacher_api_base)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or not parsed.path.endswith("/v1")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Pi DAgger teacher_api_base must be an HTTP /v1 endpoint")
        if type(self.policy_step) is not int or self.policy_step < 0:
            raise ValueError("Pi DAgger policy_step must be a nonnegative integer")
        if (
            not isinstance(self.executors, tuple)
            or not self.executors
            or any(actor not in ("student", "teacher") for actor in self.executors)
        ):
            raise ValueError("Pi DAgger needs a nonempty per-turn student/teacher plan")
        if self.protocol == "prefix_takeover" and any(
            current == "teacher" and following == "student"
            for current, following in zip(self.executors, self.executors[1:])
        ):
            raise ValueError("Pi DAgger prefix takeover cannot switch back to student")

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> "PiDaggerPlan":
        """Reject incomplete plans before any model request or tool execution."""
        required = {
            "protocol",
            "teacher_model",
            "teacher_api_base",
            "executors",
            "trial_id",
            "policy_step",
        }
        if set(value) != required:
            raise ValueError(
                "Pi DAgger needs protocol, teacher endpoint, executors, trial_id and policy_step"
            )
        raw_executors = value["executors"]
        if not isinstance(raw_executors, list):
            raise ValueError("Pi DAgger executors must be a list")
        return cls(
            value["protocol"],
            value["teacher_model"],
            value["teacher_api_base"],
            tuple(raw_executors),
            value["trial_id"],
            value["policy_step"],
        )


@dataclass(frozen=True)
class PiCritiquePlan:
    """Bounded private teacher feedback for one student attempt and revision."""

    teacher_model: str
    teacher_api_base: str
    principle: str
    trial_id: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(item, str) and item.strip()
            for item in (
                self.teacher_model,
                self.teacher_api_base,
                self.principle,
                self.trial_id,
            )
        ):
            raise ValueError(
                "Pi critique needs a teacher model, endpoint, principle and trial ID"
            )
        parsed = urlparse(self.teacher_api_base)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or not parsed.path.endswith("/v1")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Pi critique teacher_api_base must be an HTTP /v1 endpoint"
            )

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> "PiCritiquePlan":
        """Reject missing or unreviewed feedback controls."""
        if set(value) != {"teacher_model", "teacher_api_base", "principle", "trial_id"}:
            raise ValueError(
                "Pi critique needs teacher_model, teacher_api_base, principle and trial_id"
            )
        return cls(**value)


class PiAgentConfig(BaseModel):
    """Discoverable, immutable Pi trial evidence and actor capability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["literal", "teacher_trace", "dagger", "critique_revision"]
    trial_id: str | None = None
    policy_step: StrictInt | None = None
    plan: PiDaggerPlan | None = None
    critique: PiCritiquePlan | None = None

    @field_validator("plan", mode="before")
    @classmethod
    def validate_plan(cls, value: Any) -> PiDaggerPlan | None:
        if value is None or isinstance(value, PiDaggerPlan):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("Pi DAgger plan must be a mapping")
        return PiDaggerPlan.parse(value)

    @field_validator("critique", mode="before")
    @classmethod
    def validate_critique(cls, value: Any) -> PiCritiquePlan | None:
        if value is None or isinstance(value, PiCritiquePlan):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("Pi critique plan must be a mapping")
        return PiCritiquePlan.parse(value)

    @model_validator(mode="after")
    def validate_mode(self) -> "PiAgentConfig":
        if (self.trial_id is None) != (self.policy_step is None):
            raise ValueError("Pi trial identity and policy step must be bound together")
        if self.trial_id is not None and (
            not self.trial_id
            or type(self.policy_step) is not int
            or self.policy_step < 0
        ):
            raise ValueError("Pi trial identity and policy step must be valid")
        if (
            self.plan is not None
            and self.trial_id is not None
            and (
                self.trial_id != self.plan.trial_id
                or self.policy_step != self.plan.policy_step
            )
        ):
            raise ValueError("Pi DAgger plan must match the bound trial identity")
        if (
            self.critique is not None
            and self.trial_id is not None
            and self.trial_id != self.critique.trial_id
        ):
            raise ValueError("Pi critique plan must match the bound trial identity")
        if (self.mode == "dagger") != (self.plan is not None):
            raise ValueError("Pi dagger mode requires a plan; other modes forbid one")
        if (self.mode == "critique_revision") != (self.critique is not None):
            raise ValueError(
                "Pi critique_revision mode requires critique; other modes forbid one"
            )
        return self
