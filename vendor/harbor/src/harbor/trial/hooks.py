from datetime import datetime, timezone
from enum import Enum
from typing import Awaitable, Callable

from pydantic import BaseModel, Field

from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult


class TrialEvent(Enum):
    """Events in a trial's lifecycle."""

    START = "start"
    ENVIRONMENT_START = "environment_start"
    AGENT_START = "agent_start"
    VERIFICATION_START = "verification_start"
    VERIFICATION_END = "verification_end"
    RESULT_PERSISTENCE_START = "result_persistence_start"
    END = "end"
    CANCEL = "cancel"


class TrialHookEvent(BaseModel):
    """
    Unified event object passed to all trial lifecycle hooks.

    Provides context about the trial at the time of the event.
    The `result` field is populated for RESULT_PERSISTENCE_START and END events.
    """

    model_config = {"arbitrary_types_allowed": True}

    event: TrialEvent
    trial_id: str
    task_name: str
    config: TrialConfig
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    result: TrialResult | None = None


HookCallback = Callable[["TrialHookEvent"], Awaitable[None]]


async def emit_trial_hooks(
    hooks: list[HookCallback],
    *,
    event: TrialEvent,
    trial_id: str,
    task_name: str,
    config: TrialConfig,
    result: TrialResult,
) -> None:
    hook_event = TrialHookEvent(
        event=event,
        trial_id=trial_id,
        task_name=task_name,
        config=config,
        result=(
            result
            if event in {TrialEvent.RESULT_PERSISTENCE_START, TrialEvent.END}
            else None
        ),
    )
    for hook in hooks:
        await hook(hook_event)
