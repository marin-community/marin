"""Stable vocabulary for the agent-loop stop reason.

Agents that own their interaction loop publish why the loop ended in
``AgentContext.metadata["stop_reason"]`` so downstream consumers can monitor
termination outcomes even for trials that still score. The values are a
published contract: consumers match on them by name, so live here (in the
runtime-free ``harbor_config`` package) rather than as bare literals in the
agent implementation.
"""

from enum import StrEnum

__all__ = ["AgentStopReason"]


class AgentStopReason(StrEnum):
    """Why an agent loop ended."""

    TASK_COMPLETE = "task_complete"
    SESSION_ENDED = "session_ended"
    EARLY_TERMINATION = "early_termination"
    TURN_CAP_EXHAUSTED = "turn_cap_exhausted"
