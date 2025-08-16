"""Marin RL public interface with lightweight exports."""

from .datatypes import LegacyRolloutGroup as RolloutGroup, Rollout, RolloutSink, Turn

__all__ = [
    "Rollout",
    "RolloutGroup",
    "RolloutSink",
    "Turn",
]
