from enum import StrEnum


class ErrorTreatment(StrEnum):
    """How a persisted trial failure contributes to RL training."""

    MASK = "mask"
    ZERO = "zero"
    PASSTHROUGH = "passthrough"
