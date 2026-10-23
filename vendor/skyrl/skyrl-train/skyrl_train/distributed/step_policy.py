from dataclasses import dataclass
from enum import StrEnum


class NonfiniteStepPolicy(StrEnum):
    SKIP = "skip"
    FAIL = "fail"


@dataclass(frozen=True)
class OptimizerStepResult:
    grad_norm: float | None
    applied: bool


def nonfinite_step_policy(consecutive_skipped: int, limit: int | None) -> NonfiniteStepPolicy:
    """Skip a non-finite step within the allowance, or fail when it is exhausted."""
    if limit is None or consecutive_skipped >= limit:
        return NonfiniteStepPolicy.FAIL
    return NonfiniteStepPolicy.SKIP
