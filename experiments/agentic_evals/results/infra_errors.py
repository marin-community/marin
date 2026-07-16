"""Single source of truth for infrastructure-error classification.

Copied verbatim from OT-Agent ``database/unified_db/infra_errors.py``.

``INFRA_ERROR_TYPES`` is the set of exception types that represent INFRASTRUCTURE
failures (Daytona/sandbox/environment/verification-wrapper) rather than genuine
agent/task failures. Harbor's resume filters retry these, and the eval listener's
disk-based resume scanner counts them.

This module is dependency-free (stdlib only) so it can be imported from anywhere.
"""

from typing import Any, Dict, Mapping, Tuple

INFRA_ERROR_TYPES = {
    "DaytonaError",
    "DaytonaAuthenticationError",
    "DaytonaAuthorizationError",
    "DaytonaNotFoundError",
    "EnvironmentStartTimeoutError",
    "DaytonaRateLimitError",
    "CancelledError",
    "SandboxBuildFailedError",
    "AgentEnvironmentTimeoutError",
    "VerificationNotCompletedError",
    "TrialNotScoredError",
    "AddTestsDirError",
}


def compute_infra_error_stats(stats: Mapping[str, Any]) -> Tuple[int, Dict[str, int]]:
    n_infra = 0
    breakdown: Dict[str, int] = {}
    if not isinstance(stats, Mapping):
        return 0, {}
    evals = stats.get("evals")
    if not isinstance(evals, Mapping):
        return 0, {}
    for eval_data in evals.values():
        if not isinstance(eval_data, Mapping):
            continue
        exception_stats = eval_data.get("exception_stats")
        if not isinstance(exception_stats, Mapping):
            continue
        for exc_type, ids in exception_stats.items():
            if exc_type not in INFRA_ERROR_TYPES:
                continue
            n = len(ids) if isinstance(ids, list) else 1
            n_infra += n
            breakdown[exc_type] = breakdown.get(exc_type, 0) + n
    return n_infra, breakdown


def filter_error_type_flags() -> str:
    return " ".join(
        f"--filter-error-type {t}" for t in sorted(INFRA_ERROR_TYPES)
    )
