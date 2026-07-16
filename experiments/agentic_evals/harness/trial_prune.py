"""Prune infra-errored trial dirs so a subsequent ``harbor jobs start`` AUTO-RESUME re-runs them.

Extracted from OT-Agent ``hpc/harbor_utils.py``. The ``upath`` +
``harbor.models.trial.*`` imports are kept lazy (inside the function body).
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple


def prune_refire_errored_trials(
    run_dir_uri: str,
    filter_error_types: List[str],
    *,
    log_prefix: str = "",
) -> Tuple[int, Dict[str, int]]:
    """Delete infra-errored trial dirs from a Harbor run dir.

    Returns ``(n_pruned, {exception_type: count})``.
    """
    if not filter_error_types:
        return 0, {}
    try:
        from upath import UPath
        from harbor.models.trial.paths import TrialPaths
        from harbor.models.trial.result import TrialResult
        from harbor.utils.path_compat import safe_rmtree
    except ImportError as exc:
        print(
            f"{log_prefix}[refire] WARNING: harbor/upath not importable ({exc}); "
            "skipping errored-trial prune.",
            file=sys.stderr, flush=True,
        )
        return 0, {}

    filter_set = set(filter_error_types)
    job_dir = UPath(run_dir_uri)
    try:
        if not job_dir.exists():
            print(
                f"{log_prefix}[refire] no existing run dir at {run_dir_uri}; "
                "nothing to prune (fresh launch).",
                flush=True,
            )
            return 0, {}
    except Exception as exc:
        print(
            f"{log_prefix}[refire] WARNING: could not stat {run_dir_uri} "
            f"({type(exc).__name__}: {exc}); skipping prune.",
            file=sys.stderr, flush=True,
        )
        return 0, {}

    n_pruned = 0
    breakdown: Dict[str, int] = {}
    for trial_dir in job_dir.iterdir():
        try:
            if not trial_dir.is_dir():
                continue
            result_path = TrialPaths(trial_dir).result_path
            if not result_path.exists():
                continue
            try:
                trial_result = TrialResult.model_validate_json(result_path.read_text())
            except Exception as exc:
                print(
                    f"{log_prefix}[refire] WARNING: unreadable result for "
                    f"{trial_dir.name} ({type(exc).__name__}); leaving in place.",
                    file=sys.stderr, flush=True,
                )
                continue
            exc_info = trial_result.exception_info
            if exc_info is not None and exc_info.exception_type in filter_set:
                safe_rmtree(trial_dir)
                n_pruned += 1
                breakdown[exc_info.exception_type] = (
                    breakdown.get(exc_info.exception_type, 0) + 1
                )
        except Exception as exc:
            print(
                f"{log_prefix}[refire] WARNING: error handling {trial_dir} "
                f"({type(exc).__name__}: {exc}); skipping.",
                file=sys.stderr, flush=True,
            )
            continue

    if n_pruned:
        print(
            f"{log_prefix}[refire] pruned {n_pruned} infra-errored trial dir(s) "
            f"from {run_dir_uri}: {breakdown}",
            flush=True,
        )
    else:
        print(
            f"{log_prefix}[refire] no trials matched the infra filter "
            f"{sorted(filter_set)} in {run_dir_uri}; nothing to re-run.",
            flush=True,
        )
    return n_pruned, breakdown
