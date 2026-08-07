# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One table format for ablation outcomes, shared by the local sweep and the hero sweep."""

from levanter.recovery.types import RunResult

_HEADER = f"{'arm':<30} {'outcome':<14} {'attempts':>8} {'faults':<28} {'final_step':>11} {'wall_s':>9}"


def format_run_summary(rows: list[tuple[str, RunResult]]) -> str:
    """Render ``(arm name, result)`` pairs as a fixed-width table.

    Faults carry their return code because the code is what separates a deadman abort from an
    out-of-memory kill, and those are different outcomes for an ablation rather than degrees of
    the same one.
    """
    lines = [_HEADER, "-" * len(_HEADER)]
    for name, result in rows:
        faults = ",".join(f"{f.fault_class.value}:{f.returncode}" for f in result.faults) or "-"
        lines.append(
            f"{name:<30} {result.outcome.value:<14} {result.attempts:>8} {faults:<28} "
            f"{result.final_step!s:>11} {result.total_wall_time:>9.1f}"
        )
    return "\n".join(lines)
