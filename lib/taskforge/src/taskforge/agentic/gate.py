# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The evidence rules that a submitted task must satisfy. Code applies them, so the agent cannot accept its own task.

``submit`` also refuses a task that the tools cannot run or that leaks its answer, and it asks an independent
reviewer last.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Gate:
    """The evidence that a task needs before ``submit`` accepts it."""

    min_solver_runs: int
    """Solver rollouts that a task needs, so that its pass rate is a measurement."""
    solver_band: tuple[float, float]
    """The lowest and the highest permitted solver pass rate: a task outside it is broken, too hard or too easy."""
    max_format_failure_rate: float
    """The highest permitted share of solver attempts whose answer the answer format rejects."""
    min_adversary_runs: int
    """Adversary rollouts that a task needs. The review decides whether an adversary pass used a shortcut."""


@dataclass
class Evidence:
    """What the tools recorded for one version of a task, by the digest of its ``TaskSpec``."""

    controls: list[tuple[bool, bool]] = field(default_factory=list)
    """For each graded control answer: whether it must pass, and whether it passed."""
    solver_runs: int = 0
    """Solver rollouts with a graded answer. An answer that the answer format rejects is a format failure instead."""
    solver_passes: int = 0
    solver_format_failures: int = 0
    adversary_runs: int = 0
    adversary_passes: int = 0

    @property
    def solver_pass_rate(self) -> float:
        return self.solver_passes / self.solver_runs if self.solver_runs else 0.0


def missing_evidence(evidence: Evidence, gate: Gate) -> list[str]:
    """The names of the rules that ``evidence`` does not satisfy. An empty list accepts the task."""
    low, high = gate.solver_band
    rules = {
        # A wrong answer that passes shows a weak grader. A right answer that fails is often a wrong control of the
        # agent's own, so it does not block, and the solver rollouts and the review still test the grader.
        "controls": (
            any(expect and passed for expect, passed in evidence.controls)
            and any(not expect and not passed for expect, passed in evidence.controls)
            and not any(not expect and passed for expect, passed in evidence.controls)
        ),
        # A format failure tells nothing about the skill, so the band counts only the graded answers.
        "solver band": evidence.solver_runs >= gate.min_solver_runs and low <= evidence.solver_pass_rate <= high,
        "answer format": (
            evidence.solver_format_failures
            <= gate.max_format_failure_rate * (evidence.solver_runs + evidence.solver_format_failures)
        ),
        # An adversary that solves the task by the work also passes, so the review, not a count, judges the passes.
        "adversary": evidence.adversary_runs >= gate.min_adversary_runs,
    }
    return [name for name, satisfied in rules.items() if not satisfied]
