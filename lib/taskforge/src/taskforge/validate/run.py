# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One validation round of a built draft: its policy, and its evidence from disk.

The round runs ``solver.run_solver``; it replays no controls and runs no adversaries.

There is no evidence file besides the attempt files: ``load_validation`` reconstructs a round's
``ValidationEvidence`` from ``<evidence_dir>/solver/``, so review runs offline and a decision can be
re-derived from the item directory alone. A round's evidence directory is keyed by
``trials.task_digest(draft.lowered)``, so evidence can never be read against a different task,
lowering or answer format.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from taskforge.builder.run import TaskDraft
from taskforge.content_hash import digest
from taskforge.llm.policy import LLMPolicy
from taskforge.validate.attempts import trial_files
from taskforge.validate.calibration import CalibrationBand
from taskforge.validate.evidence import Evidence
from taskforge.validate.outcome import Outcome, TrialKind
from taskforge.validate.trials import Deadlines, RetryBackoff, task_digest


@dataclass(frozen=True)
class ValidationPolicy:
    """Every knob of one validation round; no field has a default.

    Attributes:
        k: Solver trials.
        band: Solve rates that count as calibrated.
        sampling: The solver's sampling; ``max_continuations`` must be 0.
        deadlines: The total-turn and attempt deadlines validation imposes on every trial.
        max_retries: Per-trial retries of ``RETRYABLE`` causes.
        token_contract_retries: Per-trial retries of ``TOKEN_CONTRACT``.
        retry_backoff: Wait between a trial's attempts.
    """

    k: int
    band: CalibrationBand
    sampling: LLMPolicy
    deadlines: Deadlines
    max_retries: int
    token_contract_retries: int
    retry_backoff: RetryBackoff

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("A validation policy needs k >= 1")
        if self.sampling.max_continuations != 0:
            raise ValueError("Validation rollouts cannot continue on length; set sampling.max_continuations=0")
        if self.max_retries < 0 or self.token_contract_retries < 0:
            raise ValueError("Retry counts must be non-negative")

    @property
    def digest(self) -> str:
        """The canonical digest of every field."""
        return digest(
            {
                "k": self.k,
                "band": self.band,
                "sampling": self.sampling,
                "deadlines": self.deadlines,
                "max_retries": self.max_retries,
                "token_contract_retries": self.token_contract_retries,
                "retry_backoff": self.retry_backoff,
            }
        )


@dataclass(frozen=True)
class ValidationEvidence:
    """A round's solver outcomes."""

    task_digest: str
    solver: tuple[Outcome, ...]

    def trial_evidence(self) -> Evidence:
        """The outcomes by trial kind, for status and ``RewardStats``."""
        return Evidence({TrialKind.SOLVER: self.solver})


def load_validation(draft: TaskDraft, evidence_dir: Path) -> ValidationEvidence:
    """A round's evidence from its attempt files, each trial's last attempt.

    Solver trials order by index. Indices must run from 0 without a gap, so every outcome keeps the
    trial name it ran under.

    Raises:
        ValueError: a solver index below the highest one on disk has no attempt.
    """
    solver = {int(name): _last(files.last) for name, files in trial_files(evidence_dir, TrialKind.SOLVER).items()}
    return ValidationEvidence(
        task_digest=task_digest(draft.lowered), solver=_by_index(TrialKind.SOLVER, solver, evidence_dir)
    )


def _by_index[T](prefix: str, trials: Mapping[int, T], evidence_dir: Path) -> tuple[T, ...]:
    missing = [f"{prefix}/{index}" for index in range(max(trials, default=-1) + 1) if index not in trials]
    if missing:
        raise ValueError(f"Trials {missing} have no attempt under {evidence_dir}")
    return tuple(trials[index] for index in range(len(trials)))


def _last(outcome: Outcome | None) -> Outcome:
    assert outcome is not None
    return outcome
