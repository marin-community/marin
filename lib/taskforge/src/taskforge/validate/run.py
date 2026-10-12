# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One validation round of a built draft: its policy, its evidence from disk, and the stage function.

``run_validation`` is the validate stage: ``solver.run_solver``'s ``k`` trials, then the round's
evidence read back from the attempt files, summarized and written as ``calibration.json``. It
replays no controls and runs no adversaries.

There is no evidence file besides the attempt files: ``load_validation`` reconstructs a round's
``ValidationEvidence`` from ``<evidence_dir>/solver/``, so review runs offline and a summary can be
re-derived from the item directory alone. A round's evidence directory is keyed by
``trials.task_digest(draft.task)``, so evidence can never be read against a different task.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter

from taskforge.builder.run import TaskDraft
from taskforge.content_hash import digest
from taskforge.llm.policy import LLMPolicy
from taskforge.validate.attempts import trial_files
from taskforge.validate.calibration import CalibrationBand, CalibrationSummary, summarize, write_summary
from taskforge.validate.outcome import Outcome, TrialKind
from taskforge.validate.solver import ModelFactory, ValidationSite, run_solver
from taskforge.validate.trials import Deadlines, EngineSettings, RetryBackoff, task_digest

CALIBRATION_FILE = "calibration.json"


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
        """The digest of every field in its schema-typed JSON form, so ``0`` and ``0.0`` in a float field agree."""
        return digest(_VALIDATION_POLICY.dump_python(self, mode="json"))


_VALIDATION_POLICY: TypeAdapter[ValidationPolicy] = TypeAdapter(ValidationPolicy)


@dataclass(frozen=True)
class ValidationEvidence:
    """A round's solver outcomes."""

    task_digest: str
    solver: tuple[Outcome, ...]


def load_validation(draft: TaskDraft, evidence_dir: Path) -> ValidationEvidence:
    """A round's evidence from its attempt files, each trial's last attempt.

    Solver trials order by index. Indices must run from 0 without a gap, so every outcome keeps the
    trial name it ran under.

    Raises:
        ValueError: a solver index below the highest one on disk has no attempt.
    """
    solver = {int(name): files.last for name, files in trial_files(evidence_dir, TrialKind.SOLVER).items()}
    return ValidationEvidence(
        task_digest=task_digest(draft.task), solver=_by_index(TrialKind.SOLVER, solver, evidence_dir)
    )


async def run_validation(
    draft: TaskDraft, policy: ValidationPolicy, site: ValidationSite, engine: EngineSettings, models: ModelFactory
) -> CalibrationSummary:
    """Run the round's solver trials, then summarize its evidence into ``site.evidence_dir / CALIBRATION_FILE``.

    Settled trials load from their attempt files (``run_solver``), so a re-run records no new trial.
    """
    await run_solver(draft, policy, site, engine, models)
    summary = summarize(load_validation(draft, site.evidence_dir), policy)
    write_summary(site.evidence_dir / CALIBRATION_FILE, summary)
    return summary


def _by_index[T](prefix: str, trials: Mapping[int, T], evidence_dir: Path) -> tuple[T, ...]:
    missing = [f"{prefix}/{index}" for index in range(max(trials, default=-1) + 1) if index not in trials]
    if missing:
        raise ValueError(f"Trials {missing} have no attempt under {evidence_dir}")
    return tuple(trials[index] for index in range(len(trials)))
