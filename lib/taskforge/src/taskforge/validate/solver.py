# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Solver trials of a built task at a fixed ``k``, resumable per trial.

The solver is the run's rollout model, built per trial by a ``ModelFactory`` so its calls are
recorded under the trial's ledger step and sample under the policy's ``sampling``. Every trial goes
through ``trials.run_trial`` with the draft's task, presented in its own answer format
(``TaskSpec.answer_format``). A trial already settled on disk is loaded rather than run again; an
unsettled one re-enters with its attempt numbers continuing after the files on disk
(``attempts.TrialFiles``).
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from taskforge.builder.run import TaskDraft
from taskforge.ledger.records import Ledger
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import CallLedger
from taskforge.validate.attempts import trial_files
from taskforge.validate.outcome import Outcome, TrialKind
from taskforge.validate.trials import EngineSettings, RetryBackoff, RolloutModel, SessionLimits, TrialPlan, run_trial

ModelFactory = Callable[[CallLedger, LLMPolicy], RolloutModel]
"""The rollout model for one trial, recording its model calls under the given ``CallLedger`` and
sampling under the given ``LLMPolicy``: the validation policy's ``sampling``, so the policy a run
records is the one its trials ran under."""


class TrialPolicy(Protocol):
    """The trial knobs of ``validate.run.ValidationPolicy`` that ``run_solver`` reads."""

    @property
    def k(self) -> int: ...

    @property
    def sampling(self) -> LLMPolicy: ...

    @property
    def limits(self) -> SessionLimits: ...

    @property
    def max_retries(self) -> int: ...

    @property
    def token_contract_retries(self) -> int: ...

    @property
    def retry_backoff(self) -> RetryBackoff: ...


@dataclass(frozen=True)
class ValidationSite:
    """Where one validation round of one item records itself."""

    item_id: str
    round: int
    evidence_dir: Path
    ledger: Ledger

    def call_ledger(self, kind: TrialKind, trial: str) -> CallLedger:
        """Where a trial's model calls are recorded: step ``<kind>/<trial>``, the prefix of its attempts' steps."""
        return CallLedger(ledger=self.ledger, item_id=self.item_id, round=self.round, step=f"{kind}/{trial}")


async def run_solver(
    draft: TaskDraft, policy: TrialPolicy, site: ValidationSite, settings: EngineSettings, models: ModelFactory
) -> tuple[Outcome, ...]:
    """``policy.k`` solver trials named ``str(index)``, all concurrent, through ``run_trial``.

    Trials already settled under ``site.evidence_dir/solver`` are loaded, not re-run; unsettled ones
    re-enter with ``first_attempt=TrialFiles.attempts``, so no file is overwritten. Each trial's model
    comes from ``models`` with ``site.call_ledger(SOLVER, <index>)`` and ``policy.sampling``. Returns
    one outcome per index.

    Every model is built before any trial starts, settled ones included, so a factory that raises
    (``GlmUnavailable``) fails the whole stage with no trial recorded, while a model that raises
    mid-trial is that attempt's retryable ``MODEL_UNAVAILABLE``.
    """
    kind = TrialKind.SOLVER
    names = [str(index) for index in range(policy.k)]
    trial_models = {name: models(site.call_ledger(kind, name), policy.sampling) for name in names}
    files = trial_files(site.evidence_dir, kind)

    async def trial(name: str) -> Outcome:
        existing = files.get(name)
        if existing is not None and existing.settled:
            return existing.last
        plan = TrialPlan(
            item_id=site.item_id,
            round=site.round,
            kind=kind,
            k=policy.k,
            limits=policy.limits,
            max_retries=policy.max_retries,
            token_contract_retries=policy.token_contract_retries,
            retry_backoff=policy.retry_backoff,
            evidence_dir=site.evidence_dir,
            ledger=site.ledger,
            first_attempt=0 if existing is None else existing.attempts,
        )
        return await run_trial(draft.task, plan, settings, trial_models[name], name)

    async with asyncio.TaskGroup() as group:
        runs = [group.create_task(trial(name)) for name in names]
    return tuple(run.result() for run in runs)
