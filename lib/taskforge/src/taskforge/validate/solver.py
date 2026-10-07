# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Solver trials of a built task at a fixed ``k``, resumable per trial.

The solver is the run's rollout model (``llm.rollout_model.GlmRolloutModel``) under the
validation policy's sampling, built per trial by a ``ModelFactory`` so its calls are recorded
under the trial's ledger step. Every trial goes through ``trials.run_trial`` with the draft's task
and execution, under ``EngineSettings`` whose conventions are pinned to the draft's own
convention, so a draft is validated with the presentation its controls were authored for. A trial
already settled on disk is loaded rather than run again; an unsettled one re-enters with its
attempt numbers continuing after the files on disk (``attempts.TrialFiles``).
"""

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from taskforge.build.run import TaskDraft
from taskforge.ledger.records import Ledger
from taskforge.llm.recording import CallLedger
from taskforge.validate.attempts import trial_files
from taskforge.validate.controls import ControlPlan
from taskforge.validate.outcome import Outcome, TrialKind
from taskforge.validate.trials import Deadlines, EngineSettings, RetryBackoff, RolloutModel, TrialPlan, run_trial

ModelFactory = Callable[[CallLedger], RolloutModel]
"""The rollout model for one trial, recording its model calls under the given ``CallLedger``.

For GLM this is ``functools.partial(GlmRolloutModel, client, sampling)``.
"""


class TrialPolicy(Protocol):
    """The trial knobs of ``validate.run.ValidationPolicy`` that solver and control plans read."""

    @property
    def k(self) -> int: ...

    @property
    def deadlines(self) -> Deadlines: ...

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

    def trial_plan(self, kind: TrialKind, k: int, policy: TrialPolicy, first_attempt: int) -> TrialPlan:
        return TrialPlan(
            item_id=self.item_id,
            round=self.round,
            kind=kind,
            k=k,
            deadlines=policy.deadlines,
            max_retries=policy.max_retries,
            token_contract_retries=policy.token_contract_retries,
            retry_backoff=policy.retry_backoff.schedule(),
            evidence_dir=self.evidence_dir,
            ledger=self.ledger,
            first_attempt=first_attempt,
        )

    def control_plan(self, policy: TrialPolicy, first_attempts: Mapping[str, int]) -> ControlPlan:
        return ControlPlan(
            item_id=self.item_id,
            round=self.round,
            deadlines=policy.deadlines,
            max_retries=policy.max_retries,
            retry_backoff=policy.retry_backoff.schedule(),
            evidence_dir=self.evidence_dir,
            ledger=self.ledger,
            first_attempts=first_attempts,
        )


def draft_settings(draft: TaskDraft, settings: EngineSettings) -> EngineSettings:
    """``settings`` with the draft's own convention as the only one, as every validation trial runs."""
    return replace(settings, conventions=(draft.convention,))


async def resume_trials(
    draft: TaskDraft,
    policy: TrialPolicy,
    site: ValidationSite,
    settings: EngineSettings,
    kind: TrialKind,
    k: int,
    models: Mapping[str, RolloutModel],
) -> dict[str, Outcome]:
    """Run each trial named in ``models`` with its model, concurrently; return outcomes by trial name.

    A trial settled under ``site.evidence_dir/<kind>`` is loaded, not run. An unsettled one runs with
    ``first_attempt`` set to the number of its attempt files, so no file is overwritten.
    """
    settings = draft_settings(draft, settings)
    files = trial_files(site.evidence_dir, kind)

    async def trial(name: str, model: RolloutModel) -> Outcome:
        existing = files.get(name)
        if existing is not None and existing.settled:
            assert existing.last is not None
            return existing.last
        plan = site.trial_plan(kind, k, policy, 0 if existing is None else existing.attempts)
        return await run_trial(draft.task, draft.execution, plan, settings, model, name)

    async with asyncio.TaskGroup() as group:
        runs = {name: group.create_task(trial(name, model)) for name, model in models.items()}
    return {name: run.result() for name, run in runs.items()}


async def run_solver(
    draft: TaskDraft, policy: TrialPolicy, site: ValidationSite, settings: EngineSettings, models: ModelFactory
) -> tuple[Outcome, ...]:
    """``policy.k`` solver trials named ``str(index)``, all concurrent, through ``run_trial``.

    Trials already settled under ``site.evidence_dir/solver`` are loaded, not re-run; unsettled ones
    re-enter with ``first_attempt=TrialFiles.attempts``. Each trial's model comes from ``models`` with
    ``site.call_ledger(SOLVER, <index>)``. Returns one outcome per index.
    """
    names = [str(index) for index in range(policy.k)]
    trial_models = {name: models(site.call_ledger(TrialKind.SOLVER, name)) for name in names}
    outcomes = await resume_trials(draft, policy, site, settings, TrialKind.SOLVER, policy.k, trial_models)
    return tuple(outcomes[name] for name in names)
