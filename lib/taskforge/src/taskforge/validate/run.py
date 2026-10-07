# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One validation round of a built draft: its policy, controls first, and its evidence from disk.

The caller drives the order: ``replay_controls`` first; only when every control is ``MET``
(``controls_passed``) do ``solver.run_solver`` and ``adversary.run_adversaries`` run, concurrently
with each other. A violated or ungraded control goes to review with partial evidence, because
rollouts against a wrong grader are evidence about the wrong grader.

There is no evidence file besides the attempt files: ``load_validation`` reconstructs a round's
``ValidationEvidence`` from ``<evidence_dir>/{control,solver,adversary}/``, so review runs offline
and a decision can be re-derived from the item directory alone. A round's evidence directory is
keyed by ``trials.task_digest(draft.task, draft.execution, draft.convention)``, so evidence can never
be read against a different task, execution or convention.
"""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from taskcompendium.models import AssistantToolCalls, TextMessage

from taskforge.build.run import TaskDraft
from taskforge.canonical import digest
from taskforge.llm.policy import LLMPolicy
from taskforge.spec.controls import Control, validate_controls
from taskforge.validate.adversary import SENTINEL_REPLIES, AdversaryRole, role_preamble
from taskforge.validate.attempts import trial_files
from taskforge.validate.calibration import CalibrationBand, TaskFacts, task_facts
from taskforge.validate.controls import (
    ControlOutcome,
    ControlVerdict,
    ScriptedModel,
    Tokenize,
    control_outcome,
    control_turns,
)
from taskforge.validate.evidence import Evidence
from taskforge.validate.outcome import Outcome, TrialKind
from taskforge.validate.solver import ValidationSite, draft_settings
from taskforge.validate.trials import Deadlines, EngineSettings, RetryBackoff, run_trial, task_digest


@dataclass(frozen=True)
class ValidationPolicy:
    """Every knob of one validation round; no field has a default.

    Attributes:
        k: Solver trials.
        adversary_k: Trials per adversary role.
        adversary_output_tokens: Served response tokens, reasoning included, each adversary attempt may spend
            before it ends with stop reason ``length``.
        roles: The adversary roles to run.
        band: Solve rates that count as calibrated.
        sampling: The solver's and adversaries' sampling; ``max_continuations`` must be 0.
        deadlines: The agent and attempt deadlines validation imposes on every trial.
        max_retries: Per-trial retries of ``RETRYABLE`` causes.
        token_contract_retries: Per-trial retries of ``TOKEN_CONTRACT``.
        retry_backoff: Wait between a trial's attempts.
    """

    k: int
    adversary_k: int
    adversary_output_tokens: int
    roles: tuple[AdversaryRole, ...]
    band: CalibrationBand
    sampling: LLMPolicy
    deadlines: Deadlines
    max_retries: int
    token_contract_retries: int
    retry_backoff: RetryBackoff

    def __post_init__(self) -> None:
        if self.k < 1 or self.adversary_k < 1:
            raise ValueError("A validation policy needs k >= 1 and adversary_k >= 1")
        if self.adversary_output_tokens < 1:
            raise ValueError(f"adversary_output_tokens must be >= 1, got {self.adversary_output_tokens}")
        if len(set(self.roles)) != len(self.roles):
            raise ValueError(f"Adversary roles repeat: {self.roles}")
        if self.sampling.max_continuations != 0:
            raise ValueError("Validation rollouts cannot continue on length; set sampling.max_continuations=0")
        if self.max_retries < 0 or self.token_contract_retries < 0:
            raise ValueError("Retry counts must be non-negative")

    @property
    def digest(self) -> str:
        """The canonical digest of every field, the role preambles and the sentinel replies."""
        return digest(
            {
                "k": self.k,
                "adversary_k": self.adversary_k,
                "adversary_output_tokens": self.adversary_output_tokens,
                "roles": self.roles,
                "band": self.band,
                "sampling": self.sampling,
                "deadlines": self.deadlines,
                "max_retries": self.max_retries,
                "token_contract_retries": self.token_contract_retries,
                "retry_backoff": self.retry_backoff,
                "preambles": {str(role): role_preamble(role, self.adversary_output_tokens) for role in self.roles},
                "sentinels": {str(role): reply for role, reply in SENTINEL_REPLIES.items()},
            }
        )


@dataclass(frozen=True)
class ValidationEvidence:
    """A round's outcomes and the task facts the adversary signals read.

    ``solver`` and ``adversaries`` are empty when the controls short-circuited.
    """

    task_digest: str
    controls: tuple[ControlOutcome, ...]
    solver: tuple[Outcome, ...]
    adversaries: Mapping[AdversaryRole, tuple[Outcome, ...]]
    facts: TaskFacts

    def trial_evidence(self) -> Evidence:
        """The outcomes by trial kind, for status and ``RewardStats``."""
        return Evidence(
            {
                TrialKind.CONTROL: tuple(c.outcome for c in self.controls),
                TrialKind.SOLVER: self.solver,
                TrialKind.ADVERSARY: tuple(o for outcomes in self.adversaries.values() for o in outcomes),
            }
        )


def controls_passed(controls: Sequence[ControlOutcome]) -> bool:
    """Whether every control's verdict is ``MET``."""
    return all(c.verdict is ControlVerdict.MET for c in controls)


async def replay_controls(
    draft: TaskDraft,
    policy: ValidationPolicy,
    site: ValidationSite,
    settings: EngineSettings,
    tokenize: Tokenize,
) -> tuple[ControlOutcome, ...]:
    """Replay every control of ``draft`` concurrently as a ``CONTROL`` trial named by its id.

    A control settled under ``site.evidence_dir/control`` is loaded, not replayed; an unsettled one
    re-enters with its attempt numbers after the files on disk.

    Raises:
        ValueError: the draft is staged, or a control needs more turns than ``settings.max_turns``.
    """
    task = draft.task
    if task.stages:
        raise ValueError("Control replay does not support staged tasks")
    validate_controls(task, draft.controls)
    too_long = [c.id for c in draft.controls if len(control_turns(c)) > settings.max_turns]
    if too_long:
        raise ValueError(f"Controls {too_long} need more than max_turns={settings.max_turns} turns")
    context_turns = sum(
        isinstance(event, AssistantToolCalls) or (isinstance(event, TextMessage) and event.role == "assistant")
        for event in task.context.events
    )
    settings = draft_settings(draft, settings)
    files = trial_files(site.evidence_dir, TrialKind.CONTROL)
    unsettled = [c.id for c in draft.controls if c.id in files and not files[c.id].settled]
    plan = site.control_plan(policy, {cid: files[cid].attempts for cid in unsettled})

    async def replay_one(control: Control) -> ControlOutcome:
        existing = files.get(control.id)
        if existing is not None and existing.settled:
            assert existing.last is not None
            return control_outcome(control, existing.last)
        model = ScriptedModel(control_turns(control), context_turns, tokenize)
        return control_outcome(
            control, await run_trial(task, draft.execution, plan.trial_plan(control.id), settings, model, control.id)
        )

    async with asyncio.TaskGroup() as group:
        runs = [group.create_task(replay_one(control)) for control in draft.controls]
    return tuple(run.result() for run in runs)


def load_validation(draft: TaskDraft, evidence_dir: Path) -> ValidationEvidence:
    """A round's evidence from its attempt files, each trial's last attempt, with ``task_facts(draft.task)``.

    Controls pair by id with ``draft.controls``; solver trials order by index; adversary trials group
    by role directory, in ``AdversaryRole`` order. Indices must run from 0 without a gap, so every
    outcome keeps the trial name it ran under.

    Raises:
        ValueError: a control of ``draft`` has no attempt file, or a solver or adversary index below the
            highest one on disk has none.
    """
    controls = trial_files(evidence_dir, TrialKind.CONTROL)
    missing = [c.id for c in draft.controls if c.id not in controls]
    if missing:
        raise ValueError(f"Controls {missing} have no attempt under {evidence_dir}")
    solver = {int(name): _last(files.last) for name, files in trial_files(evidence_dir, TrialKind.SOLVER).items()}
    adversaries: dict[AdversaryRole, dict[int, Outcome]] = {}
    for name, files in trial_files(evidence_dir, TrialKind.ADVERSARY).items():
        role, index = name.split("/")
        assert files.last is not None
        adversaries.setdefault(AdversaryRole(role), {})[int(index)] = files.last
    return ValidationEvidence(
        task_digest=task_digest(draft.task, draft.execution, draft.convention),
        controls=tuple(control_outcome(c, _last(controls[c.id].last)) for c in draft.controls),
        solver=_by_index("solver", solver, evidence_dir),
        adversaries={
            role: _by_index(f"adversary/{role}", adversaries[role], evidence_dir)
            for role in AdversaryRole
            if role in adversaries
        },
        facts=task_facts(draft.task),
    )


def _by_index(prefix: str, trials: Mapping[int, Outcome], evidence_dir: Path) -> tuple[Outcome, ...]:
    missing = [f"{prefix}/{index}" for index in range(max(trials, default=-1) + 1) if index not in trials]
    if missing:
        raise ValueError(f"Trials {missing} have no attempt under {evidence_dir}")
    return tuple(trials[index] for index in range(len(trials)))


def _last(outcome: Outcome | None) -> Outcome:
    assert outcome is not None
    return outcome
