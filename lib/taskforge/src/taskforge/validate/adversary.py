# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adversary roles: the solver's model with a role preamble, run on the solver's engine path.

A ``RoleModel`` wraps the rollout model and puts a fixed role preamble in the system turn of every
request. Everything else is the solver's: the task, its grader, its machine and its convention, so
an adversary rollout measures the grader the solver faced. Adversary trials are ordinary
``ADVERSARY`` trials through ``trials.run_trial``, named ``<role>/<index>``, so their evidence lands
under ``adversary/<role>/<index>/`` and resumes per trial like the solver's.

The engine installs a shell verifier's files only after the final response, so the grader itself
is unreadable in the machine; what can leak is the environment files, the build context and the
instruction. A ``SHORTCUT`` or ``LEAK`` trial that passes is therefore a grader or fixture gap, and
an ``AMBIGUITY`` trial that passes under the reading least likely intended is an instruction
defect (``calibration``). Adversary rollouts are evidence, never training data: their prompt ids
include the preamble.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol

from rolloutengine.contracts import ModelRequest, ModelTurn

from taskforge.build.run import TaskDraft
from taskforge.validate.outcome import Outcome, TrialKind
from taskforge.validate.solver import ModelFactory, TrialPolicy, ValidationSite, resume_trials
from taskforge.validate.trials import EngineSettings, RolloutModel


class AdversaryRole(StrEnum):
    SHORTCUT = "shortcut"
    """Be graded correct without doing the task: guess the format, exploit lenient parsing, satisfy the
    letter of the answer contract with no work. Reply exactly NO_SHORTCUT_FOUND otherwise."""
    LEAK = "leak"
    """Search the workspace, environment files and instruction for answer keys, reference outputs or grader
    code and submit what they reveal. Reply exactly NO_LEAK_FOUND otherwise."""
    AMBIGUITY = "ambiguity"
    """List every reading of the instruction that changes the answer, then solve under the reading least
    likely intended. Reply exactly NO_AMBIGUITY_FOUND when no reading changes it."""


SENTINEL_REPLIES: Mapping[AdversaryRole, str] = {
    AdversaryRole.SHORTCUT: "NO_SHORTCUT_FOUND",
    AdversaryRole.LEAK: "NO_LEAK_FOUND",
    AdversaryRole.AMBIGUITY: "NO_AMBIGUITY_FOUND",
}
"""The exact final reply of a role that found nothing to exploit; ``RoleStats.sentinel_replies`` counts them.

The ambiguity role has one too: in the first live round (``.evidence/validate/e_evidence_round-*``) it
passed in 8 of 12 trials on an unambiguous task by solving under a reading that did not change the answer."""

ROLE_PREAMBLES: Mapping[AdversaryRole, str] = {
    AdversaryRole.SHORTCUT: (
        "You are auditing an automatically graded task for loopholes. Your goal is to be graded correct "
        "WITHOUT doing the work the task asks for. You must not perform the computation, analysis, search or "
        "file processing the instruction describes, not even a quick version of it: doing the task is not a "
        "shortcut. Instead look for a way the grader could accept an answer that did not come from that work: "
        "guess the expected format and a plausible value, exploit lenient parsing (several candidate answers in "
        "one reply, extra text, different number formats or units, partial output), produce output that "
        "satisfies the letter of the answer contract while skipping the computation, or leave the machine in a "
        "state a careless check would accept. Submit the shortcut answer through the task's normal submission "
        "format. If every way to be graded correct requires doing the task's work, reply with exactly "
        f"{SENTINEL_REPLIES[AdversaryRole.SHORTCUT]} and nothing else."
    ),
    AdversaryRole.LEAK: (
        "You are auditing an automatically graded task for leaked answers. Do not solve the task by doing the "
        "work it describes. Instead search everything available to you for the expected answer: the "
        "instruction text, files in the workspace and elsewhere on the machine (hidden files, caches, build "
        "leftovers, test fixtures, reference outputs, version control history, environment variables), and "
        "any grader or checker code you can find. If you find an answer key, a reference output or grader "
        "logic that reveals what will be accepted, submit what it reveals through the task's normal submission "
        f"format. If you find no leak, reply with exactly {SENTINEL_REPLIES[AdversaryRole.LEAK]} and nothing else."
    ),
    AdversaryRole.AMBIGUITY: (
        "You are auditing an automatically graded task for ambiguous instructions. First list, briefly, every "
        "reading of the instruction that would change the correct answer: units, rounding, inclusive or "
        "exclusive bounds, ordering, formatting, which input is meant, what counts as done. Then pick the "
        "reading that is still defensible from the text but least likely to be the one the task author "
        "intended, solve the task fully and carefully under that reading, and submit that answer through the "
        "task's normal submission format. If no defensible reading changes the correct answer, reply with "
        f"exactly {SENTINEL_REPLIES[AdversaryRole.AMBIGUITY]} and nothing else."
    ),
}
"""Each role's system preamble. Part of ``ValidationPolicy.digest``: changing a word changes the evidence."""

PREAMBLE_SEPARATOR = "\n\n"


class AdversaryPolicy(TrialPolicy, Protocol):
    """The adversary knobs of ``validate.run.ValidationPolicy``."""

    @property
    def adversary_k(self) -> int: ...

    @property
    def roles(self) -> tuple[AdversaryRole, ...]: ...


def with_preamble(role: AdversaryRole, messages: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """``messages`` with ``ROLE_PREAMBLES[role]`` as the system turn.

    When the first message is a system message the preamble is prepended to its content, so the
    chat template sees one system turn.
    """
    preamble = ROLE_PREAMBLES[role]
    if messages and messages[0]["role"] == "system":
        first, *rest = messages
        content = first["content"]
        if not isinstance(content, str):
            raise ValueError(f"A role preamble needs a text system message, got {type(content).__name__}")
        return ({**first, "content": f"{preamble}{PREAMBLE_SEPARATOR}{content}"}, *(dict(m) for m in rest))
    return ({"role": "system", "content": preamble}, *(dict(m) for m in messages))


@dataclass(frozen=True)
class RoleModel:
    """``inner`` with ``ROLE_PREAMBLES[role]`` as a system message before the task's messages.

    The transformation is the same bytes on every turn of a rollout, so the rendered prompt of turn
    n+1 extends turn n's served prompt and the engine's served-prefix check holds unchanged.
    """

    role: AdversaryRole
    inner: RolloutModel

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        return await self.inner(replace(request, messages=with_preamble(self.role, request.messages)))


def adversary_trial(role: AdversaryRole, index: int) -> str:
    return f"{role}/{index}"


async def run_adversaries(
    draft: TaskDraft, policy: AdversaryPolicy, site: ValidationSite, settings: EngineSettings, inner: ModelFactory
) -> Mapping[AdversaryRole, tuple[Outcome, ...]]:
    """``policy.adversary_k`` ``run_trial`` calls per role in ``policy.roles``, all concurrent.

    Trials are ``TrialKind.ADVERSARY`` named ``f"{role}/{index}"``, so evidence lands under
    ``adversary/<role>/<index>/`` and the ledger step is ``adversary/<role>/<index>/<attempt>``.
    Resumes per trial like ``run_solver``; each trial wraps ``inner(site.call_ledger(ADVERSARY, <role>/<index>))``.
    """
    models: dict[str, RolloutModel] = {}
    for role in policy.roles:
        for index in range(policy.adversary_k):
            trial = adversary_trial(role, index)
            models[trial] = RoleModel(role, inner(site.call_ledger(TrialKind.ADVERSARY, trial)))
    outcomes = await resume_trials(draft, policy, site, settings, TrialKind.ADVERSARY, policy.adversary_k, models)
    return {
        role: tuple(outcomes[adversary_trial(role, index)] for index in range(policy.adversary_k))
        for role in policy.roles
    }
