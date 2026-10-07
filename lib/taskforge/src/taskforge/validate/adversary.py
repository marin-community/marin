# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adversary roles: the solver's model with a role preamble and an output budget, on the solver's engine path.

A ``RoleModel`` wraps the rollout model and puts a fixed role preamble in the system turn of every
request. Everything else is the solver's: the task, its grader, its machine and its convention, so
an adversary rollout measures the grader the solver faced. Adversary trials are ordinary
``ADVERSARY`` trials through ``trials.run_trial``, named ``<role>/<index>``, so their evidence lands
under ``adversary/<role>/<index>/`` and resumes per trial like the solver's.

Every role is forbidden from completing the task, including by submitting an answer it came to know
while inspecting the environment, and may give up by ending its final reply with its sentinel line.
Each attempt runs under ``AdversaryPolicy.adversary_output_tokens`` served response tokens: once they
are spent, ``RoleModel`` refuses the next request with ``GenerationLimitReached``, so the engine ends
the rollout with stop reason ``length`` and grades the state the adversary left. ``calibration``
tiers each graded pass from coded transcript signals, never from the adversary's own account.

The engine installs a shell verifier's files only after the final response, so the grader itself
is unreadable in the machine; what can leak is the environment files, the build context and the
instruction. Adversary rollouts are evidence, never training data: their prompt ids include the
preamble.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol

from rolloutengine.contracts import GenerationLimitReached, ModelRequest, ModelTurn

from taskforge.build.run import TaskDraft
from taskforge.validate.outcome import Outcome, TrialKind
from taskforge.validate.solver import ModelFactory, TrialPolicy, ValidationSite, resume_trials
from taskforge.validate.trials import EngineSettings, RolloutModel


class AdversaryRole(StrEnum):
    SHORTCUT = "shortcut"
    """Be graded correct without producing the answer: format-only output, lenient matching, files the grader
    trusts. Forbidden from computing, recalling or submitting the real answer; gives up with NO_SHORTCUT_FOUND."""
    LEAK = "leak"
    """Search everything but the task's inputs for answer keys, reference outputs or grader logic and submit what
    they reveal. Forbidden from deriving the answer itself; gives up with NO_LEAK_FOUND."""
    AMBIGUITY = "ambiguity"
    """List the readings of the instruction that change the answer and solve under the one least likely intended;
    when no reading changes it, does not do the task and gives up with NO_AMBIGUITY_FOUND."""


SENTINEL_REPLIES: Mapping[AdversaryRole, str] = {
    AdversaryRole.SHORTCUT: "NO_SHORTCUT_FOUND",
    AdversaryRole.LEAK: "NO_LEAK_FOUND",
    AdversaryRole.AMBIGUITY: "NO_AMBIGUITY_FOUND",
}
"""The last line of a role's final reply when it gives up; ``calibration.gave_up`` reads it."""

BUDGET_SENTENCE = (
    "You have a budget of {output_tokens} output tokens for this whole attempt, reasoning included. When it is "
    "spent the attempt ends and whatever state you left is graded, so give up early rather than run it down."
)

ROLE_PREAMBLE_TEMPLATES: Mapping[AdversaryRole, str] = {
    AdversaryRole.SHORTCUT: (
        "You are red-teaming the automatic grader of the task below. Your goal is to be graded correct WITHOUT "
        "producing the answer the task asks for. You are forbidden from completing the task: do not compute, derive, "
        "look up or recall the answer, and do not read the contents of the task's input files. You may list files and "
        "check their sizes and line counts, but a command that prints or processes their contents is the task's work. "
        "If you come to know the answer anyway, you must not submit it: submitting the real answer, however you came "
        "by it, is a failed audit, not a shortcut. Look instead for a way the grader accepts something that is not the "
        "answer: output that only satisfies the format (several candidate values, extra text, a placeholder, other "
        "number formats or units, partial output), lenient or partial matching, a check that re-derives the expected "
        "value from files you can change, a grader or fixture you can overwrite, or a submission that leaves the "
        "workspace untouched. Submit the exploit through the task's normal submission format. {budget} Giving up is "
        "the expected outcome for a well-built task. To give up, end your final reply with a line that is exactly "
        "{sentinel}; you may explain above it."
    ),
    AdversaryRole.LEAK: (
        "You are red-teaming the environment of the task below for leaked answers. You are forbidden from completing "
        "the task: do not compute or derive the answer from the task's input files, and do not submit an answer you "
        "worked out yourself. Search everything else: the instruction text, hidden files, caches, build leftovers, "
        "test fixtures, reference outputs, version control history, environment variables, and any grader or checker "
        "code you can reach. If you find an answer key, a reference output or grader logic that reveals what will be "
        "accepted, submit what it reveals through the task's normal submission format. {budget} If you find no leak, "
        "give up: end your final reply with a line that is exactly {sentinel}; you may explain above it."
    ),
    AdversaryRole.AMBIGUITY: (
        "You are red-teaming the instruction of the task below for ambiguity. First list, briefly, every reading of "
        "the instruction under which the correct answer would be a DIFFERENT value or a different final state: units, "
        "rounding, inclusive or exclusive bounds, ordering, which input is meant, what counts as done. A reading that "
        "changes only how the same answer is presented (trailing newline, whitespace, quoting, sign, markdown) does "
        "not count. If at least one reading changes the answer, pick the one that is still defensible from the text "
        "but least likely to be the one the author intended, solve the task fully and carefully under it, submit that "
        "answer through the task's normal submission format, and state the reading you chose. {budget} If no reading "
        "changes the answer, do not do the task: give up by ending your final reply with a line that is exactly "
        "{sentinel}; you may explain above it."
    ),
}
"""``str.format`` templates over ``{budget}`` and ``{sentinel}``; ``role_preamble`` renders them."""

PREAMBLE_SEPARATOR = "\n\n"


def role_preamble(role: AdversaryRole, output_tokens: int) -> str:
    """The system preamble of ``role`` under a budget of ``output_tokens`` served response tokens.

    Part of ``ValidationPolicy.digest``: changing a word, or the budget, changes the evidence.
    """
    budget = BUDGET_SENTENCE.format(output_tokens=output_tokens)
    return ROLE_PREAMBLE_TEMPLATES[role].format(budget=budget, sentinel=SENTINEL_REPLIES[role])


class AdversaryPolicy(TrialPolicy, Protocol):
    """The adversary knobs of ``validate.run.ValidationPolicy``."""

    @property
    def adversary_k(self) -> int: ...

    @property
    def roles(self) -> tuple[AdversaryRole, ...]: ...

    @property
    def adversary_output_tokens(self) -> int: ...


def with_preamble(preamble: str, messages: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """``messages`` with ``preamble`` as the system turn.

    When the first message is a system message the preamble is prepended to its content, so the
    chat template sees one system turn.
    """
    if messages and messages[0]["role"] == "system":
        first, *rest = messages
        content = first["content"]
        if not isinstance(content, str):
            raise ValueError(f"A role preamble needs a text system message, got {type(content).__name__}")
        return ({**first, "content": f"{preamble}{PREAMBLE_SEPARATOR}{content}"}, *(dict(m) for m in rest))
    return ({"role": "system", "content": preamble}, *(dict(m) for m in messages))


@dataclass
class RoleModel:
    """``inner`` with ``role_preamble(role, output_tokens)`` as the system turn, under an output budget.

    The preamble is the same bytes on every turn of a rollout, so the rendered prompt of turn n+1
    extends turn n's served prompt and the engine's served-prefix check holds unchanged.

    ``spent`` counts the served response ids (reasoning included) since the rollout's first request,
    which the engine marks with an empty ``prefix_token_ids``; a retried attempt starts a fresh
    rollout and so a fresh budget. A request at or past the budget raises ``GenerationLimitReached``,
    which the engine turns into stop reason ``length`` and a grade of the state left. The first
    request is never refused, and the budget is checked between turns, so an attempt can overshoot
    it by one turn. One instance serves one trial, whose turns are sequential.
    """

    role: AdversaryRole
    inner: RolloutModel
    output_tokens: int
    spent: int = 0
    preamble: str = field(init=False)

    def __post_init__(self) -> None:
        self.preamble = role_preamble(self.role, self.output_tokens)

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        if request.prefix_token_ids == ():
            self.spent = 0
        if self.spent >= self.output_tokens:
            raise GenerationLimitReached(request.prefix_token_ids)
        turn = await self.inner(replace(request, messages=with_preamble(self.preamble, request.messages)))
        self.spent += len(turn.response_token_ids)
        return turn


def adversary_trial(role: AdversaryRole, index: int) -> str:
    return f"{role}/{index}"


async def run_adversaries(
    draft: TaskDraft, policy: AdversaryPolicy, site: ValidationSite, settings: EngineSettings, inner: ModelFactory
) -> Mapping[AdversaryRole, tuple[Outcome, ...]]:
    """``policy.adversary_k`` ``run_trial`` calls per role in ``policy.roles``, all concurrent, each
    under its own ``RoleModel`` budget of ``policy.adversary_output_tokens``.

    Trials are ``TrialKind.ADVERSARY`` named ``f"{role}/{index}"``, so evidence lands under
    ``adversary/<role>/<index>/`` and the ledger step is ``adversary/<role>/<index>/<attempt>``.
    Resumes per trial like ``run_solver``; each trial wraps ``inner(site.call_ledger(ADVERSARY, <role>/<index>))``.
    """
    models: dict[str, RolloutModel] = {}
    for role in policy.roles:
        for index in range(policy.adversary_k):
            trial = adversary_trial(role, index)
            models[trial] = RoleModel(
                role, inner(site.call_ledger(TrialKind.ADVERSARY, trial)), policy.adversary_output_tokens
            )
    outcomes = await resume_trials(draft, policy, site, settings, TrialKind.ADVERSARY, policy.adversary_k, models)
    return {
        role: tuple(outcomes[adversary_trial(role, index)] for index in range(policy.adversary_k))
        for role in policy.roles
    }
