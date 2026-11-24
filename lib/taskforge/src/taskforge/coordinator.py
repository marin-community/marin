# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one workpiece through author, validate and review until it is terminal; resume at its last stage boundary.

The author submits, validation gathers evidence for that submission, and review decides. A ``Modify``
decision starts the next round with a fresh author session; ``Retry`` validates the same submission
again. An ``Accept`` goes on to the confirmation step, whose record review decides once more: a final
acceptance, or a final-review ``Revise`` (``budget.final_revisions``), which starts a round as ``Modify`` does.
"""

import logging
from collections.abc import Callable
from dataclasses import replace

from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory

from taskforge.config import TaskforgeConfig, no_files
from taskforge.effects.context import Context
from taskforge.effects.events import Note, StageBoundary, StageCompleted
from taskforge.effects.issues import DecisionModelUnavailable, Hold, Restart, Transient, describe_exception
from taskforge.effects.log import EventLog
from taskforge.harness.loop import AgenticHarness
from taskforge.harness.protocol import Stopped, Submitted
from taskforge.model.content_hash import digest
from taskforge.model.llm import Message, Role
from taskforge.model.proposal import TaskProposal
from taskforge.model.review import (
    Abandon,
    Accept,
    BandOutcome,
    Decision,
    Decline,
    FindingResponse,
    Incomplete,
    Modify,
    Reject,
    Response,
    Retry,
    Revise,
)
from taskforge.model.stage import Exit, Stage
from taskforge.model.workpiece import ENTRY, Conversation, Fallback, Terminal, Workpiece
from taskforge.resources.bundle import Resources
from taskforge.resources.ids import NewId
from taskforge.stages import STAGES
from taskforge.stages.classify import boundary
from taskforge.stages.exits import Decided, Validated
from taskforge.stages.rules import author_stopped, default_review
from taskforge.stages.solve_review import finding_lines


def reasons(error: BaseException) -> tuple[str, ...]:
    if isinstance(error, BaseExceptionGroup):
        return tuple(reason for child in error.exceptions for reason in reasons(child))
    details: list[str] = []
    seen: set[int] = set()
    cause: BaseException | None = error
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        details.append(describe_exception(cause))
        cause = cause.__cause__
    return (" <- ".join(details),)


STAGE_REPLAY_ATTEMPTS = 3
"""Bound consecutive interrupted runs of one stage visit; effect executors alone own retries."""


def digest_of(fallback: Fallback | None) -> str | None:
    return None if fallback is None else fallback.submission.digest


def apply_decision(wp: Workpiece, decision: Decision, config: TaskforgeConfig) -> Workpiece:
    """Record a review decision and apply it: terminal, a revision round, or unchanged for a retry.

    A ``Modify`` past ``budget.max_modify_rounds`` becomes ``Abandon``; one keeps the fallback its ``fallback`` digest
    names. An ``Accept`` with ``restore`` accepts the fallback, which must carry that digest; it is terminal once it
    decides on a confirmation record, or when ``solver.confirmation_k`` is 0, and never at a band ``config.band``
    refuses, whatever review rule decided it: a pending acceptance may still go to its confirmation.

    Raises:
        ValueError: The decision is a terminal acceptance at a refused band, or names a fallback it cannot keep.
    """
    budget, policy = config.budget, config.policy
    if isinstance(decision, Modify) and wp.round >= budget.max_modify_rounds:
        decision = Abandon((f"budget.max_modify_rounds exhausted after {wp.round} modify rounds", *decision.findings))
    wp = replace(wp, decisions=(*wp.decisions, decision))
    match decision:
        case Accept(band, _, restore):
            if restore is not None and (
                wp.fallback is None or (digest_of(wp.fallback), wp.fallback.band) != (restore, band)
            ):
                raise ValueError(f"Cannot restore {restore} at {band}: the fallback is {wp.fallback}")
            if restore is not None and wp.fallback is not None:  # the revisions after the restored round are discarded
                after = wp.fallback.evidence
                wp = replace(
                    wp, responses=tuple(replace(r, discarded=r.discarded or r.review >= after) for r in wp.responses)
                )
            confirmed = bool(wp.evidence) and wp.evidence[-1].confirms is not None
            final = confirmed or not policy.setting("solver.confirmation_k", int)
            if final and band not in config.band.accepted:
                setting = "band.accept_too_easy is false" if band is BandOutcome.TOO_EASY else "config.band refuses it"
                raise ValueError(f"A review rule published a terminal acceptance at {band}, but {setting}")
            return replace(
                wp,  # a pending acceptance's placement is its decision's band, published only once final
                terminal=Terminal.ACCEPTED if final else None,
                submission=wp.fallback.submission if wp.fallback and restore else wp.submission,
                accepted_with=band if final and band is not BandOutcome.IN_BAND else None,
            )
        case Reject():
            return replace(wp, terminal=Terminal.REJECTED)
        case Abandon():
            return replace(wp, terminal=Terminal.ABANDONED)
        case Modify(findings, placement, _, _, review) | Revise(findings, review, placement=placement):
            if wp.session is None:
                raise ValueError("Cannot revise without an author session")
            fallback = decision.fallback if isinstance(decision, Modify) else None  # the final revision keeps none
            current = Fallback(wp.submission, placement, len(wp.evidence) - 1) if wp.submission and placement else None
            named = [option for option in (current, wp.fallback) if option and digest_of(option) == fallback]
            kept = named[0] if named else None  # the current submission over a stored one of its digest
            if digest_of(kept) != fallback:
                raise ValueError(
                    f"Cannot keep {fallback} as the fallback: it is neither the submission nor the fallback"
                )
            system = tuple(message for message in wp.session.messages if message.role == "system")
            notes = decision.notes if isinstance(decision, Revise) else ()
            detail = "\n".join(findings)
            if review:
                detail += "\n\n" + policy.prompt("author.review").render(findings=finding_lines(review))
            if notes:  # numbered after the revise findings, so ``declined`` names either kind
                detail += "\n\n" + policy.prompt("author.notes").render(notes=finding_lines(notes, len(review)))
            kind = "review" if isinstance(decision, Modify) else "the final review"
            prompt = policy.prompt("author.revision").render(kind=kind, detail=detail)
            if isinstance(decision, Revise):  # after rendering, so no template can drop it; recorded, so replay keeps it
                ordinal = sum(isinstance(d, Revise) for d in wp.decisions)
                prompt += f"\n\nThis is final revision {ordinal} of {budget.final_revisions}."
            sent = (FindingResponse(len(wp.evidence) - 1, f, Response.REQUESTED) for f in (*review, *notes))
            return replace(
                wp,
                round=wp.round + 1,
                session=Conversation((*system, Message(Role.USER, prompt)), wp.session.tool_schemas),
                fallback=kept,
                responses=(*wp.responses, *sent),
            )
        case Retry():
            return wp
        case Incomplete():
            return replace(wp, terminal=Terminal.HELD)


def answered(wp: Workpiece, declined: tuple[Decline, ...], response: Response) -> Workpiece:
    """``wp`` with each finding still requested of the author answered: declined with its reason, else ``response``."""
    reasons = {decline.finding: decline.reason for decline in declined}
    return replace(
        wp,
        responses=tuple(
            (
                r
                if r.response is not Response.REQUESTED
                else (
                    replace(r, response=Response.DECLINED, reason=reasons[r.finding])
                    if r.finding in reasons
                    else replace(r, response=response)
                )
            )
            for r in wp.responses
        ),
    )


def advance(wp: Workpiece, stage: Stage, result: Exit, config: TaskforgeConfig) -> tuple[Workpiece, Stage | None]:
    """Apply ``stage``'s exit to the workpiece and pick the next stage, or None once the workpiece is terminal.

    Raises:
        ValueError: ``result`` is not an exit ``stage`` may return.
    """
    match stage, result:
        case Stage.AUTHOR, Submitted(submission, session):
            wp = answered(wp, submission.declined, Response.REVISED)
            return replace(wp, submission=submission, session=session), Stage.VALIDATE
        case Stage.AUTHOR, Stopped():
            wp = answered(wp, (), Response.STOPPED)
            stopped = apply_decision(replace(wp, session=result.session), author_stopped(wp, result.summary), config)
            return stopped, None if stopped.terminal is not None else Stage.CONFIRM
        case Stage.VALIDATE | Stage.CONFIRM, Validated(evidence):
            return replace(wp, evidence=(*wp.evidence, evidence)), Stage.REVIEW
        case Stage.REVIEW, Decided(decision):
            wp = apply_decision(wp, decision, config)
            if wp.terminal is not None:
                return wp, None
            following = Stage.CONFIRM if isinstance(decision, Accept) else Stage.AUTHOR
            return wp, Stage.VALIDATE if isinstance(decision, Retry) else following
    raise ValueError(f"{stage} returned {type(result).__name__}, which has no transition")


def plan(proposal: TaskProposal, config: TaskforgeConfig, resources: Resources) -> str | None:
    """Everything besides the workpiece that the stages' work depends on, or None (resume no stage) when a review rule,
    workspace renderer, machine factory or role has no stable identity; the stock stages are versioned by the journal
    format, and a stock factory's settings by the policy and the rollout identity."""
    stock = all(type(f) in (IrisMachineFactory, ShellSimMachineFactory) for f in resources.factories.values())
    if config.review is not default_review or config.workspace.rendered is not no_files or not stock:
        return None
    roles = []
    for role in (config.author, config.adversary, config.solve_review):
        if type(harness := role.harness) is not AgenticHarness:  # a subclass may run otherwise
            return None
        session = (harness.max_prompt_tokens, harness.sampling, harness.nudge, harness.output_chars)
        tools = tuple((tool.schema, type(tool).__module__, type(tool).__qualname__) for tool in role.tools.values())
        roles.append((role.name, session, tools, resources.inference_for(role.name).fingerprint()))
    used = (resources.rollouts, resources.grading, resources.decision_model, resources.web)
    fingerprints = tuple(resource.fingerprint() for resource in used if resource is not None)
    settings = (config.workspace.static, config.policy, config.band, config.budget, tuple(roles))
    return digest((proposal, settings, fingerprints, sorted(resources.factories)))


async def run_workpiece(
    proposal: TaskProposal,
    config: TaskforgeConfig,
    resources: Resources,
    log: EventLog,
    *,
    install: Callable[[Context, Resources], None],
    reached: Callable[[Workpiece], None] = lambda _: None,
) -> Workpiece:
    """Resume from the log's last stage boundary when it was recorded under this plan, else start at the author.

    ``reached`` sees the workpiece each stage visit starts from, for a caller's diagnosis of an unexpected failure.
    Validation and confirmation evidence stays provisional until the review after it continues: a cut or hold
    between them runs that stage again.
    """
    ctx = Context(log=log)
    wp = Workpiece(proposal.header.id, proposal)
    stage = Stage.AUTHOR
    config_digest = digest(config.policy)
    interrupted = False
    attempt = 1  # of the executing stage visit; a failed attempt runs the same stage and round again
    if (identity := plan(proposal, config, resources)) is None:
        logging.warning(
            "Workpiece %s resumes no stage: its review rule, workspace, a machine or a role has no identity",
            wp.id,
        )
    saved = boundary(log.records)
    try:
        async with ctx.scope("workpiece"):
            install(ctx, resources)
            wp = replace(wp, id=await ctx.emit(NewId("workpiece")))
            if saved is not None and identity is not None and saved.plan == identity:
                wp, stage = saved.workpiece, saved.following
            else:  # a new plan supersedes any earlier boundary before its first stage
                await ctx.emit(StageBoundary(identity, stage, ENTRY.dump_json((wp,)).decode()))
            while True:
                reached(wp)
                try:
                    result = await STAGES[stage](ctx, wp, config, resources)
                except* (Restart, Transient) as error:
                    # Journal each attempt's cause; a hold reason names only the last one.
                    cause = f"Stage {stage} round {wp.round} attempt {attempt} of {STAGE_REPLAY_ATTEMPTS}: " + (
                        "; ".join(reasons(error))
                    )
                    logging.warning("%s", cause)
                    await ctx.emit(Note(cause))
                    if attempt >= STAGE_REPLAY_ATTEMPTS:
                        raise
                    attempt += 1  # run the interrupted stage again
                else:
                    # A completed visit resets the budget, so a same-round re-entry starts with a full one.
                    attempt = 1
                    previous_round = wp.round
                    wp, following = advance(wp, stage, result, config)
                    if wp.round != previous_round:
                        await resources.close_round()
                    await ctx.emit(StageCompleted(stage, config_digest, type(result).__name__, wp.round, wp.terminal))
                    if following is None:
                        break  # keep the terminal stage, which a hold raised while the scope exits records
                    if following is not Stage.REVIEW:
                        await ctx.check()  # the marker's observers, before the boundary moves past them
                        await ctx.emit(StageBoundary(identity, following, ENTRY.dump_json((wp,)).decode()))
                    stage = following
    except* (Hold, Restart, Transient) as error:
        interrupted = True
        # Only a hold every leaf of which is a decision-model outage resumes without a person.
        awaits = DecisionModelUnavailable.awaits if error.split(DecisionModelUnavailable)[1] is None else None
        wp = replace(wp, terminal=Terminal.HELD, decisions=(*wp.decisions, Incomplete(reasons(error), awaits=awaits)))
    finally:
        try:
            await ctx.drain()
        finally:
            await resources.close_round()
    if interrupted:
        await log.record(StageCompleted(stage, config_digest, "Held", wp.round, Terminal.HELD))
    return wp
