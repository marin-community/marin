# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Final

from taskforge.model.workpiece import Conversation


class EffectIssue(Exception):
    """A typed execution decision with provider retry history for the event log."""

    def __init__(self, message: str, *, retries: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.retries = retries


class Transient(EffectIssue):
    """Retry this effect inside its executor; interrupted startup restarts its stage."""


class Restart(EffectIssue):
    """Run an interrupted stage again on a fresh machine; its cached effects hit by their keys.

    Lost containers and failed verifier resets need replacement, never a score.
    """


class OutputBudgetExhausted(Restart):
    """Replace this agent's machine once; repeated exhaustion is failed agent evidence."""

    session: Conversation | None = None  # the conversation of the agentic session whose command exhausted the budget


class Hold(EffectIssue):
    """Publish held until explicit resume; exhausted resources require human review.

    Refusals by the host hold; defects in authored work go back to the author for repair, not an effect retry.
    """


class DecisionModelUnavailable(Hold):
    """The decision model answered no read for ``decision_model.unavailable_wait``; defer until it answers again.

    Rationale: the decision model is one preemptible service that can be down for hours. Waiting longer holds a
    driver slot, and failing or scoring would discard authored work or invent evidence. So the read ends in this
    hold, never ``TimeoutError`` (which RolloutEngine reports as a verifier deadline); siblings finish, resolved
    effects stay journalled, and the hold names what it ``awaits`` so the driver resumes it once the model answers.
    """

    awaits: Final = "decision_model"


class RequestRejected(Hold):
    """The model server refused a solver slot's request; a retry would be refused again.

    A refusal says nothing about the task or the model's ability, so the slot holds as missing evidence instead of
    counting as a failed trial; the batch records it as that slot's result (``SLOT_LOCAL`` in ``resources.rollouts``).
    """


class LogFailure(EffectIssue):
    """Event log encoding, decoding, or persistence failed."""


class Superseded(LogFailure):
    """Another writer took the published journal; stop and publish nothing.

    A relaunch, a racing driver or a preempted worker's replacement owns the workpiece now. Holding would
    publish a terminal status over the owner's journal, so the process exits with an error instead.
    """


class UnsupportedFormat(LogFailure):
    """The root holds another journal format; leave it untouched for a migration, then rerun the proposal."""


class CorruptJournal(LogFailure):
    """The committed journal does not decode; resuming would misread or drop history, so nothing is claimed.

    Relaunching fails the same way; an operator moves the root aside and reruns the proposal.
    """


class PublicationFailed(LogFailure):
    """Storage refused a publication, such as by denied access, or the final one ran out of retries.

    The writer publishes nothing more and the head keeps its last checkpoint. A hold would need the same storage, so
    the process fails; a relaunch replays the committed history and re-executes the effects after it.
    """


class ContextExceeded(Exception):
    """The provider rejected the conversation because its context window is full.

    Not an effect decision: retrying or replacing the machine cannot shrink the conversation, so the agent harness's
    inference handler records it as a ``FinishReason.CONTEXT_EXCEEDED`` completion that ends the visit with a
    context-limit stop and replays on resume. Any other caller (the judge) lets it propagate unrecorded, so an
    unfinished grade asks the provider again. ``retries`` names the transport attempts that preceded the refusal.
    """

    def __init__(self, message: str, *, retries: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.retries = retries


def describe_exception(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__
