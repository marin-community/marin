# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from taskforge.model.workpiece import Conversation


class EffectIssue(Exception):
    """A typed execution decision with provider retry history for the event log."""

    def __init__(self, message: str, *, retries: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.retries = retries


class Transient(EffectIssue):
    """Retry this effect inside its executor; interrupted startup replays its stage."""


class Restart(EffectIssue):
    """Reconstruct an interrupted stage on a fresh machine, replaying resolved effects.

    Lost containers and failed verifier resets need replacement, never a score.
    """


class OutputBudgetExhausted(Restart):
    """Replace this agent's machine once; repeated exhaustion is failed agent evidence."""

    session: Conversation | None = None  # the conversation of the agentic session whose command exhausted the budget


class Hold(EffectIssue):
    """Publish held until explicit resume; exhausted resources require human review.

    Refusals by the host hold; defects in authored work go back to the author for repair, not an effect retry.
    """


class Unreproducible(Hold):
    """A re-executed effect failed on a fresh machine where its recording succeeded.

    The machine cannot be rebuilt to the recorded history, and every later recorded result would describe state that
    does not exist, so the workpiece holds for a person instead of continuing on it.
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

    Not an effect decision: retrying or replacing the machine cannot shrink the conversation, so the
    agent loop that owns it catches this and ends the session with a context-limit stop.
    """


def describe_exception(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__
