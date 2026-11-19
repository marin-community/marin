# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


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


class Hold(EffectIssue):
    """Publish held until explicit resume; exhausted resources require human review.

    Refusals by the host hold; defects in authored work go back to the author for repair, not an effect retry.
    """


class LogFailure(EffectIssue):
    """Event log encoding, decoding, or persistence failed."""


class ContextExceeded(Exception):
    """The provider rejected the conversation because its context window is full.

    Not an effect decision: retrying or replacing the machine cannot shrink the conversation, so the
    agent loop that owns it catches this and ends the session with a context-limit stop.
    """


def describe_exception(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__
