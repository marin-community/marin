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

    Host lowering refusals hold; InvalidTaskSpec is authored evidence returned for repair, not an effect retry.
    """


class DecisionModelUnavailable(Hold):
    """The decision model stayed unavailable for ``decision_model.unavailable_wait``; hold until it answers.

    Rationale: the decision model is one preemptible pod that can be down for hours. Waiting longer would fill
    the driver's slots with blocked workpieces, and failing would discard authored work, so the workpiece is held
    (its resolved effects stay in the log) and the driver resumes it once a decision-model read succeeds.
    """


class LogFailure(EffectIssue):
    """Event log encoding, decoding, or persistence failed."""


def describe_exception(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__
