import asyncio


class AgentSetupTimeoutError(asyncio.TimeoutError):
    pass


class AgentTimeoutError(asyncio.TimeoutError):
    pass


class AgentDiskFullError(RuntimeError):
    """The sandbox ran out of disk space during agent execution."""


class VerifierTimeoutError(asyncio.TimeoutError):
    pass


class EnvironmentStartTimeoutError(asyncio.TimeoutError):
    pass


class TrialTimeoutError(asyncio.TimeoutError):
    """An admitted trial attempt exceeded its whole-lifecycle deadline."""

    pass


class PhaseCancellationError(RuntimeError):
    """A timed-out trial phase did not terminate during cleanup grace."""

    pass


class TrialNotScoredError(Exception):
    """Base: a single-step verification-enabled trial finalized without a
    ``verifier_result``.

    This is the catch-all "never scored" marker, stamped only when no
    informative upstream ``exception_info`` was already recorded (an earlier
    failure — e.g. an ``EnvironmentStartTimeoutError`` from environment start —
    is preserved as-is, NOT overwritten). It guarantees a single-step,
    verification-enabled trial can never finalize *silently unscored*: a missing
    ``verifier_result`` with an empty ``exception_info`` would otherwise be
    invisible in ``exception_stats`` while also being dropped from the scored
    denominator (``n_trials`` keys off ``verifier_result``, not the exception
    type). ``VerificationNotCompletedError`` is the narrow subtype for when the
    verifier stage was actually reached.

    Note: the timeout classes above deliberately do NOT subclass this — being
    "not scored" is a property of ``verifier_result``, not of any timeout.
    """

    pass


class VerificationNotCompletedError(TrialNotScoredError):
    """Narrow case: the verifier stage WAS reached but produced no
    ``verifier_result``.

    Stamped only when the trial actually entered the verify step (``run``'s
    ``reached_verifier`` flag is True) and finalized with ``verifier_result is
    None`` and no prior ``exception_info`` — i.e. the verifier was invoked but
    was cut off (cancellation) or raised before producing a result. Trials that
    fell through earlier (e.g. environment-start failure) keep their original,
    informative ``exception_info`` instead; silent early fall-throughs without a
    prior error get the ``TrialNotScoredError`` base rather than this misleading
    "verification" label. (Re-port of 9203989f onto the #1672 split flow.)
    """

    pass
