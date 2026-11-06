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


class Hold(EffectIssue):
    """Publish held until explicit resume; exhausted resources require human review."""


class LogFailure(EffectIssue):
    """Event log encoding, decoding, or persistence failed."""
