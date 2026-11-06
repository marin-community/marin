# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


# Lowering rejects image-free Linux process requirements with ValueError: an unsupported
# task contract needs repair, not a retry on the same Iris host.

# resources.inference.ContextExceeded is a harness context-limit outcome, not an effect
# retry or hold: the same full conversation cannot succeed on another provider attempt.


class Transient(Exception):
    """Retry this effect inside its executor."""


class Restart(Exception):
    """The caller may rerun the function that exhausted its attempts."""


class Hold(Exception):
    """Stop the workpiece until a later relaunch."""


class LogFailure(Exception):
    """Event log encoding, decoding, or persistence failed."""
