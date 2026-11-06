# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


class Transient(Exception):
    """Retry this effect inside its executor."""


class Restart(Exception):
    """The caller may rerun the function that exhausted its attempts."""


class Hold(Exception):
    """Stop the workpiece until a later relaunch."""


class LogFailure(Exception):
    """Event log encoding, decoding, or persistence failed."""
