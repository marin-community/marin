# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The shellbox and Iris errors that say the host could not create a machine or lost one.

Shellbox raises a bare ``RuntimeError`` for several host faults, and the Iris controller refuses a
sandbox under a terminated parent job with a ``ConnectError`` whose code also covers request
errors, so both are told apart by their message text. The build (``stage40_build.infrastructure``)
and validation (``stage50_validate.classify``) read the same table here, so the two cannot drift.
"""

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from shellbox.machine import MachineTerminated

MACHINE_LOSS_MESSAGE_PREFIXES = ("Machine is closed", "ShellSim terminated")
"""Messages of the bare ``RuntimeError`` shellbox raises when a machine is gone under its caller."""
MACHINE_CREATE_MESSAGE_PREFIXES = ("Iris sandbox task failed before running",)
"""Messages of the bare ``RuntimeError`` the Iris factory raises when the sandbox task ended before it ran,
as it does when its parent job terminated."""
SUBMIT_REFUSED_MESSAGE = "Cannot submit job"
"""In the message of the controller's ``FAILED_PRECONDITION`` when it refuses a sandbox under a terminated parent job."""


def machine_lost(error: BaseException) -> bool:
    """Whether ``error`` says the machine ended under its caller: killed, expired, preempted or closed."""
    if isinstance(error, MachineTerminated):
        return True
    return isinstance(error, RuntimeError) and str(error).startswith(MACHINE_LOSS_MESSAGE_PREFIXES)


def machine_create_failed(error: BaseException) -> bool:
    """Whether ``error`` says the host could not start a sandbox, as under a terminated parent job."""
    if isinstance(error, ConnectError):
        return error.code is Code.FAILED_PRECONDITION and SUBMIT_REFUSED_MESSAGE in error.message
    return isinstance(error, RuntimeError) and str(error).startswith(MACHINE_CREATE_MESSAGE_PREFIXES)
