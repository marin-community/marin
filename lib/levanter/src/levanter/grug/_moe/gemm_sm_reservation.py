# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""SMs kept free of the persistent QuACK GEMMs traced in a ``reserved_sms`` context.

QuACK's SM100 GEMMs here use the static persistent tile scheduler: the grid is one CTA per SM (as many
clusters as the device holds) and each CTA owns a fixed share of the tiles. A GEMM CTA cannot share an SM
with the ragged all-to-all device kernel's 512-thread CTAs, so the two partition SMs by launch order: a
GEMM that starts while the all-to-all holds its 32 SMs leaves those CTAs' tiles waiting until the
all-to-all ends, and an all-to-all that starts under a full-width GEMM waits for SMs. Launching the GEMMs
on ``SM count - count`` SMs lets them run side by side. The value is read when a launcher is built (at
trace time) and never changes the computed function.

This module imports nothing GPU-specific, so the EP backends can set the context on any platform.
"""

import contextlib
import contextvars

_RESERVED_SMS: contextvars.ContextVar[int] = contextvars.ContextVar("levanter_quack_reserved_sms", default=0)


@contextlib.contextmanager
def reserved_sms(count: int):
    """Trace the QuACK GEMMs in this context to leave ``count`` SMs free."""
    if count < 0:
        raise ValueError(f"reserved SM count must be non-negative, got {count}")
    token = _RESERVED_SMS.set(count)
    try:
        yield
    finally:
        _RESERVED_SMS.reset(token)


def current_reserved_sms() -> int:
    return _RESERVED_SMS.get()
