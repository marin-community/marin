# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Formatting helpers for process failures."""

import signal
import traceback


def signal_name(signum: int) -> str:
    """Return a canonical signal name, or a numeric fallback."""
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


def format_exception_with_traceback(exc: Exception) -> str:
    """Format an exception and its traceback for a persisted task error."""
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return f"{type(exc).__name__}: {exc}\n\nTraceback:\n{trace}"
