# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import logging
import sys

import rigging.log_setup as log_setup
from rigging.log_setup import configure_logging


def test_stderr_handler_follows_stream_swaps():
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    saved_configured = log_setup._configured
    try:
        first = io.StringIO()
        sys.stderr = first
        try:
            configure_logging(level=logging.INFO)
        finally:
            sys.stderr = sys.__stderr__
        first.close()

        second = io.StringIO()
        sys.stderr = second
        try:
            logging.getLogger("rigging.log_setup_test").warning("stale stderr probe")
        finally:
            sys.stderr = sys.__stderr__

        text = second.getvalue()
        assert "--- Logging error ---" not in text
        assert "stale stderr probe" in text
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        log_setup._configured = saved_configured
