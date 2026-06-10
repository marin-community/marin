# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import sys


def flush_debug_output(*candidate_loggers: logging.Logger) -> None:
    """Best-effort flush for debug logging and standard streams."""
    seen_handlers: set[int] = set()
    for candidate in (*candidate_loggers, logging.getLogger()):
        for handler in candidate.handlers:
            handler_id = id(handler)
            if handler_id in seen_handlers:
                continue
            seen_handlers.add(handler_id)
            try:
                handler.flush()
            except Exception:
                pass

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
