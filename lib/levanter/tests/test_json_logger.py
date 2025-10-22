# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from io import StringIO

import jax.numpy as jnp

from levanter.tracker.json_logger import JsonLoggerTracker


def test_json_logger_tracker_logs_and_finishes():
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("test_json_logger")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    tracker = JsonLoggerTracker(logger)
    tracker.log({"a": 1, "b": jnp.array(2)}, step=1)
    tracker.log_summary({"c": 3})
    tracker.finish()

    logs = [json.loads(l) for l in stream.getvalue().strip().splitlines()]
    assert logs[0]["event"] == "log"
    assert logs[0]["metrics"]["a"] == 1
    assert logs[-1]["event"] == "finish"
    assert logs[-1]["summary"]["a"] == 1
    assert logs[-1]["summary"]["c"] == 3
