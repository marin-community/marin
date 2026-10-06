# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from io import StringIO

import jax.numpy as jnp
import pytest

from levanter.tracker import json_logger
from levanter.tracker.tracker import CompositeTracker, FatalTrackerError
from levanter.tracker.json_logger import JsonLoggerConfig

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


def test_durable_events_are_rank_zero_only_and_keep_distinct_runs(tmp_path, monkeypatch):
    destination = str(tmp_path / "metrics")
    monkeypatch.setattr(json_logger.jax, "process_index", lambda: 1)
    JsonLoggerConfig(metric_destination=destination).init("rank-one").log({"loss": 3}, step=0)
    assert not (tmp_path / "metrics").exists()
    monkeypatch.setattr(json_logger.jax, "process_index", lambda: 0)
    for run in ("first", "second"):
        JsonLoggerConfig(metric_destination=destination).init(run).log({"loss": 3}, step=0)
    events = [json.loads(path.read_bytes()) for path in (tmp_path / "metrics").glob("*.json")]
    assert {event["run_id"] for event in events} == {"first", "second"}


def test_durable_storage_failure_reaches_training_caller(tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    tracker = JsonLoggerConfig(metric_destination=str(blocked / "metrics")).init("failed-storage")
    with pytest.raises(FatalTrackerError) as failure:
        CompositeTracker([tracker]).log({"loss": 1}, step=0)
    assert isinstance(failure.value.__cause__, OSError)
