# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from io import StringIO

import jax
import jax.numpy as jnp
import pytest

from levanter.tracker.json_logger import JsonLoggerConfig, JsonLoggerTracker


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


@pytest.mark.parametrize("process_index", [0, 1])
def test_json_logger_config_emits_records_only_on_primary_process(monkeypatch, caplog, process_index):
    monkeypatch.setattr(jax, "process_index", lambda: process_index)
    logger_name = "test_json_logger_distributed"
    tracker = JsonLoggerConfig(logger_name=logger_name).init(run_id="distributed-run")

    with caplog.at_level(logging.INFO, logger=logger_name):
        tracker.log_hyperparameters({"batch_size": 16})
        tracker.log({"train/loss": 2.5}, step=10)
        tracker.log_summary({"tokens": 160})
        tracker.log_artifact("checkpoint", name="weights", type="model")
        tracker.finish()

    records = [json.loads(record.message) for record in caplog.records if record.name == logger_name]
    if process_index != 0:
        assert records == []
        return

    assert [record["event"] for record in records] == ["hparams", "log", "summary", "artifact", "finish"]
    assert records[0]["hparams"] == {"batch_size": 16}
    assert records[1]["step"] == 10
    assert records[1]["metrics"] == {"train/loss": 2.5}
    assert records[2]["metrics"] == {"tokens": 160}
    assert records[3]["path"] == "checkpoint"
    assert records[-1]["summary"] == {"train/loss": 2.5, "tokens": 160}
