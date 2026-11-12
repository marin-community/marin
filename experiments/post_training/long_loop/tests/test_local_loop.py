# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from experiments.post_training.long_loop.preflight import (
    EndpointEvidence,
    check_endpoint,
)
from experiments.post_training.long_loop.timing import EventLog, read_events


def test_multi_eos_serving_regression():
    evidence = EndpointEvidence(
        "step222", "step222", "hash", "hash", "user\n/think", "/think", {128001: "stop", 128009: "stop"}
    )
    check_endpoint(evidence)
    with pytest.raises(ValueError, match="Both EOS"):
        check_endpoint(replace(evidence, forced_stop_results={128001: "stop", 128009: "length"}))


def test_exception_records_failed_span_without_swallowing(tmp_path):
    log = EventLog(tmp_path / "events.jsonl", "test", "sft", "1")
    with pytest.raises(RuntimeError, match="failed save"):
        with log.span("checkpoint"):
            raise RuntimeError("failed save")
    events = read_events([log.path])
    assert [e.event for e in events] == ["start", "end"]
    assert events[-1].status == "failed"
