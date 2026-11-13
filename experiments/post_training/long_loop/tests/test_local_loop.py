# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.long_loop.timing import EventLog, read_events


def test_exception_records_failed_span_without_swallowing(tmp_path):
    log = EventLog(tmp_path / "events.jsonl", "test", "sft", "1")
    with pytest.raises(RuntimeError, match="failed save"):
        with log.span("checkpoint"):
            raise RuntimeError("failed save")
    events = read_events([log.path])
    assert [e.event for e in events] == ["start", "end"]
    assert events[-1].status == "failed"
