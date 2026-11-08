# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

import pytest
from marin.execution.lazy import OUT, apply

from experiments.post_training.long_loop.graph import StageBindings, build_loop, execute_local_graph
from experiments.post_training.long_loop.preflight import (
    EndpointEvidence,
    LearnerProjection,
    check_endpoint,
    check_projection,
)
from experiments.post_training.long_loop.timing import Event, EventLog, read_events, report


def write_checkpoint(*, output: str, predecessor: str | None, stage: str) -> None:
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    lineage = json.loads((Path(predecessor) / "checkpoint.json").read_text()) if predecessor else []
    if stage == "bad-handoff":
        raise RuntimeError("Payload missing")
    (path / "checkpoint.json").write_text(json.dumps([*lineage, stage]))


def bind_stage(predecessor, *, stage):
    return apply(
        f"local-loop/{stage}", write_checkpoint, version="2026.11.08", output=OUT, predecessor=predecessor, stage=stage
    )


def test_real_artifact_chain_uses_predecessors_and_reuses_commit(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    initial = apply(
        "local-loop/base", write_checkpoint, version="2026.11.08", output=OUT, predecessor=None, stage="base"
    )
    names = ["sft", "sft-handoff", "rl-check", "rl", "rl-handoff", "eval-check", "eval", "report"]
    final = build_loop(initial, StageBindings(*(partial(bind_stage, stage=name) for name in names)))
    execute_local_graph(final)
    output = Path(final.path()) / "checkpoint.json"
    assert json.loads(output.read_text()) == ["base", *names]
    first_write = output.stat().st_mtime_ns
    execute_local_graph(final)
    assert output.stat().st_mtime_ns == first_write


def test_failed_handoff_blocks_downstream(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    initial = apply(
        "local-loop/base", write_checkpoint, version="2026.11.08", output=OUT, predecessor=None, stage="base"
    )
    names = ["sft", "bad-handoff", "rl-check", "rl", "rl-handoff", "eval-check", "eval", "report"]
    final = build_loop(initial, StageBindings(*(partial(bind_stage, stage=name) for name in names)))
    with pytest.raises(RuntimeError, match=r"step\(s\) failed"):
        execute_local_graph(final)
    assert not (tmp_path / "local-loop/rl/2026.11.08/checkpoint.json").exists()


def test_earlier_action_loss_and_alignment_regression():
    projection = LearnerProjection((1, 2, 5), (0, 1, 1, 0, 0, 1), (1, 2, 5), (1, 2, 5), "v1", ("v1",) * 6, 100, 1000)
    check_projection(projection)
    with pytest.raises(ValueError, match="dropped"):
        check_projection(replace(projection, loss_mask=(0, 0, 0, 0, 0, 1)))
    with pytest.raises(ValueError, match="alignment"):
        check_projection(replace(projection, logprob_positions=(5,)))


def test_multi_eos_serving_regression():
    evidence = EndpointEvidence(
        "step222", "step222", "hash", "hash", "user\n/think", "/think", {128001: "stop", 128009: "stop"}
    )
    check_endpoint(evidence)
    with pytest.raises(ValueError, match="Both EOS"):
        check_endpoint(replace(evidence, forced_stop_results={128001: "stop", 128009: "length"}))


def event(span, kind, boundary, seconds, *, gpus=0, status=""):
    utc = (datetime(2026, 11, 8, tzinfo=UTC) + timedelta(seconds=seconds)).isoformat()
    return Event("synthetic", "rl", "1", span, "clock", kind, boundary, utc, seconds, gpus=gpus, status=status)


def test_overlap_report_preserves_open_and_failed_spans():
    events = [
        event("a", "allocation", "start", 0, gpus=128),
        event("a", "allocation", "end", 20, gpus=128),
        event("b", "allocation", "start", 10, gpus=128),
        event("b", "allocation", "end", 30, gpus=128),
        event("c", "generation", "start", 10),
        event("c", "generation", "end", 15, status="failed"),
        event("d", "generation", "start", 20),
    ]
    rendered = report(events, synthetic=True)
    assert "observed peak: 256 GPUs" in rendered
    assert "allocation: 30.000 seconds" in rendered
    assert "Incomplete spans: 1; failed spans: 1" in rendered
    assert "| rl | allocation | 2 | 40.000" in rendered


def test_exception_records_failed_span_without_swallowing(tmp_path):
    log = EventLog(tmp_path / "events.jsonl", "test", "sft", "1")
    with pytest.raises(RuntimeError, match="failed save"):
        with log.span("checkpoint"):
            raise RuntimeError("failed save")
    events = read_events([log.path])
    assert [e.event for e in events] == ["start", "end"]
    assert events[-1].status == "failed"
