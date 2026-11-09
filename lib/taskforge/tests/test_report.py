# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest
from rolloutengine.models import ConversationRecord, GradedRollout, ModelTurn, Rollout, RolloutStep, Transition
from rolloutengine.test_support import arithmetic_task
from taskcompendium.grading_result import GradeResult, Outcome

from taskforge.effects.events import Event, Resolved, ScopeEntered, ScopeExited, StageCompleted
from taskforge.effects.log import JsonlSink, Record
from taskforge.llm.policy import LLMPolicy
from taskforge.lowering import lower_with_settings, lowering_settings
from taskforge.model.llm import Completion, Usage
from taskforge.model.proposal import TaskProposal
from taskforge.model.review import Abandon, Incomplete
from taskforge.model.stage import Stage
from taskforge.model.workpiece import WORKPIECE, Terminal, Workpiece
from taskforge.policy import load_policy
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.report import reason_class, report
from taskforge.resources.env import CommandResult, RunCommand, WriteFile
from taskforge.resources.inference import InferenceRequest
from taskforge.resources.rollouts import SolverRollout


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("Restart: unavailable <- ConnectError: pool full", "Restart"),
        ("author stopped: generation limit after 4 turns", "author stopped"),
        ("bug: ValueError: bad", "bug:ValueError"),
        ("bug: shellbox.machine.MachineUnavailable: down", "bug:MachineUnavailable"),
        (
            "bug: Traceback\nconnectrpc.errors.ConnectError: pool full\ncaused by\nRuntimeError: wrapped",
            "bug:ConnectError",
        ),
        ("bug: ExceptionGroup: bad\n  | KeyError: missing", "bug:KeyError"),
    ],
)
def test_report_reason_class(reason: str, expected: str) -> None:
    assert reason_class(reason) == expected


async def test_report_reads_parts_counts_usage_once_and_uses_decision_time(
    tmp_path: Path, proposal: TaskProposal, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("taskforge.report.time.time", lambda: 1000000.0)
    inference = InferenceRequest((), (), LLMPolicy())
    completion = Completion("ok", usage=Usage(prompt_tokens=100, cached_tokens=20, completion_tokens=10))
    turn = ModelTurn(
        {}, (), (), None, "stop", metadata={"usage": {"prompt_tokens": 40, "cached_tokens": 15, "completion_tokens": 7}}
    )
    conversation = ConversationRecord((), (), (), (), None, (RolloutStep(turn, Transition(True), 0, ()),))
    result = GradedRollout(Rollout("r", (conversation,), "stop"), GradeResult(Outcome.GRADED, 1.0))
    spec = lower_with_settings(arithmetic_task(), lowering_settings(load_policy(DEFAULTS_DIR)), {})
    events: list[tuple[float, Event[object]]] = [
        (-2, ScopeEntered("judge")),  # An interrupted previous process left this scope open.
        (-1, ScopeEntered("workpiece")),
        (0, ScopeEntered("author")),
        (10, Resolved(inference, (), completion)),
        (11, Resolved(inference, (), completion, replayed=True)),
        (20, ScopeExited("author", False)),
        (21, Resolved(RunCommand("true", 30), (), CommandResult(0, "", "", 1, False))),
        (24, Resolved(WriteFile(PurePosixPath("x"), b"x"), (), None)),
        (25, ScopeEntered("adversary:a")),
        (30, Resolved(replace(inference, sample=1), (), completion)),
        (31, ScopeExited("adversary:a", False)),
        (32, Resolved(SolverRollout(spec, 0, 0), (), result)),
        (33, Resolved(SolverRollout(spec, 0, 0), (), result, replayed=True)),
        (34, StageCompleted(Stage.REVIEW, "", "Accepted", terminal=Terminal.ACCEPTED)),
    ]
    for name, age, terminal, reasons in (
        ("accepted", 1800, Terminal.ACCEPTED, ()),
        ("old", 8 * 3600, Terminal.ACCEPTED, ()),
        ("held", 100, Terminal.HELD, ("Restart: transport",)),
        ("abandoned", 100, Terminal.ABANDONED, ("author stopped: generation limit",)),
    ):
        directory = tmp_path / name
        directory.mkdir()
        decisions = (Abandon(reasons),) if terminal is Terminal.ABANDONED else (Incomplete(reasons),)
        (directory / "workpiece.json").write_bytes(
            WORKPIECE.dump_json(Workpiece(name, proposal, terminal=terminal, decisions=decisions))
        )
        (directory / "status.json").write_text(json.dumps({"terminal": terminal}))
        data = events if name == "accepted" else [(0, StageCompleted(Stage.REVIEW, "", "Done", terminal=terminal))]
        records = tuple(Record(i, 1000000 - age + offset, event) for i, (offset, event) in enumerate(data, 1))
        sink = JsonlSink(str(directory / "events.jsonl"))
        await sink.write(records[:1])
        if len(records) > 1:
            sink.append_bytes(b"".join(sink._line(record) for record in records[1:]))
    lines = await report(str(tmp_path))
    assert "terminal=accepted count=2" in lines
    assert "terminal=held reason=Restart count=1" in lines
    assert "terminal=abandoned reason=author stopped count=1" in lines
    assert "accepted_per_hour window_hours=1 count=1 rate=1.000" in lines
    assert "accepted_per_hour window_hours=6 count=1 rate=0.167" in lines
    assert "accepted_per_hour window_hours=24 count=2 rate=0.083" in lines
    assert "tokens role=author prompt=100 cached=20 completion=10" in lines
    assert "tokens role=adversary prompt=100 cached=20 completion=10" in lines
    assert "tokens role=solver prompt=40 cached=15 completion=7" in lines
    assert "stage=author n=1 p50_seconds=20.000 p90_seconds=20.000" in lines
    assert "event_gap=InferenceRequest n=2 p50_seconds=5.000 p90_seconds=10.000" in lines
    assert "event_gap=WriteFile n=1 p50_seconds=3.000 p90_seconds=3.000" in lines
    assert "terminal=accepted count=1" in await report(str(tmp_path), 1)


async def test_report_empty_root_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No workpiece"):
        await report(str(tmp_path))
