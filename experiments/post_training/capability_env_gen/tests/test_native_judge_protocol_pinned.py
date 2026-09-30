"""Integration checks against the exact pinned TaskCompendium judge implementation."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

taskcompendium = pytest.importorskip("taskcompendium")

from taskcompendium.grading import grade_attempt
from taskcompendium.grading_paths import submission_relative
from taskcompendium.judging import _SCORE, JudgeReply
from taskcompendium.models import Outcome
from taskcompendium.serialization import from_json, renderings_from_json

from capability_pipeline.native_judge_protocol import (
    RetryingTerminalScoreClient,
    TerminalScoreLineClient,
)

PINNED_JUDGING_SHA256 = (
    "9f9c469e65955144b6fc7e25166316d42d8a073af75d15bc3155cf4d57ca1e06"
)
RETAINED_REPLY = (
    "All four repairs name target defects; repairs 1–2 explicitly preserve exactly-once "
    "slot release (ack-independent cleanup + TTL fallback) and at-most-once completion "
    "(epoch-spanning window/ledger dedupe), with repairs 3–4 addressing semantics too. "
    "SCORE: 1"
)


class PinnedClient:
    def __init__(self, text: str):
        self.text = text

    def complete(self, prompt, policy, timeout):
        return JudgeReply(self.text, "glm-5.3", "pinned-fixture")


def _fixture_paths() -> tuple[Path, Path]:
    root = Path(__file__).parents[1]
    item = root / (
        "runs/synthesis-c22-continuation-001/review-pull-0742/items/"
        "c22.workflow_runtime_diagnosis-9-e2ede7f26d29/workspace"
    )
    return item / "task/harbor", item / "e2e/composite-smoke/smoke-gold"


def _grade_client(tmp_path: Path, client):
    task, smoke = _fixture_paths()
    specification = from_json((task / "composite-specification.json").read_bytes())
    rendering = renderings_from_json((task / "renderings.json").read_bytes())[0]
    verifier = specification.steps[0].verifier
    for path in verifier.judge.view.files:
        relative = submission_relative(
            path,
            specification.requirements.state.workdir,
            specification.requirements.state.additional_directories,
        )
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("")
    response = (smoke / "agent/response.txt").read_text()
    (tmp_path / "postmortem.md").write_text(response)
    transcript_path = smoke / "agent/transcript.json"
    transcript = tuple(json.loads(transcript_path.read_text()))
    result = grade_attempt(
        specification,
        rendering,
        response,
        tmp_path,
        transcript,
        client,
        0,
    )
    return result, client


def _grade(tmp_path: Path, text: str | None):
    client = None if text is None else TerminalScoreLineClient(PinnedClient(text))
    return _grade_client(tmp_path, client)


def test_actual_pinned_type_parser_and_grade_attempt_accept_retained_reply(tmp_path):
    source = Path(inspect.getsourcefile(JudgeReply))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == PINNED_JUDGING_SHA256
    assert hasattr(JudgeReply, "__dataclass_fields__")
    assert _SCORE.search(RETAINED_REPLY) is None

    result, client = _grade(tmp_path, RETAINED_REPLY)
    assert result.status == Outcome.GRADED
    assert result.reward == 1.0
    assert len(client.repairs) == 22
    assert all(row["original_text"] == RETAINED_REPLY for row in client.repairs)


def test_actual_native_parser_needs_no_repair_for_valid_line(tmp_path):
    valid = RETAINED_REPLY.rsplit(" SCORE: 1", 1)[0] + "\nSCORE: 1"
    result, client = _grade(tmp_path, valid)
    assert result.status == Outcome.GRADED
    assert result.reward == 1.0
    assert client.repairs == []


def test_actual_native_grade_attempt_rejects_ambiguous_scores(tmp_path):
    ambiguous = "SCORE: 0 earlier, but revised. SCORE: 1"
    result, client = _grade(tmp_path, ambiguous)
    assert result.status == Outcome.INFRA_ERROR
    assert result.reward is None
    assert result.detail["error"] == "Malformed judge verdict"
    assert client.repairs == []


def test_actual_native_grade_attempt_preserves_no_client_sentinel(tmp_path):
    result, client = _grade(tmp_path, None)
    assert client is None
    assert result.status == Outcome.INFRA_ERROR
    assert result.reward is None
    assert result.detail == {"error": "No judge client configured"}


class SequencePinnedClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.request_count = 0

    def complete(self, prompt, policy, timeout):
        del prompt, policy, timeout
        self.request_count += 1
        value = next(self.replies)
        if isinstance(value, Exception):
            raise value
        return JudgeReply(value, "glm-5.3", "pinned-fixture")


def test_actual_native_grade_attempt_reasks_missing_verdicts(tmp_path):
    valid = "Criterion evidence.\nSCORE: 1"
    raw = SequencePinnedClient(
        reply
        for _ in range(22)
        for reply in ("Criterion evidence without verdict", valid)
    )
    client = RetryingTerminalScoreClient(raw, max_reasks=3)

    result, _ = _grade_client(tmp_path, client)

    assert result.status == Outcome.GRADED
    assert result.reward == 1.0
    assert raw.request_count == 44
    evidence = client.evidence()
    assert evidence["actual_request_count"] == 44
    assert len(evidence["calls"]) == 22
    assert all(call["selected_request_index"] == 2 for call in evidence["calls"])
    assert all(
        [attempt["outcome"] for attempt in call["attempts"]]
        == ["malformed_terminal_score", "valid_terminal_score"]
        for call in evidence["calls"]
    )


def test_actual_native_grade_attempt_keeps_transport_failure_ungraded(tmp_path):
    raw = SequencePinnedClient([TimeoutError("transport stopped")])
    client = RetryingTerminalScoreClient(raw)

    result, _ = _grade_client(tmp_path, client)

    assert result.status == Outcome.INFRA_ERROR
    assert result.reward is None
    assert result.detail["error"].startswith("Judge request failed:")
    assert client.evidence()["actual_request_count"] == 1
    assert client.evidence()["calls"][0]["outcome"] == "request_or_reply_error"
