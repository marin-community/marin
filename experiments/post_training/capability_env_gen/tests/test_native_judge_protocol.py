import hashlib
from dataclasses import dataclass

import pytest

from capability_pipeline.native_judge_protocol import (
    RetryingTerminalScoreClient,
    TerminalScoreLineClient,
)


@dataclass(frozen=True)
class Reply:
    text: str
    model: str = "glm-5.3"


class Client:
    def __init__(self, text):
        self.text = text

    def complete(self, prompt, policy, timeout):
        return Reply(self.text)


def complete(text):
    client = TerminalScoreLineClient(Client(text))
    return client.complete("prompt", object(), 1), client.repairs


def test_repairs_only_missing_newline_before_exact_terminal_score():
    original = "Evidence supports the criterion. SCORE: 1"
    repaired = "Evidence supports the criterion.\nSCORE: 1"
    reply, repairs = complete(original)
    assert reply.text == repaired
    assert repairs == [
        {
            "rule": "missing_newline_before_exact_terminal_score",
            "original_text": original,
            "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "repaired_sha256": hashlib.sha256(repaired.encode()).hexdigest(),
        }
    ]


def test_preserves_already_valid_score_line_without_claiming_repair():
    reply, repairs = complete("Evidence supports the criterion.\nSCORE: 1")
    assert reply.text.endswith("\nSCORE: 1")
    assert repairs == []


def test_does_not_infer_or_rewrite_malformed_semantic_verdicts():
    for text in (
        "Evidence only",
        "Evidence. SCORE: yes",
        "SCORE: 0 earlier, but revised. SCORE: 1",
        "Evidence. SCORE: 2",
        "Evidence.SCORE: 1",
    ):
        reply, repairs = complete(text)
        assert reply.text == text
        assert repairs == []


class SequenceClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def complete(self, prompt, policy, timeout):
        self.requests.append((prompt, policy, timeout))
        value = next(self.replies)
        if isinstance(value, Exception):
            raise value
        return Reply(value)


def test_reasks_only_malformed_protocol_with_identical_prompt_and_policy():
    policy = object()
    raw = SequenceClient(["Reasoning without a verdict", "Reasoning inline SCORE: 1"])
    client = RetryingTerminalScoreClient(raw, max_reasks=3)

    reply = client.complete("same prompt", policy, 10)

    assert reply.text == "Reasoning inline\nSCORE: 1"
    assert len(raw.requests) == 2
    assert all(request[0] == "same prompt" for request in raw.requests)
    assert all(request[1] is policy for request in raw.requests)
    assert 0 < raw.requests[1][2] <= raw.requests[0][2] <= 10
    evidence = client.evidence()
    assert evidence["actual_request_count"] == 2
    assert [attempt["outcome"] for attempt in evidence["calls"][0]["attempts"]] == [
        "malformed_terminal_score",
        "inline_terminal_score_repaired",
    ]
    assert [attempt["raw_text"] for attempt in evidence["calls"][0]["attempts"]] == [
        "Reasoning without a verdict",
        "Reasoning inline SCORE: 1",
    ]
    assert evidence["repairs"][0]["request_index"] == 2


@pytest.mark.parametrize("score", ["0", "0.5", "1"])
def test_score_value_never_controls_protocol_retry(score):
    raw = SequenceClient([f"Reasoning\nSCORE: {score}", "must not be requested"])
    client = RetryingTerminalScoreClient(raw)

    reply = client.complete("prompt", object(), 10)

    assert reply.text.endswith(f"SCORE: {score}")
    assert len(raw.requests) == 1
    assert client.evidence()["actual_request_count"] == 1


def test_malformed_exhaustion_returns_raw_reply_for_native_null_outcome():
    raw = SequenceClient(["missing one", "missing two", "missing three"])
    client = RetryingTerminalScoreClient(raw, max_reasks=2)

    reply = client.complete("prompt", object(), 10)

    assert reply.text == "missing three"
    assert client.evidence()["actual_request_count"] == 3
    assert client.evidence()["calls"][0]["outcome"] == "malformed_exhausted"


def test_transport_failure_is_not_retried_or_called_truncation():
    raw = SequenceClient([TimeoutError("provider timeout"), "must not be requested"])
    client = RetryingTerminalScoreClient(raw)

    with pytest.raises(TimeoutError, match="provider timeout"):
        client.complete("prompt", object(), 10)

    evidence = client.evidence()
    assert evidence["actual_request_count"] == 1
    assert evidence["calls"][0]["outcome"] == "request_or_reply_error"
    assert evidence["calls"][0]["attempts"] == [
        {
            "request_index": 1,
            "outcome": "request_or_reply_error",
            "error_type": "TimeoutError",
        }
    ]
    assert "truncat" not in str(evidence).lower()


def test_missing_reply_envelope_remains_unknown_and_unretried():
    class MissingEnvelopeClient:
        def complete(self, prompt, policy, timeout):
            del prompt, policy, timeout
            return object()

    client = RetryingTerminalScoreClient(MissingEnvelopeClient())

    with pytest.raises(AttributeError):
        client.complete("prompt", object(), 10)

    evidence = client.evidence()
    assert evidence["actual_request_count"] == 1
    assert evidence["calls"][0]["outcome"] == "request_or_reply_error"
    assert "truncat" not in str(evidence).lower()


def test_reasks_share_one_total_timeout_budget():
    times = iter([0.0, 1.0, 4.0, 7.5])
    raw = SequenceClient(["missing", "Reasoning\nSCORE: 1"])
    client = RetryingTerminalScoreClient(raw, clock=lambda: next(times))

    client.complete("prompt", object(), 10)

    assert [request[2] for request in raw.requests] == [9.0, 6.0]
