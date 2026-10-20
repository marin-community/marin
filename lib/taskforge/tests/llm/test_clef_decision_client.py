# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Clef decision client against FakeClef: the bar, the deadline, the error split, the window, the token."""

import asyncio
import json
import logging
import time

import httpx
import pytest
from rolloutengine.models import GradedRolloutInterrupted
from rolloutengine.test_support import ReplayModel, engine, rollout_spec
from taskcompendium.chat import chat_conversation
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, GradingAttempt, PlainText, Source, TaskSpec
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from taskforge.llm.clef.budget import MAX_LENGTH, ClefTokenizer, input_tokens
from taskforge.llm.clef.decision_client import (
    WINDOW_HEADER,
    ClefDecisionClient,
    ClefDecisionConfig,
    DecisionRequestError,
    open_clef_decision_client,
)
from taskforge.llm.clef.schema import ScoreQuestion, SystemOneRequest
from taskforge.spec import decision, draft
from taskforge.spec.decision import (
    LEVELS,
    QUESTION,
    DecisionUnavailable,
    grade_decision_answer,
    host_graded,
    trial_form,
)
from taskforge.spec.rubric import Rubric, RubricItem
from tests.llm.fake_clef import FakeClef
from tests.spec.fake_decision import level_probabilities

TOKEN = "secret-bearer-123"
BASE_URL = "http://orion.test"
REFERENCE = "The approved total is 1,234.56 USD."
REPLY = "Approved total: $1,234.56."
INSTRUCTION = "Reconcile the ledger and tell me the approved total."
TRANSCRIPT = [{"role": "user", "content": INSTRUCTION}, {"role": "assistant", "content": REPLY}]
RUBRIC = Rubric(
    deliverable="The approved total of the ledger.",
    items=[
        RubricItem(
            id="total",
            intent="The answer states the approved total.",
            pass_criterion="The approved total is 1234.56 USD, within 0.01.",
            equivalences=["$1,234.56"],
            negation_handling="A total named only to be rejected does not count.",
            weight=1.0,
            role="scored",
            cap_value=None,
        )
    ],
)


def byte_tokenizer() -> ClefTokenizer:
    """One token per UTF-8 byte: offline, and exact enough to test the window arithmetic."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    return ClefTokenizer(tokenizer, "test-bytes")


TOKENIZER = byte_tokenizer()


def at_level(level: float):
    """A FakeClef prober whose score answer reads at ``level``."""

    def prober(_, question):
        return {str(i): p for i, p in enumerate(level_probabilities(level))}

    return prober


def config(window: int | None = None) -> ClefDecisionConfig:
    return ClefDecisionConfig(base_url=BASE_URL, token_env="CLEF_TEST_TOKEN", window=window, max_inflight=4)


def client(handler, window: int | None = None) -> ClefDecisionClient:
    return ClefDecisionClient(config(window), TOKEN, TOKENIZER, transport=httpx.MockTransport(handler))


def calibrated(bar: float | None) -> decision.DecisionGrader:
    grader = draft.decision_grader(RUBRIC, REFERENCE)
    return grader.model_copy(update={"acceptance_level": bar, "calibration": {} if bar is None else {"reference": bar}})


@pytest.mark.parametrize(
    ("level", "bar", "accepted"),
    [(3.2, 2.9, True), (2.5, 2.9, False), (2.5, None, True), (1.4, None, False)],
    ids=["above-the-bar", "below-the-bar", "fallback-accepts-partial", "fallback-rejects"],
)
async def test_the_read_is_accepted_against_the_calibrated_bar_or_the_fallback(level, bar, accepted):
    fake = FakeClef(tokenizer=TOKENIZER, prober=at_level(level), token=TOKEN)
    async with client(fake.handle) as clef:
        result = await graded(calibrated(bar), TRANSCRIPT, clef, timeout=30)

    assert result.status is Outcome.GRADED and result.passed is accepted
    assert result.detail is not None and result.detail["level"] == pytest.approx(level)
    assert (result.reward == 1.0) is accepted and (result.reward is not None and result.reward <= 1.0)
    if not accepted:
        assert result.reward is not None and result.reward < decision.REJECTED_REWARD_MAX


async def test_the_state_and_question_are_ws10d_v3s_form_de():
    fake = FakeClef(tokenizer=TOKENIZER, prober=at_level(3.0), token=TOKEN)
    async with client(fake.handle) as clef:
        await graded(calibrated(None), TRANSCRIPT, clef, timeout=30)

    (request,) = fake.seen
    assert request.state == {
        "task_deliverable": RUBRIC.deliverable,
        "rubric": [
            {
                "id": "total",
                "role": "scored",
                "tests": "The answer states the approved total.",
                "pass_criterion": "The approved total is 1234.56 USD, within 0.01.",
                "equivalences": ["$1,234.56"],
                "negation_handling": "A total named only to be rejected does not count.",
            }
        ],
        "y_reference_answer": REFERENCE,
        "z_candidate_answer": REPLY,
    }
    (question,) = request.questions.values()
    assert isinstance(question, ScoreQuestion)
    assert list(request.questions) == ["whole"]
    assert (question.instructions, question.levels) == (QUESTION, LEVELS)


async def test_the_deadline_bounds_the_whole_read():
    async def stalls(request):
        await asyncio.sleep(5)
        return httpx.Response(500)

    started = time.monotonic()
    async with client(stalls) as clef:
        with pytest.raises(DecisionUnavailable, match="within"):
            await graded(calibrated(None), TRANSCRIPT, clef, timeout=0.05)
    assert time.monotonic() - started < 1


def refusing_connection(request):
    raise httpx.ConnectError("connection refused", request=request)


def wrong_questions(request):
    body = FakeClef(tokenizer=TOKENIZER).answer(json.loads(request.content))
    body["answers"] = {"other": body["answers"]["whole"]}
    return httpx.Response(200, json=body)


@pytest.mark.parametrize(
    ("handler", "error"),
    [
        (FakeClef(token=TOKEN, unavailable=[503]).handle, DecisionUnavailable),
        (FakeClef(token=TOKEN, unavailable=[502]).handle, DecisionUnavailable),
        (lambda request: httpx.Response(503, json={"error": {"code": "clef_unavailable"}}), DecisionUnavailable),
        (refusing_connection, DecisionUnavailable),
        (FakeClef(token="another-token").handle, DecisionRequestError),
        (lambda request: httpx.Response(400, json={"error": "bad schema"}), DecisionRequestError),
        (wrong_questions, DecisionRequestError),
    ],
    ids=["503", "502", "front-door", "connection", "401", "400", "schema-mismatch"],
)
async def test_transient_and_permanent_failures_raise_apart(handler, error):
    async with client(handler) as clef:
        with pytest.raises(error):
            await graded(calibrated(None), TRANSCRIPT, clef, timeout=30)


def ledger_task(grader: decision.DecisionGrader) -> TaskSpec:
    source = Source(dataset="taskforge-test", revision="r1", row="0", importer_revision="test")
    return draft.assemble("ledger", INSTRUCTION, AnswerType.TEXT, PlainText(), grader, source, environment=None)


async def graded(grader: decision.DecisionGrader, messages: list[dict], client: ClefDecisionClient, *, timeout: float):
    """The host's grade of ``messages`` on the ledger task carrying ``grader``."""
    attempt = GradingAttempt(chat_conversation(messages))
    return await grade_decision_answer(ledger_task(grader), attempt, client, timeout=timeout)


async def trial():
    reply = {"role": "assistant", "content": REPLY}
    return await engine(ReplayModel([reply])).run_task(trial_form(rollout_spec(ledger_task(calibrated(None)))))


@pytest.mark.parametrize(
    ("unavailable", "graded", "calls"), [([503], True, 2), ([503, 503], False, 2)], ids=["recovers", "down-twice"]
)
async def test_a_transient_clef_failure_feeds_the_grade_retry(monkeypatch, unavailable, graded, calls):
    monkeypatch.setattr(decision, "DECISION_RETRY_BACKOFF", 0.0)
    fake = FakeClef(tokenizer=TOKENIZER, prober=at_level(3.4), token=TOKEN, unavailable=list(unavailable))
    rollout = await trial()

    async with client(fake.handle) as clef:
        if graded:
            result = await host_graded(rollout, ledger_task(calibrated(None)), clef, deadline=30)
            assert result.grade.passed is True
        else:
            with pytest.raises(GradedRolloutInterrupted) as interrupted:
                await host_graded(rollout, ledger_task(calibrated(None)), clef, deadline=30)
            assert isinstance(interrupted.value.__cause__, DecisionUnavailable)
    assert fake.calls == calls


async def test_a_permanent_clef_failure_is_not_retried(monkeypatch):
    monkeypatch.setattr(decision, "DECISION_RETRY_BACKOFF", 0.0)
    fake = FakeClef(tokenizer=TOKENIZER, token="another-token")
    rollout = await trial()

    async with client(fake.handle) as clef:
        with pytest.raises(GradedRolloutInterrupted) as interrupted:
            await host_graded(rollout, ledger_task(calibrated(None)), clef, deadline=30)

    assert isinstance(interrupted.value.__cause__, DecisionRequestError) and fake.calls == 1


class Recording:
    """FakeClef behind a handler that keeps every request's headers."""

    def __init__(self):
        self.fake = FakeClef(tokenizer=TOKENIZER, prober=at_level(3.0), token=TOKEN)
        self.headers: list[httpx.Headers] = []

    def handle(self, request):
        self.headers.append(request.headers)
        return self.fake.handle(request)


@pytest.mark.parametrize(("window", "reply_bytes"), [(16384, 40_000), (None, 200_000)], ids=["pinned-16k", "cap-128k"])
async def test_the_state_is_fitted_to_the_window_by_cutting_the_solver_output(window, reply_bytes):
    long_reply = "The approved total is 1,234.56 USD. " + "x" * reply_bytes + " Final: 1,234.56 USD."
    recording = Recording()
    async with client(recording.handle, window) as clef:
        await graded(calibrated(None), [TRANSCRIPT[0], {"role": "assistant", "content": long_reply}], clef, timeout=30)

    (request,) = recording.fake.seen
    state = request.state
    assert isinstance(state, dict)
    candidate = state["z_candidate_answer"]
    assert isinstance(candidate, str)
    assert "tokens truncated by the Clef state budget" in candidate and candidate.endswith("Final: 1,234.56 USD.")
    assert state["y_reference_answer"] == REFERENCE
    assert input_tokens(TOKENIZER, SystemOneRequest(request.state, request.questions)) <= (window or MAX_LENGTH)
    assert recording.headers[0].get(WINDOW_HEADER) == (None if window is None else str(window))


async def test_the_token_is_never_logged_or_shown(caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    long_reply = "x" * 40_000
    async with client(FakeClef(token="another-token").handle, 16384) as clef:
        assert TOKEN not in repr(clef)
        with pytest.raises(DecisionRequestError) as refused:
            await graded(
                calibrated(None), [TRANSCRIPT[0], {"role": "assistant", "content": long_reply}], clef, timeout=30
            )
    assert TOKEN not in str(refused.value)
    assert TOKEN not in caplog.text

    monkeypatch.delenv("CLEF_TEST_TOKEN", raising=False)
    with pytest.raises(ValueError, match="CLEF_TEST_TOKEN, which is not set"):
        async with open_clef_decision_client(config(), TOKENIZER):
            pass
    monkeypatch.setenv("CLEF_TEST_TOKEN", TOKEN)
    async with open_clef_decision_client(config(), TOKENIZER) as opened:
        assert TOKEN not in repr(opened)


def test_a_window_clef_does_not_run_is_refused():
    with pytest.raises(ValueError, match="window must be one of"):
        config(20000)
