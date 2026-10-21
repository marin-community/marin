"""Regression tests for the terminus-2 summarization output-budget fix.

Root cause (see agent_logs/2026-06-18_*summarization*): the summary subagent LLM
calls set NO ``max_tokens``. Proactive summarization fires when the context is
already nearly full (``free_tokens < proactive_summarization_threshold``), so the
summary completion runs out of output room, returns ``finish_reason == "length"``,
and ``lite_llm`` raises ``OutputLengthExceededError``. In the proactive path that
was swallowed (silent no-op); in the reactive path it degraded to a no-LLM
"instruction + last 1000 chars" fallback. Measured: 59% of summarizing trials hit
the degenerate fallback, reward 0.19 vs 0.82 for trials that never summarize.

The output-budget contract is:
  * a named ``RESERVED_SUMMARY_OUTPUT`` budget,
  * input unwound so ``context_limit - input_tokens >= RESERVED_SUMMARY_OUTPUT``
    before any summary call, AND that reserve passed as explicit ``max_tokens``,
  * a self-correcting retry that GROWS the reserve (deeper unwind / more output
    headroom) on ``OutputLengthExceededError`` — the only direction that can
    satisfy a thinking model that spent the budget on reasoning tokens,
  * provider rejection, rather than a guessed tokenizer margin, controls how
    much copied history each concrete recovery request can carry,
  * output exhaustion at the model limit returns a bounded handoff containing
    the partial completion and conversation excerpt,
  * other failed recovery attempts use an independent short summary and then a
    deterministic bounded handoff.

With summarization disabled, the path raises
``ContextLengthExceededError`` within ``minimal_threshold``, counts the pending
prompt, and issues ZERO summary LLM calls.

The tests bypass ``__init__`` via ``object.__new__`` and inject a fake LLM, the
pattern used by the other terminus_2 unit tests.
"""

import logging
import uuid
from unittest.mock import AsyncMock

import pytest

from harbor.agents.terminus_2.terminus_2 import (
    EpisodeLoggingPaths,
    LLMRequestContext,
    RESERVED_SUMMARY_OUTPUT,
    Terminus2,
    _OVERLONG_SUMMARY_MARKER,
    _accepted_trajectory_step_count,
)
from harbor.context_management import ContextWindowManager
from harbor.llms.base import (
    ContextLengthExceededError,
    ContextManagementInfrastructureError,
    LLMResponse,
    OutputLengthExceededError,
)


class FakeLLM:
    """Minimal async LLM stub that records call kwargs and models vLLM's
    output-truncation behaviour.

    ``call`` raises ``OutputLengthExceededError`` whenever the caller does NOT
    reserve enough output budget (``max_tokens`` missing or below
    ``min_required_budget``) — exactly the ``finish_reason == "length"`` path in
    ``lite_llm``. With a sufficient ``max_tokens`` it returns a short summary.
    """

    def __init__(
        self,
        context_limit: int,
        min_required_budget: int = 1,
        output_limit: int | None = None,
        truncated_response: str = "partial...",
    ):
        self._context_limit = context_limit
        self._min_required_budget = min_required_budget
        self._output_limit = output_limit
        self._truncated_response = truncated_response
        self.calls: list[dict] = []

    def get_model_context_limit(self) -> int:
        return self._context_limit

    def get_model_output_limit(self):
        return self._output_limit

    async def call(self, prompt: str, message_history=None, **kwargs) -> LLMResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "message_history": list(message_history or []),
                "kwargs": dict(kwargs),
            }
        )
        max_tokens = kwargs.get("max_tokens")
        if max_tokens is None or max_tokens < self._min_required_budget:
            raise OutputLengthExceededError(
                "hit max_tokens limit. Response was truncated.",
                partial_response=LLMResponse(content=self._truncated_response),
            )
        return LLMResponse(content="A short clean summary.", model_name="fake/model")


_SYS_OVERHEAD = 200
_CHARS_PER_TOKEN = 4


def _det_tokens(
    n_messages: int,
    per_msg_tokens: int,
    extra_chars: int = 0,
) -> int:
    """Deterministic token model shared by the agent stub and the window-aware
    fake so the unwind accounting and the serve-time rejection cannot drift.

    Tokens live mostly in the messages (as in reality): count = sys_overhead +
    per_msg * (n - 1), plus any appended prompt text counted by char length.
    """
    base = (
        _SYS_OVERHEAD
        if n_messages <= 1
        else _SYS_OVERHEAD + (n_messages - 1) * per_msg_tokens
    )
    return base + extra_chars // _CHARS_PER_TOKEN


class WindowAwareFakeLLM:
    """Models a provider that rejects a request whose rendered input + max_tokens
    exceeds the context window — the vLLM ``ContextLengthExceededError`` path.

    Unlike ``FakeLLM`` (which checks only ``max_tokens``), this counts the real
    request size: ``tokens(message_history) + tokens(prompt) + max_tokens``. The
    token model matches the deterministic ``_count_total_tokens`` stub in
    ``_make_agent`` (shared via ``_det_tokens``) so the unwind accounting and the
    serve-time rejection agree.
    """

    def __init__(
        self,
        context_limit: int,
        per_msg_tokens: int,
        chars_per_token: int = _CHARS_PER_TOKEN,
        sys_overhead: int = _SYS_OVERHEAD,
        request_overhead_tokens: int = 0,
        always_overflow: bool = False,
    ):
        self._context_limit = context_limit
        self._per_msg = per_msg_tokens
        self._chars_per_token = chars_per_token
        self._sys_overhead = sys_overhead
        self._request_overhead_tokens = request_overhead_tokens
        self._always_overflow = always_overflow
        self.calls: list[dict] = []

    def get_model_context_limit(self) -> int:
        return self._context_limit

    def get_model_output_limit(self):
        return None

    def _input_tokens(self, prompt: str, message_history) -> int:
        history = list(message_history or [])
        return _det_tokens(len(history), self._per_msg, len(prompt))

    async def call(self, prompt: str, message_history=None, **kwargs) -> LLMResponse:
        call = {
            "prompt": prompt,
            "kwargs": dict(kwargs),
            "message_history": list(message_history or []),
        }
        self.calls.append(call)
        input_tokens = self._input_tokens(prompt, message_history)
        max_tokens = kwargs.get("max_tokens") or 0
        rejected = (
            self._always_overflow
            or input_tokens + max_tokens + self._request_overhead_tokens
            > self._context_limit
        )
        call["accepted"] = not rejected
        if rejected:
            raise ContextLengthExceededError(
                f"This model's maximum context length is {self._context_limit} "
                f"tokens. However, your request has {input_tokens + max_tokens} "
                f"tokens."
            )
        return LLMResponse(content="A short clean summary.", model_name="fake/model")

    def assert_no_over_window_request(self) -> None:
        """Every recorded LLM call fit within the context window."""
        for c in self.calls:
            input_tokens = self._input_tokens(c["prompt"], c["message_history"])
            max_tokens = c["kwargs"].get("max_tokens") or 0
            assert (
                input_tokens + max_tokens + self._request_overhead_tokens
                <= self._context_limit
            ), (
                f"Over-window summary request dispatched: {input_tokens} input + "
                f"{max_tokens} output + {self._request_overhead_tokens} serving "
                f"overhead > {self._context_limit} limit."
            )


def _make_agent(
    *,
    enable_summarize: bool,
    context_limit: int,
    fake_llm: FakeLLM | WindowAwareFakeLLM,
    per_msg_tokens: int = 200,
    proactive_threshold: int = 8000,
) -> Terminus2:
    agent = object.__new__(Terminus2)
    agent.logger = logging.getLogger("test-terminus-2-summary-budget")
    agent._llm = fake_llm
    agent._context_window_manager = ContextWindowManager(fake_llm)
    agent._llm_call_kwargs = {}
    agent._enable_summarize = enable_summarize
    agent._proactive_summarization_threshold = proactive_threshold
    agent.model_alias = "fake/model"
    agent._session_id = str(uuid.uuid4())
    agent._summarization_count = 0
    agent._collect_rollout_details = False
    agent._interleaved_thinking = False
    agent._api_request_times = []
    agent._parser_name = "json"

    # Deterministic token accounting shared with the window-aware fake (see
    # ``_det_tokens``) so the unwind accounting and the serve-time rejection
    # agree. extra_messages (e.g. the summary prompt appended only to the
    # request) are counted by content length so the budget matches the real
    # request size.
    def fake_count_total_tokens(chat, extra_messages=None) -> int:
        extra_chars = sum(len(m.get("content", "")) for m in (extra_messages or []))
        return _det_tokens(len(chat.messages), per_msg_tokens, extra_chars)

    agent._count_total_tokens = fake_count_total_tokens  # type: ignore[method-assign]

    # The async per-turn caller (_check_proactive_summarization) uses
    # _acount_total_tokens, which is now a truly-async /tokenize probe rather
    # than a to_thread wrapper around _count_total_tokens — so stub it to the
    # same deterministic count to keep the token accounting under test.
    async def fake_acount_total_tokens(chat, extra_messages=None) -> int:
        return fake_count_total_tokens(chat, extra_messages)

    agent._acount_total_tokens = fake_acount_total_tokens  # type: ignore[method-assign]
    return agent


class _FakeChat:
    def __init__(self, n_messages: int):
        self._messages = [
            {"role": "user", "content": f"m{i}"} for i in range(n_messages)
        ]

    @property
    def messages(self):
        return self._messages

    def reset_response_chain(self) -> None:
        pass

    async def prepare_prompt_token_ids(self, prompt: str, **kwargs):
        return None


class _FakeSession:
    """Minimal async session stub for the parts of ``_summarize`` that run once
    the summary subagent succeeds (subagent 2 captures the terminal pane)."""

    async def capture_pane(self, capture_entire=False):
        return ""


def _stub_subagent_bookkeeping(agent: Terminus2) -> None:
    """Stub the trajectory bookkeeping the summary subagent path touches so a
    test can isolate the LLM-call / budget contract."""
    agent._update_subagent_metrics = lambda *a, **k: None  # type: ignore[method-assign]
    agent._append_subagent_response_step = lambda *a, **k: None  # type: ignore[method-assign]
    agent._collect_subagent_rollout_detail = lambda *a, **k: None  # type: ignore[method-assign]
    agent._save_subagent_trajectory = lambda *a, **k: None  # type: ignore[method-assign]
    agent._prepare_copied_trajectory_steps = lambda n: ([], 1)  # type: ignore[method-assign]


# --------------------------------------------------------------------------- #
# Test (a) — G1: flag-off byte-identical no-op (no summary LLM calls).
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_flag_off_raises_within_minimal_threshold_no_llm_calls():
    """enable_summarize=False: within 2048 of the limit -> raise; zero LLM calls."""
    fake_llm = FakeLLM(context_limit=32768)
    agent = _make_agent(
        enable_summarize=False,
        context_limit=32768,
        per_msg_tokens=200,
        fake_llm=fake_llm,
    )
    # 155 msgs -> ~30800 used -> ~1968 free < 2048 minimal_threshold.
    chat = _FakeChat(n_messages=155)

    with pytest.raises(ContextLengthExceededError):
        await agent._check_proactive_summarization(chat, "task", session=None)

    assert fake_llm.calls == [], "flag-off path must not issue any summary LLM call"


@pytest.mark.asyncio
async def test_flag_off_above_minimal_threshold_is_noop_no_llm_calls():
    """enable_summarize=False: with headroom -> returns None, zero LLM calls."""
    fake_llm = FakeLLM(context_limit=32768)
    agent = _make_agent(
        enable_summarize=False,
        context_limit=32768,
        per_msg_tokens=200,
        fake_llm=fake_llm,
    )
    # 10 msgs -> ~2000 used -> ~30768 free >> 2048.
    chat = _FakeChat(n_messages=10)

    result = await agent._check_proactive_summarization(chat, "task", session=None)

    assert result is None
    assert fake_llm.calls == [], "flag-off path must not issue any summary LLM call"


# --------------------------------------------------------------------------- #
# Test (b) — the fix: summary call carries an explicit max_tokens >= reserve.
# --------------------------------------------------------------------------- #


def test_accepted_answer_history_keeps_its_two_copied_summary_steps():
    parent_history_length = 9

    accepted_steps = _accepted_trajectory_step_count(
        accepted_history_length=parent_history_length + 2,
        parent_history_length=parent_history_length,
    )

    assert accepted_steps == 7


@pytest.mark.asyncio
async def test_summary_synthetic_assistant_history_supports_interleaved_thinking():
    fake_llm = FakeLLM(context_limit=32768)
    agent = _make_agent(
        enable_summarize=True,
        context_limit=32768,
        fake_llm=fake_llm,
    )
    agent._interleaved_thinking = True
    _stub_subagent_bookkeeping(agent)
    chat = _FakeChat(n_messages=3)

    await agent._summarize(chat, "task", session=_FakeSession())

    answers_history = fake_llm.calls[2]["message_history"]
    assert answers_history[-1] == {
        "role": "assistant",
        "content": "A short clean summary.",
        "reasoning_content": "",
    }
    assert chat.messages[-1] == {
        "role": "assistant",
        "content": "A short clean summary.",
        "reasoning_content": "",
    }


@pytest.mark.asyncio
async def test_run_subagent_forwards_max_tokens_to_llm_call():
    """_run_subagent must forward an explicit max_tokens to _llm.call.

    Pre-fix this fails because no max_tokens is ever passed and the FakeLLM
    raises OutputLengthExceededError.
    """
    fake_llm = FakeLLM(context_limit=32768, min_required_budget=RESERVED_SUMMARY_OUTPUT)
    agent = _make_agent(
        enable_summarize=True,
        context_limit=32768,
        per_msg_tokens=200,
        fake_llm=fake_llm,
    )

    # Stub the trajectory bookkeeping the subagent path touches so we isolate
    # the LLM-call contract.
    _stub_subagent_bookkeeping(agent)

    await agent._run_subagent(
        prompt="summarize please",
        message_history=[{"role": "user", "content": "hi"}],
        steps=[],
        session_id="s",
        agent_name="terminus-2-summarization-summary",
        filename_suffix="summary",
        summary_text="t",
        subagent_name_for_logging="summary",
        max_tokens=RESERVED_SUMMARY_OUTPUT,
    )

    assert len(fake_llm.calls) == 1
    passed = fake_llm.calls[0]["kwargs"].get("max_tokens")
    assert passed is not None, "summary LLM call issued no max_tokens (truncation bug)"
    assert passed >= RESERVED_SUMMARY_OUTPUT


def test_summary_unwind_leaves_requested_output_reserve():
    """The summary unwind removes chat history until
    context_limit - input_tokens >= RESERVED_SUMMARY_OUTPUT, so the summary
    subagent always has real output headroom regardless of starting fullness.
    """
    context_limit = 32768
    fake_llm = FakeLLM(context_limit=context_limit)
    agent = _make_agent(
        enable_summarize=True,
        context_limit=context_limit,
        per_msg_tokens=200,
        fake_llm=fake_llm,
    )
    # 155 msgs -> ~30800 used -> ~1968 free initially, so the reserve helper
    # must unwind message pairs until >= RESERVED_SUMMARY_OUTPUT is free.
    chat = _FakeChat(n_messages=155)

    agent._unwind_messages_to_free_tokens(chat, RESERVED_SUMMARY_OUTPUT)

    free = context_limit - agent._count_total_tokens(chat)
    assert free >= RESERVED_SUMMARY_OUTPUT, (
        f"reserve helper left only {free} free tokens, "
        f"need >= {RESERVED_SUMMARY_OUTPUT}"
    )


@pytest.mark.asyncio
async def test_summarize_with_retry_grows_reserve_for_thinking_model():
    """A thinking-enabled model can exhaust the default summary output budget on
    reasoning tokens. The retry must GROW the reserve (which unwinds more input
    AND grants more output headroom) until the summary fits — the only direction
    that can satisfy a summary that ran out of space.

    Pre-fix the ladder SHRANK the reserve (4096 -> 2048 -> 1024), unwinding less
    and granting less, guaranteeing re-truncation.
    """
    # Needs strictly more than the default 4096 reserve; fits once the ladder
    # grows to 8192.
    fake_llm = FakeLLM(context_limit=32768, min_required_budget=8192)
    agent = _make_agent(
        enable_summarize=True,
        context_limit=32768,
        per_msg_tokens=50,
        fake_llm=fake_llm,
    )
    _stub_subagent_bookkeeping(agent)

    chat = _FakeChat(n_messages=200)

    summary_prompt, _ = await agent._summarize_with_retry(
        chat, "task", session=_FakeSession()
    )

    # The ladder grew the reserve until the summary fit.
    reserves = [c["kwargs"].get("max_tokens") for c in fake_llm.calls]
    # First attempt truncated at the default reserve; the retry grew it.
    assert reserves[0] == RESERVED_SUMMARY_OUTPUT
    assert reserves[1] == 8192 > reserves[0]
    # The successful retry ran all three summary subagents at the grown reserve.
    assert set(reserves[1:]) == {8192}
    assert summary_prompt  # a real handoff prompt came back


@pytest.mark.asyncio
async def test_summarize_with_retry_truncates_handoff_at_model_output_limit():
    partial_summary = "model reasoning and partial summary\n" * 2_000
    output_limit = 8192
    fake_llm = FakeLLM(
        context_limit=32768,
        min_required_budget=10**9,
        output_limit=output_limit,
        truncated_response=partial_summary,
    )
    agent = _make_agent(
        enable_summarize=True,
        context_limit=32768,
        per_msg_tokens=50,
        fake_llm=fake_llm,
    )
    _stub_subagent_bookkeeping(agent)

    chat = _FakeChat(n_messages=200)

    handoff, subagent_refs = await agent._summarize_with_retry(
        chat, "task", session=object()
    )

    reserves = [c["kwargs"].get("max_tokens") for c in fake_llm.calls]
    assert reserves == [RESERVED_SUMMARY_OUTPUT, output_limit, output_limit]
    assert subagent_refs is None
    assert "model reasoning and partial summary" in handoff
    assert handoff.endswith(_OVERLONG_SUMMARY_MARKER)
    assert len(handoff.encode("utf-8")) < len(partial_summary.encode("utf-8"))
    assert len(chat.messages) == 1


@pytest.mark.asyncio
async def test_proactive_summarization_output_exhaustion_returns_handoff():
    fake_llm = FakeLLM(
        context_limit=32768,
        min_required_budget=10**9,
        output_limit=8192,
        truncated_response="partial handoff",
    )
    agent = _make_agent(
        enable_summarize=True,
        context_limit=32768,
        per_msg_tokens=200,
        fake_llm=fake_llm,
    )
    _stub_subagent_bookkeeping(agent)

    # 155 msgs -> ~30800 used -> ~1968 free < 8000 proactive threshold.
    chat = _FakeChat(n_messages=155)

    result = await agent._check_proactive_summarization(chat, "task", session=object())

    assert result is not None
    handoff, subagent_refs = result
    assert subagent_refs is None
    assert "partial handoff" in handoff
    assert handoff.endswith(_OVERLONG_SUMMARY_MARKER)
    assert all("max_tokens" in c["kwargs"] for c in fake_llm.calls), (
        "unexpected non-summary call dispatched during recovery"
    )


@pytest.mark.asyncio
async def test_output_exhaustion_without_visible_completion_uses_recent_context():
    fake_llm = FakeLLM(
        context_limit=32768,
        min_required_budget=10**9,
        output_limit=8192,
        truncated_response="",
    )
    agent = _make_agent(
        enable_summarize=True,
        context_limit=32768,
        per_msg_tokens=50,
        fake_llm=fake_llm,
    )
    _stub_subagent_bookkeeping(agent)
    chat = _FakeChat(n_messages=200)
    chat._messages[-1]["content"] = "recent-context-sentinel"

    handoff, _ = await agent._summarize_with_retry(chat, "task", session=object())

    assert "recent-context-sentinel" in handoff
    assert handoff.endswith(_OVERLONG_SUMMARY_MARKER)


@pytest.mark.asyncio
async def test_invalid_output_limit_uses_deterministic_handoff():
    fake_llm = FakeLLM(context_limit=32768, output_limit=0)
    agent = _make_agent(
        enable_summarize=True,
        context_limit=32768,
        per_msg_tokens=50,
        fake_llm=fake_llm,
    )

    chat = _FakeChat(n_messages=10)
    handoff, refs = await agent._summarize_with_retry(
        chat, "task", session=_FakeSession()
    )

    assert "ORIGINAL TASK:\ntask" in handoff
    assert refs is None
    assert len(chat.messages) == 1


# --------------------------------------------------------------------------- #
# Summary-prompt budget: the unwind must count the summary prompt, not just
# chat.messages, or the request overflows by exactly the prompt size.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_summary_prompt_counted_in_unwind_so_request_fits_window():
    """A near-full chat plus a large summary prompt must still fit the window.

    The summary prompt is appended to the retained history only for the request;
    the old reserve counted chat.messages alone, so a chat at 28,400 tokens plus
    a ~4,800-token prompt plus a 4,096 output reserve reached 37,296 > 32,768 and
    vLLM rejected it. Counting the prompt in the unwind frees enough history that
    no over-window request is dispatched.
    """
    context_limit = 32768
    per_msg = 200
    fake_llm = WindowAwareFakeLLM(context_limit=context_limit, per_msg_tokens=per_msg)
    agent = _make_agent(
        enable_summarize=True,
        context_limit=context_limit,
        per_msg_tokens=per_msg,
        fake_llm=fake_llm,
    )
    _stub_subagent_bookkeeping(agent)

    # 142 msgs -> 200 + 200*141 = 28,400 chat-only tokens (~4,368 free).
    chat = _FakeChat(n_messages=142)
    # A long original task so the generated summary prompt is large enough to
    # materially change the input count (~4,800 prompt tokens at 4 chars/token).
    long_instruction = "Do the path-tracing-reverse task thoroughly. " * 600

    summary_prompt, _ = await agent._summarize_with_retry(
        chat, long_instruction, session=_FakeSession()
    )

    assert summary_prompt
    # No summary subagent request overflowed the window.
    fake_llm.assert_no_over_window_request()


@pytest.mark.asyncio
async def test_summary_retries_against_provider_accounting_until_request_fits():
    """Recovery must use provider rejection as the admission authority.

    A local tokenizer cannot reproduce an arbitrary serving chat template. A
    fixed safety margin merely chooses one discrepancy to pass; the recovery
    contract must keep reducing copied history until the provider accepts each
    concrete summary request.
    """
    context_limit = 32768
    fake_llm = WindowAwareFakeLLM(
        context_limit=context_limit,
        per_msg_tokens=200,
        request_overhead_tokens=5000,
    )
    agent = _make_agent(
        enable_summarize=True,
        context_limit=context_limit,
        per_msg_tokens=200,
        fake_llm=fake_llm,
    )
    _stub_subagent_bookkeeping(agent)
    chat = _FakeChat(n_messages=155)

    summary_prompt, _ = await agent._summarize_with_retry(
        chat, "task", session=_FakeSession()
    )

    assert summary_prompt
    first_accepted = next(
        index for index, call in enumerate(fake_llm.calls) if call["accepted"]
    )
    recovery_attempts = fake_llm.calls[: first_accepted + 1]
    rejected_attempts = recovery_attempts[:-1]
    history_lengths = [len(call["message_history"]) for call in recovery_attempts]
    assert rejected_attempts
    assert all(
        later < earlier for earlier, later in zip(history_lengths, history_lengths[1:])
    )


@pytest.mark.asyncio
async def test_failed_summary_replaces_history_with_deterministic_handoff():
    fake_llm = WindowAwareFakeLLM(
        context_limit=32768,
        per_msg_tokens=200,
        always_overflow=True,
    )
    agent = _make_agent(
        enable_summarize=True,
        context_limit=32768,
        per_msg_tokens=200,
        fake_llm=fake_llm,
    )
    _stub_subagent_bookkeeping(agent)
    chat = _FakeChat(n_messages=155)
    handoff, refs = await agent._summarize_with_retry(
        chat,
        "task",
        session=_FakeSession(),
        pending_prompt="latest observation",
    )

    assert refs is None
    assert "latest observation" in handoff
    assert "user: m154" in handoff
    assert len(chat.messages) == 1


@pytest.mark.asyncio
async def test_pending_prompt_participates_in_proactive_context_budget():
    """The next user prompt is part of the next request, not future context."""
    fake_llm = FakeLLM(context_limit=32768)
    agent = _make_agent(
        enable_summarize=False,
        context_limit=32768,
        per_msg_tokens=200,
        fake_llm=fake_llm,
    )
    chat = _FakeChat(n_messages=150)
    pending_prompt = "x" * 8000

    with pytest.raises(ContextLengthExceededError):
        await agent._check_proactive_summarization(
            chat,
            "task",
            session=None,
            pending_prompt=pending_prompt,
        )


@pytest.mark.asyncio
async def test_proactive_summary_context_overflow_returns_bounded_handoff():
    context_limit = 32768
    fake_llm = WindowAwareFakeLLM(
        context_limit=context_limit, per_msg_tokens=200, always_overflow=True
    )
    agent = _make_agent(
        enable_summarize=True,
        context_limit=context_limit,
        per_msg_tokens=200,
        fake_llm=fake_llm,
    )
    _stub_subagent_bookkeeping(agent)

    # 155 msgs -> ~30,800 used -> ~1,968 free < 8,000 proactive threshold.
    chat = _FakeChat(n_messages=155)

    result = await agent._check_proactive_summarization(
        chat,
        "task",
        session=_FakeSession(),
        pending_prompt="latest observation",
    )

    assert result is not None
    handoff, refs = result
    assert refs is None
    assert "latest observation" in handoff
    assert len(chat.messages) == 1


@pytest.mark.asyncio
async def test_arbitrary_summary_failure_uses_short_summary_and_preserves_state(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_llm = FakeLLM(context_limit=32768, output_limit=8192)
    agent = _make_agent(
        enable_summarize=True,
        context_limit=32768,
        per_msg_tokens=50,
        fake_llm=fake_llm,
    )
    _stub_subagent_bookkeeping(agent)
    chat = _FakeChat(n_messages=20)
    monkeypatch.setattr(
        agent,
        "_summarize",
        AsyncMock(side_effect=RuntimeError("unexpected summary failure")),
    )

    class Session:
        async def capture_pane(self, capture_entire=False):
            return "terminal-state-sentinel"

    handoff, _ = await agent._summarize_with_retry(
        chat,
        "original-task-sentinel",
        session=Session(),
        pending_prompt="pending-observation-sentinel",
    )

    assert "original-task-sentinel" in handoff
    assert "pending-observation-sentinel" in handoff
    assert "terminal-state-sentinel" in handoff
    assert "A short clean summary." in handoff
    assert len(chat.messages) == 1


@pytest.mark.asyncio
async def test_reasoning_only_partial_summary_survives_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_llm = FakeLLM(context_limit=32768, output_limit=RESERVED_SUMMARY_OUTPUT)
    agent = _make_agent(
        enable_summarize=True,
        context_limit=32768,
        per_msg_tokens=50,
        fake_llm=fake_llm,
    )
    _stub_subagent_bookkeeping(agent)
    monkeypatch.setattr(
        agent,
        "_summarize",
        AsyncMock(
            side_effect=OutputLengthExceededError(
                "reasoning exhausted output",
                partial_response=LLMResponse(
                    content="",
                    reasoning_content="reasoning-state-sentinel",
                ),
            )
        ),
    )

    handoff, _ = await agent._summarize_with_retry(
        _FakeChat(n_messages=20),
        "task",
        session=_FakeSession(),
    )

    assert "reasoning-state-sentinel" in handoff


@pytest.mark.asyncio
async def test_completed_summary_stage_survives_later_stage_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_llm = FakeLLM(context_limit=32768, output_limit=8192)
    agent = _make_agent(
        enable_summarize=True,
        context_limit=32768,
        per_msg_tokens=50,
        fake_llm=fake_llm,
    )
    _stub_subagent_bookkeeping(agent)

    async def fail_after_summary(*args, **kwargs):
        kwargs["progress"].completed_summary = "completed-summary-sentinel"
        raise RuntimeError("questions stage failed")

    monkeypatch.setattr(agent, "_summarize", fail_after_summary)

    handoff, _ = await agent._summarize_with_retry(
        _FakeChat(n_messages=20),
        "task",
        session=_FakeSession(),
    )

    assert "completed-summary-sentinel" in handoff


@pytest.mark.asyncio
async def test_rejected_generated_handoff_retries_with_bounded_state(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_llm = FakeLLM(context_limit=32768, output_limit=8192)
    agent = _make_agent(
        enable_summarize=True,
        context_limit=32768,
        per_msg_tokens=50,
        fake_llm=fake_llm,
    )
    agent._pending_subagent_refs = None
    agent._pending_handoff_prompt = None
    chat = _FakeChat(n_messages=20)
    monkeypatch.setattr(
        agent,
        "_summarize_with_retry",
        AsyncMock(return_value=("generated handoff", None)),
    )
    dispatch = AsyncMock(
        side_effect=[
            ContextLengthExceededError("provider accounting drift"),
            LLMResponse(content="continued by model"),
        ]
    )
    monkeypatch.setattr(agent, "_dispatch_llm_request", dispatch)

    response = await agent._recover_context_capacity(
        chat=chat,
        error=ContextLengthExceededError("initial overflow"),
        pending_prompt="pending observation",
        request_context=LLMRequestContext(
            logging_paths=EpisodeLoggingPaths(None, None, None),
            original_instruction="original task",
            session=_FakeSession(),
        ),
    )

    assert response.content == "continued by model"
    compact_prompt = dispatch.await_args_list[1].args[1]
    assert "generated handoff" in compact_prompt
    assert "Technical difficulties" not in compact_prompt
    assert len(chat.messages) == 1


@pytest.mark.asyncio
async def test_rejected_bounded_handoff_raises_instead_of_fabricating(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_llm = FakeLLM(context_limit=32768, output_limit=8192)
    agent = _make_agent(
        enable_summarize=True,
        context_limit=32768,
        per_msg_tokens=50,
        fake_llm=fake_llm,
    )
    agent._pending_subagent_refs = None
    agent._pending_handoff_prompt = None
    monkeypatch.setattr(
        agent,
        "_summarize_with_retry",
        AsyncMock(return_value=("generated handoff", None)),
    )
    monkeypatch.setattr(
        agent,
        "_dispatch_llm_request",
        AsyncMock(side_effect=ContextLengthExceededError("still too large")),
    )

    with pytest.raises(ContextManagementInfrastructureError):
        await agent._recover_context_capacity(
            chat=_FakeChat(n_messages=20),
            error=ContextLengthExceededError("initial overflow"),
            pending_prompt="pending observation",
            request_context=LLMRequestContext(
                logging_paths=EpisodeLoggingPaths(None, None, None),
                original_instruction="original task",
                session=_FakeSession(),
            ),
        )


@pytest.mark.asyncio
async def test_disabled_summarization_context_overflow_still_raises_context_length():
    """A genuine ContextLengthExceededError (enable_summarize=False, the model
    truly exhausted the window) must stay a non-retryable model outcome — the
    summary-path translation does NOT broaden to the disabled path."""
    fake_llm = WindowAwareFakeLLM(context_limit=32768, per_msg_tokens=200)
    agent = _make_agent(
        enable_summarize=False,
        context_limit=32768,
        per_msg_tokens=200,
        fake_llm=fake_llm,
    )
    # 155 msgs -> ~30,800 used -> ~1,968 free < 2,048 minimal_threshold.
    chat = _FakeChat(n_messages=155)

    with pytest.raises(ContextLengthExceededError):
        await agent._check_proactive_summarization(chat, "task", session=None)

    # No summary calls on the disabled path.
    assert fake_llm.calls == []
