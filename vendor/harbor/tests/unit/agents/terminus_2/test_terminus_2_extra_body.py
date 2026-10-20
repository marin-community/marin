"""Regression tests: the eval-path thinking kwarg reaches the vLLM request.

Background
----------
``enable_thinking`` is a chat-template kwarg — client-specified, server-applied
(vLLM reads ``request.chat_template_kwargs`` and forwards it to
``apply_chat_template``). The RL-rollout path delivers it via a nested
``extra_body`` (YAML ``agent.extra_body.chat_template_kwargs.enable_thinking``).

The eval path historically tried to deliver it via a BARE
``--agent-kwarg enable_thinking=true``. terminus-2 has no ``enable_thinking``
parameter, so that bare kwarg fell into ``**kwargs`` and was silently
discarded — it never reached the request body. The fix routes the eval path
through the SAME live mechanism the RL path uses:

    --agent-kwarg 'extra_body={"chat_template_kwargs":{"enable_thinking":true}}'

which ``parse_kwargs`` JSON-decodes into a dict, ``AgentConfig.kwargs`` carries
to terminus-2's ``extra_body`` parameter, terminus-2 merges into
``_llm_call_kwargs["extra_body"]``, and ``LiteLLM.call`` deep-merges into the
outgoing ``litellm.acompletion`` request body.

These tests pin that the value survives end-to-end into the request, and that
the merge does not clobber a pre-existing ``extra_body`` key.

Marin's model catalog types ``agent_kwargs`` values as strings, so
``extra_body`` can also arrive at the terminus-2 constructor as a JSON object
string rather than a dict. The constructor decodes that form before merging
and rejects a string that is not a JSON object with a message naming the
field.
"""

from typing import Any

import pytest

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.cli.utils import parse_kwargs
from harbor.llms.lite_llm import LiteLLM
from harbor.models.trial.config import AgentConfig

# The exact agent-kwarg string the eval path emits when thinking is gated ON.
EVAL_THINKING_KWARG = 'extra_body={"chat_template_kwargs":{"enable_thinking":true}}'

# The extra_body value as Marin's model catalog delivers it: agent-kwargs
# values there are strings, so the JSON object stays encoded.
CATALOG_THINKING_EXTRA_BODY = '{"chat_template_kwargs":{"enable_thinking":true}}'

# The thinking kwarg in both accepted shapes: a decoded dict (the eval path
# via parse_kwargs) and the JSON object string Marin's catalog delivers.
THINKING_EXTRA_BODY_FORMS = [
    {"chat_template_kwargs": {"enable_thinking": True}},
    CATALOG_THINKING_EXTRA_BODY,
]

# hosted_vllm models require model_info (token limits + costs) to construct.
FAKE_MODEL_INFO = {
    "max_input_tokens": 8192,
    "max_output_tokens": 2048,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
}


def _fake_acompletion_response():
    """Minimal litellm-acompletion-shaped response for LiteLLM.call()."""
    return {
        "choices": [
            {
                "message": {"content": "ok", "reasoning_content": None},
                "finish_reason": "stop",
            }
        ],
        "model": "fake/model",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture(params=THINKING_EXTRA_BODY_FORMS, ids=["dict", "json-string"])
def extra_body_form(request):
    return request.param


def test_terminus_2_constructor_folds_extra_body_into_llm_call_kwargs(
    temp_dir, monkeypatch, extra_body_form
):
    """An ``extra_body=`` handed to the terminus ctor, dict or JSON object
    string, lands decoded in ``_llm_call_kwargs["extra_body"]`` ready to
    forward to every call."""
    monkeypatch.setattr(Terminus2, "_init_llm", lambda self, **kw: object())

    agent = Terminus2(
        logs_dir=temp_dir, model_name="fake/model", extra_body=extra_body_form
    )

    assert agent._llm_call_kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True}
    }


def test_terminus_2_extra_body_merges_with_preexisting_llm_call_kwargs(
    temp_dir, monkeypatch, extra_body_form
):
    """A pre-existing ``llm_call_kwargs['extra_body']`` key is preserved
    alongside the injected ``chat_template_kwargs`` (merge, not clobber),
    for both the dict and JSON-string forms."""
    monkeypatch.setattr(Terminus2, "_init_llm", lambda self, **kw: object())

    agent = Terminus2(
        logs_dir=temp_dir,
        model_name="fake/model",
        llm_call_kwargs={"extra_body": {"return_token_ids": True}},
        extra_body=extra_body_form,
    )

    assert agent._llm_call_kwargs["extra_body"] == {
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_agent_config_kwargs_carry_extra_body_to_terminus(temp_dir, monkeypatch):
    """AgentConfig.kwargs (the harbor job/trial agent-kwarg dict) carries the
    nested ``extra_body`` through to the terminus ctor unchanged."""
    monkeypatch.setattr(Terminus2, "_init_llm", lambda self, **kw: object())

    config = AgentConfig(
        name="terminus-2",
        model_name="fake/model",
        kwargs=parse_kwargs([EVAL_THINKING_KWARG]),
    )
    assert config.kwargs == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}
    }

    agent = Terminus2(logs_dir=temp_dir, model_name="fake/model", **config.kwargs)
    assert (
        agent._llm_call_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        is True
    )


@pytest.mark.parametrize(
    "bad_extra_body",
    [
        '{"chat_template_kwargs":',  # truncated JSON
        "[1, 2]",  # valid JSON, not an object
        '"chat_template_kwargs"',  # valid JSON, a bare string
    ],
)
def test_terminus_2_extra_body_bad_string_raises_naming_the_field(
    temp_dir, monkeypatch, bad_extra_body
):
    """A string extra_body that is not a JSON object is rejected with a
    ValueError naming ``extra_body``, not a bare dict.update failure."""
    monkeypatch.setattr(Terminus2, "_init_llm", lambda self, **kw: object())

    with pytest.raises(ValueError, match="extra_body"):
        Terminus2(logs_dir=temp_dir, model_name="fake/model", extra_body=bad_extra_body)


@pytest.mark.asyncio
async def test_extra_body_reaches_litellm_acompletion_request(monkeypatch):
    """End-to-end: the eval thinking kwarg arrives in the outgoing
    ``litellm.acompletion`` request body as
    ``extra_body.chat_template_kwargs.enable_thinking == True``.

    Mocks ``litellm.acompletion`` and asserts the captured call kwargs.
    """
    captured: dict[str, Any] = {}

    async def fake_chat_raw(completion_kwargs):
        captured.update(completion_kwargs)
        return _fake_acompletion_response()

    # The eval-path agent-kwargs as terminus-2 would assemble them.
    agent_kwargs = parse_kwargs([EVAL_THINKING_KWARG])
    llm_call_kwargs = dict(agent_kwargs)  # {"extra_body": {...}}

    llm = LiteLLM(
        model_name="hosted_vllm/fake/model",
        api_base="http://localhost:8000/v1",
        model_info=FAKE_MODEL_INFO,
    )

    llm._acreate_chat_raw = fake_chat_raw  # type: ignore[assignment]
    await llm.call(prompt="hi", **llm_call_kwargs)

    assert "extra_body" in captured, f"no extra_body in request: {sorted(captured)}"
    assert captured["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


@pytest.mark.asyncio
async def test_extra_body_merge_preserves_internal_keys_in_request(monkeypatch):
    """The deep-merge in LiteLLM.call preserves internally-set ``extra_body``
    keys (e.g. ``return_token_ids`` from collect_rollout_details) alongside the
    caller-supplied ``chat_template_kwargs`` — neither clobbers the other in the
    outgoing request body."""
    captured: dict[str, Any] = {}

    async def fake_chat_raw(completion_kwargs):
        captured.update(completion_kwargs)
        return _fake_acompletion_response()

    # collect_rollout_details=True makes LiteLLM.call inject return_token_ids
    # into completion_kwargs["extra_body"] internally.
    llm = LiteLLM(
        model_name="hosted_vllm/fake/model",
        api_base="http://localhost:8000/v1",
        collect_rollout_details=True,
        model_info=FAKE_MODEL_INFO,
    )

    llm_call_kwargs = parse_kwargs([EVAL_THINKING_KWARG])

    llm._acreate_chat_raw = fake_chat_raw  # type: ignore[assignment]
    await llm.call(prompt="hi", **llm_call_kwargs)

    eb = captured["extra_body"]
    assert eb["return_token_ids"] is True
    assert eb["chat_template_kwargs"]["enable_thinking"] is True
