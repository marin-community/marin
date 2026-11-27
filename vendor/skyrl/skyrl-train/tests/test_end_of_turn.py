# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextvars import Context

from rigging.inference_clock import InferenceClock, current_inference_clock

from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

import pytest

from rolloutengine.chat_template import MARIN_CHAT_TEMPLATE
from rolloutengine.models import ModelRequest, ModelTurn
from skyrl_train.rollouts.agent_backend import AgentInferenceBackend, TRIAL_HEADER
from skyrl_train.rollouts.task_worker import _model_turn
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.trajectory_runners.model_clients import DirectModelClient

PLAN, CALL, END_OF_TURN, TOOL_CALL_END, END_OF_TEXT = 3, 4, 5, 6, 9
THINK_START, THINK_END = 1, 2
HISTORY_CONTEXT_LENGTH = 1024
AGENT_TRIAL_ID = "pi"
VOCABULARY = {
    THINK_START: "<think>",
    THINK_END: "</think>",
    PLAN: "plan",
    CALL: "call",
    END_OF_TURN: "<|eot_id|>",
    TOOL_CALL_END: "<|tool_call_end|>",
    END_OF_TEXT: "<|end_of_text|>",
}


class _LlamaStyleTokenizer:
    """Ends turns with a special token that is not ``eos_token_id``."""

    eos_token_id = END_OF_TEXT

    def decode(self, token_ids, skip_special_tokens):
        return "".join(VOCABULARY[token] for token in token_ids)

    def encode(self, text, add_special_tokens):
        token_ids = [token for token, piece in VOCABULARY.items() if piece == text]
        return token_ids or [PLAN, PLAN]


class _InferenceClient:
    model_name = "policy"
    tokenizer = _LlamaStyleTokenizer()

    def __init__(self, response_ids, finish_reason):
        self.response_ids = response_ids
        self.finish_reason = finish_reason
        self.chat_requests = []

    async def tokenize(self, request):
        return {"tokens": [PLAN]}

    async def chat_completion(self, request):
        self.chat_requests.append(request["json"])
        content = _LlamaStyleTokenizer().decode(self.response_ids, skip_special_tokens=False)
        message = {"role": "assistant", "content": content}
        return {"choices": [{"message": message, "token_ids": self.response_ids, "finish_reason": self.finish_reason}]}


def _generate(response_ids, finish_reason, chat_options=None, sampling_params=None):
    client = DirectModelClient(_InferenceClient(response_ids, finish_reason))
    request = InferenceEngineInput(
        prompts=[[{"role": "user", "content": "go"}]],
        chat_completion_params=[chat_options or {}],
        sampling_params=sampling_params or {},
    )
    return asyncio.run(client.generate(request))


@pytest.mark.parametrize("terminator", [END_OF_TURN, END_OF_TEXT], ids=["end-of-turn", "eos"])
def test_the_special_token_a_turn_stopped_at_is_removed_from_the_reply(terminator):
    output = _generate([PLAN, terminator], "stop")

    assert output["assistant_messages"][0]["content"] == "plan"
    assert output["response_ids"][0] == [PLAN, terminator]


@pytest.mark.parametrize(
    "response_ids,finish_reason,expected",
    [
        ([PLAN, END_OF_TURN], "stop", "plan"),
        ([THINK_START, PLAN, THINK_END, CALL, END_OF_TURN], "stop", "call"),
        ([PLAN, END_OF_TURN], "length", "plan<|eot_id|>"),
        ([THINK_START, PLAN, THINK_END, CALL, END_OF_TURN], "length", "call<|eot_id|>"),
        ([CALL, TOOL_CALL_END], "length", "call<|tool_call_end|>"),
    ],
    ids=["plain", "reasoning", "length-stop", "reasoning-length-stop", "cut-off-delimiter"],
)
def test_explicit_special_token_retention_excludes_only_an_ended_turns_terminator(
    response_ids, finish_reason, expected
):
    # A length stop did not end its turn, so its text keeps every sampled token.
    output = _generate(response_ids, finish_reason, sampling_params={"skip_special_tokens": False})

    assert output["responses"] == [expected]
    assert output["assistant_messages"][0]["content"] == expected
    assert output["response_ids"] == [response_ids]
    assert output["stop_reasons"] == [finish_reason]


def test_a_tool_call_that_ended_its_turn_at_the_budget_is_not_a_length_stop():
    output = _generate([CALL, END_OF_TURN], "tool_calls", {"max_completion_tokens": 2})

    assert output["stop_reasons"] == ["tool_calls"]


def test_a_tool_call_cut_off_at_the_budget_on_a_special_delimiter_is_a_length_stop():
    output = _generate([CALL, TOOL_CALL_END], "tool_calls", {"max_completion_tokens": 2})

    assert output["stop_reasons"] == ["length"]


@pytest.mark.parametrize(
    ("options", "effective_cap"),
    [
        ({"max_tokens": 64}, 64),
        ({"max_tokens": 96, "max_completion_tokens": 64, "max_output_tokens": 127}, 64),
        ({"max_tokens": 256}, 128),
    ],
)
def test_agent_output_cap_reaches_the_chat_server(options, effective_cap):
    engine = _InferenceClient([PLAN], "stop")
    output = asyncio.run(
        DirectModelClient(engine).generate(
            InferenceEngineInput(
                prompts=[[{"role": "user", "content": "go"}]],
                chat_completion_params=[options],
                sampling_params={"max_tokens": 128},
            )
        )
    )
    [request] = engine.chat_requests
    assert request["max_completion_tokens"] == effective_cap
    assert "max_tokens" not in request
    assert output["generation_token_budgets"] == [effective_cap]


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_transport", [False, True], ids=["skyrl", "pi-transport"])
@pytest.mark.parametrize(
    "response_ids,content,reply",
    [
        ([PLAN, CALL, END_OF_TURN], "plan", "plan"),
        ([THINK_START, PLAN, THINK_END, CALL, END_OF_TURN], "<think>plan</think>", "call"),
    ],
    ids=["plain", "reasoning"],
)
async def test_tool_call_history_renders_without_reply_terminators(agent_transport, response_ids, content, reply):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        chat_template=MARIN_CHAT_TEMPLATE,
        bos_token="<|begin_of_text|>",
    )
    rendered = []

    class ToolClient(_InferenceClient):
        async def tokenize(self, request):
            rendered.append(
                tokenizer.apply_chat_template(request["json"]["messages"], tokenize=False, add_generation_prompt=True)
            )
            return {"tokens": [PLAN]}

        async def chat_completion(self, request):
            response = await super().chat_completion(request)
            response["choices"][0]["message"].update(
                content=content + "<|eot_id|>",
                tool_calls=[
                    {"id": "lookup", "type": "function", "function": {"name": "lookup", "arguments": '{"key": "a"}'}}
                ],
            )
            return response

    engine = ToolClient(response_ids, "tool_calls")
    client = DirectModelClient(engine)
    backend = AgentInferenceBackend("policy", HISTORY_CONTEXT_LENGTH, engine.tokenize, engine.tokenizer)

    async def complete(prompt):
        return await _model_turn(
            client,
            ModelRequest(prompt.messages, prompt.options),
            sampling_params={"skip_special_tokens": False},
            max_context_length=HISTORY_CONTEXT_LENGTH,
            max_prompt_length=HISTORY_CONTEXT_LENGTH,
            chat_template_kwargs={},
            session_id=AGENT_TRIAL_ID,
        )

    messages = [{"role": "user", "content": "go"}]
    for _ in range(3):
        if agent_transport:
            with backend.session(AGENT_TRIAL_ID, complete):
                response = await backend.chat_completion(
                    {
                        "headers": {TRIAL_HEADER: AGENT_TRIAL_ID},
                        "json": {"messages": list(messages)},
                    }
                )
            assert response["choices"][0]["token_ids"] == response_ids
            message = response["choices"][0]["message"]
        else:
            output = await client.generate(
                InferenceEngineInput(
                    prompts=[list(messages)],
                    chat_completion_params=[{}],
                    sampling_params={"skip_special_tokens": False},
                )
            )
            assert output["responses"] == [reply]
            assert output["response_ids"] == [response_ids]
            message = output["assistant_messages"][0]
        messages.extend([message, {"role": "tool", "tool_call_id": "lookup", "content": "12"}])

    initial = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\ngo<|eot_id|>\n"
    historical_turn = (
        f"<|start_header_id|>assistant<|end_header_id|>\n{content}<tool_call>\n"
        '{"name": "lookup", "arguments": {"key": "a"}}\n</tool_call><|eot_id|>'
        '<|start_header_id|>tool<|end_header_id|>\n<tool_response name="lookup">12</tool_response><|eot_id|>\n'
    )
    generation = "<|start_header_id|>assistant<|end_header_id|>\n"
    assert [text.encode() for text in rendered] == [
        (initial + historical_turn * turns + generation).encode() for turns in range(3)
    ]


@pytest.mark.parametrize(
    "finish_reason,expected",
    [("stop", "<think>plan</think>call"), ("length", "<think>plan</think>call<|eot_id|>")],
)
def test_completion_reply_excludes_the_terminator_of_an_ended_turn(finish_reason, expected):
    response_ids = [THINK_START, PLAN, THINK_END, CALL, END_OF_TURN]

    class Client(_InferenceClient):
        async def generate(self, request):
            return {
                "responses": ["<think>plan</think>call<|eot_id|>"],
                "response_ids": [self.response_ids],
                "stop_reasons": [self.finish_reason],
            }

    client = DirectModelClient(Client(response_ids, finish_reason))
    request = InferenceEngineInput(prompt_token_ids=[[PLAN]], sampling_params={"skip_special_tokens": False})
    output = asyncio.run(client.generate(request))

    assert output["responses"] == [expected]
    assert output["response_ids"] == [response_ids]


@pytest.mark.asyncio
async def test_agent_http_callback_restores_the_registering_trial_clock():
    clock = InferenceClock(2280)

    async def tokenize(_request):
        raise AssertionError("A completed turn does not need tokenization")

    async def complete(_prompt):
        assert current_inference_clock() is clock
        return ModelTurn({"role": "assistant", "content": "12"}, (1,), (2,), None, "stop")

    backend = AgentInferenceBackend("fixture", 1024, tokenize, _LlamaStyleTokenizer())
    with clock.scope(), backend.session("trial", complete):
        callback = asyncio.create_task(
            backend.chat_completion(
                {
                    "headers": {TRIAL_HEADER: "trial"},
                    "json": {"messages": [{"role": "user", "content": "six plus six"}]},
                }
            ),
            context=Context(),
        )
        result = await callback
    assert current_inference_clock() is None
    assert result["choices"][0]["message"]["content"] == "12"
