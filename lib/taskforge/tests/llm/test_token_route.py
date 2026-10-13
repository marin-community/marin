# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``GlmRolloutModel`` on the token-id route against the scripted GLM server."""

import json
from dataclasses import replace

import pytest
from rolloutengine.contracts import GenerationLimitReached, ModelRequest

from taskforge.ledger.records import EntryKind
from taskforge.llm.client import TokenEvidenceError
from taskforge.llm.glm_client import GENERATE_PATH, ROUTING_KEY_HEADER, GlmClient
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import CallLedger
from taskforge.llm.rollout_model import (
    GLM_CONTEXT_WINDOW,
    TOKEN_FIELDS,
    GlmRolloutModel,
    RolloutTransport,
    TokenContractBreak,
    served_tokens,
)
from tests.conftest import ScriptedStream, fake_source

POLICY = LLMPolicy(max_continuations=0, temperature=0.6)
OBSERVATION = 9
USER = 8
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
                "required": ["command"],
            },
        },
    }
]
OPENING = ({"role": "user", "content": "make done.txt"},)
FIRST_PROMPT = (1, 2, 3)
FIRST_RESPONSE = (4, 5, OBSERVATION)
PREFIX = FIRST_PROMPT + FIRST_RESPONSE
SUFFIX = (20, 21)
AFTER_TOOL = (
    *OPENING,
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "plan",
        "tool_calls": [{"id": "call-0", "type": "function", "function": {"name": "shell", "arguments": "{}"}}],
    },
    {"role": "tool", "tool_call_id": "call-0", "content": "ok"},
)


@pytest.fixture
async def client(fake_glm, fast_provider):
    async with GlmClient(fake_source(fake_glm), replace(fast_provider, max_attempts=3)) as glm:
        yield glm


def token_model(client, ledger) -> GlmRolloutModel:
    record = CallLedger(ledger=ledger, item_id="item", round=0, step="solver/0/0")
    return GlmRolloutModel(client, POLICY, RolloutTransport.TOKEN_ROUTE, record)


def later_turn(prefix=PREFIX, messages=AFTER_TOOL) -> ModelRequest:
    return ModelRequest(messages, {"tools": TOOLS}, prefix_token_ids=prefix, assistant_message_index=1)


def with_texts(tool_result: str, reply: str = "") -> tuple[dict, ...]:
    """``AFTER_TOOL`` with the tool result and the replayed reply's content replaced."""
    opening, assistant, tool = AFTER_TOOL
    return opening, {**assistant, "content": reply}, {**tool, "content": tool_result}


# A chat-completions stream in vLLM's shape for ``return_token_ids`` with ``logprobs``: the prompt ids
# on the first chunk, and each token-carrying chunk with its ids and full logprob entries.
VLLM_CHUNKS: list[dict | str] = [
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1760000000,
        "model": "glm-5.3",
        "prompt_token_ids": [151331, 151333, 2001],
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": None, "finish_reason": None}
        ],
    },
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1760000000,
        "model": "glm-5.3",
        "choices": [
            {
                "index": 0,
                "delta": {"reasoning": "Plan"},
                "logprobs": {
                    "content": [{"token": "Plan", "logprob": -0.0123, "bytes": [80, 108, 97, 110], "top_logprobs": []}]
                },
                "finish_reason": None,
                "token_ids": [34592],
            }
        ],
    },
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1760000000,
        "model": "glm-5.3",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "ok"},
                "logprobs": {
                    "content": [
                        {"token": "ok", "logprob": -0.5, "bytes": [111, 107], "top_logprobs": []},
                        {"token": "<|user|>", "logprob": -1.25, "bytes": None, "top_logprobs": []},
                    ]
                },
                "finish_reason": "stop",
                "token_ids": [1012, 151336],
            }
        ],
    },
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1760000000,
        "model": "glm-5.3",
        "choices": [],
        "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
    },
]


async def test_served_tokens_survive_the_sdks_reserialised_chunks(fake_glm, client):
    fake_glm.responses.append(ScriptedStream(VLLM_CHUNKS, send_done=True, stall=None))

    completion = await client.complete(list(OPENING), POLICY, TOKEN_FIELDS)

    tokens = served_tokens(completion)
    assert (tokens.prompt, tokens.response) == ((151331, 151333, 2001), (34592, 1012, 151336))
    assert tokens.logprobs == (-0.0123, -0.5, -1.25)
    stored = [json.loads(event) for event in completion.attempts[0].events]
    assert stored[2]["choices"][0]["logprobs"]["content"][0]["bytes"] == [111, 107]


def script_render(fake_glm, served: tuple[int, ...]) -> None:
    """The one-token render probe: the server reports the prompt ids it rendered."""
    fake_glm.stream(
        content="x", prompt_tokens=len(served), completion_tokens=1, prompt_token_ids=served, token_ids=(99,)
    )


def generate_bodies(fake_glm) -> list[dict]:
    return [body for body, path in zip(fake_glm.requests, fake_glm.request_paths, strict=True) if path == GENERATE_PATH]


async def test_later_turn_sends_prefix_plus_rendered_suffix_by_token_ids(fake_glm, client, ledger):
    fake_glm.stream(
        content="",
        reasoning="plan",
        tool_calls=(("shell", '{"command": "ls"}'),),
        finish="tool_calls",
        prompt_tokens=3,
        completion_tokens=3,
        prompt_token_ids=FIRST_PROMPT,
        token_ids=FIRST_RESPONSE,
    )
    script_render(fake_glm, PREFIX + SUFFIX)
    fake_glm.generate(
        (30, 31, USER), (b"fine", b"</think>done", b"<|user|>"), prompt_tokens=len(PREFIX + SUFFIX), cached_tokens=6
    )
    model = token_model(client, ledger)

    first = await model(ModelRequest(OPENING, {"tools": TOOLS}, prefix_token_ids=(), assistant_message_index=None))
    second = await model(later_turn(first.prompt_token_ids + first.response_token_ids))

    [body] = generate_bodies(fake_glm)
    assert body["token_ids"] == list(PREFIX + SUFFIX)
    assert body["sampling_params"] == {"max_tokens": POLICY.max_tokens, "temperature": 0.6, "top_p": 0.95, "logprobs": 0}
    assert body["return_token_bytes"] is True
    # The turn-0 chat call, the render probe and the token-route call carry the same key.
    assert [fake_glm.requests[0]["prompt_cache_key"], fake_glm.requests[1]["prompt_cache_key"]] == [
        "item:0:solver/0/0"
    ] * 2
    assert fake_glm.request_headers[-1][ROUTING_KEY_HEADER] == "item:0:solver/0/0"
    assert first.metadata["transport"] == "chat_completions"
    assert second.prompt_token_ids == PREFIX + SUFFIX
    assert second.response_token_ids == (30, 31, USER)
    assert second.message == {"role": "assistant", "content": "done", "reasoning_content": "fine"}
    assert (second.stop_reason, second.metadata["transport"], second.metadata["alignment"]) == (
        "stop",
        "token_route",
        "prefix",
    )
    assert [entry.attrs.get("purpose", entry.attrs.get("transport")) for entry in ledger.entries] == [
        "chat_completions",
        "render",
        "token_route",
    ]
    route = ledger.entries[-1]
    assert (route.tokens_in, route.tokens_out, route.tokens_reasoning, route.cause) == (8, 3, 2, None)
    assert route.attrs["cached_tokens"] == "6"


async def test_token_route_and_chat_turns_send_the_same_sampling_fields(fake_glm, client, ledger):
    fake_glm.stream(content="ok", prompt_tokens=3, completion_tokens=2, prompt_token_ids=FIRST_PROMPT, token_ids=(4, 5))
    script_render(fake_glm, PREFIX + SUFFIX)
    fake_glm.generate((30, USER), (b"</think>ok", b"<|user|>"), prompt_tokens=len(PREFIX + SUFFIX))
    model = token_model(client, ledger)

    await model(ModelRequest(OPENING, {}, prefix_token_ids=(), assistant_message_index=None))
    await model(later_turn())

    chat, generate = fake_glm.requests[0], generate_bodies(fake_glm)[0]["sampling_params"]
    sent = [{field: body[field] for field in ("temperature", "top_p")} for body in (chat, generate)]
    assert sent == [{"temperature": 0.6, "top_p": 0.95}] * 2


async def test_characters_split_across_tokens_decode_from_the_joined_bytes(fake_glm, client, ledger):
    script_render(fake_glm, PREFIX + SUFFIX)
    emoji, cjk = "\N{ROCKET}".encode(), "\u5b8c\u6210".encode()
    sampled = (b"</think>", emoji[:2], emoji[2:] + cjk[:1], cjk[1:4], cjk[4:], b"<|user|>")
    fake_glm.generate((30, 31, 32, 33, 34, USER), sampled, prompt_tokens=len(PREFIX + SUFFIX))

    turn = await token_model(client, ledger)(later_turn())

    assert turn.message["content"] == "\N{ROCKET}\u5b8c\u6210"


async def test_stop_text_inside_tool_results_does_not_move_the_boundary(fake_glm, client, ledger):
    # Each tool result says "<|observation|>", which the server renders as the stop id. The prefix
    # got the earlier one from a server render too, and the new one lies after the boundary.
    prefix = (1, OBSERVATION, 2, 3, 4, 5, OBSERVATION)
    suffix = (20, OBSERVATION, 21)
    script_render(fake_glm, (1, OBSERVATION, 2, 3, 45, OBSERVATION, *suffix))
    fake_glm.generate((30, USER), (b"</think>ok", b"<|user|>"), prompt_tokens=len(prefix + suffix))

    turn = await token_model(client, ledger)(later_turn(prefix, with_texts("saw <|observation|> here")))

    assert turn.metadata["alignment"] == "boundary"
    assert generate_bodies(fake_glm)[0]["token_ids"] == list(prefix + suffix)


async def test_stop_text_in_a_replayed_reply_breaks_the_contract_when_the_render_moved(fake_glm, client, ledger):
    script_render(fake_glm, (1, 2, 3, 45, OBSERVATION, *SUFFIX))

    with pytest.raises(TokenContractBreak, match="<\\|user\\|>"):
        await token_model(client, ledger)(later_turn(messages=with_texts("ok", reply="wrote <|user|> literally")))

    assert generate_bodies(fake_glm) == []
    assert ledger.entries[-1].cause == "token_contract"


async def test_retokenized_prefix_no_longer_breaks_the_rollout(fake_glm, client, ledger):
    # The server serves the sampled pair (4, 5) back as the single id 45, as it does for a sampled
    # ")," "(" that re-tokenizes to "),(".
    script_render(fake_glm, (1, 2, 3, 45, OBSERVATION, *SUFFIX))
    fake_glm.generate((30, USER), (b"</think>ok", b"<|user|>"), prompt_tokens=len(PREFIX + SUFFIX))

    turn = await token_model(client, ledger)(later_turn())

    assert turn.prompt_token_ids == PREFIX + SUFFIX
    assert turn.metadata["alignment"] == "boundary"
    assert generate_bodies(fake_glm)[0]["token_ids"] == list(PREFIX + SUFFIX)


async def test_tool_call_parsed_from_sampled_bytes(fake_glm, client, ledger):
    script_render(fake_glm, PREFIX + SUFFIX)
    sampled = (
        b"plan</think>",
        b"<tool_call>shell\n<arg_key>command</arg_key>\n<arg_value>ls -la</arg_value>\n",
        b"<arg_key>timeout</arg_key>\n<arg_value>3</arg_value>\n</tool_call>",
        b"<|observation|>",
    )
    fake_glm.generate((30, 31, 32, OBSERVATION), sampled, prompt_tokens=len(PREFIX + SUFFIX))

    turn = await token_model(client, ledger)(later_turn())

    assert turn.message["reasoning_content"] == "plan"
    [call] = turn.message["tool_calls"]
    assert call["function"]["name"] == "shell"
    assert json.loads(call["function"]["arguments"]) == {"command": "ls -la", "timeout": 3}
    assert turn.stop_reason == "tool_calls"


async def test_prompt_filling_the_window_raises_generation_limit_with_the_prompt(fake_glm, client, ledger):
    prefix = (7,) * (GLM_CONTEXT_WINDOW - 1) + (OBSERVATION,)
    script_render(fake_glm, prefix + SUFFIX)

    with pytest.raises(GenerationLimitReached) as limit:
        await token_model(client, ledger)(later_turn(prefix))

    assert limit.value.prompt_token_ids == prefix + SUFFIX
    assert generate_bodies(fake_glm) == []


async def test_route_context_overflow_raises_generation_limit_with_the_prompt(fake_glm, client, ledger):
    script_render(fake_glm, PREFIX + SUFFIX)
    fake_glm.status(400, "This model's maximum context length is 262144 tokens. However, you requested more.")

    with pytest.raises(GenerationLimitReached) as limit:
        await token_model(client, ledger)(later_turn())

    assert limit.value.prompt_token_ids == PREFIX + SUFFIX
    assert len(generate_bodies(fake_glm)) == 1


@pytest.mark.parametrize(
    ("token_bytes", "logprob_count", "detail"),
    [
        ((b"a", b"</think>b", b"<|user|>"), 2, "logprob"),
        ((b"a", None, b"<|user|>"), None, "without bytes"),
    ],
    ids=["misaligned-logprobs", "sampled-id-without-bytes"],
)
async def test_token_evidence_error_fails_the_turn_without_a_contract_break(
    fake_glm, client, ledger, token_bytes, logprob_count, detail
):
    script_render(fake_glm, PREFIX + SUFFIX)
    fake_glm.generate((30, 31, USER), token_bytes, prompt_tokens=8, logprob_count=logprob_count)

    with pytest.raises(TokenEvidenceError, match=detail):
        await token_model(client, ledger)(later_turn())

    assert len(generate_bodies(fake_glm)) == 1
    route = ledger.entries[-1]
    assert (route.kind, route.cause) == (EntryKind.LLM_CALL, "TokenEvidenceError")


async def test_chat_transport_records_a_retokenized_prefix_as_a_contract_break(fake_glm, client, ledger):
    fake_glm.stream(
        content="ok",
        prompt_tokens=7,
        completion_tokens=2,
        prompt_token_ids=(1, 2, 3, 45, OBSERVATION, *SUFFIX),
        token_ids=(30, USER),
    )
    record = CallLedger(ledger=ledger, item_id="item", round=0, step="solver/0/0")
    model = GlmRolloutModel(client, POLICY, RolloutTransport.CHAT_COMPLETIONS, record)

    with pytest.raises(TokenContractBreak) as error:
        await model(later_turn())

    assert error.value.messages == AFTER_TOOL
    [entry] = ledger.entries
    assert (entry.cause, entry.attrs["turn"], entry.attrs["transport"]) == ("token_contract", "1", "chat_completions")
    assert "re-tokenized" in entry.attrs["detail"]
