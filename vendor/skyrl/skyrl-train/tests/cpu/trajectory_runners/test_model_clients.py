# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


import functools
from unittest.mock import AsyncMock

import pytest
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from transformers import AutoTokenizer

# Public Snowball Step92 tokenizer and chat template.
SNOWBALL_TOKENIZER = (
    "open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Async-Step92",
    "929fc490219ed1f75883863518220270b40d0f2a",
)
# The Hub tokenizer that Hero's recipe pins.
MARIN_TOKENIZER = ("marin-community/marin-tokenizer", "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2")
SHELL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }
]


@functools.cache
def _tokenizer(name: str, revision: str):
    return AutoTokenizer.from_pretrained(name, revision=revision)


def _template_tokenize(tokenizer):
    async def tokenize(payload):
        body = dict(payload["json"])
        text = tokenizer.apply_chat_template(body.pop("messages"), tokenize=False, **body)
        return {"tokens": tokenizer.encode(text, add_special_tokens=False)}

    return tokenize


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,revision",
    [pytest.param(*SNOWBALL_TOKENIZER, id="snowball"), pytest.param(*MARIN_TOKENIZER, id="marin-tokenizer")],
)
@pytest.mark.parametrize("tool_call", [False, True])
@pytest.mark.parametrize("stop_string", [False, True])
async def test_grug_chat_continuation_keeps_served_tokens(model, revision, tool_call, stop_string):
    tokenizer = _tokenizer(model, revision)
    history = [{"role": "user", "content": "List files."}]
    assistant = {"role": "assistant", "content": "Paris"}
    observation = {"role": "user", "content": "Why?"}
    sampled = "Paris"
    if tool_call:
        assistant = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "shell", "arguments": '{"command":"ls"}'}}
            ],
        }
        observation = {"role": "tool", "tool_call_id": "c1", "content": "a.txt"}
        sampled = '{"name":"shell","arguments":{"command":"ls"}}'
        if model == SNOWBALL_TOKENIZER[0]:
            sampled = "<tool_call>\n" + sampled + "\n</tool_call>"
    if not stop_string:
        sampled += "<|eot_id|>"
    prompt = tokenizer.apply_chat_template(history, tools=SHELL_TOOLS, tokenize=False, add_generation_prompt=True)
    served = tokenizer.encode(prompt + sampled, add_special_tokens=False)

    response_ids = tokenizer.encode("Done.<|eot_id|>", add_special_tokens=False)
    engine = AsyncMock()
    engine.model_name = model
    engine.tokenizer = tokenizer
    engine.tokenize.side_effect = _template_tokenize(tokenizer)
    engine.chat_completion.return_value = {
        "choices": [
            {"message": {"role": "assistant", "content": "Done."}, "finish_reason": "stop", "token_ids": response_ids}
        ]
    }
    result = await DirectModelClient(engine).generate(
        {
            "prompts": [[*history, assistant, observation]],
            "chat_completion_params": [{"tools": SHELL_TOOLS}],
            "chat_continuations": [{"assistant_message_index": 1, "served_prefix_token_ids": served}],
        }
    )

    suffix = "<|eot_id|>" if stop_string else ""
    if tool_call:
        suffix += '\n<|start_header_id|>tool<|end_header_id|>\n<tool_response id="c1">a.txt</tool_response><|eot_id|>\n'
    else:
        suffix += "<|start_header_id|>user<|end_header_id|>\nWhy?<|eot_id|>\n"
    suffix += "<|start_header_id|>assistant<|end_header_id|>\n"
    assert result["prompt_ids"] == [[*served, *tokenizer.encode(suffix, add_special_tokens=False)]]
    assert result["response_ids"] == [response_ids]


@pytest.mark.asyncio
@pytest.mark.parametrize("template_case", ["eom-with-literal", "omitted-tool-turn"])
async def test_chat_continuation_rejects_closers_outside_the_assistant_boundary(template_case):
    tokenizer = _tokenizer(*SNOWBALL_TOKENIZER)
    template = tokenizer.chat_template.replace(
        "{%- endfor %}<|eot_id|>\n{% endgeneration %}", "{%- endfor %}<|eom_id|>{% endgeneration %}"
    )
    command = "ls"
    if template_case == "eom-with-literal":
        command = "echo <|eot_id|>"
    elif template_case == "omitted-tool-turn":
        template = tokenizer.chat_template.replace("{% elif has_tool_calls -%}", "{% elif has_tool_calls and false -%}")
    history = [{"role": "user", "content": "List files."}]
    messages = [
        *history,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "shell", "arguments": '{"command":"' + command + '"}'}}
            ],
        },
        {"role": "tool", "content": "a.txt"},
    ]
    prompt = tokenizer.apply_chat_template(history, tokenize=False, chat_template=template, add_generation_prompt=True)
    # The space before the closing brace is valid JSON that the template does not render.
    sampled = '<tool_call>\n{"name":"shell","arguments":{"command":"' + command + '"} }\n</tool_call>'
    sampled += "<|eot_id|>" if template_case == "omitted-tool-turn" else "<|eom_id|>"
    served = tokenizer.encode(prompt + sampled, add_special_tokens=False)

    engine = AsyncMock()
    engine.model_name = SNOWBALL_TOKENIZER[0]
    engine.tokenizer = tokenizer
    engine.tokenize.side_effect = _template_tokenize(tokenizer)
    engine.chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Done."},
                "finish_reason": "stop",
                "token_ids": tokenizer.encode("Done.<|eot_id|>", add_special_tokens=False),
            }
        ]
    }
    with pytest.raises(RuntimeError):
        await DirectModelClient(engine).generate(
            {
                "prompts": [messages],
                "chat_completion_params": [{"chat_template": template}],
                "chat_continuations": [{"assistant_message_index": 1, "served_prefix_token_ids": served}],
            }
        )
