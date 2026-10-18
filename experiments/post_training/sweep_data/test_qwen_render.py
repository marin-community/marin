# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.post_training.sweep_data.qwen_render import (
    GENERATION_PROMPT,
    QWEN_MODELS,
    every_turn_thinks,
    mask_turns,
    render,
    served_messages,
)

MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    }
]


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL, revision=QWEN_MODELS[MODEL])
    except OSError as error:
        pytest.skip(f"Qwen tokenizer unavailable: {error}")


def call(command: str) -> dict:
    return {"type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": command})}}


def tmax_row(empty_reasoning_at: int | None = None) -> list[dict]:
    """A TMax-shaped trajectory: system, one user message, then assistant tool calls and tool output."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can interact with a computer."},
        {"role": "user", "content": "Count the lines in /home/user/data.csv."},
    ]
    for turn, command in enumerate(["ls /home/user", "wc -l /home/user/data.csv"]):
        reasoning = "" if turn == empty_reasoning_at else f"Step {turn}: I should run {command}."
        messages.append(
            {
                "role": "assistant",
                "content": f"THOUGHT: run {command}.",
                "reasoning_content": reasoning,
                "tool_calls": [call(command)],
            }
        )
        messages.append({"role": "tool", "content": f"output of {command}\n"})
    messages.append({"role": "assistant", "content": "THOUGHT: done.", "reasoning_content": "The file has 3 lines."})
    return messages


def trained_texts(tokenizer, rendered) -> list[str]:
    return [tokenizer.decode(rendered.ids[start:end]) for start, end in rendered.turn_spans]


def test_single_user_row_is_byte_identical_to_the_native_template(tokenizer):
    messages = tmax_row()
    native = tokenizer.apply_chat_template(served_messages(messages), tools=TOOLS, tokenize=False)
    rendered = render(tokenizer, every_turn_thinks(tokenizer.chat_template), messages, TOOLS)
    assert rendered.text == native


def test_trained_spans_are_exactly_each_turns_output_after_the_generation_prompt(tokenizer):
    messages = tmax_row()
    template = every_turn_thinks(tokenizer.chat_template)
    rendered = render(tokenizer, template, messages, TOOLS)
    assistants = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
    assert len(rendered.turn_spans) == len(assistants)
    for (start, end), index in zip(rendered.turn_spans, assistants, strict=True):
        prompt = tokenizer.apply_chat_template(
            served_messages(messages[:index]),
            tools=TOOLS,
            chat_template=template,
            tokenize=False,
            add_generation_prompt=True,
        )
        assert prompt.endswith(GENERATION_PROMPT)
        # Context before the turn is the inference prompt, token for token; the turn is the model's output.
        assert rendered.ids[:start] == tokenizer(prompt, add_special_tokens=False)["input_ids"]
        output = rendered.text[len(prompt) :].split("<|im_end|>", 1)[0] + "<|im_end|>"
        assert rendered.ids[start:end] == tokenizer(output, add_special_tokens=False)["input_ids"]
        message = messages[index]
        assert output.startswith(message["reasoning_content"] + "\n</think>\n\n" + message["content"])
        assert all(rendered.mask[start:end]) and rendered.ids[end - 1] == tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert rendered.supervised == sum(end - start for start, end in rendered.turn_spans) - (rendered.mask[0] == 1)
    trained = {i for start, end in rendered.turn_spans for i in range(start, end)}
    assert all(rendered.mask[i] == (i in trained) for i in range(len(rendered.ids)))


def test_empty_reasoning_turn_trains_the_closing_think_as_emitted(tokenizer):
    rendered = render(tokenizer, every_turn_thinks(tokenizer.chat_template), tmax_row(empty_reasoning_at=0), TOOLS)
    start, _ = rendered.turn_spans[0]
    assert tokenizer.decode(rendered.ids[start - 2 : start]) == "<think>\n"
    assert trained_texts(tokenizer, rendered)[0].startswith("\n</think>\n\nTHOUGHT: run ls")


def test_terminus_style_row_keeps_every_think_block_unlike_native(tokenizer):
    messages = [{"role": "user", "content": "Task"}]
    for turn in range(3):
        messages.append({"role": "assistant", "content": f'{{"step": {turn}}}', "reasoning_content": f"think {turn}"})
        messages.append({"role": "user", "content": f"New Terminal Output:\n{turn}"})
    messages.pop()
    rendered = render(tokenizer, every_turn_thinks(tokenizer.chat_template), messages)
    native = tokenizer.apply_chat_template(messages, tokenize=False)
    assert rendered.text.count("<think>") == 3 and native.count("<think>") == 1


def test_tool_call_arguments_render_as_vllm_parses_them(tokenizer):
    messages = tmax_row()
    messages[2]["tool_calls"] = [
        {"type": "function", "function": {"name": "bash", "arguments": '{"command":"ls /home/user"}'}}
    ]
    rendered = render(tokenizer, every_turn_thinks(tokenizer.chat_template), messages, TOOLS)
    assert '{"name": "bash", "arguments": {"command": "ls /home/user"}}' in rendered.text


def test_mask_turns_zeroes_untrained_turns(tokenizer):
    rendered = render(tokenizer, every_turn_thinks(tokenizer.chat_template), tmax_row(), TOOLS)
    masked = mask_turns(rendered, [True, False, True])
    start, end = rendered.turn_spans[1]
    assert not any(masked.mask[start:end]) and masked.supervised == rendered.supervised - (end - start)
    with pytest.raises(ValueError):
        mask_turns(rendered, [True])


def test_patch_refuses_an_unknown_template():
    with pytest.raises(ValueError):
        every_turn_thinks("{{ messages }}")


def test_developer_messages_merge_into_the_system_message_as_vllm_does(tokenizer):
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "developer", "content": "Answer in French."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Bonjour", "reasoning_content": "Greet."},
    ]
    rendered = render(tokenizer, every_turn_thinks(tokenizer.chat_template), messages)
    assert rendered.text.startswith("<|im_start|>system\nBe brief.\n\nAnswer in French.<|im_end|>\n<|im_start|>user\nHi")
