# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from levanter.data.text.formats import ChatProcessor
from levanter.tokenizers import MarinTokenizer
from transformers import AutoTokenizer

from experiments.marin_tokenizer import MARIN_CHAT_TEMPLATE, inject_special_tokens

REASONING_TRACE = (
    "<|start_think|>User is asking how am I doing. This should be straightforward. I should reply politely.<|end_think|>"
)

CONVERSATION = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": REASONING_TRACE + "I'm doing well, thanks!"},
    {"role": "user", "content": "That's good to hear!"},
    {"role": "assistant", "content": "Great!"},
]

QUESTION = [
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "The answer is 4."},
]


@pytest.fixture
def marin_chat_tokenizer(gpt2_tokenizer) -> MarinTokenizer:
    return gpt2_tokenizer.with_chat_template(MARIN_CHAT_TEMPLATE)


def _decode(tokenizer, ids) -> str:
    return tokenizer.decode(list(ids), skip_special_tokens=False)


def test_inject_special_tokens_renames_reserved_slots(gpt2_tokenizer_path):
    base = AutoTokenizer.from_pretrained(gpt2_tokenizer_path, local_files_only=True)
    base.add_special_tokens(
        {"additional_special_tokens": ["<|reserved_special_token_0|>", "<|reserved_special_token_1|>"]}
    )
    reserved_ids = base.convert_tokens_to_ids(["<|reserved_special_token_0|>", "<|reserved_special_token_1|>"])
    plain_text = "Hello, how are you?"
    plain_ids = base.encode(plain_text, add_special_tokens=False)

    replacements = dict(zip(reserved_ids, ["<|start_think|>", "<|end_think|>"], strict=True))
    prepared = inject_special_tokens(base, replacements)

    assert prepared.encode(plain_text, add_special_tokens=False) == plain_ids
    for token_id, token_str in replacements.items():
        assert prepared.encode(token_str, add_special_tokens=False) == [token_id]
        assert prepared.decode([token_id]) == token_str


def test_assistant_mask_covers_only_assistant_turns(marin_chat_tokenizer: MarinTokenizer):
    result = marin_chat_tokenizer.apply_chat_template_with_masks([CONVERSATION])
    input_ids = np.array(result["input_ids"][0])
    assistant_mask = np.array(result["assistant_masks"][0]).astype(bool)

    masked = marin_chat_tokenizer.decode(input_ids[assistant_mask].tolist())
    assert REASONING_TRACE + "I'm doing well, thanks!" in masked
    assert "Great!" in masked
    assert "Hello, how are you?" not in masked
    assert "That's good to hear!" not in masked


def test_generation_prompt(marin_chat_tokenizer: MarinTokenizer):
    rendered = marin_chat_tokenizer.apply_chat_template(CONVERSATION, tokenize=False, add_generation_prompt=True)
    assert rendered.endswith("<|start_header_id|>assistant<|end_header_id|>\n")


@pytest.mark.parametrize(
    "enable_thinking,expected",
    [(True, "Reasoning: /think"), (False, "Reasoning: /nothink"), ("experimental", "Reasoning: experimental")],
)
def test_reasoning_mode(marin_chat_tokenizer: MarinTokenizer, enable_thinking, expected):
    rendered = marin_chat_tokenizer.apply_chat_template(QUESTION, tokenize=False, enable_thinking=enable_thinking)
    assert expected in rendered


def test_tool_definitions_rendered(marin_chat_tokenizer: MarinTokenizer):
    rendered = marin_chat_tokenizer.apply_chat_template(
        QUESTION,
        tokenize=False,
        xml_tools=[
            '{"type": "function", "function": {"name": "final_answer", "description": "Provides final answers."}}',
        ],
        python_tools=[
            '{"type": "function", "function": {"name": "python_exec", "description": "Execute Python code."}}',
        ],
        enable_thinking=True,
    )
    assert "### Tools" in rendered
    assert "<tools>" in rendered
    assert "final_answer" in rendered
    assert "When you send a message containing Python code" in rendered
    assert "python_exec" in rendered


def test_chat_processor_renders_tool_calls(marin_chat_tokenizer: MarinTokenizer):
    processor = ChatProcessor(marin_chat_tokenizer, mask_user_turns=True)
    result = processor(
        [
            {
                "messages": [
                    {"role": "user", "content": "Run the VIN check."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "check_valid_vin", "arguments": {"vin": "1FMXK92W8YPA12345"}},
                            }
                        ],
                    },
                    {"role": "tool", "name": "check_valid_vin", "tool_call_id": "call_abc", "content": {"valid": True}},
                    {"role": "assistant", "content": "VIN 1FMXK92W8YPA12345 is valid."},
                ]
            }
        ]
    )[0]

    rendered = _decode(marin_chat_tokenizer, result["input_ids"])
    assert '{"name": "check_valid_vin", "arguments": {"vin": "1FMXK92W8YPA12345"}}' in rendered
    assert '<tool_response name="check_valid_vin" id="call_abc">' in rendered
    assert result["assistant_masks"].sum() > 0


def test_chat_processor_renders_ipython_output(marin_chat_tokenizer: MarinTokenizer):
    processor = ChatProcessor(marin_chat_tokenizer, mask_user_turns=True)
    result = processor(
        [
            {
                "messages": [
                    {"role": "user", "content": "Show me the result."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_output",
                                "type": "function",
                                "function": {"name": "python_exec", "arguments": {"code": "print(1+1)"}},
                            }
                        ],
                    },
                    {"role": "ipython", "content": [{"type": "text", "text": "4\n"}]},
                    {"role": "assistant", "content": "The result is 4."},
                ]
            }
        ]
    )[0]

    rendered = _decode(marin_chat_tokenizer, result["input_ids"])
    assert '{"name": "python_exec", "arguments": {"code": "print(1+1)"}}' in rendered
    assert "<|start_header_id|>ipython<|end_header_id|>" in rendered
    assert '{"output": "4\\n"}' in rendered
    assert result["assistant_masks"].sum() > 0
