# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np
import pytest
from levanter.data.text.formats import ChatProcessor
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizer

from experiments.marin_tokenizer import (
    MARIN_CHAT_TEMPLATE,
    MARIN_CUSTOM_SPECIAL_TOKENS,
    create_marin_tokenizer,
)

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

_RESERVED_SPECIAL_TOKENS = ("<|reserved_special_token_0|>", "<|reserved_special_token_1|>")


@dataclass(frozen=True)
class MarinTokenizerFixture:
    path: str
    token_renames: dict[int, str]


@pytest.fixture(scope="module")
def marin_tokenizer_fixture(gpt2_tokenizer_path, tmp_path_factory) -> MarinTokenizerFixture:
    base = AutoTokenizer.from_pretrained(gpt2_tokenizer_path, local_files_only=True)
    base.add_special_tokens({"additional_special_tokens": list(_RESERVED_SPECIAL_TOKENS)})
    reserved_ids = base.convert_tokens_to_ids(list(_RESERVED_SPECIAL_TOKENS))
    token_renames = dict(zip(reserved_ids, MARIN_CUSTOM_SPECIAL_TOKENS.values(), strict=True))

    tokenizer = create_marin_tokenizer(base, token_renames)
    output_dir = tmp_path_factory.mktemp("marin_tokenizer")
    tokenizer.save_pretrained(output_dir)
    return MarinTokenizerFixture(path=str(output_dir), token_renames=token_renames)


@pytest.fixture(scope="module")
def marin_tokenizer(marin_tokenizer_fixture) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(marin_tokenizer_fixture.path, local_files_only=True)


@pytest.fixture(scope="module")
def marin_chat_tokenizer(marin_tokenizer_fixture) -> MarinTokenizer:
    return load_tokenizer(marin_tokenizer_fixture.path)


def _decode(tokenizer, ids) -> str:
    return tokenizer.decode(list(ids), skip_special_tokens=False)


def test_create_marin_tokenizer_preserves_base_tokens_and_renames_slots(
    gpt2_tokenizer_path,
    marin_tokenizer_fixture,
    marin_tokenizer,
):
    base = AutoTokenizer.from_pretrained(gpt2_tokenizer_path, local_files_only=True)
    plain_text = "Hello, how are you?"

    assert marin_tokenizer.encode(plain_text, add_special_tokens=False) == base.encode(
        plain_text, add_special_tokens=False
    )
    assert marin_tokenizer.chat_template == MARIN_CHAT_TEMPLATE
    for token_id, token_str in marin_tokenizer_fixture.token_renames.items():
        assert marin_tokenizer.encode(token_str, add_special_tokens=False) == [token_id]
        assert marin_tokenizer.decode([token_id]) == token_str


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
