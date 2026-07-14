# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile

import numpy as np
import pytest
from levanter.data.text.formats import ChatProcessor
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizer

from experiments.marin_tokenizer import (
    MARIN_CUSTOM_SPECIAL_TOKENS,
    create_marin_tokenizer,
    load_llama3_tokenizer,
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


@pytest.fixture(scope="module")
def marin_tokenizer_dir():
    """Build the marin tokenizer once per module and save it to disk.

    The base llama3 tokenizer is gated on the Hugging Face Hub; skip the whole
    module when it (or the network) is unavailable - these tests exercise our
    tokenizer surgery, not HF auth.
    """
    try:
        base = load_llama3_tokenizer()
    except Exception as e:
        pytest.skip(f"Llama 3 tokenizer is unavailable (gated repo or no network): {e}")
    with tempfile.TemporaryDirectory() as path:
        create_marin_tokenizer(base).save_pretrained(path)
        yield path


@pytest.fixture
def marin_tokenizer(marin_tokenizer_dir) -> PreTrainedTokenizer:
    """The marin tokenizer as a Hugging Face PreTrainedTokenizer."""
    return AutoTokenizer.from_pretrained(marin_tokenizer_dir, local_files_only=True)


@pytest.fixture
def marin_chat_tokenizer(marin_tokenizer_dir) -> MarinTokenizer:
    """The marin tokenizer wrapped as a levanter MarinTokenizer, for ChatProcessor tests."""
    load_tokenizer.cache_clear()
    return load_tokenizer(marin_tokenizer_dir)


def _decode(tokenizer, ids) -> str:
    return tokenizer.decode(list(ids), skip_special_tokens=False)


def test_special_tokens_injection(marin_tokenizer: PreTrainedTokenizer):
    """The reserved llama3 slots decode to the marin think tokens and round-trip back."""
    for token_id, token_str in MARIN_CUSTOM_SPECIAL_TOKENS.items():
        assert marin_tokenizer.decode(token_id) == token_str
        assert marin_tokenizer.convert_tokens_to_ids([token_str]) == [token_id]


def test_base_tokenization_preserved(marin_tokenizer: PreTrainedTokenizer):
    """Plain-text tokenization is unchanged from the base llama3 tokenizer."""
    assert marin_tokenizer.tokenize("Hello, how are you?") == load_llama3_tokenizer().tokenize("Hello, how are you?")


def test_assistant_mask_covers_assistant_turns(marin_tokenizer: PreTrainedTokenizer):
    """The assistant mask selects exactly the assistant turns and decodes back to them."""
    out = marin_tokenizer.apply_chat_template(
        CONVERSATION, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
    )

    expected_length = len(marin_tokenizer(REASONING_TRACE + "I'm doing well, thanks!")["input_ids"]) + len(
        marin_tokenizer("Great!")["input_ids"]
    )
    assert np.sum(out["assistant_masks"]) == expected_length

    ids = np.array(out["input_ids"])
    masked = marin_tokenizer.decode(ids[np.array(out["assistant_masks"]).astype(bool)])
    assert masked == REASONING_TRACE + "I'm doing well, thanks!<|eot_id|>Great!<|eot_id|>"


def test_generation_prompt(marin_tokenizer: PreTrainedTokenizer):
    """add_generation_prompt ends the render with an open assistant header."""
    rendered = marin_tokenizer.apply_chat_template(CONVERSATION, tokenize=False, add_generation_prompt=True)
    assert rendered.endswith("<|start_header_id|>assistant<|end_header_id|>\n")


@pytest.mark.parametrize(
    "enable_thinking,expected",
    [(True, "Reasoning: /think"), (False, "Reasoning: /nothink"), ("experimental", "Reasoning: experimental")],
)
def test_reasoning_mode(marin_tokenizer: PreTrainedTokenizer, enable_thinking, expected):
    """enable_thinking drives the Reasoning header in the system prompt."""
    rendered = marin_tokenizer.apply_chat_template(QUESTION, tokenize=False, enable_thinking=enable_thinking)
    assert expected in rendered


def test_tool_definitions_rendered(marin_tokenizer: PreTrainedTokenizer):
    """xml_tools and python_tools are emitted in the Tools section of the system prompt."""
    rendered = marin_tokenizer.apply_chat_template(
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
    """A tool-call turn and its tool response render through levanter's ChatProcessor."""
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
    """An ipython tool-output turn renders through levanter's ChatProcessor."""
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
