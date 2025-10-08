# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

import pytest
from transformers import AutoTokenizer

# Add the submodule to the path
sys.path.insert(0, "/Users/power/code/marin/submodules/levanter/src")

from levanter.inference.openai import ChatMessage


@pytest.mark.parametrize(
    "model_name",
    [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ],
)
def test_apply_chat_template_with_exclude_none(model_name):
    """Test that apply_chat_template works when using exclude_none=True."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create a simple user message like the one in the log
    messages = [
        ChatMessage(
            role="user",
            content="Bigger: 87 or 3? Just the number:",
        )
    ]

    # Use model_dump(exclude_none=True) to avoid passing None values to the template
    dict_messages = [msg.model_dump(exclude_none=True) for msg in messages]

    # This should work without errors
    tokens = tokenizer.apply_chat_template(dict_messages, tokenize=True, add_generation_prompt=True)
    assert len(tokens) > 0

    # Verify the decoded output contains the expected content
    decoded = tokenizer.decode(tokens)
    assert "Bigger: 87 or 3?" in decoded
    assert "assistant" in decoded  # Should have assistant prompt


def test_apply_chat_template_fails_with_none_values():
    """Test that apply_chat_template fails when tool_calls is None (not excluded)."""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    messages = [
        ChatMessage(
            role="user",
            content="Test message",
        )
    ]

    # Using model_dump() includes None values which breaks the template
    dict_messages_with_none = [msg.model_dump() for msg in messages]

    # This should fail because the Llama template checks 'tool_calls' in message
    # and then tries to get the length of None
    with pytest.raises(Exception, match="NoneType"):
        tokenizer.apply_chat_template(dict_messages_with_none, tokenize=True, add_generation_prompt=True)


def test_apply_chat_template_with_system_message():
    """Test apply_chat_template with system and user messages."""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="What is 2+2?"),
    ]

    dict_messages = [msg.model_dump(exclude_none=True) for msg in messages]
    tokens = tokenizer.apply_chat_template(dict_messages, tokenize=True, add_generation_prompt=True)
    decoded = tokenizer.decode(tokens)

    assert "helpful assistant" in decoded
    assert "What is 2+2?" in decoded


def test_token_roundtrip_with_decode_encode():
    """Test that decode->encode round-trips are not always consistent.

    This test demonstrates the issue where decoding a token ID to a string
    and then encoding it back may not produce the same token ID.

    The specific case from the production error:
    - Token 77755 decodes to " !!}" (with leading space)
    - But decode() strips BPE markers, producing "!!}"
    - Re-encoding "!!}" produces [3001, 92] instead of [77755]
    """
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    # Test the specific problematic token from the error
    # Token 77755 is " !!}" (with leading space in BPE)
    token_id = 77755

    # Decode it
    decoded_str = tokenizer.decode([token_id], skip_special_tokens=False)
    print(f"\nToken {token_id} decodes to: {decoded_str!r}")

    # Try to re-encode
    re_encoded = tokenizer.encode(decoded_str, add_special_tokens=False)
    print(f"Re-encoding produces: {re_encoded}")

    # This will fail - decode/encode does not round-trip for BPE tokens with spaces
    assert re_encoded != [token_id], (
        f"Expected decode/encode to fail for token {token_id}, "
        f"but it round-tripped successfully. This test documents the bug."
    )
    assert re_encoded == [3001, 92], f"Expected [3001, 92] but got {re_encoded}"


def test_token_roundtrip_with_convert_methods():
    """Test that convert_ids_to_tokens->convert_tokens_to_ids round-trips correctly.

    This demonstrates the proper way to maintain token ID consistency.
    convert_ids_to_tokens preserves BPE space markers (Ġ), allowing perfect round-trips.
    """
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    # Test the specific problematic token that fails with decode/encode
    token_id = 77755

    # Convert to token string (preserves BPE format with Ġ for space)
    token_str = tokenizer.convert_ids_to_tokens(token_id)
    print(f"\nToken {token_id} converts to: {token_str!r}")

    # Convert back to ID
    recovered_id = tokenizer.convert_tokens_to_ids(token_str)
    print(f"Converting back produces: {recovered_id}")

    # This should work - convert methods preserve BPE format
    assert recovered_id == token_id, (
        f"convert_ids_to_tokens/convert_tokens_to_ids round-trip failed: "
        f"{token_id} -> '{token_str}' -> {recovered_id}"
    )

    # Test with various other tokens to ensure general correctness
    test_texts = [
        "!!}",
        "Hello world",
        "  spaces  ",
        "123",
        "\n\n",
    ]

    for text in test_texts:
        original_tokens = tokenizer.encode(text, add_special_tokens=False)
        for token_id in original_tokens:
            token_str = tokenizer.convert_ids_to_tokens(token_id)
            recovered_id = tokenizer.convert_tokens_to_ids(token_str)
            assert recovered_id == token_id, (
                f"Round-trip failed for text '{text}': " f"{token_id} -> '{token_str}' -> {recovered_id}"
            )
