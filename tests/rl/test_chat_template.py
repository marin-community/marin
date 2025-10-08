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

import os

import pytest
from levanter.inference.openai import ChatMessage
from transformers import AutoTokenizer


def test_chat_template_and_token_conversion():
    """Test chat template with exclude_none and validate token conversion methods."""
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    # Test 1: Chat template with exclude_none=True works
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Bigger: 87 or 3? Just the number:"),
    ]
    dict_messages = [msg.model_dump(exclude_none=True) for msg in messages]
    tokens = tokenizer.apply_chat_template(dict_messages, tokenize=True, add_generation_prompt=True)
    decoded = tokenizer.decode(tokens)
    assert "helpful assistant" in decoded and "Bigger: 87 or 3?" in decoded and "assistant" in decoded

    # Test 2: Verify convert_ids_to_tokens/convert_tokens_to_ids round-trip
    token_id = 12345  # Valid token ID within Mistral's vocab
    token_str = tokenizer.convert_ids_to_tokens(token_id)
    recovered_id = tokenizer.convert_tokens_to_ids(token_str)
    assert recovered_id == token_id  # convert methods preserve token format

    # Test 3: Validate round-trip for various texts
    for text in ["!!}", "Hello world", "  spaces  ", "123", "\n\n"]:
        for token_id in tokenizer.encode(text, add_special_tokens=False):
            token_str = tokenizer.convert_ids_to_tokens(token_id)
            assert tokenizer.convert_tokens_to_ids(token_str) == token_id
