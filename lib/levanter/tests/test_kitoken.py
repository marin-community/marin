# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Kitoken integration and compatibility with HuggingFace tokenizers.
"""

import pytest

from levanter.compat.hf_checkpoints import load_tokenizer
from test_utils import skip_if_hf_model_not_accessible


@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
def test_kitoken_wrapper_valid_text():
    """Test that KitokenWrapper correctly encodes and decodes valid text."""
    tokenizer = load_tokenizer("NousResearch/Llama-2-7b-hf")

    # Test cases with valid text
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "This is a test with numbers: 12345",
        "Special characters: @#$%^&*()",
        "Unicode text: 你好世界 👍",
        "Multi\nline\ntext",
        "Tab\tseparated\tvalues",
        "",  # Empty string
        " ",  # Single space
        "A very long sentence that contains multiple words and should test the tokenizer's ability to handle longer sequences of text without any issues whatsoever.",
    ]

    for text in test_texts:
        # Test encoding
        tokens = tokenizer.encode(text, add_special_tokens=False)
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

        # Test decoding round-trip
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        assert isinstance(decoded, str)

        # Test __call__ interface
        result = tokenizer(text, add_special_tokens=False)
        assert result["input_ids"] == tokens


@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
def test_kitoken_wrapper_garbage_data():
    """Test that KitokenWrapper handles garbage/malformed data."""
    tokenizer = load_tokenizer("NousResearch/Llama-2-7b-hf")

    # Test cases with garbage/edge case data
    garbage_texts = [
        "\x00\x01\x02\x03",  # Control characters
        "�" * 10,  # Replacement characters
        "\uffff" * 5,  # Invalid unicode
        "a" * 10000,  # Very long repeated character
        "\n" * 100,  # Many newlines
        "💩" * 50,  # Emoji spam
        "\\x00\\x01",  # Escaped sequences
        "byte\xc3\x28sequence",  # Invalid UTF-8-like sequences (as string)
        "      " * 100,  # Many spaces
        "\t\t\t\t\t" * 20,  # Many tabs
    ]

    for text in garbage_texts:
        # Should not crash
        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        assert isinstance(decoded, str)


@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
def test_kitoken_wrapper_batch_decode():
    """Test batch decoding."""
    tokenizer = load_tokenizer("NousResearch/Llama-2-7b-hf")

    batch_texts = [
        "First sentence.",
        "Second sentence with more words.",
        "Third one is short.",
    ]

    # Encode all
    token_sequences = [tokenizer.encode(text, add_special_tokens=False) for text in batch_texts]

    # Batch decode
    decoded = tokenizer.batch_decode(token_sequences, skip_special_tokens=True)
    assert len(decoded) == len(batch_texts)
    assert all(isinstance(d, str) for d in decoded)
