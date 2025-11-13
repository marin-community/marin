# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Kitoken integration and compatibility with HuggingFace tokenizers.
These tests validate that the KitokenWrapper produces identical results to the underlying HF tokenizer.
"""

import pytest
from transformers import AutoTokenizer

from levanter.compat.hf_checkpoints import load_tokenizer
from test_utils import skip_if_hf_model_not_accessible


@pytest.fixture
def valid_texts():
    """Fixture providing a variety of valid text samples for testing."""
    return [
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


@pytest.fixture
def edge_case_texts():
    """Fixture providing edge case and challenging text samples."""
    return [
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


@pytest.fixture
def batch_texts():
    """Fixture providing a batch of texts for batch processing tests."""
    return [
        "First sentence.",
        "Second sentence with more words.",
        "Third one is short.",
        "Fourth has some numbers: 42, 1337.",
        "Fifth contains unicode: 日本語テスト",
    ]


@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
def test_kitoken_single_encode_matches_hf(valid_texts):
    """Test that single text encoding matches HF tokenizer exactly."""
    kitoken_wrapper = load_tokenizer("NousResearch/Llama-2-7b-hf")
    hf_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

    for text in valid_texts:
        kitoken_tokens = kitoken_wrapper.encode(text, add_special_tokens=False)
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        assert kitoken_tokens == hf_tokens, f"Encoding mismatch for text: {text!r}"


@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
def test_kitoken_single_decode_matches_hf(valid_texts):
    """Test that single sequence decoding matches HF tokenizer exactly."""
    kitoken_wrapper = load_tokenizer("NousResearch/Llama-2-7b-hf")
    hf_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

    for text in valid_texts:
        # Use HF tokenizer to get tokens for consistent starting point
        tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        kitoken_decoded = kitoken_wrapper.decode(tokens, skip_special_tokens=False)
        hf_decoded = hf_tokenizer.decode(tokens, skip_special_tokens=False)

        assert kitoken_decoded == hf_decoded, f"Decoding mismatch for tokens from text: {text!r}"


@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
def test_kitoken_call_interface_matches_hf(valid_texts):
    """Test that __call__ interface matches HF tokenizer exactly."""
    kitoken_wrapper = load_tokenizer("NousResearch/Llama-2-7b-hf")
    hf_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

    for text in valid_texts:
        kitoken_result = kitoken_wrapper(text, add_special_tokens=False)
        hf_result = hf_tokenizer(text, add_special_tokens=False)

        assert kitoken_result["input_ids"] == hf_result["input_ids"], f"__call__ mismatch for text: {text!r}"


@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
def test_kitoken_batch_encode_matches_hf(batch_texts):
    """Test that batch encoding matches HF tokenizer exactly."""
    kitoken_wrapper = load_tokenizer("NousResearch/Llama-2-7b-hf")
    hf_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

    # Test using __call__ with list of texts
    kitoken_result = kitoken_wrapper(batch_texts, add_special_tokens=False)
    hf_result = hf_tokenizer(batch_texts, add_special_tokens=False)

    assert kitoken_result["input_ids"] == hf_result["input_ids"], "Batch encoding mismatch"


@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
def test_kitoken_batch_decode_matches_hf(batch_texts):
    """Test that batch decoding matches HF tokenizer exactly."""
    kitoken_wrapper = load_tokenizer("NousResearch/Llama-2-7b-hf")
    hf_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

    # Encode all texts using HF tokenizer for consistent starting point
    token_sequences = [hf_tokenizer.encode(text, add_special_tokens=False) for text in batch_texts]

    kitoken_decoded = kitoken_wrapper.batch_decode(token_sequences, skip_special_tokens=False)
    hf_decoded = hf_tokenizer.batch_decode(token_sequences, skip_special_tokens=False)

    assert kitoken_decoded == hf_decoded, "Batch decoding mismatch"


@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
def test_kitoken_edge_cases_matches_hf(edge_case_texts):
    """Test that edge cases and garbage data are handled identically to HF tokenizer."""
    kitoken_wrapper = load_tokenizer("NousResearch/Llama-2-7b-hf")
    hf_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

    for text in edge_case_texts:
        # Test encoding
        kitoken_tokens = kitoken_wrapper.encode(text, add_special_tokens=False)
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)
        assert kitoken_tokens == hf_tokens, f"Edge case encoding mismatch for: {text!r}"

        # Test decoding round-trip
        kitoken_decoded = kitoken_wrapper.decode(kitoken_tokens, skip_special_tokens=False)
        hf_decoded = hf_tokenizer.decode(hf_tokens, skip_special_tokens=False)
        assert kitoken_decoded == hf_decoded, f"Edge case decoding mismatch for: {text!r}"


@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
def test_kitoken_round_trip_consistency(valid_texts):
    """Test encode-decode round-trip consistency between Kitoken and HF."""
    kitoken_wrapper = load_tokenizer("NousResearch/Llama-2-7b-hf")
    hf_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

    for text in valid_texts:
        # Kitoken round-trip
        kitoken_tokens = kitoken_wrapper.encode(text, add_special_tokens=False)
        kitoken_decoded = kitoken_wrapper.decode(kitoken_tokens, skip_special_tokens=False)

        # HF round-trip
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)
        hf_decoded = hf_tokenizer.decode(hf_tokens, skip_special_tokens=False)

        # Both should match
        assert kitoken_decoded == hf_decoded, f"Round-trip mismatch for text: {text!r}"
