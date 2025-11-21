# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Kitoken integration and compatibility with HuggingFace tokenizers.
These tests validate that the KitokenWrapper produces identical results to the underlying HF tokenizer.
"""

import pytest
from test_utils import skip_if_hf_model_not_accessible
from transformers import AutoTokenizer

from levanter.compat.hf_checkpoints import load_tokenizer


def valid_texts():
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


def edge_case_texts():
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


@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
@pytest.mark.parametrize("add_special_tokens", [True, False])
@pytest.mark.parametrize("texts", [valid_texts(), edge_case_texts()])
def test_kitoken_single_encode_matches_hf(texts, add_special_tokens):
    """Test that single text encoding matches HF tokenizer exactly."""
    kitoken_wrapper = load_tokenizer("NousResearch/Llama-2-7b-hf")
    hf_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

    for text in texts:
        kitokens = kitoken_wrapper.encode(text, add_special_tokens=add_special_tokens)
        hftokens = hf_tokenizer.encode(text, add_special_tokens=add_special_tokens)
        assert (
            kitokens == hftokens
        ), f"Encoding mismatch for text: {text!r} {kitokens} != {hftokens} -- {add_special_tokens}"

        kitokens = kitoken_wrapper(text, add_special_tokens=add_special_tokens)
        hftokens = hf_tokenizer(text, add_special_tokens=add_special_tokens)
        assert (
            kitokens["input_ids"] == hftokens["input_ids"]
        ), f"__call__ encoding mismatch for text: {text!r} {kitokens['input_ids']} != {hftokens['input_ids']} -- {add_special_tokens}"

    kitokens = kitoken_wrapper(texts, add_special_tokens=add_special_tokens)
    hftokens = hf_tokenizer(texts, add_special_tokens=add_special_tokens)
    for i, text in enumerate(texts):
        assert (
            kitokens["input_ids"][i] == hftokens["input_ids"][i]
        ), f"Batch encoding mismatch for text: {text!r} {kitokens['input_ids'][i]} != {hftokens['input_ids'][i]} -- {add_special_tokens}"


@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
@pytest.mark.parametrize("texts", [valid_texts(), edge_case_texts()])
def test_kitoken_round_trip_consistency(texts):
    """Test encode-decode round-trip consistency between Kitoken and HF."""
    kitoken_wrapper = load_tokenizer("NousResearch/Llama-2-7b-hf")
    hf_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

    for text in texts:
        # Kitoken round-trip
        kitoken_tokens = kitoken_wrapper.encode(text, add_special_tokens=False)
        kitoken_decoded = kitoken_wrapper.decode(kitoken_tokens, skip_special_tokens=False)

        # HF round-trip
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)
        hf_decoded = hf_tokenizer.decode(hf_tokens, skip_special_tokens=False)

        # Both should match
        assert kitoken_decoded == hf_decoded, f"Round-trip mismatch for text: {text!r}"
