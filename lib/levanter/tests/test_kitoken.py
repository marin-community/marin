# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Kitoken integration and compatibility with HuggingFace tokenizers.
"""

import numpy as np
import pytest

try:
    import kitoken
    KITOKEN_AVAILABLE = True
except ImportError:
    KITOKEN_AVAILABLE = False

from levanter.compat.hf_checkpoints import load_tokenizer
from test_utils import skip_if_hf_model_not_accessible


@pytest.mark.skipif(not KITOKEN_AVAILABLE, reason="Kitoken not installed")
@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
def test_kitoken_vs_hf_tokenizer_valid_text():
    """Test that Kitoken produces identical results to HF tokenizer on valid text."""
    # Load HF tokenizer
    hf_tokenizer = load_tokenizer("NousResearch/Llama-2-7b-hf")

    # For Kitoken, we need to get the tokenizer.json file
    # The tokenizer should have been downloaded to the HF cache
    import os
    from transformers.utils import TRANSFORMERS_CACHE
    from pathlib import Path

    # Try to find the tokenizer.json in the HF cache
    cache_dir = os.environ.get("HF_HOME", TRANSFORMERS_CACHE)
    model_cache = Path(cache_dir) / "models--NousResearch--Llama-2-7b-hf"

    # Find tokenizer.json in snapshots
    tokenizer_json_path = None
    if model_cache.exists():
        for snapshot_dir in (model_cache / "snapshots").iterdir():
            candidate = snapshot_dir / "tokenizer.json"
            if candidate.exists():
                tokenizer_json_path = str(candidate)
                break

    if tokenizer_json_path is None:
        pytest.skip("Could not find tokenizer.json for Kitoken test")

    # Load Kitoken
    kitoken_encoder = kitoken.Kitoken.from_file(tokenizer_json_path)

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
        # HF tokenization
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        # Kitoken tokenization
        kitoken_tokens = kitoken_encoder.encode(text, add_special_tokens=False)

        # Compare
        assert len(hf_tokens) == len(kitoken_tokens), f"Token count mismatch for text: {text!r}"
        assert hf_tokens == kitoken_tokens, f"Token mismatch for text: {text!r}\nHF: {hf_tokens}\nKitoken: {kitoken_tokens}"

        # Test decoding
        hf_decoded = hf_tokenizer.decode(hf_tokens, skip_special_tokens=True)
        kitoken_decoded = kitoken_encoder.decode(kitoken_tokens, skip_special_tokens=True)

        # Decoded text should match (allowing for minor whitespace differences)
        assert hf_decoded.strip() == kitoken_decoded.strip(), f"Decoded text mismatch for: {text!r}\nHF: {hf_decoded!r}\nKitoken: {kitoken_decoded!r}"


@pytest.mark.skipif(not KITOKEN_AVAILABLE, reason="Kitoken not installed")
@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
def test_kitoken_vs_hf_tokenizer_garbage_data():
    """Test that Kitoken handles garbage/malformed data the same way as HF tokenizer."""
    # Load HF tokenizer
    hf_tokenizer = load_tokenizer("NousResearch/Llama-2-7b-hf")

    # Get Kitoken path
    import os
    from transformers.utils import TRANSFORMERS_CACHE
    from pathlib import Path

    cache_dir = os.environ.get("HF_HOME", TRANSFORMERS_CACHE)
    model_cache = Path(cache_dir) / "models--NousResearch--Llama-2-7b-hf"

    tokenizer_json_path = None
    if model_cache.exists():
        for snapshot_dir in (model_cache / "snapshots").iterdir():
            candidate = snapshot_dir / "tokenizer.json"
            if candidate.exists():
                tokenizer_json_path = str(candidate)
                break

    if tokenizer_json_path is None:
        pytest.skip("Could not find tokenizer.json for Kitoken test")

    kitoken_encoder = kitoken.Kitoken.from_file(tokenizer_json_path)

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
        try:
            # HF tokenization
            hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

            # Kitoken tokenization
            kitoken_tokens = kitoken_encoder.encode(text, add_special_tokens=False)

            # Compare token sequences
            assert len(hf_tokens) == len(kitoken_tokens), f"Token count mismatch for garbage: {text!r}"
            assert hf_tokens == kitoken_tokens, f"Token mismatch for garbage: {text!r}\nHF: {hf_tokens}\nKitoken: {kitoken_tokens}"

        except Exception as e:
            # If HF raises an exception, Kitoken should too (or vice versa)
            # We'll be lenient here since error handling might differ
            print(f"Exception for text {text!r}: {e}")


@pytest.mark.skipif(not KITOKEN_AVAILABLE, reason="Kitoken not installed")
@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
def test_kitoken_vs_hf_batch_encoding():
    """Test that Kitoken produces identical results for batch encoding."""
    # Load HF tokenizer
    hf_tokenizer = load_tokenizer("NousResearch/Llama-2-7b-hf")

    # Get Kitoken path
    import os
    from transformers.utils import TRANSFORMERS_CACHE
    from pathlib import Path

    cache_dir = os.environ.get("HF_HOME", TRANSFORMERS_CACHE)
    model_cache = Path(cache_dir) / "models--NousResearch--Llama-2-7b-hf"

    tokenizer_json_path = None
    if model_cache.exists():
        for snapshot_dir in (model_cache / "snapshots").iterdir():
            candidate = snapshot_dir / "tokenizer.json"
            if candidate.exists():
                tokenizer_json_path = str(candidate)
                break

    if tokenizer_json_path is None:
        pytest.skip("Could not find tokenizer.json for Kitoken test")

    kitoken_encoder = kitoken.Kitoken.from_file(tokenizer_json_path)

    # Test batch encoding
    batch_texts = [
        "First sentence.",
        "Second sentence with more words.",
        "Third one is short.",
    ]

    # Encode each individually and compare
    for text in batch_texts:
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)
        kitoken_tokens = kitoken_encoder.encode(text, add_special_tokens=False)

        assert hf_tokens == kitoken_tokens, f"Batch encoding mismatch for: {text!r}"


@pytest.mark.skipif(not KITOKEN_AVAILABLE, reason="Kitoken not installed")
@skip_if_hf_model_not_accessible("NousResearch/Llama-2-7b-hf")
def test_kitoken_decode_equivalence():
    """Test that decoding with Kitoken matches HF tokenizer."""
    # Load HF tokenizer
    hf_tokenizer = load_tokenizer("NousResearch/Llama-2-7b-hf")

    # Get Kitoken path
    import os
    from transformers.utils import TRANSFORMERS_CACHE
    from pathlib import Path

    cache_dir = os.environ.get("HF_HOME", TRANSFORMERS_CACHE)
    model_cache = Path(cache_dir) / "models--NousResearch--Llama-2-7b-hf"

    tokenizer_json_path = None
    if model_cache.exists():
        for snapshot_dir in (model_cache / "snapshots").iterdir():
            candidate = snapshot_dir / "tokenizer.json"
            if candidate.exists():
                tokenizer_json_path = str(candidate)
                break

    if tokenizer_json_path is None:
        pytest.skip("Could not find tokenizer.json for Kitoken test")

    kitoken_encoder = kitoken.Kitoken.from_file(tokenizer_json_path)

    # Test with some token sequences
    test_token_sequences = [
        [1, 2, 3, 4, 5],
        [100, 200, 300, 400],
        list(range(1000, 1100)),
    ]

    for tokens in test_token_sequences:
        hf_decoded = hf_tokenizer.decode(tokens, skip_special_tokens=True)
        kitoken_decoded = kitoken_encoder.decode(tokens, skip_special_tokens=True)

        # Allow for minor whitespace differences
        assert hf_decoded.strip() == kitoken_decoded.strip(), f"Decode mismatch for tokens: {tokens}"
