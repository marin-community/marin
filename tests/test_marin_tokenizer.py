# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizer

from experiments.create_marin_tokenizer import (
    create_marin_tokenizer,
    load_llama3_tokenizer,
    special_tokens_injection_check,
)


@pytest.fixture
def marin_tokenizer():
    """Fixture that provides a configured marin tokenizer for testing."""
    try:
        llama3_tokenizer = load_llama3_tokenizer()
    except Exception as e:
        if os.getenv("CI", False) in ["true", "1"]:
            pytest.skip("Llama 3 tokenizer repository is gated")
        raise e
    tokenizer = create_marin_tokenizer(llama3_tokenizer)

    # Roundtrip write-read to ensure consistency
    with tempfile.TemporaryDirectory() as temp_path:
        tokenizer.save_pretrained(temp_path)
        tokenizer = AutoTokenizer.from_pretrained(temp_path, local_files_only=True)

    return tokenizer


def test_special_tokens_injection(marin_tokenizer: PreTrainedTokenizer):
    """Test that special tokens are correctly replaced."""
    special_tokens_injection_check(marin_tokenizer)
