# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

import pytest
from transformers import AutoTokenizer

from experiments.create_marin_tokenizer import (
    create_marin_tokenizer,
    load_llama3_tokenizer,
    special_tokens_injection_check,
)

pytestmark = pytest.mark.skipif("CI" in os.environ, reason="Requires HF tokenizer download")


def test_special_tokens_injection():
    """Test that special tokens are correctly replaced after a save/load roundtrip."""
    tokenizer = create_marin_tokenizer(load_llama3_tokenizer())

    with tempfile.TemporaryDirectory() as temp_path:
        tokenizer.save_pretrained(temp_path)
        tokenizer = AutoTokenizer.from_pretrained(temp_path, local_files_only=True)

    special_tokens_injection_check(tokenizer)
