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
