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

import pytest

from marin.processing.tokenize.data_configs import _are_tokenizers_equivalent


def test_are_tokenizers_equivalent():
    # Test cases where tokenizers should be equivalent
    equivalent_pairs = [
        ("meta-llama/Meta-Llama-3.1-8B", "stanford-crfm/marin-tokenizer"),
        ("meta-llama/Meta-Llama-3.1-8B", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        ("stanford-crfm/marin-tokenizer", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ]

    for t1, t2 in equivalent_pairs:
        try:
            assert _are_tokenizers_equivalent(t1, t2), f"Tokenizers {t1} and {t2} should be equivalent"
        except Exception as e:
            pytest.skip(f"Skipping test because models are not accessible: {e}")

    # Test cases where tokenizers should be different
    different_pairs = [
        ("meta-llama/Meta-Llama-3.1-8B", "EleutherAI/gpt-neox-20b"),
        ("stanford-crfm/marin-tokenizer", "EleutherAI/gpt-neox-20b"),
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", "EleutherAI/gpt-neox-20b"),
    ]

    for t1, t2 in different_pairs:
        try:
            assert not _are_tokenizers_equivalent(t1, t2), f"Tokenizers {t1} and {t2} should be different"
        except Exception as e:
            pytest.skip(f"Skipping test because models are not accessible: {e}")

    # Test that a tokenizer is equivalent to itself
    for t in [
        "meta-llama/Meta-Llama-3.1-8B",
        "stanford-crfm/marin-tokenizer",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "EleutherAI/gpt-neox-20b",
    ]:
        try:
            assert _are_tokenizers_equivalent(t, t), f"Tokenizer {t} should be equivalent to itself"
        except Exception as e:
            pytest.skip(f"Skipping test because model is not accessible: {e}")
