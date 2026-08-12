# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for tokenizer loading across donor repos."""

import pytest

from experiments.datakit.cluster.quality.fast_transformer.data import encode_texts, encode_texts_fast, load_tokenizer

# Prose, code, CJK, and punctuation-dense text: the cases where a tokenizer
# built from the wrong file (or a mis-parsed config) diverges.
SAMPLE_TEXTS = [
    "An ordinary English sentence about a topic.",
    "def f(x):\n    return x**2  # square",
    "日本語のテキストです。",
    "mixed 123 ¡Hola! ~~~",
]


@pytest.mark.data_integration
def test_nemotron_tokenizer_loads_despite_an_unparseable_model_config():
    """``nvidia/Nemotron-Flash-1B`` ships a custom architecture whose config
    AutoTokenizer cannot parse, but its tokenizer.json is complete: the loader
    must still return a working fast tokenizer, or a scorer trained on this
    vocabulary cannot be scored with."""
    tokenizer = load_tokenizer("nvidia/Nemotron-Flash-1B")
    assert tokenizer.vocab_size == 131072
    encoded = tokenizer(SAMPLE_TEXTS[1], add_special_tokens=False)["input_ids"]
    assert tokenizer.decode(encoded, clean_up_tokenization_spaces=False) == SAMPLE_TEXTS[1]


@pytest.mark.data_integration
def test_gigatoken_matches_the_nemotron_tokenizer_exactly():
    """The training path gates on gigatoken parity, so the fast backend must
    reproduce this tokenizer's ids rather than merely accept it."""
    hf = encode_texts("nvidia/Nemotron-Flash-1B", SAMPLE_TEXTS, 512)
    fast = encode_texts_fast("nvidia/Nemotron-Flash-1B", SAMPLE_TEXTS, 512)
    assert [list(row) for row in hf] == [list(row) for row in fast]
