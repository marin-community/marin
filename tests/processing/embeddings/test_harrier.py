# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

torch = pytest.importorskip("torch")

from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

from marin.processing.embeddings.harrier import HarrierConfig, load_harrier


def test_concurrent_harrier_loads_share_a_usable_backend(tmp_path):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", model_max_length=32
    ).save_pretrained(tmp_path)
    Qwen3Model(
        Qwen3Config(
            vocab_size=4,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
        )
    ).save_pretrained(tmp_path)
    config = HarrierConfig(
        model=str(tmp_path),
        revision="local",
        tokenizer_revision="local",
        dimensions=16,
        max_tokens=32,
        compute_dtype="bfloat16",
        device="cpu",
        cpu_threads=1,
    )
    callers = Barrier(4)

    def load():
        callers.wait(timeout=10)
        return load_harrier(config)

    with ThreadPoolExecutor(max_workers=4) as pool:
        backends = list(pool.map(lambda _: load(), range(4)))

    assert all(backend is backends[0] for backend in backends)
    vectors = backends[0].embed(["hello world"])
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(1), rtol=1e-5)
