# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from jax import local_devices
from marin.processing.embeddings.harrier_jax import HarrierJaxConfig, HarrierJaxEmbedder
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen3 import Qwen3Config, Qwen3Model


@pytest.mark.tpu
def test_harrier_jax_matches_hf_base_model_across_length_buckets(tmp_path):
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "short": 2, "long": 3, "query": 4}
    backend = Tokenizer(WordLevel(vocab=vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]")
    tokenizer.save_pretrained(tmp_path)

    torch.manual_seed(0)
    hf_config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=256,
        tie_word_embeddings=True,
        use_cache=False,
    )
    reference = Qwen3Model(hf_config).eval()
    reference.save_pretrained(tmp_path)

    texts = ["short query", "long " * 130, "short", "query long short"]
    expected = []
    with torch.inference_mode():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt")["input_ids"]
            hidden = reference(ids).last_hidden_state[0, -1].float()
            expected.append(torch.nn.functional.normalize(hidden, dim=0).numpy())

    embedder = HarrierJaxEmbedder(
        HarrierJaxConfig(
            model=str(tmp_path),
            revision="main",
            tokenizer_revision="main",
            dimensions=hf_config.hidden_size,
            max_tokens=256,
            compute_dtype="float32",
            platform=local_devices()[0].platform,
            batch_size_per_device=2,
            batch_token_limit_per_device=256,
            cache_ttl_days=0,
        )
    )
    actual = embedder.embed(texts)

    assert actual.shape == (len(texts), hf_config.hidden_size)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, np.stack(expected), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-5)
