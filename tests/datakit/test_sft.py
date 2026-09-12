# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from haliax import Axis
from levanter.data.text._batch_tokenizer import BatchTokenizer
from levanter.store.cache import TreeCache
from levanter.tokenizers import load_tokenizer
from marin.datakit.sft import SftInput, SftTokenStore, build_sft_store, sft_data_config
from marin.execution.artifact import read_artifact


def test_sft_store_retains_conversations_and_packs_without_boundary_loss(tmp_path, gpt2_tokenizer_path):
    sources = []
    texts = []
    for name, records in [("a", ["Hello", "A longer answer", "x " * 100]), ("b", ["Hello", "Bonjour"])]:
        path = tmp_path / name
        path.mkdir()
        pq.write_table(
            pa.Table.from_pylist([{"id": str(i), "text": text} for i, text in enumerate(records)]), path / "part.parquet"
        )
        sources.append(SftInput(name, str(path)))
        texts.extend(records)
    tokenizer = load_tokenizer(gpt2_tokenizer_path)
    encoded = BatchTokenizer(tokenizer)([{"text": text} for text in texts])
    expected = [row["input_ids"] for row in encoded if len(row["input_ids"]) <= 24]
    output = str(tmp_path / "store")
    result = build_sft_store(
        sources, output_path=output, tokenizer=gpt2_tokenizer_path, max_length=24, seed=42, num_shards=2, max_workers=2
    )
    assert read_artifact(output, SftTokenStore) == result
    assert result.sources["a"].conversations == result.sources["b"].conversations == 2
    assert result.sources["a"].overlength_conversations == 1
    assert result.sources["a"].overlength_tokens == len(encoded[2]["input_ids"])
    cache = TreeCache.load(output + "/train", {"input_ids": np.zeros(0, dtype=np.int32)}).as_sync_dataset()
    assert Counter(tuple(cache[i]["input_ids"]) for i in range(len(cache))) == Counter(map(tuple, expected))
    config = sft_data_config(result)
    packed = config.train_sets(Axis("position", 24), key=jax.random.PRNGKey(42))["sft"].as_sync_dataset()
    recovered = []
    for i in range(len(packed)):
        example = packed[i]
        tokens = np.asarray(example.tokens)
        segments = np.asarray(example.attn_mask.segment_ids[0])
        weights = np.asarray(example.loss_weight)
        same_document = (segments[:-1] >= 0) & (segments[:-1] == segments[1:])
        np.testing.assert_array_equal(weights[:-1], same_document.astype(np.float32))
        assert weights[-1] == 0
        for segment in np.unique(segments[segments >= 0]):
            recovered.append(tuple(tokens[segments == segment]))
    assert Counter(recovered) == Counter(map(tuple, expected))
    assert config.validation_sets(Axis("position", 24)) == {}
