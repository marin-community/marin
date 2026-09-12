# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from dataclasses import replace

import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from haliax import Axis
from levanter.data.text._batch_tokenizer import BatchTokenizer
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.schedule import BatchSchedule
from levanter.store.cache import SerialCacheWriter, TreeCache
from levanter.tokenizers import load_tokenizer
from marin.datakit.sft import SftInput, SftTokenStore, build_sft_store, sft_data_config
from marin.execution.artifact import read_artifact

from experiments.june_tpu_67b_a2b.moe.train import build_train_dataset


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


@pytest.mark.asyncio
async def test_replay_preserves_rare_sources_and_batches_with_packed_sft(tmp_path, gpt2_tokenizer_path):
    for name, rows in {
        "sft": [[11, 12, 13], [14, 15, 16]],
        "common": [[21] * 32],
        "rare": [[31] * 32],
    }.items():
        with SerialCacheWriter(str(tmp_path / name / "train"), {"input_ids": np.zeros(0, dtype=np.int32)}) as writer:
            writer.write_batch([{"input_ids": np.array(row, dtype=np.int32)} for row in rows])
    store = SftTokenStore(
        cache_path=str(tmp_path / "sft"), tokenizer=gpt2_tokenizer_path, max_length=8, seed=0, sources={}
    )
    sft = replace(sft_data_config(store), mixture_block_size=100)
    replay = LmDataConfig(
        tokenizer=gpt2_tokenizer_path,
        components={
            name: DatasetComponent(
                source=UrlDatasetSourceConfig(train_urls=[], validation_urls=[]),
                cache_dir=str(tmp_path / name),
                format=TextLmDatasetFormat(),
                pack=False,
            )
            for name in ("common", "rare")
        },
        train_weights={"common": 0.99, "rare": 0.01},
        mixture_block_size=100,
        auto_build_caches=False,
        block_cross_document_attention=True,
    )
    data = replace(
        sft,
        components={**sft.components, **replay.components},
        train_weights={"sft": 0.8, **{name: 0.2 * weight for name, weight in replay.train_weights.items()}},
        # Five times the replay block preserves its rare-source counts at 20%.
        mixture_block_size=5 * replay.mixture_block_size,
    )
    dataset = build_train_dataset(
        data,
        max_seq_len=8,
        batch_schedule=BatchSchedule(10),
        key=jax.random.PRNGKey(42),
    )
    examples = await dataset.get_batch(list(range(500)))
    counts = Counter()
    for example in examples:
        first_token = int(example.tokens[0])
        if first_token < 20:
            counts["sft"] += 1
            np.testing.assert_array_equal(example.loss_weight, [1, 1, 0, 1, 1, 0, 0, 0])
            segments = np.asarray(example.attn_mask.segment_ids[0])
            assert segments[0] != segments[3]
        else:
            counts["common" if first_token == 21 else "rare"] += 1
            np.testing.assert_array_equal(example.loss_weight, [1, 1, 1, 1, 1, 1, 1, 0])
        assert example.attn_mask.thd_segment_metadata is None
    assert counts == {"sft": 400, "common": 99, "rare": 1}
    # This is the tree stacking operation required to put both formats in a batch.
    batch = jax.tree.map(lambda *values: np.stack(values), *examples)
    assert batch.tokens.shape == (500, 8)
