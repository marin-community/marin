# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from haliax import Axis

from levanter.data.packing import lookahead_pack_documents, pack_documents
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, dataset_for_component
from levanter.data.text.formats import PrebuiltLmDatasetFormat
from levanter.store.cache import SerialCacheWriter, TreeCache, consolidate_shard_cache_ledgers
from levanter.utils.thread_utils import blocking_wait


def test_lookahead_fills_gaps_without_losing_documents_and_honors_segment_cap():
    lengths = np.asarray([6, 6, 4, 4])
    greedy = pack_documents(lengths, 10, max_segments_per_example=256)
    packed = lookahead_pack_documents(lengths, 10, lookahead=2, max_segments_per_example=256)
    assert len(greedy) == 3 and len(packed) == 2
    np.testing.assert_array_equal(packed[0], [0, 2])
    np.testing.assert_array_equal(packed[1], [1, 3])
    assert sum(10 - lengths[pack].sum() for pack in packed) == 0

    lengths = np.ones(1000, dtype=np.int64)
    packed = lookahead_pack_documents(lengths, 1024, lookahead=16, max_segments_per_example=256)
    assert [len(pack) for pack in packed] == [256, 256, 256, 232]
    np.testing.assert_array_equal(np.sort(np.concatenate(packed)), np.arange(1000))


@pytest.mark.parametrize("shards", [1, 2])
def test_prebuilt_lookahead_preserves_prediction_weights_and_seeded_random_access(tmp_path, shards):
    # Chat, full-token replay, teacher-only relay, then chat. Weights already
    # index predictions, so neither packing nor the dataset may shift them again.
    tokens = [np.arange(100 + 10 * i, 100 + 10 * i + n, dtype=np.int32) for i, n in enumerate([6, 6, 4, 4])]
    weights = [
        np.asarray(w, dtype=np.float32) for w in ([0, 1, 0, 1, 1, 0], [1, 1, 1, 1, 1, 0], [0, 0, 1, 0], [0, 1, 1, 0])
    ]
    exemplar = {"input_ids": np.zeros(0, dtype=np.int32), "loss_weights": np.zeros(0, dtype=np.float32)}
    records = [{"input_ids": t, "loss_weights": w} for t, w in zip(tokens, weights, strict=True)]
    paths = []
    for shard in range(shards):
        path = str(tmp_path) if shards == 1 else str(tmp_path / f"part-{shard}")
        with SerialCacheWriter(path, exemplar) as writer:
            writer.write_batch(records[shard * 4 // shards : (shard + 1) * 4 // shards])
        paths.append(path)
    if shards > 1:
        consolidate_shard_cache_ledgers(paths, str(tmp_path), exemplar)
    cache = TreeCache.load(str(tmp_path), exemplar)
    component = DatasetComponent(
        cache_dir=str(tmp_path),
        pack=256,
        packing_lookahead=2,
        format=PrebuiltLmDatasetFormat(loss_weights_key="loss_weights"),
    )
    dataset = dataset_for_component(
        component, Axis("position", 10), cache, eos_id=None, block_cross_document_attention=True
    )
    assert blocking_wait(dataset.async_len()) == 2
    examples = blocking_wait(dataset.get_batch([1, 0, 1]))
    for example, documents in zip(examples, [(1, 3), (0, 2), (1, 3)], strict=True):
        np.testing.assert_array_equal(example.tokens, np.concatenate([tokens[i] for i in documents]))
        np.testing.assert_array_equal(example.loss_weight, np.concatenate([weights[i] for i in documents]))
        np.testing.assert_array_equal(
            example.attn_mask.segment_ids[0], np.repeat(documents, [len(tokens[i]) for i in documents])
        )
    first = dataset.shuffle(jax.random.PRNGKey(17), perm_type="feistel")
    second = dataset.shuffle(jax.random.PRNGKey(17), perm_type="feistel")
    for left, right in zip(
        blocking_wait(first.get_batch([1, 0])), blocking_wait(second.get_batch([1, 0])), strict=True
    ):
        np.testing.assert_array_equal(left.tokens, right.tokens)
        np.testing.assert_array_equal(left.loss_weight, right.loss_weight)


def test_lookahead_rejects_oversized_document_instead_of_silently_clipping():
    with pytest.raises(ValueError, match="Document 1 has 11 tokens"):
        lookahead_pack_documents(np.asarray([6, 11, 4]), 10, lookahead=2, max_segments_per_example=256)


def test_lookahead_rejects_misaligned_token_and_supervision_rows(tmp_path):
    with SerialCacheWriter(
        str(tmp_path), {"input_ids": np.zeros(0, dtype=np.int32), "loss_weights": np.zeros(0, dtype=np.float32)}
    ) as writer:
        writer.write_batch(
            [{"input_ids": np.asarray([10, 11, 12], dtype=np.int32), "loss_weights": np.ones(2, dtype=np.float32)}]
        )
    component = DatasetComponent(
        cache_dir=str(tmp_path),
        pack=256,
        packing_lookahead=2,
        format=PrebuiltLmDatasetFormat(loss_weights_key="loss_weights"),
    )
    with pytest.raises(ValueError, match="identical per-document lengths"):
        dataset_for_component(
            component, Axis("position", 10), writer.result(), eos_id=None, block_cross_document_attention=True
        )


@pytest.mark.parametrize("shuffle", [False, True])
def test_iid_validation_packs_only_reserved_sources_and_preserves_masks(tmp_path, shuffle):
    exemplar = {"input_ids": np.zeros(0, dtype=np.int32), "loss_weights": np.zeros(0, dtype=np.float32)}
    components = {}
    caches = {}
    for name, lengths in {"sft/reserved": [6, 4, 3] * 4, "train-only": [12] * 3}.items():
        path = str(tmp_path / name)
        with SerialCacheWriter(path, exemplar) as writer:
            writer.write_batch(
                [
                    {
                        "input_ids": np.arange(100 * i + 1, 100 * i + n + 1, dtype=np.int32),
                        "loss_weights": np.asarray([j % 2 if j < n - 1 else 0 for j in range(n)], dtype=np.float32),
                    }
                    for i, n in enumerate(lengths)
                ]
            )
        caches[name] = writer.result()
        components[name] = DatasetComponent(
            cache_dir=path,
            flat_cache=True,
            pack=256,
            packing_lookahead=3,
            format=PrebuiltLmDatasetFormat(loss_weights_key="loss_weights"),
        )
    config = LmDataConfig(
        tokenizer="passthrough",
        vocab_size=2048,
        components=components,
        num_validation_sequences={"sft/reserved": 2},
        shuffle_before_trainval_split=shuffle,
    )
    # The unrelated source is valid at its training context but exceeds the
    # evaluation context. It has no IID reservation and must not be packed there.
    train_only = config.build_token_datasets(caches, Axis("position", 16), split="train")["train-only"]
    assert len(train_only.as_sync_dataset()) == 3
    Pos = Axis("position", 8)
    reference = dataset_for_component(
        components["sft/reserved"], Pos, caches["sft/reserved"], eos_id=None, block_cross_document_attention=False
    )
    if shuffle:
        reference = reference.shuffle(jax.random.PRNGKey(0), perm_type="feistel")
    length = len(reference.as_sync_dataset())
    expected = blocking_wait(reference.get_batch([length - 2, length - 1]))
    actual = config.validation_sets(Pos)["sft/reserved"].as_sync_dataset()
    assert len(actual) == 2
    for index, example in enumerate(expected):
        np.testing.assert_array_equal(actual[index].tokens.array, example.tokens)
        np.testing.assert_array_equal(actual[index].loss_weight.array, example.loss_weight)
