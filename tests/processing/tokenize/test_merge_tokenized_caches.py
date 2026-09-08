# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib

import numpy as np
from levanter.data.text.datasets import UrlDatasetSourceConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.store.cache import CACHE_LAYOUT_SHARDED, CacheLedger, SerialCacheWriter, TreeCache
from marin.processing.tokenize.merge_tokenized_caches import MergeTokenizedCachesConfig, _merge_tokenized_caches

merge_tokenized_caches_module = importlib.import_module("marin.processing.tokenize.merge_tokenized_caches")


class _DummyTokenizer:
    is_fast = False
    model_max_length = 8192
    bos_token_id = 1
    eos_token_id = 2
    bos_token = "<bos>"
    eos_token = "<eos>"
    name_or_path = "dummy"
    vocab_size = 128

    def __len__(self):
        return self.vocab_size

    def encode(self, text: str) -> list[int]:
        del text
        return [self.bos_token_id, 10, self.eos_token_id]

    def __call__(self, text, return_attention_mask=False, verbose=False, **kwargs):
        del verbose, kwargs
        if isinstance(text, str):
            input_ids = [self.bos_token_id, 10, self.eos_token_id]
            out = {"input_ids": input_ids}
            if return_attention_mask:
                out["attention_mask"] = [1] * len(input_ids)
            return out

        texts = list(text)
        input_ids = [[self.bos_token_id, 10 + i, self.eos_token_id] for i, _ in enumerate(texts)]
        out = {"input_ids": input_ids}
        if return_attention_mask:
            out["attention_mask"] = [[1] * len(ids) for ids in input_ids]
        return out


def _write_cache(cache_root: str, docs: list[list[int]]) -> None:
    exemplar = {"input_ids": np.zeros((0,), dtype=np.int32)}
    with SerialCacheWriter(f"{cache_root}/train", exemplar) as writer:
        writer.write_batch([{"input_ids": np.asarray(doc, dtype=np.int32)} for doc in docs])


def _write_sharded_cache(cache_root: str, shards: list[list[list[int]]]) -> None:
    shard_rows: dict[str, int] = {}
    field_counts_by_shard: dict[str, dict[str, int]] = {}
    finished_shards: list[str] = []
    exemplar = {"input_ids": np.zeros((0,), dtype=np.int32)}
    for index, docs in enumerate(shards):
        shard_name = f"part-{index:05d}-of-{len(shards):05d}"
        with SerialCacheWriter(f"{cache_root}/{shard_name}", exemplar) as writer:
            writer.write_batch([{"input_ids": np.asarray(doc, dtype=np.int32)} for doc in docs])
        shard_rows[shard_name] = len(docs)
        field_counts_by_shard[shard_name] = {"input_ids": sum(len(doc) for doc in docs)}
        finished_shards.append(shard_name)

    CacheLedger(
        total_num_rows=sum(shard_rows.values()),
        shard_rows=shard_rows,
        is_finished=True,
        finished_shards=finished_shards,
        field_counts={"input_ids": sum(counts["input_ids"] for counts in field_counts_by_shard.values())},
        field_counts_by_shard=field_counts_by_shard,
        layout=CACHE_LAYOUT_SHARDED,
    )._serialize_and_commit(cache_root)


def test_merge_tokenized_caches_merges_train_split(monkeypatch, tmp_path):
    monkeypatch.setattr(
        merge_tokenized_caches_module.transformers.AutoTokenizer,
        "from_pretrained",
        lambda _: _DummyTokenizer(),
    )

    left_cache = tmp_path / "left"
    right_cache = tmp_path / "right"
    merged_cache = tmp_path / "merged"

    _write_cache(str(left_cache), [[1, 2, 3], [4, 5]])
    _write_cache(str(right_cache), [[6], [7, 8, 9, 10]])

    cfg = MergeTokenizedCachesConfig(
        input_configs={
            "left": UrlDatasetSourceConfig(cache_dir=str(left_cache), train_urls=[], validation_urls=[]),
            "right": UrlDatasetSourceConfig(cache_dir=str(right_cache), train_urls=[], validation_urls=[]),
        },
        cache_path=str(merged_cache),
        tokenizer="dummy",
        format=TextLmDatasetFormat(),
    )

    _merge_tokenized_caches(cfg)

    merged = TreeCache.load(str(merged_cache / "train"), {"input_ids": np.zeros((0,), dtype=np.int32)})
    assert len(merged) == 4
    assert merged.store.tree["input_ids"].data_size == 10

    first = merged[0]["input_ids"].tolist()
    last = merged[3]["input_ids"].tolist()
    assert first == [1, 2, 3]
    assert last == [7, 8, 9, 10]


def test_merge_tokenized_caches_expands_sharded_input_layout(monkeypatch, tmp_path):
    monkeypatch.setattr(
        merge_tokenized_caches_module.transformers.AutoTokenizer,
        "from_pretrained",
        lambda _: _DummyTokenizer(),
    )

    source_cache = tmp_path / "source"
    merged_cache = tmp_path / "merged"
    _write_sharded_cache(
        str(source_cache / "train"),
        [
            [[1, 2, 3], [4, 5]],
            [[6], [7, 8, 9, 10]],
        ],
    )

    cfg = MergeTokenizedCachesConfig(
        input_configs={
            "source": UrlDatasetSourceConfig(cache_dir=str(source_cache), train_urls=[], validation_urls=[]),
        },
        cache_path=str(merged_cache),
        tokenizer="dummy",
        format=TextLmDatasetFormat(),
    )

    _merge_tokenized_caches(cfg)

    merged = TreeCache.load(str(merged_cache / "train"), {"input_ids": np.zeros((0,), dtype=np.int32)})
    assert len(merged) == 4
    assert merged.store.tree["input_ids"].data_size == 10
    assert merged[0]["input_ids"].tolist() == [1, 2, 3]
    assert merged[3]["input_ids"].tolist() == [7, 8, 9, 10]


def test_merge_tokenized_caches_persists_explicit_preprocessor_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(
        merge_tokenized_caches_module.transformers.AutoTokenizer,
        "from_pretrained",
        lambda _: _DummyTokenizer(),
    )

    source_cache = tmp_path / "source"
    merged_cache = tmp_path / "merged"
    _write_cache(str(source_cache), [[1, 2, 3]])
    historical_metadata = {
        "tokenizer": "historical-tokenizer",
        "vocab_size": 128256,
        "return_attention_mask": False,
        "padding": False,
        "max_length": 131072,
        "append_bos": False,
        "append_eos": True,
    }
    cfg = MergeTokenizedCachesConfig(
        input_configs={
            "source": UrlDatasetSourceConfig(cache_dir=str(source_cache), train_urls=[], validation_urls=[]),
        },
        cache_path=str(merged_cache),
        tokenizer="dummy",
        format=TextLmDatasetFormat(),
        preprocessor_metadata=historical_metadata,
    )

    _merge_tokenized_caches(cfg)

    ledger = CacheLedger.load(str(merged_cache / "train"))
    assert ledger.metadata is not None
    assert ledger.metadata.preprocessor_metadata == historical_metadata
