# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib

import numpy as np
from levanter.data.text import TextLmDatasetFormat, UrlDatasetSourceConfig
from levanter.store import SerialCacheWriter, TreeCache

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
