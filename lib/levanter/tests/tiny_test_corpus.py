# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os

import numpy as np

from levanter.data.audio import AudioIODatasetConfig
from levanter.data.text import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.store.cache import TreeCache


def _write_tiny_corpus(path):
    os.makedirs(f"{path}/train", exist_ok=True)
    with open(f"{path}/train/docs.jsonl", "w") as f:
        for i in range(10):
            f.write(json.dumps({"text": f"hello world {i} " * 100}))
            f.write("\n")

    os.makedirs(f"{path}/validation", exist_ok=True)
    with open(f"{path}/validation/docs.jsonl", "w") as f:
        for i in range(10):
            f.write(json.dumps({"text": f"bye world {i} " * 100}))
            f.write("\n")


def tiny_corpus_config(path):
    _write_tiny_corpus(path)
    component = DatasetComponent(
        source=UrlDatasetSourceConfig(
            train_urls=[f"file://{path}/train/docs.jsonl"],
            validation_urls=[f"file://{path}/validation/docs.jsonl"],
        ),
        cache_dir=f"{path}/cache",
    )
    return LmDataConfig(components={"tiny": component})


def tiny_asr_corpus_config(path):
    return AudioIODatasetConfig(
        id="WillHeld/test_librispeech_parquet",
        text_key="text",
        train_split="validation",
        validation_split="validation",
        cache_dir=f"{path}/cache_asr",
    )


def construct_small_data_cache(
    path, num_shards=8, chunk_size=512, doc_len=128, vocab_size=1024
) -> tuple[LmDataConfig, dict[str, TreeCache]]:
    from levanter.store.cache import SerialCacheWriter

    rng = np.random.default_rng(0)

    caches: dict[str, TreeCache] = {}
    exemplar = {"input_ids": np.zeros((doc_len,), dtype=np.int32)}

    for split in ["train", "validation"]:
        with SerialCacheWriter(f"{path}/cache/{split}", exemplar) as writer:
            for shard in range(num_shards):
                writer.write_batch(
                    [
                        {"input_ids": rng.integers(0, vocab_size, size=(doc_len,), dtype=np.int32)}
                        for _ in range(chunk_size)
                    ]
                )
        caches[split] = writer.result()

    component = DatasetComponent(source=None, cache_dir=f"{path}/cache")
    config = LmDataConfig(components={"tiny": component}, vocab_size=vocab_size, tokenizer="gpt2")

    return config, caches
