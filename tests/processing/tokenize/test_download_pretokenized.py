# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from levanter.store.cache import SerialCacheWriter, TreeCache
from marin.processing.tokenize.download_pretokenized import (
    PretokenizedCacheDownloadConfig,
    fetch_pretokenized_cache,
)


def test_download_and_load_cache_from_local_source(tmp_path):
    exemplar = {"input_ids": np.array([0], dtype=np.int32)}
    expected_rows = [
        np.array([11, 12, 13], dtype=np.int32),
        np.array([21, 22], dtype=np.int32),
    ]
    source_path = tmp_path / "source"
    with SerialCacheWriter(str(source_path / "train"), exemplar) as writer:
        writer.write_batch([{"input_ids": row} for row in expected_rows])

    output_path = tmp_path / "downloaded"
    config = PretokenizedCacheDownloadConfig(
        cache_path=str(output_path),
        tokenizer="local-test-tokenizer",
        hf_repo_id="test/cache",
        hf_revision="test-revision",
        source_url_override=str(source_path),
    )

    fetch_pretokenized_cache(config)

    cache = TreeCache.load(str(output_path / "train"), exemplar=exemplar)
    assert len(cache) == len(expected_rows)
    for actual, expected in zip(cache, expected_rows, strict=True):
        np.testing.assert_array_equal(actual["input_ids"], expected)
