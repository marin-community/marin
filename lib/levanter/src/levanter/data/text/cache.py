# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import numpy as np

from levanter.data.sharded_datasource import ShardedDataSource
from levanter.store.cache import CacheMetadata, CacheOptions, TreeCache, build_or_load_cache
from levanter.utils.hf_utils import HfTokenizer

from .formats import LmDatasetFormatBase, preprocessor_for_format

logger = logging.getLogger("levanter.data.text.cache")


def build_lm_dataset_cache(
    cache_dir: str,
    source: ShardedDataSource[dict],
    format: LmDatasetFormatBase,
    tokenizer: HfTokenizer,
    options: CacheOptions = CacheOptions.default(),
    enforce_eos: bool = True,
) -> TreeCache[dict]:
    """
    Creates a cache for a dataset. If the cache already exists, it will be loaded. Otherwise, it will be built.
    """
    name = os.path.join(*cache_dir.split("/")[-2:])
    processor = preprocessor_for_format(format, tokenizer, enforce_bos=True, enforce_eos=enforce_eos)
    try:
        return TreeCache.load(
            cache_dir,
            exemplar=processor.output_exemplar,
            options=CacheMetadata(preprocessor_metadata=processor.metadata),
        )
    except FileNotFoundError:
        pass

    logger.info(f"Building cache for {name}...")
    return build_or_load_cache(cache_dir, source, processor, options=options)


def load_lm_dataset_cache(
    cache_dir: str,
    format: LmDatasetFormatBase,
    tokenizer: HfTokenizer,
    enforce_eos: bool = True,
) -> TreeCache[dict]:
    """Load an existing cache, raising if not present."""
    processor = preprocessor_for_format(format, tokenizer, enforce_bos=True, enforce_eos=enforce_eos)
    cache = TreeCache.load(
        cache_dir,
        exemplar=processor.output_exemplar,
        options=CacheMetadata(preprocessor_metadata=processor.metadata),
    )
    return cache


def cached_token_count(cache_path: str, field: str = "input_ids") -> int:
    """Return the total number of tokens stored in a finished TreeCache."""
    cache = TreeCache.load(cache_path, {field: np.zeros((0,), dtype=np.int32)})
    return cache.store.tree[field].data_size
