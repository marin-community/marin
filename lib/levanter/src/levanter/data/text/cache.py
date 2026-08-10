# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from levanter.data.sharded_datasource import ShardedDataSource
from levanter.store.cache import CacheMetadata, CacheOptions, TreeCache, build_or_load_cache
from levanter.tokenizers import MarinTokenizer

from .formats import LmDatasetFormatBase, preprocessor_for_format

logger = logging.getLogger("levanter.data.text.cache")


def build_lm_dataset_cache(
    cache_dir: str,
    source: ShardedDataSource[dict],
    format: LmDatasetFormatBase,
    tokenizer: MarinTokenizer,
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
    tokenizer: MarinTokenizer,
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
