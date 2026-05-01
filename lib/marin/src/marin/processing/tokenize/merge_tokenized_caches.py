# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge multiple finished tokenized caches into one reusable cache."""

from __future__ import annotations

import dataclasses
import json
import logging
import os

from rigging.filesystem import open_url
from levanter.data.text import (
    LmDatasetFormatBase,
    LmDatasetSourceConfigBase,
    TextLmDatasetFormat,
    UrlDatasetSourceConfig,
    preprocessor_for_format,
)
from levanter.store.cache import CacheMetadata, consolidate_shard_caches
from levanter.store.tree_store import TreeStore
import transformers

from marin.execution import THIS_OUTPUT_PATH, ExecutorStep, InputName, ensure_versioned
from marin.processing.tokenize.data_configs import TokenizerConfigLike, step_to_lm_dataset_source_config
from marin.processing.tokenize.tokenize import TokenizeConfigBase
from marin.utils import fsspec_exists

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class MergeTokenizedCachesConfig(TokenizeConfigBase):
    """Configuration for merging existing tokenized caches into one cache."""

    input_configs: dict[str, LmDatasetSourceConfigBase]
    cache_path: str = THIS_OUTPUT_PATH
    tokenizer: str = "stanford-crfm/marin-tokenizer"
    tags: list[str] = dataclasses.field(default_factory=list)
    format: LmDatasetFormatBase = dataclasses.field(default_factory=TextLmDatasetFormat)
    enforce_eos: bool = True

    def as_lm_dataset_source_config(
        self, actual_output_path: str | InputName | None, *, include_raw_paths: bool = True
    ) -> LmDatasetSourceConfigBase:
        if actual_output_path is None:
            raise ValueError("actual_output_path must be provided for a merged cache.")

        source_tags: list[str] = []
        for input_config in self.input_configs.values():
            source_tags.extend(input_config.tags or [])

        return UrlDatasetSourceConfig(
            tags=list(dict.fromkeys([*source_tags, *self.tags])),
            train_urls=[],
            validation_urls=[],
            cache_dir=actual_output_path,
            format=self.format,
        )


def _cache_paths_for_split(cfg: MergeTokenizedCachesConfig, split: str) -> list[str]:
    cache_paths: list[str] = []
    for name, input_config in cfg.input_configs.items():
        if input_config.cache_dir is None:
            raise ValueError(f"Input config {name} does not have a cache_dir")

        split_cache_path = os.path.join(input_config.cache_dir, split)
        ledger_path = os.path.join(split_cache_path, "shard_ledger.json")
        if fsspec_exists(ledger_path):
            cache_paths.append(split_cache_path)
    return cache_paths


def _write_split_stats(output_split_path: str, exemplar: dict) -> None:
    store = TreeStore.open(exemplar, output_split_path, mode="r", cache_metadata=True)
    total_tokens = store.tree["input_ids"].data_size if "input_ids" in store.tree else 0
    total_elements = len(store)
    stats_path = os.path.join(output_split_path, ".stats.json")
    with open_url(stats_path, "w") as f:
        json.dump({"total_tokens": total_tokens, "total_elements": total_elements}, f)


def _validate_input_formats(cfg: MergeTokenizedCachesConfig) -> None:
    mismatched = [name for name, input_config in cfg.input_configs.items() if input_config.format != cfg.format]
    if mismatched:
        raise ValueError(
            f"All merged caches must use format {cfg.format!r}; mismatched inputs: {', '.join(sorted(mismatched))}"
        )


def _merge_split(
    *,
    cfg: MergeTokenizedCachesConfig,
    split: str,
    exemplar: dict,
    metadata: CacheMetadata,
) -> bool:
    source_cache_paths = _cache_paths_for_split(cfg, split)
    if not source_cache_paths:
        return False

    output_split_path = os.path.join(cfg.cache_path, split)
    ledger_path = os.path.join(output_split_path, "shard_ledger.json")
    if fsspec_exists(ledger_path):
        logger.info("Merged %s cache already exists at %s; refreshing stats only.", split, output_split_path)
        _write_split_stats(output_split_path, exemplar)
        return True

    logger.info(
        "Merging %d tokenized %s caches into %s",
        len(source_cache_paths),
        split,
        output_split_path,
    )
    consolidate_shard_caches(
        shard_cache_paths=source_cache_paths,
        output_path=output_split_path,
        exemplar=exemplar,
        metadata=metadata,
    )
    _write_split_stats(output_split_path, exemplar)
    return True


def _merge_tokenized_caches(cfg: MergeTokenizedCachesConfig) -> MergeTokenizedCachesConfig:
    if not cfg.input_configs:
        raise ValueError("input_configs cannot be empty")

    _validate_input_formats(cfg)

    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.tokenizer)
    processor = preprocessor_for_format(cfg.format, tokenizer, enforce_bos=True, enforce_eos=cfg.enforce_eos)
    metadata = CacheMetadata(preprocessor_metadata=processor.metadata)
    exemplar = processor.output_exemplar

    merged_any = False
    for split in ("train", "validation"):
        merged_any = _merge_split(cfg=cfg, split=split, exemplar=exemplar, metadata=metadata) or merged_any

    if not merged_any:
        raise ValueError("None of the input caches contained a finished train or validation split to merge.")

    return cfg


def merge_tokenized_caches(
    output_cache_path_name: str,
    input_steps: dict[str, TokenizerConfigLike],
    *,
    tokenizer: str,
    tags: list[str] | None = None,
    dataset_format: LmDatasetFormatBase | None = None,
) -> ExecutorStep[MergeTokenizedCachesConfig]:
    """Create a shared cache-merge step from existing tokenized cache steps."""
    if not input_steps:
        raise ValueError("input_steps cannot be empty")

    input_configs = {
        name: step_to_lm_dataset_source_config(step, include_raw_paths=False) for name, step in input_steps.items()
    }
    resolved_format = dataset_format or next(iter(input_configs.values())).format
    config = MergeTokenizedCachesConfig(
        input_configs=input_configs,
        cache_path=THIS_OUTPUT_PATH,
        tokenizer=ensure_versioned(tokenizer),
        tags=tags or [],
        format=resolved_format,
    )
    return ExecutorStep(
        name=os.path.join("tokenized", "merged", output_cache_path_name),
        fn=_merge_tokenized_caches,
        config=config,
    )
