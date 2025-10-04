# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from dataclasses import dataclass, field

import levanter
from levanter.data.metrics_monitor import LoggingMetricsMonitor
from levanter.data.text import BatchTokenizer, SingleDatasetLMConfigBase
from levanter.distributed import RayConfig
from levanter.store.cache import build_or_load_cache
from levanter.tracker import NoopConfig, TrackerConfig
from levanter.utils.logging import init_logging


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RayCachedLMDatasetConfig(SingleDatasetLMConfigBase, RayConfig):
    tracker: TrackerConfig = field(default_factory=NoopConfig)


@levanter.config.main()
def main(args: RayCachedLMDatasetConfig):
    """Caches two different kinds of datasets. It can cache a dataset from a list of urls, or a dataset from a hf dataset"""
    init_logging(".", "cache_dataset.log")
    args.initialize()

    tokenizer = args.the_tokenizer

    for split in ["train", "validation"]:
        print(f"Caching {split} to {args.cache_dir}.")
        # connect or start the actor
        batch_tokenizer = BatchTokenizer(tokenizer, enforce_eos=args.enforce_eos)
        split_cache_dir = os.path.join(args.cache_dir, split)  # type: ignore
        source = args.get_shard_source(split)

        if source is None:
            logger.warning(f"Skipping {split} because it is empty.")
            continue

        monitors: list = []
        if not isinstance(args.tracker, NoopConfig):
            monitors.append(LoggingMetricsMonitor("preprocess/" + split, commit=True))

        cache = build_or_load_cache(
            cache_dir=split_cache_dir,
            source=source,
            processor=batch_tokenizer,
            await_finished=False,
            monitors=monitors,
        )

        cache.await_finished()
        print(f"Finished caching {split} to {split_cache_dir}.")


if __name__ == "__main__":
    main()
