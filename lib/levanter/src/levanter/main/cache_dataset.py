# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass, field

import levanter
from levanter.data.text import LmDataConfig
from levanter.distributed import RayConfig
from levanter.tracker import NoopConfig, TrackerConfig
from levanter.utils.logging import init_logging


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RayCachedLMDatasetConfig(LmDataConfig, RayConfig):
    tracker: TrackerConfig = field(default_factory=NoopConfig)


@levanter.config.main()
def main(args: RayCachedLMDatasetConfig):
    """Caches two different kinds of datasets. It can cache a dataset from a list of urls, or a dataset from a hf dataset"""
    init_logging(".", "cache_dataset.log")
    args.initialize()

    tokenizer = args.the_tokenizer
    for split in ["train", "validation"]:
        print(f"Caching {split} for all components.")
        # build_caches will build or load as needed
        args.build_caches(split)
        print(f"Finished caching {split}.")


if __name__ == "__main__":
    main()
