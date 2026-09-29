# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure packed sequence capacity for the full Grug SFT mix."""

import argparse
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from haliax import Axis
from levanter.data.text.datasets import PackedTokenDataset
from levanter.store.cache import TreeCache
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging

from experiments.grug_sft.special_token_full_ab import STORE_MANIFEST
from experiments.grug_sft.special_token_lr import CONTEXT

logger = logging.getLogger(__name__)
EXEMPLAR = {"input_ids": np.zeros(0, dtype=np.int32)}


async def packed_sequences(cache_path: str) -> int:
    cache = TreeCache.load(prefix_join(cache_path, "train"), EXEMPLAR)
    dataset = PackedTokenDataset(
        cache,
        Axis("position", CONTEXT),
        max_segments_per_example=CONTEXT,
        block_cross_document_attention=True,
    )
    return await dataset.async_len()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=STORE_MANIFEST)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = json.loads(StoragePath(args.manifest).read_text())
    with ThreadPoolExecutor(max_workers=16) as pool:
        lengths = pool.map(lambda record: asyncio.run(packed_sequences(record["cache_path"])), manifest.values())
        packed = dict(zip(manifest, lengths, strict=True))

    sources = {
        name: {
            "raw_tokens": sum(int(counts["tokens"]) for counts in record["sources"].values()),
            "packed_sequences": packed[name],
            "packed_positions": packed[name] * CONTEXT,
        }
        for name, record in manifest.items()
    }
    raw_tokens = sum(source["raw_tokens"] for source in sources.values())
    packed_positions = sum(source["packed_positions"] for source in sources.values())
    report = {
        "context_length": CONTEXT,
        "raw_tokens": raw_tokens,
        "packed_sequences": sum(packed.values()),
        "packed_positions": packed_positions,
        "packing_utilization": raw_tokens / packed_positions,
        "sources": sources,
    }
    StoragePath(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    logger.info(
        "Packed %d raw tokens into %d positions (%.4f utilization)",
        raw_tokens,
        packed_positions,
        raw_tokens / packed_positions,
    )


if __name__ == "__main__":
    configure_logging(logging.INFO)
    main()
