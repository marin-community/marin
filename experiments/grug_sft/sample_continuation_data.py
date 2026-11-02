# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample a small, fixed slice of the 10.05 SFT mixture for short continuation runs.

The 10.05 SFT (``special_token_full_ab_grouped.py``) mixed 216 packed SFT stores with pretraining replay.
A continuation of a few steps needs only a few hundred million tokens, so this script draws whole documents
from every source in proportion to its share of the 10.05 mixture and writes two caches:

* ``<output>/sft/train``: SFT documents, read back as a packed text component.
* ``<output>/replay``: replay documents, read back as a flat (concatenated) cache.

SFT stores keep their cache under ``<cache_path>/train``; replay paths are flat caches.

Documents are written in a shuffled order, and ``manifest.json`` records the tokens drawn per source.
"""

import argparse
import json
import logging
import math
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from levanter.store.cache import SerialCacheWriter, TreeCache
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

MIX_PATH = Path(__file__).with_name("special_token_full_ab_grouped_mix.json")
REPLAY_PATH = Path(__file__).with_name("replay_skew8.json")
STORES_MANIFEST = "gs://marin-us-central2/grug_sft/stores/full-ab-packed-2026.09.29.json"
EXEMPLAR = {"input_ids": np.zeros((0,), dtype=np.int32)}
READ_BATCH = 512
WRITE_BATCH = 256
READ_WORKERS = 32


@dataclass(frozen=True)
class SourceDraw:
    name: str
    cache_path: str
    target_tokens: int


def source_draws(total_tokens: int) -> tuple[list[SourceDraw], list[SourceDraw]]:
    """Split ``total_tokens`` across SFT stores and replay caches by their 10.05 mixture shares."""
    plan = json.loads(MIX_PATH.read_text())
    stores = json.loads(StoragePath(STORES_MANIFEST).read_text())
    sft_fraction = float(plan["sft_fraction"])
    allocations = {name: int(tokens) for name, tokens in plan["allocations_tokens"].items()}
    budget = sum(allocations.values())
    sft = [
        SourceDraw(name, f"{stores[name]['cache_path']}/train", math.ceil(total_tokens * sft_fraction * tokens / budget))
        for name, tokens in sorted(allocations.items())
    ]

    replay = []
    for name, record in sorted(json.loads(REPLAY_PATH.read_text()).items()):
        copies = record["copies"]
        total_copies = sum(copies.values())
        for path, count in sorted(copies.items()):
            share = (1.0 - sft_fraction) * record["weight"] * count / total_copies
            replay.append(SourceDraw(f"{name}:{path}", path, math.ceil(total_tokens * share)))
    return sft, replay


def draw_documents(draw: SourceDraw, rng: np.random.Generator) -> list[np.ndarray]:
    """Draw whole documents without replacement until the source's token target is met."""
    cache = TreeCache.load(draw.cache_path, EXEMPLAR)
    order = rng.permutation(len(cache))
    docs: list[np.ndarray] = []
    tokens = 0
    for start in range(0, len(order), READ_BATCH):
        for row in cache.get_batch_sync([int(i) for i in order[start : start + READ_BATCH]]):
            docs.append(np.asarray(row["input_ids"], dtype=np.int32))
            tokens += len(docs[-1])
            if tokens >= draw.target_tokens:
                logger.info("%s: %d docs, %d tokens", draw.name, len(docs), tokens)
                return docs
    logger.warning("%s exhausted after %d of %d tokens", draw.name, tokens, draw.target_tokens)
    return docs


def write_cache(path: str, docs: list[np.ndarray], rng: np.random.Generator) -> None:
    """Write locally, then upload: cache writers use the ambient s3fs, which cannot reach CoreWeave from GCP."""
    with tempfile.TemporaryDirectory() as local:
        with SerialCacheWriter(local, EXEMPLAR) as writer:
            order = rng.permutation(len(docs))
            for start in range(0, len(order), WRITE_BATCH):
                writer.write_batch({"input_ids": [docs[i] for i in order[start : start + WRITE_BATCH]]})
        fs, remote = filesystem_for(path)
        fs.put(f"{local}/", remote, recursive=True)


def sample(output: str, total_tokens: int, seed: int) -> None:
    """Each source draws with its own seeded generator, so the sample does not depend on read order."""
    sft_draws, replay_draws = source_draws(total_tokens)
    manifest: dict[str, dict] = {"seed": seed, "total_tokens_requested": total_tokens, "sources": {}}
    caches = {"sft/train": sft_draws, "replay": replay_draws}
    for cache_index, (cache_name, draws) in enumerate(caches.items()):
        with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
            drawn_per_source = list(
                pool.map(
                    lambda item, c=cache_index: draw_documents(item[1], np.random.default_rng([seed, c, item[0]])),
                    enumerate(draws),
                )
            )
        docs: list[np.ndarray] = []
        for draw, drawn in zip(draws, drawn_per_source, strict=True):
            docs.extend(drawn)
            manifest["sources"][draw.name] = {
                "cache": cache_name,
                "cache_path": draw.cache_path,
                "target_tokens": draw.target_tokens,
                "drawn_tokens": int(sum(len(d) for d in drawn)),
                "drawn_documents": len(drawn),
            }
        logger.info("%s: %d docs from %d sources", cache_name, len(docs), len(draws))
        write_cache(f"{output}/{cache_name}", docs, np.random.default_rng([seed, cache_index]))
    fs, remote = filesystem_for(f"{output}/manifest.json")
    with fs.open(remote, "w") as handle:
        json.dump(manifest, handle, indent=1)


if __name__ == "__main__":
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--total-tokens", type=int, default=1_400_000_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sample(args.output, args.total_tokens, args.seed)
