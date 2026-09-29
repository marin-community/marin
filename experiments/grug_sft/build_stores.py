# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the Levanter caches for the full Grug SFT mix and publish their counts.

Run on an Iris CPU coordinator in us-central2. Completed normalization and
tokenization steps are reused through their content-addressed artifacts.
"""

import argparse
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fray.types import ResourceConfig
from levanter.tokenizers import TokenizerBackend
from marin.datakit.sft_sources import all_sft_sources
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner
from marin.processing.tokenize.attributes import tokenize_attributes_step
from marin.processing.tokenize.store_builder import LevanterStoreData, build_levanter_store_step
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.grug_sft.report_packing import packed_sequences
from experiments.grug_sft.special_token_full_ab import MIX_PATH, STORE_MANIFEST
from experiments.grug_sft.special_token_lr import CONTEXT, TOKENIZER

REUSABLE_STORES_PATH = Path(__file__).with_name("reusable_stores.json")
MAX_WORKERS_PER_STEP = 64
WORKER_RESOURCES = ResourceConfig(cpu=2, ram="32g", disk="20g")


def selected_sources(requested: list[str]) -> list[str]:
    allocations = json.loads(MIX_PATH.read_text())["allocations_tokens"]
    selected = sorted(requested or allocations)
    unknown = set(selected) - set(allocations)
    if unknown:
        raise ValueError(f"Sources absent from the full SFT mix: {sorted(unknown)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--manifest")
    args = parser.parse_args()
    if args.max_concurrent < 1:
        raise ValueError("--max-concurrent must be positive")

    registry = all_sft_sources()
    names = selected_sources(args.source)
    reusable = json.loads(REUSABLE_STORES_PATH.read_text())
    missing = set(names) - set(registry)
    if missing:
        raise ValueError(f"SFT sources absent from registry: {sorted(missing)}")

    stores = {}
    existing = {}
    for name in names:
        tokenize = tokenize_attributes_step(
            name=f"datakit/tokenize/sft/{name}",
            train_normalize=registry[name].normalized,
            tokenizer=TOKENIZER,
            tokenizer_backend=TokenizerBackend.HF,
            max_workers=MAX_WORKERS_PER_STEP,
            worker_resources=WORKER_RESOURCES,
        )
        if name in reusable:
            record = reusable[name]
            if tokenize.output_path.rsplit("/", 1)[-1] != record["tokenized_artifact"].rsplit("/", 1)[-1]:
                raise ValueError(f"Reusable store for {name} has a stale tokenization dependency")
            existing[name] = record
            continue
        stores[name] = build_levanter_store_step(
            name=f"datakit/levanter-store/sft/{name}",
            tokenize_steps=[tokenize],
            max_workers=MAX_WORKERS_PER_STEP,
            levanter_batch_size=256,
            worker_resources=WORKER_RESOURCES,
        )

    logging.info("Reusing %d stores and building %d stores", len(existing), len(stores))
    if stores:
        StepRunner().run(list(stores.values()), max_concurrent=args.max_concurrent)

    manifest = {
        name: {
            "cache_path": record["cache_path"],
            "tokenizer": TOKENIZER,
            "max_length": CONTEXT,
            "sources": {name: {"tokens": record["tokens"]}},
        }
        for name, record in existing.items()
    }
    for name, step in stores.items():
        artifact = read_artifact(step.output_path, LevanterStoreData)
        if artifact.tokenizer != TOKENIZER:
            raise ValueError(f"Unexpected tokenizer in {name}: {artifact.tokenizer}")
        train = artifact.splits["train"]
        if train.total_tokens <= 0:
            raise ValueError(f"{name} has no training tokens")
        manifest[name] = {
            "cache_path": artifact.cache_path,
            "tokenizer": artifact.tokenizer,
            "max_length": CONTEXT,
            "sources": {name: {"tokens": train.total_tokens}},
        }

    with ThreadPoolExecutor(max_workers=16) as pool:
        lengths = pool.map(lambda item: asyncio.run(packed_sequences(item[1]["cache_path"])), manifest.items())
        for record, length in zip(manifest.values(), lengths, strict=True):
            record["packed_sequences"] = length

    if not args.source and set(manifest) != set(json.loads(MIX_PATH.read_text())["allocations_tokens"]):
        raise ValueError("Full store manifest does not cover the mix")
    manifest_path = args.manifest or (STORE_MANIFEST if not args.source else None)
    if manifest_path is not None:
        StoragePath(manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        logging.info("Published %d store records to %s", len(manifest), manifest_path)


if __name__ == "__main__":
    configure_logging(logging.INFO)
    main()
