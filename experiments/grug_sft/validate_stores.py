# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the full Grug SFT store manifest and read representative packed examples."""

import argparse
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import jax
import numpy as np
from haliax import Axis
from levanter.data.text.datasets import PackedTokenDataset
from levanter.schedule import BatchSchedule
from levanter.store.cache import CacheLedger, TreeCache
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging

from experiments.grug_sft.special_token_full_ab import MIX_PATH, STEPS, STORE_MANIFEST
from experiments.grug_sft.special_token_lr import BATCH, CONTEXT, TOKENIZER, data_config
from experiments.grug_sft.special_token_train import build_train_dataset, verify_data_epochs

logger = logging.getLogger(__name__)
SAMPLE_SOURCES = (
    "agenttrove-glm53-compactions",
    "penfever-traces/glm52-terminus2/exp_rpt_unitsyn-python-v3",
    "science-tool-use-conversations",
)
EXEMPLAR = {"input_ids": np.zeros(0, dtype=np.int32)}


def validate_ledger(item: tuple[str, dict]) -> tuple[str, int, int]:
    name, record = item
    if record["tokenizer"] != TOKENIZER or record["max_length"] != CONTEXT:
        raise ValueError(f"{name} has incompatible tokenizer or context length")
    path = prefix_join(record["cache_path"], "train")
    ledger = CacheLedger.load(path)
    if not ledger.is_finished or ledger.total_num_rows <= 0:
        raise ValueError(f"{name} has no finished training cache")
    actual_tokens = ledger.field_counts["input_ids"]
    declared_tokens = sum(int(value["tokens"]) for value in record["sources"].values())
    if actual_tokens != declared_tokens:
        raise ValueError(f"{name} ledger has {actual_tokens} tokens, manifest declares {declared_tokens}")
    return name, ledger.total_num_rows, actual_tokens


async def read_packed_sample(name: str, record: dict) -> None:
    path = prefix_join(record["cache_path"], "train")
    cache = TreeCache.load(path, EXEMPLAR)
    dataset = PackedTokenDataset(
        cache,
        Axis("position", CONTEXT),
        max_segments_per_example=CONTEXT,
        block_cross_document_attention=True,
    )
    if await dataset.async_len() <= 0:
        raise ValueError(f"{name} has no packed examples")
    example = (await dataset.get_batch([0]))[0]
    if example.tokens.shape != (CONTEXT,) or not np.any(np.asarray(example.loss_weight)):
        raise ValueError(f"{name} produced an empty or incorrectly shaped training example")
    logger.info("Read packed example from %s", name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=STORE_MANIFEST)
    args = parser.parse_args()

    manifest = json.loads(StoragePath(args.manifest).read_text())
    allocations = json.loads(MIX_PATH.read_text())["allocations_tokens"]
    if set(manifest) != set(allocations):
        raise ValueError("Store manifest does not cover the full SFT mix")

    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(validate_ledger, manifest.items()))
    logger.info(
        "Validated %d finished training caches: %d records, %d tokens",
        len(rows),
        sum(row[1] for row in rows),
        sum(row[2] for row in rows),
    )

    data, _ = data_config(STEPS, args.manifest, mix_path=MIX_PATH)
    data_key, _ = jax.random.split(jax.random.PRNGKey(0), 2)
    train_dataset = build_train_dataset(data, max_seq_len=CONTEXT, batch_schedule=BatchSchedule(BATCH), key=data_key)
    verify_data_epochs(train_dataset, STEPS * BATCH, 1)
    mixed_examples = asyncio.run(train_dataset.get_batch(list(range(8))))
    if len(mixed_examples) != 8 or any(example.tokens.shape != (CONTEXT,) for example in mixed_examples):
        raise ValueError("Mixed SFT reader returned malformed packed examples")
    logger.info("Read %d examples from the shuffled SFT/replay mixture", len(mixed_examples))
    for name in SAMPLE_SOURCES:
        asyncio.run(read_packed_sample(name, manifest[name]))
    logger.info("Full Grug SFT store smoke passed")


if __name__ == "__main__":
    configure_logging(logging.INFO)
    main()
