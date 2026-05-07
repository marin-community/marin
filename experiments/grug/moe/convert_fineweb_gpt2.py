# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert modded-nanogpt FineWeb10B GPT-2 tokenized .bin files to Levanter TreeCache format.

Downloads from HuggingFace (kjj0/fineweb10B-gpt2), splits on BOS token (50256),
converts to TreeCache with train/validation splits, and writes to GCS.

Usage (as iris job):
    .venv/bin/iris --config lib/iris/examples/marin.yaml job run --no-wait \
        -- python -m experiments.grug.moe.convert_fineweb_gpt2 \
            --output gs://marin-us-central1/data/fineweb10B-gpt2
"""

import argparse
import json
import logging
import os
import tempfile

import numpy as np
from huggingface_hub import hf_hub_download
from levanter.store.cache import CacheLedger
from levanter.store.tree_store import TreeStore

logger = logging.getLogger(__name__)

HF_REPO = "kjj0/fineweb10B-gpt2"
NUM_TRAIN_SHARDS = 103
BOS_TOKEN_ID = 50256
MIN_DOC_LEN = 16  # Skip very short docs


def _download_shard(shard_name: str, local_dir: str) -> str:
    return hf_hub_download(repo_id=HF_REPO, filename=shard_name, repo_type="dataset", local_dir=local_dir)


def _read_bin(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint16).astype(np.int32)


def _split_on_bos(tokens: np.ndarray) -> list[np.ndarray]:
    """Split a flat token array into documents at BOS token boundaries."""
    bos_positions = np.where(tokens == BOS_TOKEN_ID)[0]
    if len(bos_positions) == 0:
        return [tokens]
    docs = []
    for i in range(len(bos_positions)):
        start = bos_positions[i]
        end = bos_positions[i + 1] if i + 1 < len(bos_positions) else len(tokens)
        if end - start >= MIN_DOC_LEN:
            docs.append(tokens[start:end])
    return docs


def _write_docs_to_cache(store: TreeStore, docs: list[np.ndarray]) -> int:
    """Write documents to a TreeStore, return number written."""
    batch = [{"input_ids": doc} for doc in docs]
    store.extend(batch)
    return len(batch)


def _write_ledger(output_dir: str, total_rows: int, shard_rows: dict[str, int]):
    ledger = CacheLedger(total_num_rows=total_rows, shard_rows=shard_rows, is_finished=True)
    ledger_path = os.path.join(output_dir, "cache_ledger.json")
    if output_dir.startswith("gs://"):
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        with fs.open(ledger_path, "w") as f:
            json.dump(ledger.to_dict(), f)
    else:
        os.makedirs(output_dir, exist_ok=True)
        with open(ledger_path, "w") as f:
            json.dump(ledger.to_dict(), f)


def convert(output_base: str, num_shards: int = NUM_TRAIN_SHARDS):
    """Download and convert FineWeb10B-GPT2 to Levanter TreeCache."""
    train_dir = os.path.join(output_base, "train")
    val_dir = os.path.join(output_base, "validation")

    exemplar = {"input_ids": np.zeros(0, dtype=np.int32)}
    train_store = TreeStore.open(exemplar, train_dir, mode="a", cache_metadata=False)
    val_store = TreeStore.open(exemplar, val_dir, mode="a", cache_metadata=False)

    train_total = 0
    val_total = 0
    train_shard_rows: dict[str, int] = {}
    val_shard_rows: dict[str, int] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Process validation shard
        val_name = "fineweb_val_000000.bin"
        logger.info(f"Processing validation: {val_name}")
        path = _download_shard(val_name, tmpdir)
        tokens = _read_bin(path)
        docs = _split_on_bos(tokens)
        n = _write_docs_to_cache(val_store, docs)
        val_shard_rows[val_name] = n
        val_total += n
        os.remove(path)
        logger.info(f"  -> {n} docs from validation")

        # Process train shards
        for i in range(1, num_shards + 1):
            shard_name = f"fineweb_train_{i:06d}.bin"
            logger.info(f"Processing train {shard_name} ({i}/{num_shards})")
            path = _download_shard(shard_name, tmpdir)
            tokens = _read_bin(path)
            docs = _split_on_bos(tokens)
            n = _write_docs_to_cache(train_store, docs)
            train_shard_rows[shard_name] = n
            train_total += n
            os.remove(path)
            logger.info(f"  -> {n} docs, running total: {train_total}")

    _write_ledger(train_dir, train_total, train_shard_rows)
    _write_ledger(val_dir, val_total, val_shard_rows)
    logger.info(f"Done. train={train_total} docs, val={val_total} docs at {output_base}")


def main():
    parser = argparse.ArgumentParser(description="Convert FineWeb10B-GPT2 to Levanter TreeCache")
    parser.add_argument("--output", required=True, help="Output directory (local or gs://)")
    parser.add_argument("--num-shards", type=int, default=NUM_TRAIN_SHARDS, help="Number of train shards")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    convert(args.output, args.num_shards)


if __name__ == "__main__":
    main()
