# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count every token id over the first shards of a flat token cache and write the ``[vocab]`` int64 counts.

    python -m experiments.grug.fast_track.token_frequency --shards 8 --out s3://.../token_counts.npz

Run it in the cache's region (an Iris CPU job); it reads each shard's ``input_ids/data`` zarr3 array in chunks.
"""

import io
import json
import logging

import click
import fsspec
import numpy as np
import tensorstore as ts
from levanter.tensorstore_serialization import build_kvstore_spec

from experiments.grug.fast_track.launch import V16384_CACHE_DIR, V16384_VOCAB

logger = logging.getLogger(__name__)

NUM_SHARDS = 128
READ_TOKENS = 1 << 27


def shard_counts(shard_dir: str, vocab: int) -> tuple[np.ndarray, int]:
    """Token-id counts of one cache shard and its token count."""
    with fsspec.open(f"{shard_dir}/shard_ledger.json") as f:
        num_tokens = json.load(f)["field_counts"]["input_ids"]
    array = ts.open({"driver": "zarr3", "kvstore": build_kvstore_spec(f"{shard_dir}/input_ids/data")}).result()
    counts = np.zeros(vocab, np.int64)
    for start in range(0, num_tokens, READ_TOKENS):
        ids = array[start : min(start + READ_TOKENS, num_tokens)].read().result()
        counts += np.bincount(ids, minlength=vocab)[:vocab]
    return counts, num_tokens


@click.command()
@click.option("--shards", type=int, required=True, help="How many shards, from part-00000, to count.")
@click.option("--out", required=True, help="Output .npz path.")
def main(shards: int, out: str) -> None:
    logging.basicConfig(level=logging.INFO)
    total = np.zeros(V16384_VOCAB, np.int64)
    tokens = 0
    for i in range(shards):
        counts, n = shard_counts(f"{V16384_CACHE_DIR}/part-{i:05d}-of-{NUM_SHARDS:05d}", V16384_VOCAB)
        total += counts
        tokens += n
        logger.info("shard %d: %d tokens, %d ids unseen so far", i, n, int(np.sum(total == 0)))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, counts=total, num_tokens=np.asarray(tokens), shards=np.asarray(shards))
    with fsspec.open(out, "wb") as f:
        f.write(buffer.getvalue())
    logger.info("wrote %s (%d tokens)", out, tokens)


if __name__ == "__main__":
    main()
