# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pack the shootout contributor rows (8 x 32,768) into 262,144-token prebuilt Levanter caches.

Row r of packed epoch k is rows 8r..8r+7 of the source epoch file, in file order, so two packed
rows hold exactly the 16 rows of one shootout update. Tokens and loss weights are copied
unchanged. Each 32k row ends in 128001 padding with zero loss weight, so EOS-derived segments
keep every trajectory in its own attention segment.
"""

import argparse
import gzip
import hashlib
import json

import numpy as np
from levanter.store.cache import CacheMetadata, SerialCacheWriter
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join

ROWS = "s3://marin-us-east-02a/marin/experiments/sft-shootout/rows/t202-contrib-v01-20261022/plain"
# manifest.json output_hashes and splits, manifest sha 507e73f0...
FILE_SHA256 = {
    0: "fcb68466da3dbcc7d08d326404ff3a8ba78e455ce7f3e4b943af17fadff5965d",
    1: "cd140d6d6de3db29ca6f3e01a3ff10eebb25373b3bb6342ceb866091dbfe6209",
    2: "8ba702bb108f613cc385575d9fd7fe7c736d4a36fc314ccc1199b15d65bd2256",
}
SUPERVISED_PER_EPOCH = 16137263
SOURCE_ROWS = 3616
ROW_TOKENS = 32768
PACK = 8
ROWS_PER_UPDATE = 16
PAD = 128001
METADATA = {"input_ids_key": "input_ids", "loss_weights_key": "loss_weights"}
EXEMPLAR = {"input_ids": np.zeros((0,), dtype=np.int32), "loss_weights": np.zeros((0,), dtype=np.float32)}


def source_rows(epoch: int) -> list[dict]:
    fs, key = filesystem_for(prefix_join(ROWS, f"train-epoch-{epoch}-prebuilt.jsonl.gz"))
    content = fs.cat_file(key)
    if hashlib.sha256(content).hexdigest() != FILE_SHA256[epoch]:
        raise ValueError(f"Epoch {epoch} rows changed")
    rows = [json.loads(line) for line in gzip.decompress(content).splitlines()]
    if len(rows) != SOURCE_ROWS:
        raise ValueError(f"Epoch {epoch} has {len(rows)} rows")
    for row in rows:
        ids, weights = row["input_ids"], row["loss_weights"]
        if len(ids) != ROW_TOKENS or len(weights) != ROW_TOKENS or ids[-1] != PAD or weights[-1] != 0:
            raise ValueError("Every source row must be 32,768 tokens ending in zero-weight padding")
        end = len(ids)
        while end and ids[end - 1] == PAD:
            end -= 1
        if PAD in ids[:end] or any(weights[end:]):
            raise ValueError("A trajectory holds EOS before its padding, or padding carries loss")
    return rows


def build_epoch(epoch: int, output: str, updates: int | None) -> dict:
    rows = source_rows(epoch)
    if updates is not None:
        rows = rows[: updates * ROWS_PER_UPDATE]
    per_update = [
        sum(sum(row["loss_weights"][:-1]) for row in rows[start : start + ROWS_PER_UPDATE])
        for start in range(0, len(rows), ROWS_PER_UPDATE)
    ]
    packed_supervised = []
    path = prefix_join(output, f"epoch-{epoch}/train")
    with SerialCacheWriter(path, EXEMPLAR, metadata=CacheMetadata(METADATA)) as writer:
        for start in range(0, len(rows), PACK):
            group = rows[start : start + PACK]
            ids = np.concatenate([np.asarray(row["input_ids"], dtype=np.int32) for row in group])
            weights = np.concatenate([np.asarray(row["loss_weights"], dtype=np.float32) for row in group])
            packed_supervised.append(float(weights[:-1].sum()))
            writer.write_batch([{"input_ids": ids, "loss_weights": weights}])
    packed_per_update = [sum(packed_supervised[i : i + 2]) for i in range(0, len(packed_supervised), 2)]
    if packed_per_update != per_update:
        raise ValueError("Packing changed the supervised tokens of an update")
    if updates is None and sum(per_update) != SUPERVISED_PER_EPOCH:
        raise ValueError("Epoch supervised total differs from the row manifest")
    return {
        "epoch": epoch,
        "cache": path,
        "packed_rows": len(packed_supervised),
        "updates": len(per_update),
        "supervised_tokens": sum(per_update),
        "supervised_per_update": per_update,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--updates", type=int, default=None, help="Only the first N updates, for a local check")
    args = parser.parse_args()
    configure_coreweave_s3()
    summary = [build_epoch(epoch, args.output, args.updates) for epoch in range(args.epochs)]
    fs, key = filesystem_for(prefix_join(args.output, "summary.json"))
    fs.pipe_file(key, json.dumps(summary, indent=2).encode())
    print(json.dumps([{k: v for k, v in item.items() if k != "supervised_per_update"} for item in summary]))


if __name__ == "__main__":
    main()
