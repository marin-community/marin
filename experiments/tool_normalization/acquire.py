# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read pinned pilot source rows from the existing CoreWeave mirror."""

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from levanter.store.tree_store import TreeStore
from rigging.filesystem.s3_compat import configure_coreweave_s3
from tokenizers import Tokenizer

from experiments.tool_normalization.scope import EXCLUDED_SOURCES


def save_source(row: dict, ids: np.ndarray, tokenizer: Tokenizer, output: Path) -> dict:
    if row["source"] in EXCLUDED_SOURCES:
        raise ValueError("Excluded source in acquisition manifest")
    identity = f'{row["source"]}\n{row["shard"]}\n{row["row"]}'
    key = hashlib.sha256(identity.encode()).hexdigest()
    directory = output / key
    record = directory / "source.json"
    if record.exists():
        saved = json.loads(record.read_text())
        if hashlib.sha256((directory / "input_ids.i32").read_bytes()).hexdigest() != saved["input_ids_sha256"]:
            raise ValueError("Existing source payload hash mismatch")
        if hashlib.sha256(ids.tobytes()).hexdigest() != saved["input_ids_sha256"]:
            raise ValueError("Source shard changed since acquisition")
        if hashlib.sha256(saved["text"].encode()).hexdigest() != saved["text_sha256"]:
            raise ValueError("Existing source text hash mismatch")
        return saved
    text = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
    saved = {
        **row,
        "id": key,
        "tokens": len(ids),
        "input_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
        "source_loss_masks": "not stored in source TreeStore; must reconstruct and validate from original template",
        "length_disposition": "over_limit" if len(ids) > 262144 else "eligible",
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "input_ids.i32").write_bytes(ids.tobytes())
    record.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"id": key, "tokens": len(ids), "source": row["source"]}), flush=True)
    return saved


def acquire_shard(rows: list[dict], tokenizer: Tokenizer, output: Path) -> list[dict]:
    """Read one shard once and retain only metadata in the returned index."""
    shard = rows[0]["shard"]
    if any(row["shard"] != shard or row["source"] in EXCLUDED_SOURCES for row in rows):
        raise ValueError("Acquisition group contains an excluded source or another shard")
    store = TreeStore.open({"input_ids": np.zeros(0, dtype=np.int32)}, shard, mode="r", cache_metadata=True).tree[
        "input_ids"
    ]
    index = []
    for start in range(0, len(rows), 32):
        batch = rows[start : start + 32]
        arrays = store.get_batch_sync([row["row"] for row in batch])
        for row, ids in zip(batch, arrays, strict=True):
            saved = save_source(row, np.asarray(ids, dtype="<i4"), tokenizer, output)
            index.append({key: value for key, value in saved.items() if key != "text"})
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rows = json.loads(args.manifest.read_text())["rows"]
    args.output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row["shard"]].append(row)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(lambda group: acquire_shard(group, tokenizer, args.output), groups.values())
        index = [row for group in results for row in group]
    (args.output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(
        json.dumps(
            {
                "rows": len(index),
                "tokens": sum(row["tokens"] for row in index),
                "long_rows_32k": sum(row["tokens"] > 32768 for row in index),
            }
        )
    )


if __name__ == "__main__":
    main()
