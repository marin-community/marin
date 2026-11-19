# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze a stratified replay review set without reading whole giant documents."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from levanter.store.jagged_array import JaggedArrayStore
from rigging.filesystem.storage_path import prefix_join
from tokenizers import Tokenizer

from experiments.sft_new_mix.exp3_runtime import configure_storage

HEAD_TOKENS = 4096
TAIL_TOKENS = 2048
SCORE_WINDOW_TOKENS = 512


def sample_frame(manifest: list[dict], count: int, seed: int) -> list[dict]:
    """Sample each nonempty cache once, then top up uniformly over caches."""
    strata = sorted((row for row in manifest if row["rows"]), key=lambda row: row["cache_dir"])
    if count < len(strata):
        raise ValueError(f"{count} documents cannot cover {len(strata)} nonempty strata")
    rng = np.random.default_rng(seed)
    selections = [*range(len(strata)), *rng.integers(len(strata), size=count - len(strata)).tolist()]
    seen: set[tuple[str, int]] = set()
    result = []
    for stratum in selections:
        entry = strata[stratum]
        index = int(rng.integers(entry["rows"]))
        while (entry["cache_dir"], index) in seen:
            index = int(rng.integers(entry["rows"]))
        seen.add((entry["cache_dir"], index))
        result.append({**entry, "doc_index": index, "sample_id": len(result), "seed": seed})
    return result


def extract_sample(entry: dict, tokenizer: Tokenizer, output: Path) -> dict:
    """Read review excerpts and complete-document BME windows at the sampled row."""
    store = JaggedArrayStore.open(prefix_join(entry["cache_dir"], "input_ids"), mode="r", dtype=np.int32)
    index = entry["doc_index"]
    # Offset zero is the row-count sentinel; offsets[1] is the end of row zero.
    start = int(store.offsets[index].read().result()) if index else 0
    end = int(store.offsets[index + 1].read().result())
    size = end - start
    head = np.asarray(store.data[start : min(end, start + HEAD_TOKENS)].read().result())
    tail_start = max(start + len(head), end - TAIL_TOKENS)
    tail = np.asarray(store.data[tail_start:end].read().result())
    offsets = sorted({0, max(0, (size - SCORE_WINDOW_TOKENS) // 2), max(0, size - SCORE_WINDOW_TOKENS)})
    windows = [
        np.asarray(store.data[start + offset : min(end, start + offset + SCORE_WINDOW_TOKENS)].read().result())
        for offset in offsets
    ]
    record = {
        "sample_id": entry["sample_id"],
        "seed": entry["seed"],
        "cache_dir": entry["cache_dir"],
        "doc_index": index,
        "tokens": size,
        "population_rows": entry["rows"],
        "population_tokens": entry["tokens"],
        "lengths_sha256": entry["lengths_sha256"],
        "review_head_tokens": len(head),
        "review_tail_tokens": len(tail),
        "review_clipped": len(head) + len(tail) < size,
        "text": (
            tokenizer.decode(head.tolist(), skip_special_tokens=False)
            + ("\n\n[OMITTED DOCUMENT MIDDLE]\n\n" if tail_start > start + len(head) else "")
            + tokenizer.decode(tail.tolist(), skip_special_tokens=False)
        ),
        "bme_window_starts": offsets,
        "bme_token_ids": [window.tolist() for window in windows],
    }
    content = json.dumps(record, ensure_ascii=False, sort_keys=True).encode()
    (output / f"sample-{entry['sample_id']:04d}.json").write_bytes(content)
    return {key: value for key, value in record.items() if key not in ("text", "bme_token_ids")} | {
        "sha256": hashlib.sha256(content).hexdigest()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    configure_storage()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_bytes = args.manifest.read_bytes()
    frame = sample_frame(json.loads(manifest_bytes), args.count, args.seed)
    (args.output / "frame.json").write_text(json.dumps(frame, indent=2))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        samples = list(pool.map(lambda entry: extract_sample(entry, tokenizer, args.output), frame))
    receipt = {
        "seed": args.seed,
        "strata": len({entry["cache_dir"] for entry in frame}),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "samples": samples,
    }
    (args.output / "manifest.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({"samples": len(samples), "strata": receipt["strata"]}))


if __name__ == "__main__":
    main()
