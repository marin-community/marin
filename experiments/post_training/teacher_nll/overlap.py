# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check whether teacher-NLL rows appear in the 10/05 SFT stores.

    python overlap.py --cells gs://.../contrib-full.jsonl.gz [...] --output gs://.../run --shard K --num-shards N

Work items are whole agentic stores and single parts of the other stores, spread over shards by token
count. Each shard streams its items once, in document-aligned windows:

- Exact (every store): a store document matches a row when both token sequences are equal after
  stripping the render tail (any of 198, 220, 128001, 128009).
- 50-gram (agentic stores): every 50-token window is hashed and looked up in the 50-grams that lie
  fully inside a row's loss-bearing span and occur in at most two tasks (shared ones are template or
  harness boilerplate; attempts of one task may share grams). Per store, each row's covered grams
  are counted once; documents that share at least STRONG_GRAMS grams with a row are listed. This
  catches a trace serialized differently.

Writes ``shard-K.json`` under ``--output``.
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import time
from collections import defaultdict

import fsspec
import numpy as np
import tensorstore as ts

TAIL = (198, 220, 128001, 128009)
GRAM = 50
BASE = np.uint64(0x9E3779B97F4A7C15)
BASE_INV = np.uint64(pow(int(BASE), -1, 1 << 64))
WINDOW_TOKENS = 1 << 25
MAX_TASKS_PER_GRAM = 2
MAX_ROWS_PER_DOC = 20
STRONG_GRAMS = 200
AGENTIC = re.compile(r"agent|swe|penfever|terminus|tool|opencode|interactive")
MANIFEST = "gs://marin-us-central2/grug_sft/stores/full-ab-packed-2026.09.29.json"
TS_CONTEXT = {"cache_pool": {"total_bytes_limit": 0}, "data_copy_concurrency": {"limit": 4}}


def normalize(ids: np.ndarray) -> np.ndarray:
    end = len(ids)
    while end and int(ids[end - 1]) in TAIL:
        end -= 1
    return ids[:end]


def digest(ids: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(ids, dtype=np.int64).tobytes()).hexdigest()


def powers(base: np.uint64, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.uint64)
    out[0] = 1
    with np.errstate(over="ignore"):
        np.cumprod(np.full(n - 1, base, dtype=np.uint64), out=out[1:])
    return out


class GramHasher:
    """Hash every GRAM-token window: H[i] = sum_j (t[i+j] + 1) * BASE^j (mod 2^64)."""

    def __init__(self, max_tokens: int):
        self.pow = powers(BASE, max_tokens)
        self.inv = powers(BASE_INV, max_tokens)

    def __call__(self, tokens: np.ndarray) -> np.ndarray:
        n = len(tokens)
        if n < GRAM:
            return np.empty(0, dtype=np.uint64)
        with np.errstate(over="ignore"):
            prefix = np.empty(n + 1, dtype=np.uint64)
            prefix[0] = 0
            np.cumsum((tokens.astype(np.uint64) + np.uint64(1)) * self.pow[:n], out=prefix[1:])
            return (prefix[GRAM:] - prefix[: n - GRAM + 1]) * self.inv[: n - GRAM + 1]


def read_rows(paths: list[str]) -> list[dict]:
    """Rows keyed by trial (first file wins); token arrays are kept as numpy to bound memory."""
    rows: dict[str, dict] = {}
    for path in paths:
        with fsspec.open(path, "rb") as raw, gzip.open(raw, "rt") as f:
            for line in f:
                row = json.loads(line)
                if row["trial"] in rows:
                    continue
                rows[row["trial"]] = {
                    "trial": row["trial"],
                    "task": row.get("task_id") or row["trial"].split("__")[0],
                    "input_ids": np.asarray(row["input_ids"], dtype=np.int64),
                    "loss_weights": np.asarray(row["loss_weights"], dtype=np.float32),
                }
    return list(rows.values())


class RowIndex:
    """Exact digests by normalized length, and the loss-span 50-gram query set with each gram's rows."""

    def __init__(self, rows: list[dict], hasher: GramHasher):
        self.trials = [row["trial"] for row in rows]
        tasks = {task: k for k, task in enumerate(sorted({row["task"] for row in rows}))}
        self.exact: dict[int, dict[str, int]] = defaultdict(dict)
        grams, owners = [], []
        for r, row in enumerate(rows):
            ids = row["input_ids"]
            norm = normalize(ids)
            self.exact[len(norm)][digest(norm)] = r
            # weight i scores token i+1: the loss-bearing tokens are ids[1:][weights[:-1] > 0].
            target = np.zeros(len(ids), dtype=bool)
            target[1:] = row["loss_weights"][:-1] > 0
            if len(ids) < GRAM:
                continue
            inside = np.lib.stride_tricks.sliding_window_view(target, GRAM).all(axis=1)
            unique = np.unique(hasher(ids)[inside])
            grams.append(unique)
            owners.append(np.full(len(unique), r, dtype=np.int32))
        task_of = np.array([tasks[row["task"]] for row in rows], dtype=np.int32)
        values = np.concatenate(grams)
        rows_of = np.concatenate(owners)
        order = np.lexsort((rows_of, values))
        values, rows_of = values[order], rows_of[order]
        # Distinct tasks per gram: count (gram, task) pairs.
        pair_order = np.lexsort((task_of[rows_of], values))
        pv, pt = values[pair_order], task_of[rows_of][pair_order]
        new_pair = np.ones(len(pv), dtype=bool)
        new_pair[1:] = (pv[1:] != pv[:-1]) | (pt[1:] != pt[:-1])
        gram_ids, tasks_per_gram = np.unique(pv[new_pair], return_counts=True)
        keep_gram = gram_ids[tasks_per_gram <= MAX_TASKS_PER_GRAM]
        keep = np.isin(values, keep_gram)
        values, rows_of = values[keep], rows_of[keep]
        self.query, first = np.unique(values, return_index=True)
        # CSR: rows of query[q] are owner_rows[owner_ptr[q]:owner_ptr[q + 1]].
        self.owner_ptr = np.append(first, len(values)).astype(np.int64)
        self.owner_rows = rows_of
        self.row_grams = np.bincount(rows_of, minlength=len(rows))
        self.bitmap = np.zeros(1 << 26, dtype=np.uint64)  # 2^32 bits over the low hash bits
        low = (self.query & np.uint64(0xFFFFFFFF)).astype(np.int64)
        np.bitwise_or.at(self.bitmap, low >> 6, np.left_shift(np.uint64(1), (low & 63).astype(np.uint64)))

    def lookup(self, hashes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (positions in hashes, query indices) of the hashes that are in the query set."""
        low = (hashes & np.uint64(0xFFFFFFFF)).astype(np.int64)
        maybe = np.nonzero((self.bitmap[low >> 6] >> (low & 63).astype(np.uint64)) & np.uint64(1))[0]
        if not len(maybe) or not len(self.query):
            return maybe[:0], maybe[:0]
        pos = np.minimum(np.searchsorted(self.query, hashes[maybe]), len(self.query) - 1)
        hit = self.query[pos] == hashes[maybe]
        return maybe[hit], pos[hit]

    def row_counts(self, query_indices: np.ndarray) -> np.ndarray:
        """Per-row count of the given (distinct) query grams."""
        starts = self.owner_ptr[query_indices]
        sizes = self.owner_ptr[query_indices + 1] - starts
        if not sizes.sum():
            return np.zeros(len(self.trials), dtype=np.int64)
        offsets = np.repeat(starts - np.concatenate([[0], np.cumsum(sizes)[:-1]]), sizes)
        flat = np.arange(int(sizes.sum())) + offsets
        return np.bincount(self.owner_rows[flat], minlength=len(self.trials))


def list_parts(root: str) -> list[str]:
    kv = ts.KvStore.open(root).result()
    keys = kv.list(ts.KvStore.KeyRange(inclusive_min="part-", exclusive_max="part.")).result()
    names = sorted({k.decode().split("/", 1)[0] for k in keys if k.decode().endswith("offsets/zarr.json")})
    return [root + n + "/input_ids/" for n in names]


def work_items(manifest: dict) -> list[dict]:
    items = []
    for name, entry in sorted(manifest.items()):
        tokens = sum(source["tokens"] for source in entry["sources"].values())
        parts = list_parts(entry["cache_path"] + "/train/")
        if AGENTIC.search(name):
            items.append({"store": name, "parts": parts, "grams": True, "tokens": tokens})
        else:
            items += [{"store": name, "parts": [p], "grams": False, "tokens": tokens / len(parts)} for p in parts]
    return items


def assign(items: list[dict], num_shards: int) -> list[list[dict]]:
    """Greedy longest-first; gram items cost about 3x the tokens of an exact-only item."""
    shards: list[list[dict]] = [[] for _ in range(num_shards)]
    load = [0.0] * num_shards
    for item in sorted(items, key=lambda i: (-i["tokens"] * (3 if i["grams"] else 1), i["store"], i["parts"][0])):
        k = min(range(num_shards), key=lambda s: (load[s], s))
        shards[k].append(item)
        load[k] += item["tokens"] * (3 if item["grams"] else 1)
    return shards


def scan_item(item: dict, index: RowIndex, hasher: GramHasher, record: dict) -> None:
    name = item["store"]
    seen = np.zeros(len(index.query), dtype=bool) if item["grams"] else None
    lengths = set(index.exact)
    for base in item["parts"]:
        offsets = ts.open({"driver": "zarr3", "kvstore": base + "offsets", "context": TS_CONTEXT}).result()
        count = int(offsets[0].read().result())
        ends = offsets[1 : count + 1].read().result().astype(np.int64)
        starts = np.concatenate([[0], ends[:-1]])
        data = ts.open({"driver": "zarr3", "kvstore": base + "data", "context": TS_CONTEXT}).result()
        doc = 0
        while doc < count:
            last = max(int(np.searchsorted(ends, starts[doc] + WINDOW_TOKENS, side="right")), doc + 1)
            lo, hi = int(starts[doc]), int(ends[last - 1])
            tokens = data[lo:hi].read().result().astype(np.int64)
            record["tokens"] += hi - lo
            for i in range(doc, last):
                s, e = int(starts[i]) - lo, int(ends[i]) - lo
                if not any(e - s - k in lengths for k in range(7)):
                    continue
                norm = normalize(tokens[s:e])
                if len(norm) not in lengths:
                    continue
                record["exact_compared"] += 1
                r = index.exact[len(norm)].get(digest(norm))
                if r is not None:
                    record["exact_hits"].append({"store": name, "part": base, "doc": i, "trial": index.trials[r]})
            if seen is not None:
                record["gram_tokens"] += hi - lo
                found, qidx = index.lookup(hasher(tokens))
                local_starts = starts[doc:last] - lo
                docs = np.searchsorted(local_starts, found, side="right") - 1 + doc
                inside = found + GRAM <= ends[docs] - lo
                found, qidx, docs = found[inside], qidx[inside], docs[inside]
                seen[qidx] = True
                for d in np.unique(docs):
                    counts = index.row_counts(np.unique(qidx[docs == d]))
                    strong = np.nonzero(counts >= STRONG_GRAMS)[0]
                    if len(strong):
                        strong = strong[np.argsort(-counts[strong])][:MAX_ROWS_PER_DOC]
                        record["strong_docs"].append(
                            {
                                "store": name,
                                "part": base,
                                "doc": int(d),
                                "length": int(ends[d] - starts[d]),
                                "rows": {index.trials[r]: int(counts[r]) for r in strong},
                            }
                        )
            doc = last
        record["docs"] += count
    if seen is not None:
        counts = index.row_counts(np.nonzero(seen)[0])
        record["store_rows"][name] = {index.trials[r]: int(c) for r, c in enumerate(counts) if c}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--part-slice", default=None, help="smoke only: START:STOP of each store's parts")
    args = parser.parse_args()
    started = time.time()
    hasher = GramHasher(WINDOW_TOKENS + (1 << 19))
    rows = read_rows(args.cells)
    index = RowIndex(rows, hasher)
    with fsspec.open(args.manifest, "rt") as f:
        manifest = json.load(f)
    items = work_items(manifest)
    if args.part_slice:
        start, stop = (int(x) for x in args.part_slice.split(":"))
        items = [{**i, "parts": i["parts"][start:stop]} for i in items if i["parts"][start:stop]]
    items = assign(items, args.num_shards)[args.shard]
    record = {
        "shard": args.shard,
        "num_shards": args.num_shards,
        "rows": len(rows),
        "query_grams": len(index.query),
        "row_grams": {t: int(n) for t, n in zip(index.trials, index.row_grams, strict=True)},
        "items": [[i["store"], len(i["parts"]), i["grams"]] for i in items],
        "docs": 0,
        "tokens": 0,
        "gram_tokens": 0,
        "exact_compared": 0,
        "exact_hits": [],
        "strong_docs": [],
        "store_rows": {},
    }
    print(f"shard {args.shard}: {len(items)} items, {sum(i['tokens'] for i in items) / 1e9:.1f}B tokens", flush=True)
    for k, item in enumerate(items, 1):
        scan_item(item, index, hasher, record)
        print(
            f"{k}/{len(items)} {item['store']} tokens {record['tokens'] / 1e9:.2f}B "
            f"exact {len(record['exact_hits'])} strong {len(record['strong_docs'])} "
            f"{time.time() - started:.0f}s",
            flush=True,
        )
    record["seconds"] = round(time.time() - started, 1)
    record["task"] = os.environ.get("IRIS_TASK_ID", "local")
    with fsspec.open(f"{args.output.rstrip('/')}/shard-{args.shard}.json", "wt") as f:
        json.dump(record, f)
    print(json.dumps({k: v if not isinstance(v, (list, dict)) else len(v) for k, v in record.items()}), flush=True)


if __name__ == "__main__":
    main()
