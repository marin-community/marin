# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select a fixed 10/05 SFT text slice for offline Mochi distillation and pack it into 262,144-token rows.

Documents come from the 10/05 math and agentic/SWE caches, at roughly their mixture shares. Each document gets
an assistant-only shifted loss mask (weight i scores token i + 1). Documents are packed whole into rows of
exactly 262,144 tokens; leftover space is EOS padding with zero weight. Outputs:

- ``<plan>/<split>/rows.jsonl``: per packed row, the ordered ``(doc, offset)`` list.
- ``<plan>/<split>/docs-NNN.jsonl.gz``: per document ``input_ids`` and ``loss_weights``, eight whole packed rows
  per cell, in row order (the scorer's cells, also written under ``--cells-root`` in the scoring region).

    python -m experiments.post_training.teacher_distill.prep_slice --plan s3://.../plan --cells-root s3://.../cells
"""

import argparse
import gzip
import json
import logging

import numpy as np
from levanter.store.cache import TreeCache
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

logger = logging.getLogger(__name__)

POS = 262_144
EOS = 128_001
START_HEADER, END_HEADER, EOT, EOM = 128_006, 128_007, 128_009, 128_008
ASSISTANT = 78_191
MAX_DOC = 65_536  # fits the teacher serve's max_model_len of 73,728
MIN_TARGET = 16
BASE = "s3://marin-us-east-02a/marin/datakit/"
# (name, cache, share of selected tokens). Math : agentic/SWE ~ 63 : 37, as in the 10/05 mixture weights.
SOURCES = (
    ("math_v3", "levanter-store/sft/nemotron_sft_v3/math_v3/train_c0e873eb/train", 0.40),
    ("math_v2_high", "levanter-store/sft/nemotron_sft_v3/math_v2/high_1384dbf7/train", 0.23),
    ("agenttrove", "levanter-store/sft/agenttrove_64b65e16/train", 0.17),
    ("swe_rebench_v2", "store/sft/open_swe_traces/v1_2/minisweagent/qwen38_27b/swe_rebench_v2_f35daad0/train", 0.20),
)
SPLITS = (("train", 480), ("heldout", 32))
ROWS_PER_CELL = 8
SEED = 504


def assistant_weights(ids: np.ndarray) -> np.ndarray:
    """Shifted weights: 1 at i when token i + 1 is assistant-generated (content through <|eot_id|>)."""
    mask = np.zeros(len(ids), np.float32)
    i, n = 0, len(ids)
    while i < n - 2:
        if ids[i] == START_HEADER and ids[i + 1] == ASSISTANT and ids[i + 2] == END_HEADER:
            # The template writes one newline after the header; generation starts after it.
            j = i + 4
            while j < n and ids[j] not in (EOT, EOM, START_HEADER):
                mask[j] = 1
                j += 1
            if j < n and ids[j] in (EOT, EOM):
                mask[j] = 1
                j += 1
            i = j
        else:
            i += 1
    weights = np.zeros(len(ids), np.float32)
    weights[:-1] = mask[1:]
    return weights


def select(split_index: int, rows_needed: int, rng: np.random.Generator, taken: dict[str, set[int]]):
    budget = rows_needed * POS * 1.04
    docs = []
    for name, path, share in SOURCES:
        cache = TreeCache.load(BASE + path, {"input_ids": np.zeros(0, np.int32)})
        total = len(cache)
        want, got = budget * share, 0
        while got < want:
            candidates = [int(x) for x in rng.choice(total, size=512, replace=False) if int(x) not in taken[name]]
            batch = cache.get_batch_sync(candidates)
            for index, entry in zip(candidates, batch):
                taken[name].add(index)
                ids = np.asarray(entry["input_ids"], np.int32)
                if len(ids) > MAX_DOC or (ids[:-1] == EOS).any():
                    continue
                if ids[-1] != EOS:
                    ids = np.append(ids, np.int32(EOS))
                weights = assistant_weights(ids)
                if weights.sum() < MIN_TARGET:
                    continue
                docs.append({"doc": f"{name}/{index}", "source": name, "input_ids": ids, "loss_weights": weights})
                got += len(ids)
                if got >= want:
                    break
        logger.info("split %d %s: %d tokens from %d rows", split_index, name, got, len(taken[name]))
    return docs


def pack(docs: list[dict], rows_needed: int, rng: np.random.Generator):
    """First fit over open rows, in a seeded random order. Returns rows of (doc index, offset)."""
    order = rng.permutation(len(docs))
    rows: list[list[tuple[int, int]]] = []
    used: list[int] = []
    for k in order:
        length = len(docs[k]["input_ids"])
        for r in range(len(rows)):
            if used[r] + length <= POS:
                rows[r].append((int(k), used[r]))
                used[r] += length
                break
        else:
            if len(rows) == rows_needed:
                continue
            rows.append([(int(k), 0)])
            used.append(length)
    fill = sum(used) / (len(rows) * POS)
    return rows, fill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="02a prefix for the packing plan")
    parser.add_argument("--cells-root", required=True, help="scoring-region prefix for the scorer cells")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    rng = np.random.default_rng(SEED)
    taken: dict[str, set[int]] = {name: set() for name, _, _ in SOURCES}
    summary = {}
    for split_index, (split, rows_needed) in enumerate(SPLITS):
        docs = select(split_index, rows_needed, rng, taken)
        rows, fill = pack(docs, rows_needed, rng)
        if len(rows) != rows_needed:
            raise ValueError(f"{split}: packed {len(rows)} rows, need {rows_needed}")
        # Renumber documents in packed-row order so each cell holds whole rows: a scoring time cap then
        # drops whole rows, never part of one.
        order = [k for row in rows for k, _ in row]
        renumber = {k: n for n, k in enumerate(order)}
        docs = [docs[k] for k in order]
        plan_rows = [[{"doc": renumber[k], "offset": o} for k, o in row] for row in rows]
        StoragePath(prefix_join(args.plan, f"{split}/rows.jsonl")).write_text(
            "".join(json.dumps(r) + "\n" for r in plan_rows)
        )
        cells = []
        for cell_index, start in enumerate(range(0, len(rows), ROWS_PER_CELL)):
            cell_docs = [renumber[k] for row in rows[start : start + ROWS_PER_CELL] for k, _ in row]
            lines = [
                json.dumps(
                    {
                        "cell": f"{split}-{cell_index:03d}",
                        "row": n,
                        "doc": docs[n]["doc"],
                        "source": docs[n]["source"],
                        "input_ids": docs[n]["input_ids"].tolist(),
                        "loss_weights": docs[n]["loss_weights"].tolist(),
                    }
                )
                for n in cell_docs
            ]
            payload = gzip.compress(("\n".join(lines) + "\n").encode())
            name = f"{split}/docs-{cell_index:03d}.jsonl.gz"
            for root in (args.plan, args.cells_root):
                StoragePath(prefix_join(root, name)).write_bytes(payload)
            cells.append(prefix_join(args.cells_root, name))
        by_source = {}
        for d in docs:
            s = by_source.setdefault(d["source"], {"docs": 0, "tokens": 0, "loss_tokens": 0})
            s["docs"] += 1
            s["tokens"] += len(d["input_ids"])
            s["loss_tokens"] += int(d["loss_weights"].sum())
        summary[split] = {"rows": len(rows), "fill": fill, "docs": len(docs), "sources": by_source, "cells": cells}
        logger.info("%s: %s", split, json.dumps({k: v for k, v in summary[split].items() if k != "cells"}))
    StoragePath(prefix_join(args.plan, "summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
