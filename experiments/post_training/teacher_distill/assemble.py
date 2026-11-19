# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Join the packing plan with Mochi's stored top-20 and write the trainer's teacher-scored row caches.

Each cache entry is one packed row of exactly 262,144 positions with ``input_ids``, ``loss_weights``,
``teacher_ids`` and ``teacher_logprobs`` (both flattened [262144, 20]; row i is the teacher distribution for
token i + 1, zero where the position carries no loss). Packed rows whose cell was not scored are dropped whole.

    python -m experiments.post_training.teacher_distill.assemble --plan s3://.../plan \
        --teacher s3://.../teacher-top20 --output s3://.../rows
"""

import argparse
import datetime
import faulthandler
import gzip
import io
import json
import logging
import time

import numpy as np
from levanter.store.cache import SerialCacheWriter
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

logger = logging.getLogger(__name__)

POS = 262_144
TOP = 20
EOS = 128_001
ROWS_PER_CELL = 8
WRITE_BATCH = 8


def load_cell(plan: str, teacher: str, split: str, index: int):
    name = f"{split}/docs-{index:03d}"
    docs = {
        d["row"]: d
        for d in (
            json.loads(x)
            for x in gzip.decompress(StoragePath(prefix_join(plan, name + ".jsonl.gz")).read_bytes()).splitlines()
        )
    }
    # Load each array once: every NpzFile item access decompresses a fresh full copy, and a per-document
    # slice of it would keep that whole copy alive.
    with np.load(io.BytesIO(StoragePath(prefix_join(teacher, name + ".npz")).read_bytes())) as npz:
        scored = {key: npz[key] for key in npz.files}
    offsets = scored["doc_offsets"]
    for n, row in enumerate(scored["doc_rows"].tolist()):
        lo, hi = offsets[n], offsets[n + 1]
        weights = np.asarray(docs[row]["loss_weights"], np.float32)
        positions = scored["positions"][lo:hi]
        if not np.array_equal(positions, np.flatnonzero(weights)):
            raise ValueError(f"{name} doc {row}: teacher positions differ from the loss mask")
        docs[row]["teacher"] = (positions, scored["ids"][lo:hi], scored["logprobs"][lo:hi])
    return docs


def build_split(plan: str, teacher: str, output: str, name: str, splits: list[str], scored_cells: set[str]) -> dict:
    """Write the scored rows of ``splits``, in order, as one cache named ``name``."""
    exemplar = {
        "input_ids": np.zeros(0, np.int32),
        "loss_weights": np.zeros(0, np.float32),
        "teacher_ids": np.zeros(0, np.int32),
        "teacher_logprobs": np.zeros(0, np.float32),
    }
    written = loss_tokens = planned = 0
    with SerialCacheWriter(prefix_join(output, name), exemplar) as writer:
        for split in splits:
            rows = [
                json.loads(x) for x in StoragePath(prefix_join(plan, f"{split}/rows.jsonl")).read_text().splitlines()
            ]
            planned += len(rows)
            for cell_index, start in enumerate(range(0, len(rows), ROWS_PER_CELL)):
                if f"{split}/docs-{cell_index:03d}" not in scored_cells:
                    logger.info("%s cell %d not scored; dropping its rows", split, cell_index)
                    continue
                logger.info("%s cell %d: loading", split, cell_index)
                docs = load_cell(plan, teacher, split, cell_index)
                logger.info("%s cell %d: loaded %d docs", split, cell_index, len(docs))
                batch = {k: [] for k in exemplar}
                for row in rows[start : start + ROWS_PER_CELL]:
                    ids = np.full(POS, EOS, np.int32)
                    weights = np.zeros(POS, np.float32)
                    t_ids = np.zeros((POS, TOP), np.int32)
                    t_lp = np.zeros((POS, TOP), np.float32)
                    for item in row:
                        doc, offset = docs[item["doc"]], item["offset"]
                        n = len(doc["input_ids"])
                        ids[offset : offset + n] = doc["input_ids"]
                        weights[offset : offset + n] = doc["loss_weights"]
                        positions, top_ids, top_lp = doc["teacher"]
                        t_ids[offset + positions] = top_ids
                        t_lp[offset + positions] = top_lp
                    batch["input_ids"].append(ids)
                    batch["loss_weights"].append(weights)
                    batch["teacher_ids"].append(t_ids.reshape(-1))
                    batch["teacher_logprobs"].append(t_lp.reshape(-1))
                    loss_tokens += int(weights.sum())
                writer.write_batch(batch)
                written += len(batch["input_ids"])
    summary = {"rows": written, "planned_rows": planned, "loss_tokens": loss_tokens, "splits": splits}
    logger.info("%s: %s", name, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", default=["heldout", "train"], help="SPLIT or NAME=SPLIT+SPLIT")
    parser.add_argument(
        "--wait-idle-seconds", type=float, default=0, help="first wait until the scorer summary is this long unchanged"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    faulthandler.dump_traceback_later(300, repeat=True)
    logger.info("assembling %s", args.splits)
    configure_coreweave_s3()
    if args.wait_idle_seconds:
        summary_path = StoragePath(prefix_join(args.teacher, "summary.json"))
        while (
            age := (datetime.datetime.now(datetime.UTC) - summary_path.mtime()).total_seconds()
        ) < args.wait_idle_seconds:
            logger.info("scorer summary updated %.0f s ago; waiting", age)
            time.sleep(60)
    # One summary per scoring job (or shard); a cell is scored if any of them lists it.
    scored = {
        c["cell"]
        for path in StoragePath(args.teacher).ls()
        if str(path).rsplit("/", 1)[-1].startswith("summary") and str(path).endswith(".json")
        for c in json.loads(StoragePath(str(path)).read_text())["cells"]
    }
    # NAME=SPLIT[+SPLIT...] writes several plan splits as one cache; a bare SPLIT writes itself.
    targets = dict(spec.split("=", 1) if "=" in spec else (spec, spec) for spec in args.splits)
    summary = {
        name: build_split(args.plan, args.teacher, args.output, name, splits.split("+"), scored)
        for name, splits in targets.items()
    }
    for split, values in summary.items():
        StoragePath(prefix_join(args.output, f"{split}-summary.json")).write_text(json.dumps(values, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
