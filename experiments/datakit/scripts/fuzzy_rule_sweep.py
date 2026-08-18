# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure what each candidate duplicate rule would remove, per content type.

Reads the grouped cluster-text dataset and, for every cluster, runs the same
greedy longest-first cover under several thresholds at once. One pass over the
data prices the whole threshold curve instead of one point, which is what the
precision/recall trade-off needs.

Containment is measured on word n-grams in the direction a removal would go:
the shorter document must be held by the longer one. A pair that shares no
n-gram at all is a connected-components artifact, and counting those separately
shows how much of a cluster is spurious rather than duplicated.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \
        --no-wait --priority interactive --cpu 4 --memory 16GB \
        -- python experiments/datakit/scripts/fuzzy_rule_sweep.py \
            --text s3://.../user/rav/dedup/cluster_text/v3/text \
            --verified datakit/verify_fuzzy_dups_c757e4f0 \
            --out s3://.../user/rav/dedup/rule_sweep/v1
"""

import argparse
import json
import logging
from collections import Counter
from collections.abc import Iterator
from typing import Any

import dupekit
import numpy as np
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

logger = logging.getLogger(__name__)

COUNTER_PREFIX = "fuzzy/rule_sweep"
NGRAM_SIZE = 3
THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 1.00)
EXHAUSTIVE_MAXIMUM = 512
COLUMNS = ["cluster_key", "id", "text", "source_tag"]

_TYPE_RULES = (
    ("code", ("code", "stack-v3", "starcoder", "coderforge", "opencoder", "github")),
    ("math", ("math", "openwebmath", "proof")),
    ("sft_rollout", ("sft_", "pretraining_sft", "rollout", "agent", "tulu", "smoltalk")),
    ("synthetic_qa", ("diverse_qa", "synthetic", "wiki_rewrite", "rewrite")),
    ("reference", ("wikiteam", "stackexchange", "wikipedia", "arxiv", "pes2o", "books", "gutenberg")),
)


def content_type(source_key: str) -> str:
    lowered = source_key.lower()
    for name, needles in _TYPE_RULES:
        if any(needle in lowered for needle in needles):
            return name
    return "web_prose"


def ngram_hashes(text: str, size: int = NGRAM_SIZE) -> np.ndarray:
    tokens = text.casefold().split()
    if not tokens:
        return np.empty(0, dtype=np.uint64)
    if len(tokens) < size:
        joined = [" ".join(tokens).encode("utf-8", "ignore")]
        return np.unique(np.array(dupekit.hash_xxh3_64_batch(joined), dtype=np.uint64))
    grams = [" ".join(tokens[i : i + size]).encode("utf-8", "ignore") for i in range(len(tokens) - size + 1)]
    return np.unique(np.array(dupekit.hash_xxh3_64_batch(grams), dtype=np.uint64))


def solve_cluster(members: list[dict[str, Any]]) -> dict[str, int]:
    """Greedy longest-first cover under every threshold at once.

    Returns removal counts keyed by threshold, plus the artifact counters. The
    survivor set is rebuilt per threshold because a lower threshold removes
    more members and so leaves fewer representatives standing.
    """
    order = sorted(range(len(members)), key=lambda i: -len(members[i]["text"]))
    grams = [ngram_hashes(members[i]["text"]) for i in order]
    sizes = [g.size for g in grams]

    # Containment of the shorter document in the longer one, for every ordered
    # pair. Computed once and reused by all thresholds.
    contained: dict[tuple[int, int], float] = {}
    zero_overlap = 0
    total_pairs = 0
    for j in range(len(order)):
        for i in range(j + 1, len(order)):
            # order is longest-first, so j is the longer document.
            total_pairs += 1
            if sizes[i] == 0 or sizes[j] == 0:
                continue
            shared = int(np.intersect1d(grams[i], grams[j], assume_unique=True).size)
            if shared == 0:
                zero_overlap += 1
                continue
            contained[(i, j)] = shared / sizes[i]

    result = {"members": len(members), "pairs": total_pairs, "zero_overlap_pairs": zero_overlap}
    for threshold in THRESHOLDS:
        removed = set()
        for j in range(len(order)):
            if j in removed:
                continue
            for i in range(j + 1, len(order)):
                if i in removed:
                    continue
                if contained.get((i, j), 0.0) >= threshold:
                    removed.add(i)
        result[f"removed_{threshold:.2f}"] = len(removed)
    return result


def sweep_file(spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Stream one grouped text file and price every threshold on its clusters."""
    tally: Counter = Counter()
    current: str | None = None
    members: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal members
        if len(members) >= 2 and len(members) <= EXHAUSTIVE_MAXIMUM:
            kind = content_type(spec["tag_to_key"].get(members[0]["source_tag"], members[0]["source_tag"]))
            outcome = solve_cluster(members)
            for key, value in outcome.items():
                tally[f"{kind}/{key}"] += value
            tally[f"{kind}/clusters"] += 1
        elif len(members) > EXHAUSTIVE_MAXIMUM:
            tally["skipped_large_clusters"] += 1
            tally["skipped_large_members"] += len(members)
        members = []

    with StoragePath(spec["path"]).open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group, columns=COLUMNS)
            for row in table.to_pylist():
                if row["cluster_key"] != current:
                    flush()
                    current = row["cluster_key"]
                members.append(row)
            del table
        flush()

    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/files", 1)
    yield {"path": spec["path"], "tally": dict(tally)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Grouped cluster-text directory")
    parser.add_argument("--verified", required=True, help="Verified artifact that pins source order")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--files", type=int, default=200, help="Sample this many part files")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--worker-cpu", type=float, default=16)
    parser.add_argument("--worker-ram", default="48g")
    parser.add_argument("--task-cpu", type=float, default=1)
    parser.add_argument("--task-ram", default="6g")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    artifact_path = prefix_join(prefix_join(args.prefix, args.verified), ".artifact.json")
    sources = json.loads(StoragePath(artifact_path).read_bytes())["result"]["sources"]
    tag_to_key = {entry["source_tag"]: key for key, entry in sources.items()}

    fs, directory = url_to_fs(args.text)
    names = sorted(str(n).rsplit("/", 1)[-1] for n in fs.ls(directory, detail=False) if str(n).endswith(".parquet"))
    rng = np.random.default_rng(args.seed)
    if args.files and args.files < len(names):
        names = [names[i] for i in sorted(rng.choice(len(names), size=args.files, replace=False))]
    logger.info("Sweeping %d files from %s", len(names), args.text)

    specs = [{"path": prefix_join(args.text, name), "tag_to_key": tag_to_key} for name in names]
    worker = ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram, disk="64g")
    task = ResourceConfig(cpu=args.task_cpu, ram=args.task_ram, disk="16g")
    context = ZephyrContext(name="fuzzy-rule-sweep", resources=worker, max_workers=args.max_workers)
    outcome = context.execute(Dataset.from_list(specs).flat_map(sweep_file), verbose=True, map_task_resources=task)

    total: Counter = Counter()
    for result in outcome.results:
        if isinstance(result, dict):
            total.update(result["tally"])
    payload = {"text": args.text, "files": len(names), "thresholds": list(THRESHOLDS), "tally": dict(total)}
    StoragePath(prefix_join(args.out, "sweep.json")).write_bytes(json.dumps(payload, indent=1).encode())
    logger.info("Wrote %s", prefix_join(args.out, "sweep.json"))


if __name__ == "__main__":
    main()
