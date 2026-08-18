# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample the duplicate decisions a verification run actually recorded.

Reads the markers a run wrote rather than recomputing the rule, so every pair
is a document the store would delete beside the document kept in its place.
Writes JSON that :mod:`experiments.datakit.reports` renders as a spot-check page.

Both documents of a pair sit in the same grouped text file, because a cluster is
solved as a unit and the grouping put the whole cluster in one file. That is
what makes sampling cheap: pick a grouped file, read markers for a sample of the
normalized shards its rows came from, and join on id.

Runs on the cluster that owns the data. A grouped partition reaches 8 GiB and
one of its row groups does not fit in a laptop, so the join streams in batches
and the job asks for real memory.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \
        --no-wait --priority interactive --cpu 8 --memory 128GB \
        -- python experiments/datakit/scripts/sample_verified_duplicates.py \
            --cluster-text s3://.../user/rav/dedup/cluster_text/v11 \
            --markers s3://.../user/rav/dedup/verified/v11-c060 \
            --out s3://.../user/rav/dedup/reports/c060.json
"""

import argparse
import collections
import concurrent.futures
import json
import logging
import random

import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

logger = logging.getLogger(__name__)

TEXT_COLUMNS = ["dup_cluster_id", "id", "text", "file_idx", "source_tag"]
MARKER_COLUMNS = ["id", "dup_representative_id", "dup_containment", "dup_jaccard", "dup_novel_tokens"]
TEXT_CAP = 6000
# Documents run to 64 MB, so a row group can be gigabytes. Stream the join.
TEXT_BATCH_ROWS = 128
MARKER_READ_CONCURRENCY = 32

_TYPE_RULES = (
    ("code", ("code", "stack-v3", "starcoder", "coderforge", "opencoder", "github")),
    ("math", ("math", "openwebmath", "proof")),
    ("sft_rollout", ("sft_", "pretraining_sft", "rollout", "agent", "tulu", "smoltalk")),
    ("synthetic_qa", ("diverse_qa", "synthetic", "wiki_rewrite", "rewrite")),
    ("reference", ("wikiteam", "stackexchange", "wikipedia", "arxiv", "pes2o", "books", "gutenberg")),
)


def content_type(source_key: str) -> str:
    """Coarse type for a normalized source key; first match wins."""
    lowered = source_key.lower()
    for name, needles in _TYPE_RULES:
        if any(needle in lowered for needle in needles):
            return name
    return "web_prose"


def band(containment: float) -> str:
    if containment >= 0.999:
        return "1.00"
    if containment >= 0.90:
        return "0.90-1.00"
    if containment >= 0.75:
        return "0.75-0.90"
    return "under 0.75"


def read_markers(markers_root: str, shards: dict, file_indices: list[int]) -> list[dict]:
    """Every marker the run wrote for the given normalized shards."""

    def one(file_idx: int) -> list[dict]:
        shard = shards.get(file_idx)
        if shard is None:
            return []
        path = prefix_join(markers_root, f"outputs/{shard['source_tag']}/{shard['basename']}")
        try:
            with StoragePath(path).open("rb") as handle:
                parquet = pq.ParquetFile(handle)
                if parquet.metadata.num_rows == 0:
                    return []
                return parquet.read(columns=MARKER_COLUMNS).to_pylist()
        except FileNotFoundError:
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MARKER_READ_CONCURRENCY) as pool:
        return [row for batch in pool.map(one, file_indices) for row in batch]


def harvest_file(text_path: str, shards: dict, markers_root: str, marker_shards: int, rng: random.Random) -> list[dict]:
    """Join one grouped text file to the markers for a sample of its shards."""
    with StoragePath(text_path).open("rb") as handle:
        routing = pq.ParquetFile(handle).read(columns=["file_idx"]).column("file_idx").to_pylist()

    # Weight the shard draw by how many rows each put in this file, so the reads
    # land where the join can hit. Taking the busiest outright would report one
    # giant source over and over.
    counts = collections.Counter(routing)
    ranked = sorted(counts, key=lambda idx: rng.random() ** (1.0 / counts[idx]), reverse=True)
    del routing

    marker_rows = read_markers(markers_root, shards, ranked[:marker_shards])
    if not marker_rows:
        return []

    needed = {row["id"] for row in marker_rows} | {row["dup_representative_id"] for row in marker_rows}
    by_id: dict[str, dict] = {}
    with StoragePath(text_path).open("rb") as handle:
        for batch in pq.ParquetFile(handle).iter_batches(batch_size=TEXT_BATCH_ROWS, columns=TEXT_COLUMNS):
            ids = batch.column("id").to_pylist()
            keep = [position for position, doc_id in enumerate(ids) if doc_id in needed]
            for row in batch.take(keep).to_pylist() if keep else []:
                row["chars"] = len(row["text"])
                row["text"] = row["text"][:TEXT_CAP]
                by_id[row["id"]] = row

    pairs = []
    for mark in marker_rows:
        member, representative = by_id.get(mark["id"]), by_id.get(mark["dup_representative_id"])
        if member is None or representative is None:
            continue
        containment = float(mark["dup_containment"])
        pairs.append(
            {
                "containment": round(containment, 4),
                "jaccard": round(float(mark["dup_jaccard"]), 4),
                "novel_tokens": int(mark["dup_novel_tokens"]),
                "band": band(containment),
                "cluster_id": member["dup_cluster_id"],
                "deleted_id": member["id"],
                "kept_id": representative["id"],
                "deleted_source": member["source_tag"],
                "kept_source": representative["source_tag"],
                "cross_source": member["source_tag"] != representative["source_tag"],
                "deleted_chars": member["chars"],
                "kept_chars": representative["chars"],
                "deleted_text": member["text"],
                "kept_text": representative["text"],
            }
        )
    rng.shuffle(pairs)
    return pairs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-text", required=True, help="Root holding manifest.json beside text/")
    parser.add_argument("--markers", required=True, help="Root of the verified marker tree")
    parser.add_argument("--out", required=True, help="Where to write the sampled pairs as JSON")
    parser.add_argument("--files", type=int, default=6, help="Grouped text files to read")
    parser.add_argument("--marker-shards", type=int, default=400, help="Normalized shards sampled per file")
    parser.add_argument("--per-bucket", type=int, default=5, help="Pairs kept per content type and band")
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configure_coreweave_s3()

    manifest = json.loads(StoragePath(prefix_join(args.cluster_text, "manifest.json")).read_bytes())
    shards = {shard["file_idx"]: shard for shard in manifest["shards"]}
    tag_to_key = {shard["source_tag"]: shard["source_key"] for shard in manifest["shards"]}

    rng = random.Random(args.seed)
    names = sorted(str(p).rsplit("/", 1)[-1] for p in StoragePath(f"{args.cluster_text}/text/*.parquet").glob())
    chosen = rng.sample(names, min(args.files, len(names)))
    logger.info("Sampling %d of %d grouped text files against %s", len(chosen), len(names), args.markers)

    buckets: dict[tuple, list] = collections.defaultdict(list)
    seen = 0
    for position, name in enumerate(chosen, start=1):
        path = prefix_join(f"{args.cluster_text}/text", name)
        for pair in harvest_file(path, shards, args.markers, args.marker_shards, rng):
            seen += 1
            pair["deleted_source_name"] = tag_to_key.get(pair["deleted_source"], pair["deleted_source"])
            pair["kept_source_name"] = tag_to_key.get(pair["kept_source"], pair["kept_source"])
            pair["kind"] = content_type(pair["deleted_source_name"])
            slot = buckets[(pair["kind"], pair["band"])]
            if len(slot) < args.per_bucket and sum(1 for p in slot if p["deleted_source"] == pair["deleted_source"]) < 2:
                slot.append(pair)
        kept = sum(len(v) for v in buckets.values())
        logger.info("[%d/%d] %s: %d removals seen, %d kept", position, len(chosen), name, seen, kept)

    payload = {
        "cluster_text": args.cluster_text,
        "markers": args.markers,
        "pairs_seen": seen,
        "text_cap": TEXT_CAP,
        "samples": [pair for slot in buckets.values() for pair in slot],
    }
    StoragePath(args.out).write_bytes(json.dumps(payload, ensure_ascii=False, indent=1).encode())
    logger.info("Wrote %d samples from %d removals to %s", len(payload["samples"]), seen, args.out)


if __name__ == "__main__":
    main()
