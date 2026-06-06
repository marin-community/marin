# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""High-recall scan of the math val set against any nemotron math subset.

Same pipeline as nemotron_math_val_contamination.py + verify_math_val_pairs_zephyr.py
in one driver, but parameterized over corpus subset and LSH banding. Default
banding is 284 perms / 71 bands (r=4): candidate recall ~99% at Jaccard 0.5
(the original 286/26 banding has ~1.3% recall there — fine for >=0.75, useless
at 0.5).

Stages:
1. Copy val docs in-region if needed (~150 MB, one-time per region).
2. MinHash attrs over the corpus.
3. Collect LSH buckets containing >=1 val doc.
4. Bucket join -> candidate pairs (val_id, other_id).
5. Exact 5-char-shingle Jaccard verification; keep pairs >= 0.5.

Launch (per subset, in the subset's region):

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
        --cpu 4 --memory 32GB --disk 20GB --priority interactive --extra cpu \
        --enable-extra-resources --preemptible --region <region> \
        --job-name val-scan-<subset> \
        -- python scripts/analysis/nemotron_math_val_full_scan.py --subset <subset>
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.processing.classification.deduplication.fuzzy_minhash import compute_minhash_attrs
from marin.utils import fsspec_glob
from zephyr import Dataset, ZephyrContext, counters

logger = logging.getLogger(__name__)

VAL_DOCS_EAST5 = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/val_docs"
MIN_REPORT_JACCARD = 0.5
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Subset:
    corpus: str
    region: str
    shards: int


# 3 and 4plus_mind were rsync'd from us-central1 (no CPU pool there) on 2026-06-06.
SUBSETS = {
    "4plus": Subset("gs://marin-us-east5/normalized/nemotron_cc_math_v1/4plus_b05688a8/outputs/main", "us-east5", 231),
    "3": Subset("gs://marin-us-east5/normalized/nemotron_cc_math_v1/3_f8007d22/outputs/main", "us-east5", 400),
    "4plus_mind": Subset(
        "gs://marin-us-east5/normalized/nemotron_cc_math_v1/4plus_mind_1173e12a/outputs/main", "us-east5", 336
    ),
}

_VAL_TEXT: dict[str, str] | None = None


def ensure_val_docs(region: str) -> str:
    """Copy the 231 val-doc shards into the target region once (~150 MB)."""
    if region == "us-east5":
        return VAL_DOCS_EAST5
    dest = f"gs://marin-{region}/scratch/ahmed/midtrain_dedup/val_docs"
    fs = fsspec.filesystem("gcs")
    src_files = fsspec_glob(f"{VAL_DOCS_EAST5}/*.parquet")
    if len(fsspec_glob(f"{dest}/*.parquet")) < len(src_files):
        for f in src_files:
            fs.copy(f, f"{dest}/{f.rsplit('/', 1)[1]}")
        logger.info("copied %d val doc shards to %s", len(src_files), dest)
    return dest


def load_val_ids(val_docs: str) -> frozenset[str]:
    ids: set[str] = set()
    for path in fsspec_glob(f"{val_docs}/*.parquet"):
        ids.update(pq.read_table(path, columns=["id"]).column("id").to_pylist())
    return frozenset(ids)


def _bucket_records(attr_file: str, val_ids: frozenset[str], val_buckets: frozenset[str]) -> Iterator[dict]:
    for batch in pq.ParquetFile(fsspec.open(attr_file, "rb").open()).iter_batches(columns=["id", "buckets"]):
        for doc_id, buckets in zip(batch.column("id"), batch.column("buckets"), strict=True):
            id_str = doc_id.as_py()
            is_val = id_str in val_ids
            for bucket in buckets.as_py():
                if bucket in val_buckets:
                    yield {"bucket": bucket, "id": id_str, "is_val": is_val}


def _emit_candidate_pairs(_key: str, items: Iterator[dict]) -> Iterator[dict]:
    ids: set[str] = set()
    val_ids: set[str] = set()
    for item in items:
        ids.add(item["id"])
        if item["is_val"]:
            val_ids.add(item["id"])
    if not val_ids or len(ids) < 2:
        counters.increment("valbuckets/skipped")
        return
    counters.increment("valbuckets/with_val")
    for val_id in val_ids:
        for other in ids - {val_id}:
            yield {"val_id": val_id, "other_id": other}


def _shingles(text: str, n: int = 5) -> set[str]:
    cleaned = _WS.sub(" ", text.lower()).strip()
    return {cleaned[i : i + n] for i in range(max(1, len(cleaned) - n + 1))}


def _load_val_text(val_docs: str) -> dict[str, str]:
    global _VAL_TEXT
    if _VAL_TEXT is None:
        val_text: dict[str, str] = {}
        for path in fsspec_glob(f"{val_docs}/*.parquet"):
            t = pq.read_table(path, columns=["id", "text"])
            val_text.update(zip(t.column("id").to_pylist(), t.column("text").to_pylist(), strict=True))
        _VAL_TEXT = val_text
    return _VAL_TEXT


def _shard_pairs(pairs_dir: str, shard_ids: set[str]) -> dict[str, list[str]]:
    """Stream candidate pairs, keep only this shard's other_ids, deduped.

    High-recall banding emits billions of (val_id, other_id) rows corpus-wide
    (one per shared bucket); broadcasting that dict OOMs workers. The deduped
    per-shard slice is ~1/n_shards of it.
    """
    id_array = pa.array(shard_ids, type=pa.string())
    by_other: dict[str, set[str]] = defaultdict(set)
    for path in fsspec_glob(f"{pairs_dir}/*.parquet"):
        for batch in pq.ParquetFile(fsspec.open(path, "rb").open()).iter_batches(batch_size=262144):
            mask = pc.is_in(batch.column("other_id"), value_set=id_array)
            hits = pa.RecordBatch.from_arrays(batch.columns, batch.schema.names).filter(mask)
            for val_id, other_id in zip(
                hits.column("val_id").to_pylist(), hits.column("other_id").to_pylist(), strict=True
            ):
                by_other[other_id].add(val_id)
    return {k: sorted(v) for k, v in by_other.items()}


def _verify_shard(shard_path: str, pairs_dir: str, val_docs: str) -> Iterator[dict]:
    val_text = _load_val_text(val_docs)
    table = pq.read_table(shard_path, columns=["id", "text"])
    texts = dict(zip(table.column("id").to_pylist(), table.column("text").to_pylist(), strict=True))
    del table
    pairs_by_other = _shard_pairs(pairs_dir, set(texts))
    counters.increment("verify/shard_pairs", sum(len(v) for v in pairs_by_other.values()))
    val_shingles: dict[str, set[str]] = {}
    for id_str, val_ids in pairs_by_other.items():
        text = texts[id_str]
        other: set[str] | None = None
        for val_id in val_ids:
            # Length bound: shingle Jaccard <= min(len)/max(len); prune cheap.
            la, lb = len(text), len(val_text[val_id])
            if min(la, lb) / max(la, lb, 1) < MIN_REPORT_JACCARD:
                counters.increment("verify/len_pruned")
                continue
            if other is None:
                other = _shingles(text)
            if val_id not in val_shingles:
                val_shingles[val_id] = _shingles(val_text[val_id])
            vs = val_shingles[val_id]
            union = len(vs | other)
            jaccard = len(vs & other) / union if union else 0.0
            counters.increment("verify/exact")
            if jaccard >= MIN_REPORT_JACCARD:
                counters.increment("verify/reported")
                yield {"val_id": val_id, "other_id": id_str, "jaccard": round(jaccard, 4)}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=sorted(SUBSETS), required=True)
    parser.add_argument("--num-perms", type=int, default=284)
    parser.add_argument("--num-bands", type=int, default=71)
    args = parser.parse_args()

    sub = SUBSETS[args.subset]
    scratch = f"gs://marin-{sub.region}/scratch/ahmed/midtrain_dedup/{args.subset}_{args.num_perms}x{args.num_bands}"
    val_docs = ensure_val_docs(sub.region)
    val_ids = load_val_ids(val_docs)
    logger.info("subset=%s shards=%d val_ids=%d scratch=%s", args.subset, sub.shards, len(val_ids), scratch)

    # Val buckets come from the val docs' own minhash, not the corpus minhash:
    # for subsets other than 4plus, val ids never appear in the corpus.
    region_scratch = f"gs://marin-{sub.region}/scratch/ahmed/midtrain_dedup"
    val_source = NormalizedData(main_output_dir=val_docs, dup_output_dir="", counters={})
    val_minhash = compute_minhash_attrs(
        source=val_source,
        output_path=f"{region_scratch}/val_docs_minhash_{args.num_perms}x{args.num_bands}",
        num_perms=args.num_perms,
        num_bands=args.num_bands,
        worker_resources=ResourceConfig(cpu=2, ram="8g", disk="5g"),
        max_workers=64,
    )
    bucket_set: set[str] = set()
    for path in fsspec_glob(f"{val_minhash.attr_dir}/*.parquet"):
        for buckets in pq.read_table(path, columns=["buckets"]).column("buckets").to_pylist():
            bucket_set.update(buckets)
    val_buckets = frozenset(bucket_set)
    logger.info("collected %d val buckets", len(val_buckets))

    source = NormalizedData(main_output_dir=sub.corpus, dup_output_dir="", counters={})
    minhash = compute_minhash_attrs(
        source=source,
        output_path=f"{scratch}/minhash",
        num_perms=args.num_perms,
        num_bands=args.num_bands,
        worker_resources=ResourceConfig(cpu=4, ram="24g", disk="5g"),
        max_workers=sub.shards,
    )
    attr_files = sorted(fsspec_glob(f"{minhash.attr_dir}/*.parquet"))

    # 71 bands -> ~4.1M val bucket keys broadcast to every worker; 64g held at 1.5M.
    join_ctx = ZephyrContext(
        name=f"val-bucket-join-{args.subset}",
        max_workers=sub.shards,
        resources=ResourceConfig(cpu=2, ram="64g", disk="10g"),
        coordinator_resources=ResourceConfig(cpu=4, ram="32g", disk="10g"),
    )
    # Include the val docs' minhash rows so val ids appear in every bucket
    # group (corpus shards alone lack them outside 4plus). _emit_candidate_pairs
    # dedups via sets, so the redundant val rows for 4plus are harmless.
    join_files = attr_files + sorted(fsspec_glob(f"{val_minhash.attr_dir}/*.parquet"))
    pairs_dir = f"{scratch}/val_candidate_pairs"
    join_ctx.execute(
        Dataset.from_list(join_files)
        .flat_map(lambda path, ids=val_ids, vb=val_buckets: _bucket_records(path, ids, vb))
        .group_by(lambda r: r["bucket"], reducer=_emit_candidate_pairs)
        .write_parquet(f"{pairs_dir}/pairs-{{shard:05d}}-of-{{total:05d}}.parquet")
    )

    # Dedup the (val_id, other_id) pairs once (3-4B rows -> ~10M); verify
    # workers then read ~200 MB instead of re-streaming the raw pair list.
    dedup_dir = f"{scratch}/val_pairs_dedup"
    dedup_ctx = ZephyrContext(
        name=f"dedup-pairs-{args.subset}",
        max_workers=sub.shards,
        resources=ResourceConfig(cpu=2, ram="32g", disk="10g"),
        coordinator_resources=ResourceConfig(cpu=4, ram="32g", disk="10g"),
    )
    dedup_ctx.execute(
        Dataset.from_files(f"{pairs_dir}/*.parquet")
        .load_parquet()
        .group_by(lambda r: f"{r['val_id']}|{r['other_id']}", reducer=lambda _k, items: [next(iter(items))])
        .write_parquet(f"{dedup_dir}/pairs-{{shard:05d}}-of-{{total:05d}}.parquet")
    )
    pairs_dir = dedup_dir

    corpus_files = sorted(fsspec_glob(f"{sub.corpus}/*.parquet"))
    verify_ctx = ZephyrContext(
        name=f"verify-val-pairs-{args.subset}",
        max_workers=sub.shards,
        resources=ResourceConfig(cpu=2, ram="48g", disk="10g"),
        coordinator_resources=ResourceConfig(cpu=4, ram="32g", disk="10g"),
    )
    outcome = verify_ctx.execute(
        Dataset.from_list(corpus_files)
        .flat_map(lambda path, p=pairs_dir, v=val_docs: _verify_shard(path, p, v))
        .write_parquet(f"{scratch}/verified_pairs/verified-{{shard:05d}}-of-{{total:05d}}.parquet")
    )

    stats = {
        "subset": args.subset,
        "num_perms": args.num_perms,
        "num_bands": args.num_bands,
        "num_val_ids": len(val_ids),
        "num_val_buckets": len(val_buckets),
        "counters": dict(outcome.counters),
        "verified_out": f"{scratch}/verified_pairs",
    }
    with fsspec.open(f"{scratch}/scan_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("stats: %s", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
