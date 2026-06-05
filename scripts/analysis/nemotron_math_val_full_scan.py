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


SUBSETS = {
    "4plus": Subset("gs://marin-us-east5/normalized/nemotron_cc_math_v1/4plus_b05688a8/outputs/main", "us-east5", 231),
    "3": Subset("gs://marin-us-central1/normalized/nemotron_cc_math_v1/3_f8007d22/outputs/main", "us-central1", 400),
    "4plus_mind": Subset(
        "gs://marin-us-central1/normalized/nemotron_cc_math_v1/4plus_mind_1173e12a/outputs/main", "us-central1", 336
    ),
}

_PAIRS: dict[str, list[str]] | None = None
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


def _val_buckets(attr_file: str, val_ids: frozenset[str]) -> Iterator[dict]:
    for batch in pq.ParquetFile(fsspec.open(attr_file, "rb").open()).iter_batches(columns=["id", "buckets"]):
        for doc_id, buckets in zip(batch.column("id"), batch.column("buckets"), strict=True):
            if doc_id.as_py() in val_ids:
                for bucket in buckets.as_py():
                    yield {"bucket": bucket}


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


def _load_verify_state(pairs_dir: str, val_docs: str) -> tuple[dict[str, list[str]], dict[str, str]]:
    global _PAIRS, _VAL_TEXT
    if _PAIRS is None:
        by_other = defaultdict(set)
        for path in fsspec_glob(f"{pairs_dir}/*.parquet"):
            t = pq.read_table(path, columns=["val_id", "other_id"])
            for val_id, other_id in zip(t.column("val_id").to_pylist(), t.column("other_id").to_pylist(), strict=True):
                by_other[other_id].add(val_id)
        _PAIRS = {k: sorted(v) for k, v in by_other.items()}
        val_text: dict[str, str] = {}
        for path in fsspec_glob(f"{val_docs}/*.parquet"):
            t = pq.read_table(path, columns=["id", "text"])
            val_text.update(zip(t.column("id").to_pylist(), t.column("text").to_pylist(), strict=True))
        _VAL_TEXT = val_text
    assert _PAIRS is not None and _VAL_TEXT is not None
    return _PAIRS, _VAL_TEXT


def _verify_shard(shard_path: str, pairs_dir: str, val_docs: str) -> Iterator[dict]:
    pairs_by_other, val_text = _load_verify_state(pairs_dir, val_docs)
    val_shingles: dict[str, set[str]] = {}
    for batch in pq.ParquetFile(shard_path).iter_batches(columns=["id", "text"], batch_size=2048):
        for doc_id, text in zip(batch.column("id"), batch.column("text"), strict=True):
            id_str = doc_id.as_py()
            val_ids = pairs_by_other.get(id_str)
            if not val_ids:
                continue
            counters.increment("verify/candidates")
            other = _shingles(text.as_py())
            for val_id in val_ids:
                if val_id not in val_shingles:
                    val_shingles[val_id] = _shingles(val_text[val_id])
                vs = val_shingles[val_id]
                union = len(vs | other)
                jaccard = len(vs & other) / union if union else 0.0
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

    collect_ctx = ZephyrContext(
        name=f"collect-val-buckets-{args.subset}",
        max_workers=sub.shards,
        resources=ResourceConfig(cpu=1, ram="8g", disk="5g"),
        coordinator_resources=ResourceConfig(cpu=2, ram="16g", disk="10g"),
    )
    collect_ctx.execute(
        Dataset.from_list(attr_files)
        .flat_map(lambda path, ids=val_ids: _val_buckets(path, ids))
        .write_parquet(f"{scratch}/val_buckets/buckets-{{shard:05d}}-of-{{total:05d}}.parquet")
    )
    bucket_set: set[str] = set()
    for path in fsspec_glob(f"{scratch}/val_buckets/*.parquet"):
        bucket_set.update(pq.read_table(path, columns=["bucket"]).column("bucket").to_pylist())
    val_buckets = frozenset(bucket_set)
    logger.info("collected %d val buckets", len(val_buckets))

    # 71 bands -> ~4.1M val bucket keys broadcast to every worker; 64g held at 1.5M.
    join_ctx = ZephyrContext(
        name=f"val-bucket-join-{args.subset}",
        max_workers=sub.shards,
        resources=ResourceConfig(cpu=2, ram="64g", disk="10g"),
        coordinator_resources=ResourceConfig(cpu=4, ram="32g", disk="10g"),
    )
    pairs_dir = f"{scratch}/val_candidate_pairs"
    join_ctx.execute(
        Dataset.from_list(attr_files)
        .flat_map(lambda path, ids=val_ids, vb=val_buckets: _bucket_records(path, ids, vb))
        .group_by(lambda r: r["bucket"], reducer=_emit_candidate_pairs)
        .write_parquet(f"{pairs_dir}/pairs-{{shard:05d}}-of-{{total:05d}}.parquet")
    )

    corpus_files = sorted(fsspec_glob(f"{sub.corpus}/*.parquet"))
    verify_ctx = ZephyrContext(
        name=f"verify-val-pairs-{args.subset}",
        max_workers=sub.shards,
        resources=ResourceConfig(cpu=2, ram="32g", disk="10g"),
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
