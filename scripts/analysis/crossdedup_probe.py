# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quick cross-corpus near-dup probe: a sample of subset `3` vs a sample of `4plus`.

Sizing probe for the full 3-vs-4plus decontamination. Datakit shards are
content-hash ordered, so a `3` doc's near-duplicate lands in a *random* `4plus`
shard — one-shard-vs-one-shard would miss almost everything. Instead this scans
a subsample of ONE `3` shard against the first N `4plus` shards, reusing the
existing 284x71 MinHash for the `4plus` side (no re-signing) and computing
MinHash only for the small query subsample. Output is verified
``(three_id, fourplus_id, jaccard)`` pairs to histogram, so we can pick the
Jaccard cutoff before committing to the full job.

The bucket-join + verify mirror ``nemotron_math_val_full_scan`` but the helpers
are inlined (not imported) because Zephyr ships functions defined in this
entrypoint module to workers by value; a sibling-imported function would need
its module importable on the worker, which it is not.

Launch (us-east5, in-region):

    uv run iris --controller-url=http://localhost:10000 --cluster=marin job run --no-wait \
        --cpu 4 --memory 32GB --disk 20GB --priority interactive --extra cpu \
        --enable-extra-resources --preemptible --region us-east5 \
        --job-name crossdedup-probe \
        -- python scripts/analysis/crossdedup_probe.py
"""

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from collections.abc import Iterator

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

THREE_CORPUS = "gs://marin-us-east5/normalized/nemotron_cc_math_v1/3_f8007d22/outputs/main"
FOURP_CORPUS = "gs://marin-us-east5/normalized/nemotron_cc_math_v1/4plus_b05688a8/outputs/main"
FOURP_MINHASH = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/4plus_284x71/minhash/outputs"
PROBE = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/crossdedup_probe"

_WS = re.compile(r"\s+")
QUERY_TEXT_SCHEMA = pa.schema([("id", pa.string()), ("text", pa.string())])
PAIR_SCHEMA = pa.schema([("val_id", pa.string()), ("other_id", pa.string())])
VERIFIED_PAIR_SCHEMA = pa.schema([("val_id", pa.string()), ("other_id", pa.string()), ("jaccard", pa.float64())])

_QUERY_TEXT: dict[str, str] | None = None  # per-worker text cache


def load_query_ids(query_docs: str) -> frozenset[str]:
    ids: set[str] = set()
    for path in fsspec_glob(f"{query_docs}/*.parquet"):
        with fsspec.open(path, "rb") as f:
            ids.update(pq.read_table(f, columns=["id"]).column("id").to_pylist())
    return frozenset(ids)


def bucket_records(attr_file: str, query_ids: frozenset[str], query_buckets: frozenset[str]) -> Iterator[dict]:
    """Emit (bucket, id, is_query) for docs landing in a query-touched bucket."""
    with fsspec.open(attr_file, "rb") as f:
        for batch in pq.ParquetFile(f).iter_batches(columns=["id", "buckets"]):
            for doc_id, buckets in zip(batch.column("id"), batch.column("buckets"), strict=True):
                id_str = doc_id.as_py()
                is_query = id_str in query_ids
                for bucket in buckets.as_py():
                    if bucket in query_buckets:
                        yield {"bucket": bucket, "id": id_str, "is_query": is_query}


def emit_candidate_pairs(_key: str, items: Iterator[dict]) -> Iterator[dict]:
    """Per bucket, emit (query_id, corpus_id) pairs if the bucket has >=1 query doc + >=1 other."""
    ids: set[str] = set()
    query_ids: set[str] = set()
    for item in items:
        ids.add(item["id"])
        if item["is_query"]:
            query_ids.add(item["id"])
    if not query_ids or len(ids) < 2:
        counters.increment("buckets/skipped")
        return
    counters.increment("buckets/with_query")
    for query_id in query_ids:
        for other in ids - {query_id}:
            yield {"val_id": query_id, "other_id": other}


def pair_key(record: dict) -> tuple[str, str]:
    return record["val_id"], record["other_id"]


def first_pair(_key: tuple[str, str], items: Iterator[dict]) -> Iterator[dict]:
    yield next(items)


def shingles(text: str, n: int = 5) -> set[str]:
    cleaned = _WS.sub(" ", text.lower()).strip()
    return {cleaned[i : i + n] for i in range(max(1, len(cleaned) - n + 1))}


def load_query_text(query_docs: str) -> dict[str, str]:
    global _QUERY_TEXT
    if _QUERY_TEXT is None:
        text: dict[str, str] = {}
        for path in fsspec_glob(f"{query_docs}/*.parquet"):
            with fsspec.open(path, "rb") as f:
                t = pq.read_table(f, columns=["id", "text"])
            text.update(zip(t.column("id").to_pylist(), t.column("text").to_pylist(), strict=True))
        _QUERY_TEXT = text
    return _QUERY_TEXT


def shard_pairs(pairs_dir: str, shard_ids: set[str]) -> dict[str, list[str]]:
    """Candidate pairs whose corpus id is in this shard, grouped corpus_id -> [query_id]."""
    id_array = pa.array(shard_ids, type=pa.string())
    by_other: dict[str, set[str]] = defaultdict(set)
    for path in fsspec_glob(f"{pairs_dir}/*.parquet"):
        with fsspec.open(path, "rb") as f:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=262144):
                mask = pc.is_in(batch.column("other_id"), value_set=id_array)
                hits = pa.RecordBatch.from_arrays(batch.columns, batch.schema.names).filter(mask)
                for query_id, corpus_id in zip(
                    hits.column("val_id").to_pylist(), hits.column("other_id").to_pylist(), strict=True
                ):
                    by_other[corpus_id].add(query_id)
    return {k: sorted(v) for k, v in by_other.items()}


def verify_shard(shard_path: str, pairs_dir: str, query_docs: str, min_jaccard: float) -> Iterator[dict]:
    """Exact 5-char-shingle Jaccard for this corpus shard's candidate pairs, at an explicit threshold."""
    query_text = load_query_text(query_docs)
    with fsspec.open(shard_path, "rb") as f:
        table = pq.read_table(f, columns=["id", "text"])
    texts = dict(zip(table.column("id").to_pylist(), table.column("text").to_pylist(), strict=True))
    del table
    pairs_by_other = shard_pairs(pairs_dir, set(texts))
    query_shingles: dict[str, set[str]] = {}
    for corpus_id, query_ids in pairs_by_other.items():
        corpus_text = texts[corpus_id]
        corpus_shingles: set[str] | None = None
        for query_id in query_ids:
            la, lb = len(corpus_text), len(query_text[query_id])
            if min(la, lb) / max(la, lb, 1) < min_jaccard:
                counters.increment("verify/len_pruned")
                continue
            if corpus_shingles is None:
                corpus_shingles = shingles(corpus_text)
            if query_id not in query_shingles:
                query_shingles[query_id] = shingles(query_text[query_id])
            qs = query_shingles[query_id]
            union = len(qs | corpus_shingles)
            jaccard = len(qs & corpus_shingles) / union if union else 0.0
            counters.increment("verify/exact")
            if jaccard >= min_jaccard:
                counters.increment("verify/reported")
                yield {"val_id": query_id, "other_id": corpus_id, "jaccard": round(jaccard, 4)}


def build_query(query_shard: str, subsample: int, seed: int) -> str:
    """Subsample one `3` shard into the probe query dir; return that dir."""
    qdir = f"{PROBE}/query"
    if fsspec_glob(f"{qdir}/*.parquet"):
        logger.info("reusing existing probe query at %s", qdir)
        return qdir
    with fsspec.open(query_shard, "rb") as f:
        table = pq.read_table(f, columns=["id", "text"])
    n = table.num_rows
    take = min(subsample, n)
    idx = sorted(random.Random(seed).sample(range(n), take))
    sample = table.take(idx)
    with fsspec.open(f"{qdir}/{query_shard.rsplit('/', 1)[1]}", "wb") as f:
        pq.write_table(sample.cast(QUERY_TEXT_SCHEMA), f)
    logger.info("wrote %d/%d query docs from %s", take, n, query_shard)
    return qdir


def _preemptible(cpu: int, ram: str) -> ResourceConfig:
    return ResourceConfig(cpu=cpu, ram=ram, disk="10g", preemptible=True, regions=("us-east5",))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-shard-idx", type=int, default=0, help="Which `3` shard to subsample as the query.")
    parser.add_argument("--query-subsample", type=int, default=25_000, help="Docs to sample from the query shard.")
    parser.add_argument("--corpus-shards", type=int, default=24, help="First N `4plus` shards to scan against.")
    parser.add_argument("--min-jaccard", type=float, default=0.3, help="Report+prune threshold (low, to see the tail).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    three_shards = sorted(fsspec_glob(f"{THREE_CORPUS}/*.parquet"))
    fourp_shards = sorted(fsspec_glob(f"{FOURP_CORPUS}/*.parquet"))
    qdir = build_query(three_shards[args.query_shard_idx], args.query_subsample, args.seed)
    query_ids = load_query_ids(qdir)

    qmh = compute_minhash_attrs(
        source=NormalizedData(main_output_dir=qdir, dup_output_dir="", counters={}),
        output_path=f"{PROBE}/query_minhash",
        num_perms=284,
        num_bands=71,
        worker_resources=_preemptible(2, "8g"),
        max_workers=8,
    )
    query_mh_files = sorted(fsspec_glob(f"{qmh.attr_dir}/*.parquet"))
    bucket_set: set[str] = set()
    for path in query_mh_files:
        with fsspec.open(path, "rb") as f:
            for buckets in pq.read_table(f, columns=["buckets"]).column("buckets").to_pylist():
                bucket_set.update(buckets)
    query_buckets = frozenset(bucket_set)
    logger.info("query=%d docs, %d buckets; corpus=%d shards", len(query_ids), len(query_buckets), args.corpus_shards)

    corpus_text = fourp_shards[: args.corpus_shards]
    corpus_minhash = [f"{FOURP_MINHASH}/{s.rsplit('/', 1)[1]}" for s in corpus_text]

    pairs_dir = f"{PROBE}/pairs"
    join_ctx = ZephyrContext(
        name="probe-join",
        max_workers=args.corpus_shards,
        resources=_preemptible(2, "24g"),
        coordinator_resources=_preemptible(2, "12g"),
    )
    join_ctx.execute(
        Dataset.from_list(corpus_minhash + query_mh_files)
        .flat_map(lambda path, ids=query_ids, vb=query_buckets: bucket_records(path, ids, vb))
        .group_by(lambda r: r["bucket"], reducer=emit_candidate_pairs)
        .write_parquet(
            f"{pairs_dir}/pairs-{{shard:05d}}-of-{{total:05d}}.parquet", schema=PAIR_SCHEMA, skip_existing=True
        )
    )

    dedup_dir = f"{PROBE}/dedup"
    dedup_ctx = ZephyrContext(
        name="probe-dedup",
        max_workers=args.corpus_shards,
        resources=_preemptible(2, "16g"),
        coordinator_resources=_preemptible(2, "12g"),
    )
    dedup_ctx.execute(
        Dataset.from_files(f"{pairs_dir}/*.parquet")
        .load_parquet()
        .group_by(pair_key, reducer=first_pair)
        .write_parquet(
            f"{dedup_dir}/pairs-{{shard:05d}}-of-{{total:05d}}.parquet", schema=PAIR_SCHEMA, skip_existing=True
        )
    )

    verified_dir = f"{PROBE}/verified"
    verify_ctx = ZephyrContext(
        name="probe-verify",
        max_workers=args.corpus_shards,
        resources=_preemptible(2, "24g"),
        coordinator_resources=_preemptible(2, "12g"),
    )
    outcome = verify_ctx.execute(
        Dataset.from_list(corpus_text)
        .flat_map(lambda path, pd=dedup_dir, q=qdir, mj=args.min_jaccard: verify_shard(path, pd, q, mj))
        .write_parquet(
            f"{verified_dir}/verified-{{shard:05d}}-of-{{total:05d}}.parquet",
            schema=VERIFIED_PAIR_SCHEMA,
            skip_existing=True,
        )
    )

    stats = {
        "query_shard": three_shards[args.query_shard_idx],
        "query_docs": len(query_ids),
        "query_buckets": len(query_buckets),
        "corpus_shards": args.corpus_shards,
        "min_jaccard": args.min_jaccard,
        "counters": dict(outcome.counters),
        "verified_dir": verified_dir,
    }
    with fsspec.open(f"{PROBE}/probe_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("probe stats: %s", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
