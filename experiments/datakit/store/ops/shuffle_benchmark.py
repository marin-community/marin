# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded benchmark of a SHUFFLE-based datakit clustered store build.

Issue: marin-community/marin#6687 -- the current map-side store produces one
tiny Levanter leaf cache per (input_shard, cluster, quality), ~14.2M leaves /
~70M objects for the v0.1 store. The alternative is to ``group_by`` the
(cluster, quality) key so a single reducer writes one big cache per bucket --
no tiny leaves ever. The open question is how expensive / how long that shuffle
is, and whether bucket skew (top bucket ~651B tokens) wrecks the reduce stage.

This driver measures exactly that on a STRIDED ~1/STRIDE sample of the real
input shards, doing the real 5-way join (so the routed keys carry real
cluster/quality skew). The reducer only COUNTS tokens/rows per output key --
the reduce-side cache write is identical work in the shuffle and the
map+compaction designs, so leaving it out isolates the shuffle cost (scatter
write + gather read) and the realized skew.

Output keys are ``(cluster, quality, sub)`` where ``sub`` round-robins each map
shard's docs across ``--subshards`` reducers, so a hot (cluster, quality)
bucket is split into ``subshards`` parallel reducers instead of one. The
per-key token counts returned to the driver give the realized max-reducer load,
from which we size the full-scale shuffle.

Scatter intermediate is written under ``<output>/_scatter`` (a path we control)
because Zephyr deletes ``chunk_storage_prefix/<execution_id>`` in a ``finally``;
poll ``gsutil du`` on that prefix DURING the run to capture the scatter
footprint before cleanup.

Run on iris (eu-west4 pinned by the worker's ``MARIN_PREFIX``)::

    uv run iris --cluster=marin job run --region europe-west4 --extra=cpu \\
        --priority best_effort --cpu 2 --memory 8GB --enable-extra-resources \\
        --no-preemptible \\
        -- python experiments/datakit/store/ops/shuffle_benchmark.py \\
           --stride 50 --subshards 16 --max-workers 512
"""

import argparse
import bisect
import json
import logging
import random
import time
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from fray import ResourceConfig
from marin.datakit.decon import DeconAttributes
from marin.datakit.sources import all_sources
from marin.execution.artifact import Artifact
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.tokenize.attributes import TokenizedAttrData
from rigging.filesystem import url_to_fs
from rigging.log_setup import configure_logging
from zephyr import Dataset, ZephyrContext, counters

from experiments.datakit.cluster.domain.v0.assign import AssignmentAttrData
from experiments.datakit.cluster.quality.v0.all_sources_quality_llm import LlmQualityOutput
from experiments.datakit.store.all_sources_store import (
    CLUSTER_ASSIGN_ROOT,
    DECONTAM_ROOT,
    DEDUP_PATH,
    QUALITY_ROOT,
    SPLIT,
    TOKENIZE_ROOT,
    _build_resolution_index,
)
from experiments.datakit.store.datakit_store import (
    _TOKENIZE_BATCH_SIZE,
    ClusteredStoreData,
    _per_source_shard_tuples,
    _resolve_dedup_attr_dir,
)

logger = logging.getLogger(__name__)

OUTPUT_ROOT = "gs://marin-eu-west4/datakit/store/_shuffle_bench"


def _emit_shard(
    items: Iterator[dict[str, str]],
    shard_info,
    *,
    bucket_keys: list[tuple[int, int]],
    cum_weights: list[float],
    subshards: int,
) -> Iterator[dict[str, object]]:
    """Read one tokenize shard's real ``input_ids`` and emit one shuffle record per doc.

    The 5-way join is intentionally dropped: today's decon/cluster/quality
    inputs have drifted from what the v0.1 store was built against (quality
    re-emitted under ``part-`` names with an incompatible schema), so the real
    join no longer runs against them. The join is map-side CPU feeding the
    shuffle, not the shuffle itself -- so to measure the shuffle faithfully we
    keep the REAL token payload (real doc-size distribution, real volume) and
    assign each doc a ``(cluster, quality)`` bucket sampled from the REAL
    per-bucket token-mass distribution taken from the v0.1 artifact. That
    reproduces the true skew (top bucket ~651B tokens) the reducers must absorb.

    ``sub`` round-robins each shard's docs across ``subshards`` reducers so a
    hot bucket is split ``subshards`` ways instead of landing on one reducer.
    """
    spec = next(iter(items))
    rng = random.Random(shard_info.shard_idx)

    n_out = 0
    n_tokens = 0
    sub_counter = 0
    fs, resolved = url_to_fs(spec["tokenize"])
    with fs.open(resolved, "rb") as fh:
        pf = pq.ParquetFile(fh)
        for batch in pf.iter_batches(batch_size=_TOKENIZE_BATCH_SIZE, columns=["input_ids"]):
            tok_input_ids = batch.column("input_ids")
            for i in range(len(tok_input_ids)):
                ids = tok_input_ids[i].values.to_numpy()
                c, q = bucket_keys[bisect.bisect_right(cum_weights, rng.random())]
                yield {
                    "c": c,
                    "q": q,
                    "sub": sub_counter % subshards,
                    # numpy int32: zephyr scatter pickles payloads, so this is
                    # ~7x more compact than a boxed python int list, both in the
                    # map's pending buffer and on the wire (wire-realistic).
                    "input_ids": ids,
                }
                sub_counter += 1
                n_out += 1
                n_tokens += len(ids)

    counters.increment("bench/records_out", n_out)
    counters.increment("bench/tokens_out", n_tokens)
    # uncompressed int32 token bytes that must cross the shuffle
    counters.increment("bench/token_bytes_out", n_tokens * 4)


def _count_reducer(key: tuple[int, int, int], items: Iterator[dict[str, object]]) -> dict[str, int]:
    """Reducer: count rows + tokens for one (cluster, quality, sub) output key.

    Reading every item's ``input_ids`` forces the full gather (scatter read),
    so the reduce stage exercises the shuffle's read side even though it never
    writes a cache.
    """
    c, q, sub = key
    rows = 0
    tokens = 0
    for rec in items:
        rows += 1
        tokens += len(rec["input_ids"])
    counters.increment("bench/reduce_rows", rows)
    return {"c": c, "q": q, "sub": sub, "rows": rows, "tokens": tokens}


def _resolve_sources() -> tuple[dict, dict, dict, dict, FuzzyDupsAttrData]:
    """Resolve all datakit sources to typed artifacts (mirrors all_sources_store.main)."""
    dedup = Artifact.from_path(DEDUP_PATH, FuzzyDupsAttrData)
    source_names = list(all_sources())

    logger.info("indexing 4 roots via shallow fs.ls")
    tok_idx = _build_resolution_index(TOKENIZE_ROOT)
    dec_idx = _build_resolution_index(DECONTAM_ROOT)
    clu_idx = _build_resolution_index(CLUSTER_ASSIGN_ROOT)
    qual_idx = _build_resolution_index(QUALITY_ROOT)

    # Keep only sources materialized in all four attribute roots. Sources
    # missing from any (e.g. ``climblab-ja`` has no quality output, ``safety_pt/*``
    # is missing from dedup) are exactly the ones the real store omits.
    resolvable = [n for n in source_names if n in tok_idx and n in dec_idx and n in clu_idx and n in qual_idx]
    skipped = sorted(set(source_names) - set(resolvable))
    if skipped:
        logger.info("skipping %d sources missing from an attribute root: %s", len(skipped), skipped)

    def _resolve(name):
        # Tolerate sources with an incomplete attribute artifact (e.g. a decon
        # dir with only .executor_status). The benchmark wants a representative
        # sample, not every last source; skip failures with a warning.
        try:
            return (
                name,
                Artifact.from_path(tok_idx[name], TokenizedAttrData),
                Artifact.from_path(dec_idx[name], DeconAttributes),
                Artifact.from_path(clu_idx[name], AssignmentAttrData),
                Artifact.from_path(qual_idx[name], LlmQualityOutput),
            )
        except FileNotFoundError as e:
            logger.warning("skipping %s: %s", name, e)
            return None

    tokenize, decontam, cluster_assign, quality = {}, {}, {}, {}
    logger.info("loading typed artifacts for %d resolvable sources", len(resolvable))
    with ThreadPoolExecutor(max_workers=16) as pool:
        for result in pool.map(_resolve, resolvable):
            if result is None:
                continue
            name, tok, dec, clu, qual = result
            tokenize[name] = tok
            decontam[name] = dec
            cluster_assign[name] = clu
            quality[name] = qual
    logger.info("resolved %d sources", len(tokenize))
    return tokenize, decontam, cluster_assign, quality, dedup


def _build_shard_specs(tokenize, decontam, cluster_assign, quality, dedup) -> list[dict[str, str]]:
    """Flat per-source-shard spec list (mirrors build_clustered_store's resolution)."""
    specs: list[dict[str, str]] = []
    for name in sorted(tokenize):
        tok = tokenize[name]
        main_dir = tok.source_main_dirs.get(SPLIT)
        if main_dir is None:
            raise ValueError(f"{name}: tokenize has no source_main_dir for split={SPLIT!r}")
        try:
            dedup_attr_dir = _resolve_dedup_attr_dir(source_name=name, main_output_dir=main_dir, dedup=dedup)
            shard_tuples = _per_source_shard_tuples(
                source_name=name,
                tokenize=tok,
                decontam=decontam[name],
                cluster_assign=cluster_assign[name],
                quality=quality[name],
                dedup_attr_dir=dedup_attr_dir,
                split=SPLIT,
            )
        except KeyError:
            # Sources absent from dedup (e.g. safety_pt/*) are omitted from the real store.
            logger.info("skipping %s: no dedup entry", name)
            continue
        except FileNotFoundError as e:
            logger.warning("skipping %s: %s", name, e)
            continue
        specs.extend(shard_tuples)
    return specs


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=50, help="Keep every Nth shard spec (~1/N sample).")
    parser.add_argument("--subshards", type=int, default=16, help="Split each (cluster,quality) bucket K ways.")
    parser.add_argument("--max-workers", type=int, default=512)
    parser.add_argument("--num-output-shards", type=int, default=256)
    parser.add_argument("--output", default=None, help="Output root; default OUTPUT_ROOT/<stride>x.")
    parser.add_argument(
        "--store-artifact",
        default="gs://marin-eu-west4/datakit/store/v0.1_20260518",
        help="v0.1 store artifact whose per-bucket token-mass defines the real skew distribution.",
    )
    args = parser.parse_args()

    output = args.output or f"{OUTPUT_ROOT}/stride{args.stride}_sub{args.subshards}"
    scatter_prefix = f"{output}/_scatter"

    # Real skew: weight each (cluster, quality) bucket by its token mass in the
    # built v0.1 store, so sampled keys reproduce the true reduce-side load.
    store = Artifact.from_path(args.store_artifact, ClusteredStoreData)
    buckets = [b for b in store.buckets if b.total_tokens > 0]
    bucket_keys = [(b.cluster_id, b.quality_bucket) for b in buckets]
    total_tok = sum(b.total_tokens for b in buckets)
    cum = 0.0
    cum_weights: list[float] = []
    for b in buckets:
        cum += b.total_tokens / total_tok
        cum_weights.append(cum)
    cum_weights[-1] = 1.0  # guard fp drift so bisect never overflows
    logger.info("loaded skew over %d buckets (%.3e tokens) from %s", len(buckets), total_tok, args.store_artifact)

    tokenize, decontam, cluster_assign, quality, dedup = _resolve_sources()

    all_specs = _build_shard_specs(tokenize, decontam, cluster_assign, quality, dedup)
    sub_specs = all_specs[:: args.stride]
    logger.info(
        "BENCH: %d total shards -> %d sampled (stride=%d), subshards=%d, scatter_prefix=%s",
        len(all_specs),
        len(sub_specs),
        args.stride,
        args.subshards,
        scatter_prefix,
    )

    ctx = ZephyrContext(
        resources=ResourceConfig(cpu=2, ram="16g", disk="10g"),
        coordinator_resources=ResourceConfig(cpu=1, ram="3g", preemptible=False),
        max_workers=args.max_workers,
        chunk_storage_prefix=scatter_prefix,
        name="datakit-shuffle-bench",
    )

    ds = (
        Dataset.from_list(sub_specs)
        .map_shard(
            lambda items, shard, bk=bucket_keys, cw=cum_weights, k=args.subshards: _emit_shard(
                items, shard, bucket_keys=bk, cum_weights=cw, subshards=k
            )
        )
        .group_by(
            key=lambda r: (r["c"], r["q"], r["sub"]),
            reducer=_count_reducer,
            num_output_shards=args.num_output_shards,
        )
    )

    t0 = time.monotonic()
    outcome = ctx.execute(ds, verbose=True)
    wall = time.monotonic() - t0

    per_key = outcome.results
    by_bucket: dict[tuple[int, int], int] = defaultdict(int)
    max_reducer_tokens = 0
    for r in per_key:
        by_bucket[(r["c"], r["q"])] += r["tokens"]
        max_reducer_tokens = max(max_reducer_tokens, r["tokens"])

    bucket_tokens = sorted(by_bucket.values(), reverse=True)
    metrics = {
        "stride": args.stride,
        "subshards": args.subshards,
        "num_output_shards": args.num_output_shards,
        "max_workers": args.max_workers,
        "n_total_shards": len(all_specs),
        "n_sampled_shards": len(sub_specs),
        "wall_seconds": round(wall, 1),
        "counters": dict(outcome.counters),
        "n_output_keys": len(per_key),
        "n_nonempty_buckets": len(by_bucket),
        "max_reducer_tokens": max_reducer_tokens,
        "top5_bucket_tokens_sampled": bucket_tokens[:5],
        "scatter_prefix": scatter_prefix,
        "output": output,
    }
    logger.info("BENCH_METRICS_JSON %s", json.dumps(metrics))
    print("BENCH_METRICS_JSON", json.dumps(metrics))


if __name__ == "__main__":
    main()
