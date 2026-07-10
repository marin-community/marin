# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-source spot check of a trained LLM-quality classifier's inference.

For each active Datakit source, samples one random shard from
``inference/<name>/quality-llm/<source>_<hash>/`` and aggregates the
classifier's ``attributes.score`` (= ``P(label == '1')``):

  * Mean, median, std, fraction of docs above ``--threshold``
  * Top-K and bottom-K docs by P(high), joined back to the normalized
    parquet so we can eyeball the text (~200 char preview)

The point is qualitative -- sources we expect to be high quality (e.g.
``cp/peS2o``, ``cp/arxiv_papers``, ``starcoder2/documentation``) should
score above sources we expect to be low quality (e.g. ``cp/foodista``,
short-form synthetic Q&A). Surprises in this ranking are calibration
bugs worth fixing before consolidation.

Submit on iris in eu-west4 so the GCS reads stay in-region:

    uv run iris --cluster=marin job run --no-wait --memory=2G --extra=cpu \\
        --region europe-west4 \\
        --job-name "llmq-spot-check-$(date +%Y%m%d-%H%M%S)" -- \\
        python -m experiments.datakit.cluster.quality.v0.ops.spot_check \\
          --inference-base gs://marin-eu-west4/datakit/llm-quality-classifier/inference/sonnet46-thr05 \\
          --top-k 3 --bottom-k 3
"""

import argparse
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from statistics import mean, median, stdev

import pyarrow.parquet as pq
from marin.datakit.normalize import NormalizedData
from marin.datakit.sources import all_sources
from marin.execution.artifact import read_artifact
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)


DEFAULT_TOP_K = 3
DEFAULT_BOTTOM_K = 3
DEFAULT_THRESHOLD = 0.5
# Scaling law: per source we read row group 0 of one attribute shard
# (~MBs) plus row group 0 of the corresponding normalized shard for
# text lookups. The text column dominates -- decompressed it can hit
# 300-500 MB for large web sources (cp/stackv2_code, finepdfs,
# nemotron_cc_v2/*). Even 4 concurrent workers OOM a 2 GB driver;
# sequential keeps peak bounded to one row group at a time (~1 GB
# inclusive of pyarrow overhead). The whole 104-source pass runs in
# ~5-10 min sequential on in-region iris.
DEFAULT_NUM_WORKERS = 1
DEFAULT_TEXT_PREVIEW_CHARS = 200


@dataclass
class SourceStats:
    name: str
    n: int
    mean: float
    median: float
    std: float
    frac_high: float
    attr_shard: str
    top: list[tuple[str, int, float, str]]  # (id, partition_id, p_high, text_preview)
    bottom: list[tuple[str, int, float, str]]


def _list_parquet(d: str) -> list[str]:
    out: list[str] = []
    for root, _, files in StoragePath(d).walk():
        for f in files:
            if f.endswith(".parquet") and not f.startswith("."):
                out.append(str(root / f))
    return sorted(out)


def _find_attr_dir(inference_base: str, source_name: str) -> str | None:
    parent_rel = "/".join(source_name.split("/")[:-1])
    parent_path = StoragePath(inference_base) / "quality-llm" / parent_rel
    if not parent_path.exists():
        return None
    leaf_prefix = source_name.split("/")[-1] + "_"
    for entry in parent_path.ls():
        if entry.name.startswith(leaf_prefix):
            return str(entry)
    return None


def _read_attr_shard(path: str) -> list[tuple[str, int, float]]:
    """Read row group 0 of an attribute parquet shard.

    Row-group-bounded reads cap per-worker memory regardless of the
    source's total shard size. The attributes parquet is co-partitioned
    with the source's normalized parquet, so sampling within row group
    0 here lets us join against row group 0 of the normalized shard
    without buffering multi-GB tables.
    """
    rows: list[tuple[str, int, float]] = []
    with StoragePath(path).open("rb") as fh:
        pfile = pq.ParquetFile(fh)
        if pfile.num_row_groups == 0:
            return rows
        for batch in pfile.iter_batches(
            columns=["id", "partition_id", "attributes"],
            batch_size=2048,
            row_groups=[0],
        ):
            ids = batch.column("id").to_pylist()
            pids = batch.column("partition_id").to_pylist()
            attrs = batch.column("attributes").to_pylist()
            for doc_id, pid, attr in zip(ids, pids, attrs, strict=True):
                if not attr:
                    continue
                score = attr.get("score")
                if score is None:
                    continue
                rows.append((str(doc_id), int(pid), float(score)))
    return rows


def _normalized_shard_for(norm_dir: str, partition_id: int) -> str | None:
    for f in StoragePath(norm_dir).ls():
        if f.name.startswith(f"part-{partition_id:05d}-of-") and f.name.endswith(".parquet"):
            return str(f)
    return None


def _lookup_texts(norm_dir: str, by_pid: dict[int, set[str]], preview_chars: int) -> dict[str, str]:
    """Look up text previews for the requested ids by streaming row groups.

    The attribute parquet preserves input row order, but the new writer
    chooses its own row group sizes — so attribute row group 0 does NOT
    align with normalized row group 0. We iterate row groups one at a
    time (bounded memory) and break out as soon as every requested id
    is found. Most lookups hit within the first 1-2 row groups.
    """
    out: dict[str, str] = {}
    for pid, ids in by_pid.items():
        shard = _normalized_shard_for(norm_dir, pid)
        if not shard:
            continue
        target = set(ids)
        with StoragePath(shard).open("rb") as fh:
            pfile = pq.ParquetFile(fh)
            for rg_idx in range(pfile.num_row_groups):
                if not target:
                    break
                for batch in pfile.iter_batches(columns=["id", "text"], batch_size=512, row_groups=[rg_idx]):
                    bids = batch.column("id").to_pylist()
                    btexts = batch.column("text").to_pylist()
                    for doc_id, text in zip(bids, btexts, strict=True):
                        sid = str(doc_id)
                        if sid in target:
                            preview = (str(text) or "")[:preview_chars].replace("\n", " ").replace("\r", " ")
                            out[sid] = preview
                            target.discard(sid)
                            if not target:
                                break
                    if not target:
                        break
    return out


def _spot_check_one(source_name: str, inference_base: str, top_k: int, bottom_k: int, threshold: float, preview: int):
    attr_dir = _find_attr_dir(inference_base, source_name)
    if not attr_dir:
        return (source_name, None, "no attr dir")
    shards = _list_parquet(attr_dir)
    if not shards:
        return (source_name, None, "no shards")
    rng = random.Random(hash(("spot-check", source_name)) & 0x7FFF_FFFF)
    chosen = rng.choice(shards)
    rows = _read_attr_shard(chosen)
    if not rows:
        return (source_name, None, "empty shard")

    scores = [r[2] for r in rows]
    rows_sorted = sorted(rows, key=lambda r: r[2], reverse=True)
    top_rows = rows_sorted[:top_k]
    bot_rows = rows_sorted[-bottom_k:] if bottom_k else []

    by_pid: dict[int, set[str]] = {}
    for doc_id, pid, _ in top_rows + bot_rows:
        by_pid.setdefault(pid, set()).add(doc_id)

    src = all_sources()[source_name]
    nd = read_artifact(src.normalized.output_path, NormalizedData)
    text_by_id = _lookup_texts(nd.main_output_dir, by_pid, preview)

    def _enrich(rs):
        return [(i, p, s, text_by_id.get(i, "")) for i, p, s in rs]

    stats = SourceStats(
        name=source_name,
        n=len(rows),
        mean=mean(scores),
        median=median(scores),
        std=stdev(scores) if len(scores) > 1 else 0.0,
        frac_high=sum(1 for s in scores if s >= threshold) / len(scores),
        attr_shard=chosen.rsplit("/", 1)[-1],
        top=_enrich(top_rows),
        bottom=_enrich(bot_rows),
    )
    return (source_name, stats, None)


def run(
    *,
    inference_base: str,
    top_k: int,
    bottom_k: int,
    threshold: float,
    num_workers: int,
    preview: int,
) -> list[SourceStats]:
    names = sorted(all_sources())
    logger.info("spot-checking %d active sources from %s", len(names), inference_base)
    results: list[SourceStats] = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futs = {pool.submit(_spot_check_one, n, inference_base, top_k, bottom_k, threshold, preview): n for n in names}
        for fut in as_completed(futs):
            name, stats, err = fut.result()
            if err:
                logger.warning("%s: %s", name, err)
                continue
            results.append(stats)
    results.sort(key=lambda s: s.mean, reverse=True)
    return results


def print_ranking(results: list[SourceStats]) -> None:
    print(f"\n{'='*100}\nPER-SOURCE MEAN P(high), sorted high -> low\n{'='*100}")
    print(f"{'source':50s} {'n':>7s} {'mean':>7s} {'med':>7s} {'std':>7s} {'frac>=0.5':>10s}")
    for s in results:
        print(f"{s.name:50s} {s.n:>7d} {s.mean:>7.3f} {s.median:>7.3f} {s.std:>7.3f} {s.frac_high:>10.3f}")


def print_examples(results: list[SourceStats], n_each: int = 5) -> None:
    print(f"\n{'='*100}\nTOP {n_each} SOURCES BY MEAN P(high) -- highest-scored docs\n{'='*100}")
    for s in results[:n_each]:
        print(f"\n--- {s.name}  (mean={s.mean:.3f}, n={s.n}) ---")
        for doc_id, pid, score, text in s.top:
            print(f"  p={score:.3f} pid={pid} id={doc_id}")
            print(f"    {text}")

    print(f"\n{'='*100}\nBOTTOM {n_each} SOURCES BY MEAN P(high) -- lowest-scored docs\n{'='*100}")
    for s in results[-n_each:]:
        print(f"\n--- {s.name}  (mean={s.mean:.3f}, n={s.n}) ---")
        for doc_id, pid, score, text in s.bottom:
            print(f"  p={score:.3f} pid={pid} id={doc_id}")
            print(f"    {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-base", required=True, help="e.g. gs://...llm-quality-classifier/inference/sonnet46-thr05"
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--bottom-k", type=int, default=DEFAULT_BOTTOM_K)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--preview-chars", type=int, default=DEFAULT_TEXT_PREVIEW_CHARS)
    args = parser.parse_args()

    configure_logging(logging.INFO)
    results = run(
        inference_base=args.inference_base,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        threshold=args.threshold,
        num_workers=args.num_workers,
        preview=args.preview_chars,
    )
    print_ranking(results)
    print_examples(results)


if __name__ == "__main__":
    main()
