# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count documents — and eval ngrams — for the all-sources decon plan.

Three numbers we care about:

1. **Eval ngram count** -- exact ``bf.add`` call count for a single
   combined bloom built over the parquet eval tree under ``EVAL_ROOT``
   (same ``_extract_features`` and ``_bloom_hash`` paths as ``decon.py``).
   We report both total inserts and unique hashes; the bloom math cares
   about uniques but oversizing for total inserts is safe.

2. **Eval record count** — orthogonal sanity check.

3. **Corpus record count** (every entry of
   :func:`marin.datakit.sources.all_sources`). Each ``.normalized`` step's
   ``.artifact`` is loaded under the active ``MARIN_PREFIX`` to find
   ``main_output_dir``, then ``part-*.parquet`` row counts are summed via
   parquet footers (no payload reads). Sources not materialized under the
   active prefix are reported as ``(unresolved)``.

Submit on iris (eu-west4 worker pins ``MARIN_PREFIX`` to gs://marin-eu-west4):

    uv run iris --cluster=marin job run --region europe-west4 \\
        --extra=cpu --priority interactive --memory 8GB --cpu 2 \\
        --enable-extra-resources \\
        -- python experiments/datakit/decontam/ops/count_docs.py

Or run locally against a known prefix:

    MARIN_PREFIX=gs://marin-eu-west4 uv run python experiments/datakit/decontam/ops/count_docs.py
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pyarrow.parquet as pq
from marin.datakit.decon import NGramConfig, _bloom_hash, _extract_features
from marin.datakit.normalize import NormalizedData
from marin.datakit.sources import all_sources
from marin.execution.artifact import Artifact
from pyarrow import fs
from rigging.filesystem import marin_prefix

from experiments.datakit.decontam.all_sources_decon import EVAL_ROOT, NGRAM_LENGTH, OVERLAP_THRESHOLD

NGRAM = NGramConfig(ngram_length=NGRAM_LENGTH, overlap_threshold=OVERLAP_THRESHOLD)

logger = logging.getLogger(__name__)

PARTITION_CONCURRENCY = 32
SOURCE_CONCURRENCY = 16
EVAL_FILE_CONCURRENCY = 16


def _row_count(path: str) -> int:
    gcs = fs.GcsFileSystem()
    # pyarrow's GcsFileSystem expects bare ``bucket/key``, not a ``gs://...`` URI.
    bare = path.removeprefix("gs://")
    return pq.ParquetFile(bare, filesystem=gcs).metadata.num_rows


def _list_parquet_shards(main_output_dir: str) -> list[str]:
    gcs = fs.GcsFileSystem()
    bare = main_output_dir.removeprefix("gs://")
    entries = gcs.get_file_info(fs.FileSelector(bare, recursive=False))
    return sorted(f"gs://{e.path}" for e in entries if e.path.endswith(".parquet"))


def _list_parquet_recursive(root: str) -> list[str]:
    gcs = fs.GcsFileSystem()
    bare = root.removeprefix("gs://").rstrip("/")
    entries = gcs.get_file_info(fs.FileSelector(bare, recursive=True))
    return sorted(f"gs://{e.path}" for e in entries if e.path.endswith(".parquet"))


@dataclass
class EvalStats:
    n_records: int
    n_ngram_inserts: int
    unique_hashes: set[int]


def _file_ngram_stats(path: str) -> EvalStats:
    """Walk a parquet file's ``text`` column and compute exact ngram stats."""
    gcs = fs.GcsFileSystem()
    bare = path.removeprefix("gs://")
    n_records = 0
    n_ngrams = 0
    unique: set[int] = set()
    pf = pq.ParquetFile(bare, filesystem=gcs)
    for batch in pf.iter_batches(columns=["text"], batch_size=1024):
        for v in batch.column("text").to_pylist():
            if not v:
                continue
            n_records += 1
            for feat in _extract_features(str(v), NGRAM):
                n_ngrams += 1
                unique.add(_bloom_hash(feat))
    return EvalStats(n_records=n_records, n_ngram_inserts=n_ngrams, unique_hashes=unique)


def count_corpus_source(name: str, normalized_step) -> tuple[str, int | None, str]:
    """Return (name, total_rows or None, status). None = unresolved."""
    try:
        nd: NormalizedData = Artifact.from_path(normalized_step, NormalizedData)
    except FileNotFoundError:
        return (name, None, "unresolved (no .artifact at output_path)")
    except Exception as e:
        return (name, None, f"artifact load failed: {e!r}")

    try:
        shards = _list_parquet_shards(nd.main_output_dir)
    except Exception as e:
        return (name, None, f"list failed at {nd.main_output_dir}: {e!r}")
    if not shards:
        return (name, None, f"no parquet shards under {nd.main_output_dir}")

    with ThreadPoolExecutor(max_workers=PARTITION_CONCURRENCY) as pool:
        counts = list(pool.map(_row_count, shards))
    return (name, sum(counts), f"{len(shards)} shard(s) @ {nd.main_output_dir}")


def count_eval_subtree(name: str, files: list[str]) -> tuple[str, EvalStats | None, str]:
    """Aggregate ngram stats over a list of parquet files (one eval subdir)."""
    if not files:
        return (name, None, "no parquet files")
    with ThreadPoolExecutor(max_workers=EVAL_FILE_CONCURRENCY) as pool:
        per_file = list(pool.map(_file_ngram_stats, files))
    combined = EvalStats(n_records=0, n_ngram_inserts=0, unique_hashes=set())
    for s in per_file:
        combined.n_records += s.n_records
        combined.n_ngram_inserts += s.n_ngram_inserts
        combined.unique_hashes |= s.unique_hashes
    return (name, combined, f"{len(files)} file(s)")


def _report_corpus(results: list[tuple[str, int | None, str]]) -> int:
    results.sort(key=lambda r: (-(r[1] or -1), r[0]))
    print(f"\n# CORPUS SOURCES (decon scan targets) -- {len(results)} entries\n")
    print(f"{'rows':>15}  {'name':<40s}  status")
    print(f"{'-' * 15}  {'-' * 40}  {'-' * 40}")
    total = 0
    unresolved = 0
    for name, rows, status in results:
        rows_str = f"{rows:>15,}" if rows is not None else f"{'(unresolved)':>15}"
        print(f"{rows_str}  {name:<40s}  {status}")
        if rows is not None:
            total += rows
        else:
            unresolved += 1
    print(f"\n# corpus total rows: {total:,}")
    print(f"# resolved: {len(results) - unresolved}/{len(results)}")
    return total


def _report_evals(results: list[tuple[str, EvalStats | None, str]]) -> tuple[int, int, int]:
    results.sort(key=lambda r: -(r[1].n_ngram_inserts if r[1] is not None else -1))
    print(f"\n# EVAL SOURCES (bloom inputs) -- {len(results)} entries\n")
    print(f"{'records':>10}  {'inserts':>12}  {'unique':>12}  {'name':<40s}  status")
    print(f"{'-' * 10}  {'-' * 12}  {'-' * 12}  {'-' * 40}  {'-' * 30}")
    total_records = 0
    total_inserts = 0
    global_unique: set[int] = set()
    for name, stats, status in results:
        if stats is None:
            print(f"{'(none)':>10}  {'(none)':>12}  {'(none)':>12}  {name:<40s}  {status}")
            continue
        row = f"{stats.n_records:>10,}  {stats.n_ngram_inserts:>12,}  {len(stats.unique_hashes):>12,}"
        print(f"{row}  {name:<40s}  {status}")
        total_records += stats.n_records
        total_inserts += stats.n_ngram_inserts
        global_unique |= stats.unique_hashes
    print()
    print(f"# eval total records: {total_records:,}")
    print(f"# eval total ngram inserts (bf.add calls): {total_inserts:,}")
    print(f"# eval unique ngram hashes (bloom capacity floor): {len(global_unique):,}")
    return total_records, total_inserts, len(global_unique)


def _group_files_by_eval(files: list[str], eval_root: str) -> dict[str, list[str]]:
    """Group parquet paths by ``<bucket>/<aa-or-lmh>/<eval-name>/`` prefix.

    For ``gs://.../evals/aa/hle/test.parquet`` the group key is ``aa/hle``;
    for ``gs://.../evals/lmh/mmlu_anatomy/eval.parquet`` it's ``lmh/mmlu_anatomy``.
    """
    root = eval_root.rstrip("/") + "/"
    groups: dict[str, list[str]] = {}
    for p in files:
        rel = p.removeprefix(root)
        parts = rel.split("/")
        if len(parts) < 3:
            continue  # skip stray top-level files
        key = "/".join(parts[:2])  # e.g. "aa/hle" or "lmh/mmlu_anatomy"
        groups.setdefault(key, []).append(p)
    return groups


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    prefix = marin_prefix()
    logger.info("using MARIN_PREFIX=%s", prefix)

    # Pass 1: eval corpus (under EVAL_ROOT, parquet tree, ngram-exact).
    logger.info("listing eval parquet files under %s", EVAL_ROOT)
    all_eval_files = _list_parquet_recursive(EVAL_ROOT)
    groups = _group_files_by_eval(all_eval_files, EVAL_ROOT)
    logger.info("found %d eval subtrees across %d parquet files", len(groups), len(all_eval_files))
    eval_results: list[tuple[str, EvalStats | None, str]] = []
    with ThreadPoolExecutor(max_workers=SOURCE_CONCURRENCY) as pool:
        futures = {pool.submit(count_eval_subtree, name, files): name for name, files in groups.items()}
        for fut in futures:
            eval_results.append(fut.result())
    _, eval_inserts, eval_unique = _report_evals(eval_results)

    # Pass 2: corpus sources (large, MARIN_PREFIX-relative)
    sources = all_sources()
    logger.info("counting %d corpus sources", len(sources))
    corpus_results: list[tuple[str, int | None, str]] = []
    with ThreadPoolExecutor(max_workers=SOURCE_CONCURRENCY) as pool:
        futures = [pool.submit(count_corpus_source, name, src.normalized) for name, src in sources.items()]
        for fut in futures:
            corpus_results.append(fut.result())
    corpus_total = _report_corpus(corpus_results)

    # Bloom sizing recommendation: at least the unique hash count, with 2x
    # headroom so growth or new evals don't blow the FPR. Round to a clean
    # 1e6-aligned value.
    recommended = max(1_000_000, ((eval_unique * 2) + 999_999) // 1_000_000 * 1_000_000)
    print("\n# bloom sizing recommendation")
    print(f"# eval unique ngrams: {eval_unique:,} (real bloom capacity needed)")
    print(f"# eval total inserts: {eval_inserts:,} (bf.add call count)")
    print(f"# recommended ESTIMATED_DOC_COUNT: {recommended:,} (= ~2x unique, rounded to 1e6)")
    print(f"# corpus records (decon scan targets): {corpus_total:,}")


if __name__ == "__main__":
    # Make sure pyarrow GcsFileSystem picks up ambient ADC.
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "")
    main()
