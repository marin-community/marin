# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tally row counts for every staged eval dataset under ``EVAL_ROOT``.

For each ``aa/<eval>`` and ``lmh/<task>`` subdir, read the parquet
footer (no payload reads) to get ``num_rows`` and report per-eval +
total. AA subdirs are cross-checked against ``AA_EVALS`` from
``prepare_eval_corpus.py``; any expected AA eval missing from the
staged corpus is flagged.

Submit on iris (eu-west4 worker pins ``MARIN_PREFIX`` to gs://marin-eu-west4):

    uv run iris --cluster=marin job run --region europe-west4 \\
        --extra=cpu --priority interactive \\
        -- python experiments/datakit/decontam/ops/count_eval_rows.py

Or run locally against a known prefix:

    MARIN_PREFIX=gs://marin-eu-west4 \\
        uv run python experiments/datakit/decontam/ops/count_eval_rows.py
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from pyarrow import fs
from rigging.log_setup import configure_logging

from experiments.datakit.decontam.all_sources_decon import EVAL_ROOT
from experiments.datakit.decontam.prepare_eval_corpus import AA_EVALS

logger = logging.getLogger(__name__)

FILE_CONCURRENCY = 32


def _row_count(path: str) -> int:
    gcs = fs.GcsFileSystem()
    bare = path.removeprefix("gs://")
    return pq.ParquetFile(bare, filesystem=gcs).metadata.num_rows


def _list_parquet_recursive(root: str) -> list[str]:
    gcs = fs.GcsFileSystem()
    bare = root.removeprefix("gs://").rstrip("/")
    entries = gcs.get_file_info(fs.FileSelector(bare, recursive=True))
    return sorted(f"gs://{e.path}" for e in entries if e.path.endswith(".parquet"))


def _group_files_by_eval(files: list[str], eval_root: str) -> dict[str, list[str]]:
    """Group parquet paths into ``aa/<eval>`` / ``lmh/<task>`` buckets."""
    root = eval_root.rstrip("/") + "/"
    groups: dict[str, list[str]] = {}
    for p in files:
        rel = p.removeprefix(root)
        parts = rel.split("/")
        if len(parts) < 3:
            continue
        groups.setdefault("/".join(parts[:2]), []).append(p)
    return groups


def _report(kind: str, rows_by_name: dict[str, int]) -> int:
    print(f"\n# {kind.upper()} -- {len(rows_by_name)} datasets\n")
    print(f"{'rows':>10}  name")
    print(f"{'-' * 10}  {'-' * 50}")
    total = 0
    for name in sorted(rows_by_name, key=lambda n: (-rows_by_name[n], n)):
        n = rows_by_name[name]
        print(f"{n:>10,}  {name}")
        total += n
    print(f"\n# {kind} total rows: {total:,}")
    return total


def main() -> None:
    configure_logging(logging.INFO)
    logger.info("listing eval parquet files under %s", EVAL_ROOT)
    all_files = _list_parquet_recursive(EVAL_ROOT)
    groups = _group_files_by_eval(all_files, EVAL_ROOT)
    logger.info("found %d datasets across %d parquet files", len(groups), len(all_files))

    with ThreadPoolExecutor(max_workers=FILE_CONCURRENCY) as pool:
        per_file = dict(zip(all_files, pool.map(_row_count, all_files), strict=True))

    rows_by_dataset: dict[str, int] = {}
    for name, files in groups.items():
        rows_by_dataset[name] = sum(per_file[f] for f in files)

    aa_rows = {n: r for n, r in rows_by_dataset.items() if n.startswith("aa/")}
    lmh_rows = {n: r for n, r in rows_by_dataset.items() if n.startswith("lmh/")}

    aa_total = _report("aa", aa_rows)
    lmh_total = _report("lmh", lmh_rows)

    expected_aa = {f"aa/{cfg.subdir}" for cfg in AA_EVALS}
    missing_aa = expected_aa - set(aa_rows)
    extra_aa = set(aa_rows) - expected_aa
    if missing_aa:
        print(f"\n# MISSING aa datasets ({len(missing_aa)}):")
        for name in sorted(missing_aa):
            print(f"  {name}")
    if extra_aa:
        print(f"\n# UNEXPECTED aa datasets ({len(extra_aa)}):")
        for name in sorted(extra_aa):
            print(f"  {name}")
    if not missing_aa and not extra_aa:
        print(f"\n# aa: all {len(expected_aa)} expected datasets present")

    print(f"\n# TOTAL across {len(rows_by_dataset)} datasets: {aa_total + lmh_total:,} rows")


if __name__ == "__main__":
    main()
