# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export document ids marked contaminated by a Datakit decon run.

This is a post-processing step over existing Datakit attributes. It does not
rescan the normalized text corpus. For each contaminated row, it writes the
document id, co-partitioning metadata, the normalized source parquet path, the
Datakit attribute parquet path, max overlap, matched hashes, and eval ids
attributed through Datakit's ``_bloom/eval_hash_index.parquet`` sidecar.

Default paths target the Math500-only Nemotron-CC math decon run:

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
        --cpu 8 --memory 32GB --disk 20GB --priority interactive --extra cpu \
        --enable-extra-resources --preemptible --region us-east5 \
        --job-name export-nemotron-math500-contam-docs \
        -e PYTHONUNBUFFERED 1 \
        -- python scripts/analysis/export_decon_contaminated_docs.py --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from marin.utils import fsspec_exists, fsspec_glob

logger = logging.getLogger(__name__)

NORMALIZED_NEMOTRON_CC_MATH_4PLUS = "gs://marin-us-east5/normalized/nemotron_cc_math_v1/4plus_b05688a8/outputs/main"
MATH500_DECON_ATTRS = (
    "gs://marin-us-east5/scratch/ahmed/math-decontamin-dir/decon_nemotron_cc_math_4plus/"
    "math500__test/datakit_decon_attrs"
)
DEFAULT_OUTPUT_ROOT = (
    "gs://marin-us-east5/scratch/ahmed/math-decontamin-dir/decon_nemotron_cc_math_4plus/"
    "math500__test/contaminated_docs"
)

CONTAMINATED_PARQUET = "contaminated_docs.parquet"
CONTAMINATED_CSV = "contaminated_docs.csv"
SUMMARY_JSON = "summary.json"

OUTPUT_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("partition_id", pa.int64()),
        ("row_index_in_partition", pa.int64()),
        ("source_parquet", pa.string()),
        ("attr_parquet", pa.string()),
        ("max_overlap", pa.float64()),
        ("matched_hashes", pa.list_(pa.uint64())),
        ("matched_eval_ids", pa.list_(pa.string())),
        ("matched_eval_id_count", pa.int64()),
    ]
)


def write_json(path: str, payload: dict[str, Any]) -> None:
    fs, _, paths = fsspec.get_fs_token_paths(path)
    parent = os.path.dirname(paths[0])
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(paths[0], "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_parquet_table(path: str, table: pa.Table) -> None:
    fs, _, paths = fsspec.get_fs_token_paths(path)
    parent = os.path.dirname(paths[0])
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(paths[0], "wb") as f:
        pq.write_table(table, f)


def write_csv(path: str, table: pa.Table) -> None:
    fs, _, paths = fsspec.get_fs_token_paths(path)
    parent = os.path.dirname(paths[0])
    if parent:
        fs.makedirs(parent, exist_ok=True)
    rows = table.to_pylist()
    columns = [
        "id",
        "partition_id",
        "row_index_in_partition",
        "source_parquet",
        "attr_parquet",
        "max_overlap",
        "matched_eval_id_count",
        "matched_eval_ids",
    ]
    with fs.open(paths[0], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["matched_eval_ids"] = "|".join(row.get("matched_eval_ids") or [])
            writer.writerow(row)


def fail_if_existing(paths: list[str], *, resume: bool, force: bool) -> None:
    existing = [path for path in paths if fsspec_exists(path)]
    if not existing:
        return
    if force:
        for path in existing:
            fs, _, resolved = fsspec.get_fs_token_paths(path)
            logger.warning("removing existing output because --force was set: %s", path)
            fs.rm(resolved[0], recursive=True)
        return
    if resume:
        return
    raise RuntimeError(f"outputs already exist; pass --resume or --force: {existing}")


def _attribute_column(table: pa.Table, name: str) -> pa.ChunkedArray:
    if name in table.column_names:
        return table[name]
    if "attributes" in table.column_names:
        return pc.struct_field(table["attributes"], name)
    raise ValueError(f"attribute column {name!r} not found in table schema: {table.schema}")


def normalized_files_by_basename(normalized_root: str) -> dict[str, str]:
    files = sorted(fsspec_glob(f"{normalized_root.rstrip('/')}/**/*.parquet"))
    by_basename: dict[str, str] = {}
    for path in files:
        basename = os.path.basename(path)
        if basename in by_basename:
            raise ValueError(f"duplicate normalized parquet basename {basename!r}: {by_basename[basename]}, {path}")
        by_basename[basename] = path
    if not by_basename:
        raise FileNotFoundError(f"No normalized parquet files found under {normalized_root}")
    return by_basename


def load_hash_to_eval_ids(eval_hash_index_path: str) -> dict[int, list[str]]:
    logger.info("loading eval hash index: %s", eval_hash_index_path)
    hash_to_eval_ids: defaultdict[int, set[str]] = defaultdict(set)
    rows = 0
    with fsspec.open(eval_hash_index_path, "rb") as f:
        parquet_file = pq.ParquetFile(f)
        for row_group in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group, columns=["hash", "eval_id"])
            rows += table.num_rows
            for hash_value, eval_id in zip(table["hash"].to_pylist(), table["eval_id"].to_pylist(), strict=True):
                hash_to_eval_ids[int(hash_value)].add(str(eval_id))
    logger.info("loaded %d eval hash-index rows covering %d unique hashes", rows, len(hash_to_eval_ids))
    return {hash_value: sorted(eval_ids) for hash_value, eval_ids in hash_to_eval_ids.items()}


def matched_eval_ids_for_hashes(hashes: list[int] | None, hash_to_eval_ids: dict[int, list[str]]) -> list[str]:
    eval_ids: set[str] = set()
    for hash_value in hashes or []:
        eval_ids.update(hash_to_eval_ids.get(int(hash_value), ()))
    return sorted(eval_ids)


def contaminated_rows_for_attr_file(
    *,
    attr_path: str,
    source_path: str,
    hash_to_eval_ids: dict[int, list[str]],
) -> list[dict[str, Any]]:
    with fsspec.open(attr_path, "rb") as f:
        table = pq.read_table(f)

    contaminated = _attribute_column(table, "contaminated").combine_chunks()
    if pc.sum(contaminated.cast(pa.int64())).as_py() == 0:
        return []

    row_indices = pc.indices_nonzero(contaminated).cast(pa.int64())
    filtered = pa.table(
        {
            "id": table["id"].filter(contaminated),
            "partition_id": table["partition_id"].filter(contaminated),
            "row_index_in_partition": row_indices,
            "max_overlap": _attribute_column(table, "max_overlap").filter(contaminated),
            "matched_hashes": _attribute_column(table, "matched_hashes").filter(contaminated),
        }
    )

    rows = []
    for row in filtered.to_pylist():
        matched_hashes = [int(hash_value) for hash_value in row["matched_hashes"] or []]
        matched_eval_ids = matched_eval_ids_for_hashes(matched_hashes, hash_to_eval_ids)
        rows.append(
            {
                "id": row["id"],
                "partition_id": int(row["partition_id"]),
                "row_index_in_partition": int(row["row_index_in_partition"]),
                "source_parquet": source_path,
                "attr_parquet": attr_path,
                "max_overlap": float(row["max_overlap"]),
                "matched_hashes": matched_hashes,
                "matched_eval_ids": matched_eval_ids,
                "matched_eval_id_count": len(matched_eval_ids),
            }
        )
    return rows


def export_contaminated_docs(
    *,
    decon_attrs: str,
    normalized_root: str,
    output_root: str,
    resume: bool,
    force: bool,
) -> dict[str, Any]:
    output_parquet = f"{output_root.rstrip('/')}/{CONTAMINATED_PARQUET}"
    output_csv = f"{output_root.rstrip('/')}/{CONTAMINATED_CSV}"
    summary_path = f"{output_root.rstrip('/')}/{SUMMARY_JSON}"
    fail_if_existing([output_parquet, output_csv, summary_path], resume=resume, force=force)
    if resume and fsspec_exists(summary_path):
        logger.info("using existing contaminated-doc export: %s", summary_path)
        with fsspec.open(summary_path) as f:
            return json.load(f)

    attr_paths = sorted(path for path in fsspec_glob(f"{decon_attrs.rstrip('/')}/*.parquet") if "/_bloom/" not in path)
    if not attr_paths:
        raise FileNotFoundError(f"No Datakit attribute parquet files found under {decon_attrs}")
    source_by_basename = normalized_files_by_basename(normalized_root)
    hash_to_eval_ids = load_hash_to_eval_ids(f"{decon_attrs.rstrip('/')}/_bloom/eval_hash_index.parquet")

    rows: list[dict[str, Any]] = []
    total_attr_rows = 0
    for index, attr_path in enumerate(attr_paths):
        basename = os.path.basename(attr_path)
        source_path = source_by_basename.get(basename)
        if source_path is None:
            raise ValueError(f"attribute parquet {attr_path} has no matching normalized source parquet")
        logger.info("extracting contaminated docs from attr file %d/%d: %s", index + 1, len(attr_paths), attr_path)
        with fsspec.open(attr_path, "rb") as f:
            total_attr_rows += pq.ParquetFile(f).metadata.num_rows
        rows.extend(
            contaminated_rows_for_attr_file(
                attr_path=attr_path,
                source_path=source_path,
                hash_to_eval_ids=hash_to_eval_ids,
            )
        )

    rows.sort(key=lambda row: (row["partition_id"], row["row_index_in_partition"], row["id"]))
    table = pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)
    write_parquet_table(output_parquet, table)
    write_csv(output_csv, table)

    summary = {
        "decon_attrs": decon_attrs,
        "normalized_root": normalized_root,
        "attr_files": len(attr_paths),
        "total_attr_rows": total_attr_rows,
        "contaminated_docs": len(rows),
        "output_parquet": output_parquet,
        "output_csv": output_csv,
        "max_overlap": float(np.max([row["max_overlap"] for row in rows])) if rows else 0.0,
        "matched_eval_records": len({eval_id for row in rows for eval_id in row["matched_eval_ids"]}),
    }
    write_json(summary_path, summary)
    logger.info("wrote contaminated-doc export: %s", json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decon-attrs", default=MATH500_DECON_ATTRS)
    parser.add_argument("--normalized-root", default=NORMALIZED_NEMOTRON_CC_MATH_4PLUS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    export_contaminated_docs(
        decon_attrs=args.decon_attrs,
        normalized_root=args.normalized_root,
        output_root=args.output_root,
        resume=args.resume,
        force=args.force,
    )


if __name__ == "__main__":
    main()
