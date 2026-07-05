# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Datakit decontamination against staged math benchmark eval splits.

This consumes the Datakit-ready directory produced by
``stage_math_decontam_sources.py`` and runs one independent Datakit decon pass
per eval split.  Separate passes are intentional: a combined bloom filter would
answer "contaminated by any split", but it would not tell us whether each split
independently crosses the paragraph overlap threshold.

Launch in us-east5 so normalized Nemotron CC math reads stay in-region:

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
        --cpu 8 --memory 64GB --disk 50GB --priority interactive --extra cpu \
        --enable-extra-resources --preemptible --region us-east5 \
        --job-name decon-nemotron-math-benchmarks \
        -e PYTHONUNBUFFERED 1 \
        -- python scripts/analysis/decon_nemotron_math_against_benchmarks.py --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import posixpath
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray import ResourceConfig
from marin.datakit.decon import NGramConfig, decon_to_parquet
from marin.datakit.normalize import NormalizedData
from marin.utils import fsspec_exists, fsspec_glob
from zephyr.readers import SUPPORTED_EXTENSIONS, load_file

logger = logging.getLogger(__name__)

NORMALIZED_NEMOTRON_CC_MATH_4PLUS = "gs://marin-us-east5/normalized/nemotron_cc_math_v1/4plus_b05688a8/outputs/main"
MATH_DECONTAM_EVAL_ROOT = "gs://marin-us-east5/scratch/ahmed/math-decontamin-dir"
DEFAULT_OUTPUT_ROOT = "gs://marin-us-east5/scratch/ahmed/math-decontamin-dir/decon_nemotron_cc_math_4plus"

RUN_SUMMARY_FILENAME = "run_summary.json"
REPORT_FILENAME = "contamination_report.json"
AGGREGATE_JSON_FILENAME = "contamination_report.all.json"
AGGREGATE_CSV_FILENAME = "contamination_report.all.csv"
MANIFEST_FILENAME = ".manifest.json"


@dataclass(frozen=True)
class EvalSource:
    """One staged benchmark source split."""

    key: str
    source_dir: str
    data_file: str
    dataset_id: str | None
    revision: str | None
    config: str | None
    split: str | None
    record_count_hint: int | None = None

    @property
    def report_group(self) -> str:
        return self.key.replace("__", "/")


def write_json(path: str, payload: dict[str, Any]) -> None:
    fs, _, paths = fsspec.get_fs_token_paths(path)
    parent = os.path.dirname(paths[0])
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(paths[0], "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: str) -> dict[str, Any]:
    with fsspec.open(path) as f:
        return json.load(f)


def uri_has_children(path: str) -> bool:
    return bool(fsspec_glob(f"{path.rstrip('/')}/*"))


def remove_uri(path: str) -> None:
    fs, _, paths = fsspec.get_fs_token_paths(path)
    if fs.exists(paths[0]):
        fs.rm(paths[0], recursive=True)


def fail_if_existing(path: str, *, resume: bool, force: bool) -> None:
    if not (fsspec_exists(path) or uri_has_children(path)):
        return
    if force:
        logger.warning("removing existing output because --force was set: %s", path)
        remove_uri(path)
        return
    if resume:
        return
    raise RuntimeError(f"target already exists; pass --resume or --force: {path}")


def _is_hidden_dir(root: str, resolved: str) -> bool:
    rel = os.path.relpath(root, resolved)
    if rel == ".":
        return False
    return any(part.startswith(".") for part in rel.split(os.sep))


def _safe_key(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in value)
    safe = safe.strip("_")
    if not safe:
        raise ValueError(f"cannot build safe key from {value!r}")
    return safe


def _relative_data_dir(eval_root: str, data_file: str) -> str:
    root = eval_root.rstrip("/") + "/"
    if not data_file.startswith(root):
        raise ValueError(f"data file {data_file!r} is not under eval root {eval_root!r}")
    rel_file = data_file[len(root) :]
    return rel_file.rsplit("/", 1)[0]


def _source_key_from_data_file(eval_root: str, data_file: str) -> str:
    rel_dir = _relative_data_dir(eval_root, data_file)
    return "__".join(_safe_key(part) for part in rel_dir.split("/"))


def _source_dir_from_file(data_file: str) -> str:
    return data_file.rsplit("/", 1)[0]


def discover_eval_sources(eval_root: str) -> list[EvalSource]:
    """Discover staged benchmark split directories under ``eval_root``."""
    manifest_path = posixpath.join(eval_root.rstrip("/"), MANIFEST_FILENAME)
    if fsspec_exists(manifest_path):
        manifest = read_json(manifest_path)
        sources: list[EvalSource] = []
        for entry in manifest.get("entries", []):
            data_file = str(entry["output_file"])
            sources.append(
                EvalSource(
                    key=_source_key_from_data_file(eval_root, data_file),
                    source_dir=_source_dir_from_file(data_file),
                    data_file=data_file,
                    dataset_id=entry.get("dataset_id"),
                    revision=entry.get("revision"),
                    config=entry.get("config"),
                    split=entry.get("split"),
                    record_count_hint=entry.get("record_count"),
                )
            )
        if sources:
            return sorted(sources, key=lambda source: source.key)

    fs, resolved = fsspec.get_fs_token_paths(eval_root)[0], fsspec.get_fs_token_paths(eval_root)[2][0]
    protocol = eval_root.split("://")[0] if "://" in eval_root else ""
    sources = []
    for root, _dirs, files in fs.walk(resolved):
        if _is_hidden_dir(root, resolved):
            continue
        for filename in files:
            if filename.startswith(".") or not filename.endswith(SUPPORTED_EXTENSIONS):
                continue
            full = f"{protocol}://{os.path.join(root, filename)}" if protocol else os.path.join(root, filename)
            sources.append(
                EvalSource(
                    key=_source_key_from_data_file(eval_root, full),
                    source_dir=_source_dir_from_file(full),
                    data_file=full,
                    dataset_id=None,
                    revision=None,
                    config=None,
                    split=None,
                )
            )
    if not sources:
        raise FileNotFoundError(f"No eval data files found under {eval_root}")
    return sorted(sources, key=lambda source: source.key)


def count_eval_records(source: EvalSource) -> int:
    """Count zephyr-readable records in one staged eval source directory."""
    count = 0
    for record in load_file(source.data_file):
        if record.get("text"):
            count += 1
    return count


def _unique_uint64_values(values: pa.Array | pa.ChunkedArray) -> set[int]:
    if len(values) == 0:
        return set()
    array = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
    if len(array) == 0:
        return set()
    numpy_values = array.to_numpy(zero_copy_only=False)
    if numpy_values.size == 0:
        return set()
    return {int(value) for value in np.unique(numpy_values)}


def _sum_scalar(value: pa.Scalar | None) -> float:
    if value is None or not value.is_valid:
        return 0.0
    return float(value.as_py())


def _max_scalar(value: pa.Scalar | None) -> float:
    if value is None or not value.is_valid:
        return 0.0
    return float(value.as_py())


def _attribute_column(table: pa.Table, name: str) -> pa.ChunkedArray:
    """Return a Datakit decon attribute column from either supported layout."""
    if name in table.column_names:
        return table[name]
    if "attributes" in table.column_names:
        return pc.struct_field(table["attributes"], name)
    raise ValueError(f"attribute column {name!r} not found in table schema: {table.schema}")


def attr_paths(output_dir: str) -> list[str]:
    paths = sorted(path for path in fsspec_glob(f"{output_dir.rstrip('/')}/*.parquet") if "/_bloom/" not in path)
    if not paths:
        raise FileNotFoundError(f"No Datakit attribute parquet files found under {output_dir}")
    return paths


def collect_attr_stats(paths: list[str]) -> dict[str, Any]:
    """Summarize Datakit attribute files and collect hit hashes for attribution."""
    hit_hashes: set[int] = set()
    total_docs = 0
    contaminated_docs = 0
    max_overlap_sum = 0.0
    contaminated_overlap_sum = 0.0
    max_overlap = 0.0
    raw_matched_hashes = 0

    for index, path in enumerate(paths):
        logger.info("reading attr file %d/%d: %s", index + 1, len(paths), path)
        with fsspec.open(path, "rb") as f:
            table = pq.read_table(f)

        total_docs += table.num_rows
        contaminated_column = _attribute_column(table, "contaminated")
        max_overlap_column = _attribute_column(table, "max_overlap")
        matched_hashes_column = _attribute_column(table, "matched_hashes")
        attr_table = pa.table(
            {
                "contaminated": contaminated_column,
                "max_overlap": max_overlap_column,
                "matched_hashes": matched_hashes_column,
            }
        )

        contaminated = contaminated_column.combine_chunks()
        contaminated_count = int(pc.sum(contaminated.cast(pa.int64())).as_py() or 0)
        contaminated_docs += contaminated_count
        max_overlap_sum += _sum_scalar(pc.sum(max_overlap_column))
        max_overlap = max(max_overlap, _max_scalar(pc.max(max_overlap_column)))

        if contaminated_count == 0:
            continue
        contaminated_table = attr_table.filter(contaminated)
        contaminated_overlap_sum += _sum_scalar(pc.sum(contaminated_table["max_overlap"]))
        flat_hashes = pc.list_flatten(contaminated_table["matched_hashes"])
        raw_matched_hashes += len(flat_hashes)
        hit_hashes.update(_unique_uint64_values(flat_hashes))

    return {
        "attr_files": len(paths),
        "total_docs": total_docs,
        "contaminated_docs": contaminated_docs,
        "clean_docs": total_docs - contaminated_docs,
        "contamination_rate": contaminated_docs / total_docs if total_docs else 0.0,
        "mean_max_overlap": max_overlap_sum / total_docs if total_docs else 0.0,
        "mean_contaminated_max_overlap": contaminated_overlap_sum / contaminated_docs if contaminated_docs else 0.0,
        "max_overlap": max_overlap,
        "raw_matched_hashes_in_contaminated_docs": raw_matched_hashes,
        "unique_matched_hashes_in_contaminated_docs": len(hit_hashes),
        "hit_hashes": hit_hashes,
    }


def collect_eval_attribution(eval_hash_index_path: str, hit_hashes: set[int], top_k: int) -> dict[str, Any]:
    """Map contaminated-doc feature hits back to eval IDs through Datakit's sidecar."""
    if not hit_hashes:
        return {
            "eval_hash_index_rows": 0,
            "matched_eval_hash_index_rows": 0,
            "eval_records_with_feature_hit": 0,
            "top_eval_ids_by_matched_features": [],
        }

    value_set = pa.array(np.fromiter(hit_hashes, dtype=np.uint64, count=len(hit_hashes)), type=pa.uint64())
    eval_id_counts: Counter[str] = Counter()
    eval_hash_index_rows = 0
    matched_eval_hash_index_rows = 0

    with fsspec.open(eval_hash_index_path, "rb") as f:
        parquet_file = pq.ParquetFile(f)
        for row_group in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group, columns=["hash", "eval_id"])
            eval_hash_index_rows += table.num_rows
            matched = table.filter(pc.is_in(table["hash"], value_set=value_set))
            matched_eval_hash_index_rows += matched.num_rows
            eval_id_counts.update(str(eval_id) for eval_id in matched["eval_id"].to_pylist())

    return {
        "eval_hash_index_rows": eval_hash_index_rows,
        "matched_eval_hash_index_rows": matched_eval_hash_index_rows,
        "eval_records_with_feature_hit": len(eval_id_counts),
        "top_eval_ids_by_matched_features": [
            {"eval_id": eval_id, "matched_feature_rows": count} for eval_id, count in eval_id_counts.most_common(top_k)
        ],
    }


def summarize_decon_output(
    *,
    source: EvalSource,
    decon_output: str,
    eval_record_count: int,
    top_k_eval_ids: int,
) -> dict[str, Any]:
    paths = attr_paths(decon_output)
    attr_stats = collect_attr_stats(paths)
    hit_hashes = attr_stats.pop("hit_hashes")
    attribution = collect_eval_attribution(
        posixpath.join(decon_output.rstrip("/"), "_bloom", "eval_hash_index.parquet"),
        hit_hashes,
        top_k_eval_ids,
    )
    eval_records_with_hit = int(attribution["eval_records_with_feature_hit"])
    return {
        "key": source.key,
        "report_group": source.report_group,
        "source_dir": source.source_dir,
        "data_file": source.data_file,
        "dataset_id": source.dataset_id,
        "revision": source.revision,
        "config": source.config,
        "split": source.split,
        "eval_records": eval_record_count,
        "eval_record_hit_rate": eval_records_with_hit / eval_record_count if eval_record_count else 0.0,
        "decon_output": decon_output,
        **attr_stats,
        **attribution,
    }


def run_decon_for_source(
    *,
    source: EvalSource,
    normalized_root: str,
    output_root: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_root = posixpath.join(output_root.rstrip("/"), source.key)
    decon_output = posixpath.join(source_root, "datakit_decon_attrs")
    run_summary_path = posixpath.join(source_root, RUN_SUMMARY_FILENAME)
    report_path = posixpath.join(source_root, REPORT_FILENAME)

    fail_if_existing(source_root, resume=args.resume, force=args.force)

    ngram = None
    if args.ngram_length > 0:
        ngram = NGramConfig(
            ngram_length=args.ngram_length,
            stride=args.ngram_stride,
            overlap_threshold=args.overlap_threshold,
        )

    if args.resume and fsspec_exists(run_summary_path):
        logger.info("using existing decon run for %s: %s", source.key, run_summary_path)
        run_summary = read_json(run_summary_path)
    else:
        attrs = decon_to_parquet(
            normalized_data=NormalizedData(main_output_dir=normalized_root, dup_output_dir="", counters={}),
            eval_data_sources=source.source_dir,
            output_path=decon_output,
            ngram=ngram,
            estimated_doc_count=args.estimated_eval_features,
            false_positive_rate=args.false_positive_rate,
            worker_resources=ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram),
            max_workers=args.max_workers,
        )
        run_summary = {
            "key": source.key,
            "source": source.__dict__,
            "normalized_root": normalized_root,
            "decon_output": decon_output,
            "ngram_length": args.ngram_length,
            "ngram_stride": args.ngram_stride,
            "overlap_threshold": args.overlap_threshold,
            "estimated_eval_features": args.estimated_eval_features,
            "false_positive_rate": args.false_positive_rate,
            "attributes": attrs.model_dump(),
        }
        write_json(run_summary_path, run_summary)

    eval_record_count = source.record_count_hint if source.record_count_hint is not None else count_eval_records(source)
    report = summarize_decon_output(
        source=source,
        decon_output=str(run_summary["decon_output"]),
        eval_record_count=eval_record_count,
        top_k_eval_ids=args.top_k_eval_ids,
    )
    report["run_summary_path"] = run_summary_path
    write_json(report_path, report)
    logger.info(
        "%s: contaminated %d/%d docs (%.6f), eval hits %d/%d",
        source.key,
        report["contaminated_docs"],
        report["total_docs"],
        report["contamination_rate"],
        report["eval_records_with_feature_hit"],
        report["eval_records"],
    )
    return report


def write_aggregate_reports(output_root: str, reports: list[dict[str, Any]]) -> dict[str, str]:
    json_path = posixpath.join(output_root.rstrip("/"), AGGREGATE_JSON_FILENAME)
    csv_path = posixpath.join(output_root.rstrip("/"), AGGREGATE_CSV_FILENAME)
    write_json(
        json_path,
        {
            "schema_version": 1,
            "reports": reports,
        },
    )

    columns = [
        "key",
        "report_group",
        "dataset_id",
        "revision",
        "config",
        "split",
        "eval_records",
        "eval_records_with_feature_hit",
        "eval_record_hit_rate",
        "total_docs",
        "contaminated_docs",
        "clean_docs",
        "contamination_rate",
        "mean_max_overlap",
        "mean_contaminated_max_overlap",
        "max_overlap",
        "unique_matched_hashes_in_contaminated_docs",
        "eval_hash_index_rows",
        "matched_eval_hash_index_rows",
        "decon_output",
    ]
    fs, _, paths = fsspec.get_fs_token_paths(csv_path)
    parent = os.path.dirname(paths[0])
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(paths[0], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reports)
    return {"json": json_path, "csv": csv_path}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-root", default=NORMALIZED_NEMOTRON_CC_MATH_4PLUS)
    parser.add_argument("--eval-root", default=MATH_DECONTAM_EVAL_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--only", nargs="*", default=None, help="Optional source keys to run, e.g. gsm8k__main__test.")
    parser.add_argument("--dry-run", action="store_true", help="Only print discovered sources.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed per-source decon runs.")
    parser.add_argument("--force", action="store_true", help="Remove existing per-source outputs before running.")
    parser.add_argument("--ngram-length", type=int, default=13)
    parser.add_argument("--ngram-stride", type=int, default=0)
    parser.add_argument("--overlap-threshold", type=float, default=0.5)
    parser.add_argument("--estimated-eval-features", type=int, default=1_000_000)
    parser.add_argument("--false-positive-rate", type=float, default=1e-9)
    parser.add_argument("--max-workers", type=int, default=256)
    parser.add_argument("--worker-cpu", type=int, default=2)
    parser.add_argument("--worker-ram", default="6g")
    parser.add_argument("--top-k-eval-ids", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sources = discover_eval_sources(args.eval_root)
    if args.only:
        requested = set(args.only)
        sources = [source for source in sources if source.key in requested]
        missing = requested - {source.key for source in sources}
        if missing:
            raise ValueError(f"Requested source keys not found under {args.eval_root}: {sorted(missing)}")
    if not sources:
        raise ValueError("No eval sources selected")

    logger.info("selected %d eval sources:", len(sources))
    for source in sources:
        logger.info("  %s -> %s", source.key, source.source_dir)

    if args.dry_run:
        print(json.dumps([source.__dict__ for source in sources], indent=2, sort_keys=True))
        return

    reports = [
        run_decon_for_source(
            source=source,
            normalized_root=args.normalized_root,
            output_root=args.output_root,
            args=args,
        )
        for source in sources
    ]
    paths = write_aggregate_reports(args.output_root, reports)
    logger.info("wrote aggregate reports: %s", paths)


if __name__ == "__main__":
    main()
