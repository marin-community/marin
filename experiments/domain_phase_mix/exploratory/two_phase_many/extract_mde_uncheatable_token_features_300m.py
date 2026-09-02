# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "gcsfs", "numpy", "pandas", "pyarrow", "tqdm"]
# ///
"""Materialize compact token-level uncheatable features from raw MDE score files.

The full raw-text checkpoint-feature surface stores one ``scored_documents``
Parquet per checkpoint.  Each file is about 1GB because it contains per-byte
loss arrays for all raw-text evaluation bundles.  This script extracts a
bounded, aligned token sketch for the uncheatable eval datasets, plus exact
per-document aggregates, into compact local or GCS Parquet files.

This should normally run in the same region as the source artifacts.  By
default the script refuses to read ``gs://`` inputs; pass ``--allow-gcs-read``
only for an intentional in-region extraction job.
"""

from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import math
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURE_INDEX = (
    SCRIPT_DIR / "reference_outputs/mde_checkpoint_features_full_swarm_20260529/feature_surface_index.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/mde_uncheatable_token_features_300m_20260530"
TEXT_SURFACE = "raw_text_loss_features"
RAW_ARTIFACT = "scored_documents.parquet"
DATASET_PREFIX = "uncheatable_eval/"
DEFAULT_REFERENCE_RUN = "baseline_proportional"
LOG2E = float(np.log2(np.e))
SELECTED_TOKENS_FILE = "selected_tokens.parquet"
TOKEN_SHARD_DIR = "token_nll_shards"
DOCUMENT_SHARD_DIR = "document_loss_shards"
RUN_MANIFEST_DIR = "run_manifests"
SHARD_MANIFEST_FILE = "shard_manifest.csv"
SUMMARY_FILE = "summary.json"


@dataclass(frozen=True)
class TokenRecord:
    """One selected token coordinate shared across checkpoint rows."""

    token_key: str
    dataset_name: str
    request_id: str
    token_index: int
    token_bytes: int
    hash_value: int


@dataclass(frozen=True)
class ReverseTokenSortKey:
    """Reverse ordering so heap root stores the worst selected token."""

    hash_value: int
    token_key: str

    def __lt__(self, other: "ReverseTokenSortKey") -> bool:
        return (self.hash_value, self.token_key) > (other.hash_value, other.token_key)


@dataclass(frozen=True)
class SelectTokenSketchConfig:
    """Executor config for deterministic reference-token sketch selection."""

    output_path: str
    feature_index: str
    reference_run: str = DEFAULT_REFERENCE_RUN
    sample_tokens_per_dataset: int = 4096
    dataset_prefix: str = DATASET_PREFIX
    batch_size: int = 1
    allow_gcs_read: bool = False


@dataclass(frozen=True)
class ExtractRunFeaturesConfig:
    """Executor config for one run's uncheatable token/document features."""

    output_path: str
    feature_index: str
    selected_tokens_path: str
    run_name: str
    dataset_prefix: str = DATASET_PREFIX
    batch_size: int = 1
    allow_gcs_read: bool = False
    progress_every_batches: int = 250


@dataclass(frozen=True)
class CollectShardedFeaturesConfig:
    """Executor config for collecting sharded extraction manifests."""

    output_path: str
    feature_index: str
    selected_tokens_path: str
    run_output_paths: dict[str, str]
    reference_run: str = DEFAULT_REFERENCE_RUN
    sample_tokens_per_dataset: int = 4096
    dataset_prefix: str = DATASET_PREFIX


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-index", default=str(DEFAULT_FEATURE_INDEX))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--reference-run", default=DEFAULT_REFERENCE_RUN)
    parser.add_argument("--sample-tokens-per-dataset", type=int, default=4096)
    parser.add_argument("--dataset-prefix", default=DATASET_PREFIX)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--run-name", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--allow-gcs-read", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def stable_hash(value: str) -> int:
    """Return a deterministic 64-bit hash for token sampling."""
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def selected_surface_rows(feature_index: str, run_names: Iterable[str], max_runs: int | None) -> pd.DataFrame:
    """Load raw-text feature index rows in stable run-name order."""
    frame = pd.read_csv(feature_index)
    required = {"run_name", "surface", "artifact_name", "artifact_uri"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Feature index {feature_index} is missing columns: {missing}")
    rows = frame.loc[
        frame["surface"].eq(TEXT_SURFACE) & frame["artifact_name"].eq(RAW_ARTIFACT),
        ["run_name", "artifact_uri", "size_bytes"],
    ].copy()
    rows = rows.sort_values("run_name").reset_index(drop=True)
    requested = set(run_names)
    if requested:
        rows = rows.loc[rows["run_name"].isin(requested)].copy()
        missing_runs = sorted(requested - set(rows["run_name"]))
        if missing_runs:
            raise ValueError(f"Requested run names missing from raw-text feature index: {missing_runs}")
    if max_runs is not None:
        rows = rows.head(max_runs).copy()
    if rows.empty:
        raise ValueError("No raw-text feature rows selected")
    return rows.reset_index(drop=True)


def guard_remote_reads(rows: pd.DataFrame, allow_gcs_read: bool) -> None:
    """Prevent accidental large local reads from GCS."""
    has_gcs = rows["artifact_uri"].astype(str).str.startswith("gs://").any()
    if has_gcs and not allow_gcs_read:
        raise ValueError(
            "Selected raw-text artifacts are on gs://. This extraction can read large inputs; "
            "rerun with --allow-gcs-read only inside the intended region-local job."
        )


def join_path_or_uri(base: str, name: str) -> str:
    """Join a local path or fsspec URI with one path component."""
    if "://" in base:
        return f"{base.rstrip('/')}/{name}"
    return str(Path(base) / name)


def safe_run_filename(run_name: str) -> str:
    """Return a filename-safe representation of a run name."""
    return run_name.replace("/", "_").replace(":", "_")


def token_schema() -> pa.Schema:
    """Return the selected-token feature output schema."""
    return pa.schema(
        [
            ("run_name", pa.string()),
            ("token_key", pa.string()),
            ("dataset_name", pa.string()),
            ("token_nll", pa.float64()),
            ("token_bytes", pa.int64()),
        ]
    )


def document_schema() -> pa.Schema:
    """Return the document aggregate output schema."""
    return pa.schema(
        [
            ("run_name", pa.string()),
            ("document_key", pa.string()),
            ("dataset_name", pa.string()),
            ("document_nll", pa.float64()),
            ("document_bytes", pa.int64()),
            ("document_bpb", pa.float64()),
        ]
    )


def write_json(path: str, payload: object) -> None:
    """Write JSON to a local path or fsspec URI."""
    with fsspec.open(path, "wt") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path: str) -> dict:
    """Read JSON from a local path or fsspec URI."""
    with fsspec.open(path, "rt") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def scored_document_columns() -> list[str]:
    """Return the raw score columns needed for token/document loss extraction."""
    return [
        "request_id",
        "dataset_name",
        "score_byte_start",
        "score_byte_end",
        "per_byte_loss",
        "token_byte_starts",
        "token_byte_ends",
    ]


def iter_scored_document_batches(uri: str, dataset_prefix: str, batch_size: int) -> Iterable[pd.DataFrame]:
    """Yield filtered scored-document batches without materializing a full score file."""
    with fsspec.open(uri, "rb") as handle:
        parquet = pq.ParquetFile(handle)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=scored_document_columns()):
            frame = batch.to_pandas()
            mask = frame["dataset_name"].astype(str).str.startswith(dataset_prefix)
            filtered = frame.loc[mask].reset_index(drop=True)
            if not filtered.empty:
                yield filtered


def iter_token_losses(row: pd.Series) -> Iterable[tuple[int, int, float]]:
    """Yield ``(token_index, token_bytes, token_nll)`` for the scored span."""
    per_byte_loss = np.asarray(row["per_byte_loss"], dtype=np.float64)
    starts = np.asarray(row["token_byte_starts"], dtype=np.int64)
    ends = np.asarray(row["token_byte_ends"], dtype=np.int64)
    score_start = int(row["score_byte_start"])
    score_end = int(row["score_byte_end"])
    if score_end <= score_start:
        return
    for token_index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        overlap_start = max(int(start), score_start)
        overlap_end = min(int(end), score_end)
        if overlap_end <= overlap_start:
            continue
        token_nll = float(per_byte_loss[overlap_start:overlap_end].sum())
        if not math.isfinite(token_nll):
            continue
        yield token_index, int(overlap_end - overlap_start), token_nll


def choose_token_sketch(
    reference_batches: Iterable[pd.DataFrame],
    *,
    sample_tokens_per_dataset: int,
) -> pd.DataFrame:
    """Choose a deterministic per-dataset token sketch from the reference run."""
    if sample_tokens_per_dataset <= 0:
        raise ValueError(f"sample_tokens_per_dataset must be positive, got {sample_tokens_per_dataset}")
    by_dataset: dict[str, list[tuple[ReverseTokenSortKey, TokenRecord]]] = {}
    for frame in tqdm(reference_batches, desc="select reference token batches"):
        for _, row in frame.iterrows():
            request_id = str(row["request_id"])
            dataset_name = str(row["dataset_name"])
            for token_index, token_bytes, _token_nll in iter_token_losses(row):
                token_key = f"{request_id}:{token_index}"
                hash_value = stable_hash(token_key)
                heap = by_dataset.setdefault(dataset_name, [])
                if len(heap) >= sample_tokens_per_dataset:
                    worst_record = heap[0][1]
                    if (hash_value, token_key) >= (worst_record.hash_value, worst_record.token_key):
                        continue
                record = TokenRecord(
                    token_key=token_key,
                    dataset_name=dataset_name,
                    request_id=request_id,
                    token_index=token_index,
                    token_bytes=token_bytes,
                    hash_value=hash_value,
                )
                entry = (ReverseTokenSortKey(hash_value, token_key), record)
                if len(heap) < sample_tokens_per_dataset:
                    heapq.heappush(heap, entry)
                else:
                    heapq.heapreplace(heap, entry)

    selected: list[TokenRecord] = []
    for dataset_name, heap in sorted(by_dataset.items()):
        records = sorted((record for _key, record in heap), key=lambda record: (record.hash_value, record.token_key))
        selected.extend(records[:sample_tokens_per_dataset])
        if len(records) < sample_tokens_per_dataset:
            print(
                f"warning: dataset {dataset_name} has only {len(records)} scored tokens; "
                f"requested {sample_tokens_per_dataset}"
            )

    if not selected:
        raise ValueError("Reference run produced no selected uncheatable tokens")
    return pd.DataFrame.from_records([record.__dict__ for record in selected])


def document_rows(run_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Build exact per-document score-span aggregates."""
    rows = []
    for _, row in frame.iterrows():
        score_start = int(row["score_byte_start"])
        score_end = int(row["score_byte_end"])
        num_bytes = max(0, score_end - score_start)
        if num_bytes <= 0:
            continue
        per_byte_loss = np.asarray(row["per_byte_loss"], dtype=np.float64)
        nll = float(per_byte_loss[score_start:score_end].sum())
        rows.append(
            {
                "run_name": run_name,
                "document_key": str(row["request_id"]),
                "dataset_name": str(row["dataset_name"]),
                "document_nll": nll,
                "document_bytes": num_bytes,
                "document_bpb": nll * LOG2E / num_bytes,
            }
        )
    return pd.DataFrame.from_records(rows)


def token_rows(run_name: str, frame: pd.DataFrame, selected_tokens: pd.DataFrame) -> pd.DataFrame:
    """Build selected token NLL rows for one checkpoint run."""
    selected = set(selected_tokens["token_key"].astype(str))
    token_bytes_by_key = dict(zip(selected_tokens["token_key"], selected_tokens["token_bytes"], strict=True))
    dataset_by_key = dict(zip(selected_tokens["token_key"], selected_tokens["dataset_name"], strict=True))
    rows = []
    for _, row in frame.iterrows():
        request_id = str(row["request_id"])
        for token_index, token_bytes, token_nll in iter_token_losses(row):
            token_key = f"{request_id}:{token_index}"
            if token_key not in selected:
                continue
            expected_bytes = int(token_bytes_by_key[token_key])
            if token_bytes != expected_bytes:
                raise ValueError(f"Token byte mismatch for {token_key}: {token_bytes} != {expected_bytes}")
            rows.append(
                {
                    "run_name": run_name,
                    "token_key": token_key,
                    "dataset_name": str(dataset_by_key[token_key]),
                    "token_nll": token_nll,
                    "token_bytes": expected_bytes,
                }
            )
    return pd.DataFrame.from_records(rows)


def writer_for(path: str, schema: pa.Schema) -> pq.ParquetWriter:
    """Create a Parquet writer for local or fsspec-supported output paths."""
    fs, _, paths = fsspec.get_fs_token_paths(path)
    if len(paths) != 1:
        raise ValueError(f"Expected one output path, got {paths}")
    parent = os.path.dirname(paths[0])
    if parent:
        fs.makedirs(parent, exist_ok=True)
    handle = fs.open(paths[0], "wb")
    return pq.ParquetWriter(handle, schema=schema)


def write_table(path: str, frame: pd.DataFrame) -> None:
    """Write a single DataFrame to Parquet."""
    table = pa.Table.from_pandas(frame, preserve_index=False)
    with fsspec.open(path, "wb") as handle:
        pq.write_table(table, handle)


def select_token_sketch(config: SelectTokenSketchConfig) -> None:
    """Select and persist a deterministic reference-token sketch."""
    rows = selected_surface_rows(config.feature_index, [config.reference_run], None)
    guard_remote_reads(rows, config.allow_gcs_read)
    print(
        json.dumps(
            {
                "event": "select_token_sketch_start",
                "reference_run": config.reference_run,
                "artifact_uri": str(rows.iloc[0]["artifact_uri"]),
                "sample_tokens_per_dataset": config.sample_tokens_per_dataset,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    started = time.time()
    selected_tokens = choose_token_sketch(
        iter_scored_document_batches(
            str(rows.iloc[0]["artifact_uri"]),
            config.dataset_prefix,
            config.batch_size,
        ),
        sample_tokens_per_dataset=config.sample_tokens_per_dataset,
    )
    output_path = config.output_path.rstrip("/")
    write_table(join_path_or_uri(output_path, SELECTED_TOKENS_FILE), selected_tokens)
    summary = {
        "dataset_prefix": config.dataset_prefix,
        "elapsed_seconds": time.time() - started,
        "reference_run": config.reference_run,
        "selected_tokens": len(selected_tokens),
        "selected_token_datasets": selected_tokens.groupby("dataset_name").size().astype(int).to_dict(),
        "selected_tokens_path": join_path_or_uri(output_path, SELECTED_TOKENS_FILE),
    }
    write_json(join_path_or_uri(output_path, SUMMARY_FILE), summary)
    print(json.dumps({"event": "select_token_sketch_finish", **summary}, sort_keys=True), flush=True)


def extract_one_run(
    *,
    run_name: str,
    artifact_uri: str,
    selected_tokens: pd.DataFrame,
    output_path: str,
    dataset_prefix: str,
    batch_size: int,
    progress_every_batches: int,
) -> dict[str, object]:
    """Extract sharded token/document features for one checkpoint run."""
    started = time.time()
    output_path = output_path.rstrip("/")
    run_file = safe_run_filename(run_name)
    token_path = join_path_or_uri(join_path_or_uri(output_path, TOKEN_SHARD_DIR), f"{run_file}.parquet")
    document_path = join_path_or_uri(join_path_or_uri(output_path, DOCUMENT_SHARD_DIR), f"{run_file}.parquet")
    manifest_path = join_path_or_uri(join_path_or_uri(output_path, RUN_MANIFEST_DIR), f"{run_file}.json")
    print(
        json.dumps(
            {
                "event": "extract_run_start",
                "run_name": run_name,
                "artifact_uri": artifact_uri,
                "token_path": token_path,
                "document_path": document_path,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    document_count = 0
    selected_tokens_found = 0
    batch_count = 0
    token_writer = writer_for(token_path, token_schema())
    document_writer = writer_for(document_path, document_schema())
    try:
        for frame in iter_scored_document_batches(artifact_uri, dataset_prefix, batch_size):
            batch_count += 1
            docs = document_rows(run_name, frame)
            toks = token_rows(run_name, frame, selected_tokens)
            document_count += len(docs)
            selected_tokens_found += len(toks)
            if not toks.empty:
                token_writer.write_table(pa.Table.from_pandas(toks, schema=token_schema(), preserve_index=False))
            if not docs.empty:
                document_writer.write_table(pa.Table.from_pandas(docs, schema=document_schema(), preserve_index=False))
            if progress_every_batches > 0 and batch_count % progress_every_batches == 0:
                print(
                    json.dumps(
                        {
                            "event": "extract_run_progress",
                            "run_name": run_name,
                            "batches": batch_count,
                            "documents": document_count,
                            "selected_tokens_found": selected_tokens_found,
                            "elapsed_seconds": time.time() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        token_writer.close()
        document_writer.close()

    missing_selected_tokens = int(len(selected_tokens) - selected_tokens_found)
    payload = {
        "run_name": run_name,
        "artifact_uri": artifact_uri,
        "batches": batch_count,
        "documents": document_count,
        "selected_tokens_found": selected_tokens_found,
        "missing_selected_tokens": missing_selected_tokens,
        "elapsed_seconds": time.time() - started,
        "token_path": token_path,
        "document_path": document_path,
        "manifest_path": manifest_path,
    }
    write_json(manifest_path, payload)
    print(json.dumps({"event": "extract_run_finish", **payload}, sort_keys=True), flush=True)
    return payload


def extract_run_features(config: ExtractRunFeaturesConfig) -> None:
    """Executor entrypoint for one run's sharded feature extraction."""
    rows = selected_surface_rows(config.feature_index, [config.run_name], None)
    guard_remote_reads(rows, config.allow_gcs_read)
    selected_tokens = pd.read_parquet(config.selected_tokens_path)
    extract_one_run(
        run_name=config.run_name,
        artifact_uri=str(rows.iloc[0]["artifact_uri"]),
        selected_tokens=selected_tokens,
        output_path=config.output_path,
        dataset_prefix=config.dataset_prefix,
        batch_size=config.batch_size,
        progress_every_batches=config.progress_every_batches,
    )


def collect_sharded_features(config: CollectShardedFeaturesConfig) -> None:
    """Collect shard manifests and write a lightweight dataset summary."""
    output_path = config.output_path.rstrip("/")
    manifest_rows = []
    for run_name, run_output_path in sorted(config.run_output_paths.items()):
        manifest_path = join_path_or_uri(
            join_path_or_uri(run_output_path.rstrip("/"), RUN_MANIFEST_DIR),
            f"{safe_run_filename(run_name)}.json",
        )
        manifest_rows.append(read_json(manifest_path))
    manifest = pd.DataFrame.from_records(manifest_rows)
    manifest_path = join_path_or_uri(output_path, SHARD_MANIFEST_FILE)
    with fsspec.open(manifest_path, "wt") as handle:
        manifest.to_csv(handle, index=False)
    summary = {
        "dataset_prefix": config.dataset_prefix,
        "feature_index": config.feature_index,
        "reference_run": config.reference_run,
        "sample_tokens_per_dataset": config.sample_tokens_per_dataset,
        "selected_runs": len(manifest),
        "selected_tokens_path": config.selected_tokens_path,
        "shard_manifest_path": manifest_path,
        "token_shard_paths": manifest["token_path"].astype(str).tolist(),
        "document_shard_paths": manifest["document_path"].astype(str).tolist(),
        "semantics": (
            "Sharded token NLLs are checkpoint-observed gold-token losses on a deterministic "
            "uncheatable token sketch. They must be used through queryable train-fold interpolation "
            "or another mixture-to-feature model; using a held-out checkpoint's own token losses as "
            "predictors leaks the target."
        ),
    }
    write_json(join_path_or_uri(output_path, SUMMARY_FILE), summary)
    print(json.dumps({"event": "collect_sharded_features_finish", **summary}, sort_keys=True), flush=True)


def main() -> None:
    """Run extraction."""
    args = parse_args()
    rows = selected_surface_rows(args.feature_index, args.run_name, args.max_runs)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "selected_runs": len(rows),
                    "total_source_size_bytes": int(rows["size_bytes"].fillna(0).sum()),
                    "first_runs": rows["run_name"].head(10).tolist(),
                    "output_dir": args.output_dir,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    guard_remote_reads(rows, args.allow_gcs_read)

    reference = rows.loc[rows["run_name"].eq(args.reference_run)]
    if reference.empty:
        raise ValueError(f"Reference run {args.reference_run!r} is not selected")
    selected_tokens = choose_token_sketch(
        iter_scored_document_batches(
            str(reference.iloc[0]["artifact_uri"]),
            args.dataset_prefix,
            args.batch_size,
        ),
        sample_tokens_per_dataset=args.sample_tokens_per_dataset,
    )

    output_dir = args.output_dir.rstrip("/")
    write_table(os.path.join(output_dir, SELECTED_TOKENS_FILE), selected_tokens)

    token_output_schema = token_schema()
    document_output_schema = document_schema()

    token_path = os.path.join(output_dir, "uncheatable_token_nll_long.parquet")
    document_path = os.path.join(output_dir, "uncheatable_document_loss_long.parquet")
    token_writer = writer_for(token_path, token_output_schema)
    document_writer = writer_for(document_path, document_output_schema)

    processed = []
    try:
        for _, source in tqdm(rows.iterrows(), total=len(rows), desc="extract runs"):
            run_name = str(source["run_name"])
            document_count = 0
            selected_tokens_found = 0
            for frame in iter_scored_document_batches(
                str(source["artifact_uri"]),
                args.dataset_prefix,
                args.batch_size,
            ):
                docs = document_rows(run_name, frame)
                toks = token_rows(run_name, frame, selected_tokens)
                document_count += len(docs)
                selected_tokens_found += len(toks)
                if not toks.empty:
                    token_writer.write_table(
                        pa.Table.from_pandas(toks, schema=token_output_schema, preserve_index=False)
                    )
                if not docs.empty:
                    document_writer.write_table(
                        pa.Table.from_pandas(docs, schema=document_output_schema, preserve_index=False)
                    )
            if selected_tokens_found != len(selected_tokens):
                missing = len(selected_tokens) - selected_tokens_found
                print(f"warning: {run_name} missing {missing} selected tokens")
            processed.append(
                {
                    "run_name": run_name,
                    "documents": document_count,
                    "selected_tokens_found": selected_tokens_found,
                    "source_uri": str(source["artifact_uri"]),
                }
            )
    finally:
        token_writer.close()
        document_writer.close()

    summary = {
        "feature_index": str(args.feature_index),
        "reference_run": args.reference_run,
        "sample_tokens_per_dataset": args.sample_tokens_per_dataset,
        "dataset_prefix": args.dataset_prefix,
        "selected_runs": len(rows),
        "selected_tokens": len(selected_tokens),
        "selected_token_datasets": selected_tokens.groupby("dataset_name").size().astype(int).to_dict(),
        "token_path": token_path,
        "document_path": document_path,
        "processed": processed,
        "semantics": (
            "Token NLLs are checkpoint-observed gold-token losses on a deterministic uncheatable token sketch. "
            "They must be used through queryable train-fold interpolation or another mixture-to-feature model; "
            "using a held-out checkpoint's own token losses as predictors leaks the target."
        ),
    }
    with fsspec.open(os.path.join(output_dir, SUMMARY_FILE), "wt") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
