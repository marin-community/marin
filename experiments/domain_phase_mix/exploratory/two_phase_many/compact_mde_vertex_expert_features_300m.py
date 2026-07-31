# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "gcsfs", "numpy", "pandas", "pyarrow", "tqdm"]
# ///
"""Compact 300M MDE vertex-expert feature surfaces into dense matrices.

The vertex-expert scoring graph emits one raw-text, teacher-forced, and MCQ
feature artifact per expert checkpoint.  This script converts those sharded
Parquet outputs into small aligned matrices suitable for local modeling.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from experiments.domain_phase_mix.exploratory.two_phase_many.compact_mde_uncheatable_token_matrix_300m import (
    DENSE_RUN_NAMES_FILE,
    join_path_or_uri,
    read_source_paths,
    write_json,
    write_numpy,
)
from experiments.domain_phase_mix.launch_300m_checkpoint_features_canary import (
    MCQ_REQUEST_FEATURES_PARQUET,
    MCQ_SURFACE,
    TEACHER_FORCED_REQUEST_FEATURES_PARQUET,
    TEACHER_FORCED_SURFACE,
    TEXT_FEATURE_SURFACE,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/mde_vertex_expert_dense_features_300m_20260531"
FEATURE_INDEX_FILE = "feature_surface_index.csv"
EXPERT_MANIFEST_FILE = "expert_manifest.csv"
RAW_TEXT_DOCUMENT_NLL_MATRIX_FILE = "raw_text_document_nll_matrix.npy"
RAW_TEXT_DOCUMENT_BPB_MATRIX_FILE = "raw_text_document_bpb_matrix.npy"
RAW_TEXT_DOCUMENT_METADATA_FILE = "raw_text_document_metadata.parquet"
RAW_TEXT_TOKEN_NLL_MATRIX_FILE = "raw_text_token_nll_matrix.npy"
RAW_TEXT_TOKEN_METADATA_FILE = "raw_text_token_metadata.parquet"
TEACHER_FORCED_REQUEST_MATRIX_FILE = "teacher_forced_request_matrix.npy"
TEACHER_FORCED_REQUEST_METADATA_FILE = "teacher_forced_request_metadata.parquet"
MCQ_CHOICE_MATRIX_FILE = "mcq_choice_matrix.npy"
MCQ_CHOICE_METADATA_FILE = "mcq_choice_metadata.parquet"
SUMMARY_FILE = "summary.json"
LOG2E = float(np.log2(np.e))


@dataclass(frozen=True)
class CompactMdeVertexFeaturesConfig:
    """Executor config for dense vertex-expert feature compaction."""

    output_path: str
    feature_index: str
    token_feature_dir: str | None = None
    dtype: str = "float32"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-index", required=True)
    parser.add_argument("--token-feature-dir")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    return parser.parse_args()


def write_parquet(path: str, frame: pd.DataFrame) -> None:
    """Write a parquet frame to local disk or object storage."""
    fs, _, paths = fsspec.get_fs_token_paths(path)
    if len(paths) != 1:
        raise ValueError(f"Expected one path, got {paths}")
    parent = str(Path(paths[0]).parent)
    if parent and parent != ".":
        fs.makedirs(parent, exist_ok=True)
    with fs.open(paths[0], "wb") as handle:
        frame.to_parquet(handle, index=False)


def write_csv(path: str, frame: pd.DataFrame) -> None:
    """Write CSV to local disk or object storage."""
    fs, _, paths = fsspec.get_fs_token_paths(path)
    if len(paths) != 1:
        raise ValueError(f"Expected one path, got {paths}")
    parent = str(Path(paths[0]).parent)
    if parent and parent != ".":
        fs.makedirs(parent, exist_ok=True)
    with fs.open(paths[0], "wt") as handle:
        frame.to_csv(handle, index=False)


def _surface_rows(feature_index: pd.DataFrame, surface: str, artifact_name: str) -> pd.DataFrame:
    rows = feature_index.loc[
        feature_index["surface"].eq(surface) & feature_index["artifact_name"].eq(artifact_name)
    ].copy()
    if rows.empty:
        raise ValueError(f"Feature index has no rows for surface={surface} artifact={artifact_name}")
    rows = rows.sort_values("run_order").reset_index(drop=True)
    if rows["run_name"].duplicated().any():
        duplicated = rows.loc[rows["run_name"].duplicated(), "run_name"].tolist()
        raise ValueError(f"Duplicate {surface} rows for run names: {duplicated[:10]}")
    return rows


def _document_frame(path: str, run_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with fsspec.open(path, "rb") as handle:
        parquet = pq.ParquetFile(handle)
        for batch in parquet.iter_batches(
            batch_size=32,
            columns=[
                "request_id",
                "dataset_name",
                "score_byte_start",
                "score_byte_end",
                "per_byte_loss",
            ],
        ):
            frame = batch.to_pandas()
            for _, row in frame.iterrows():
                score_start = int(row["score_byte_start"])
                score_end = int(row["score_byte_end"])
                document_bytes = score_end - score_start
                if document_bytes <= 0:
                    continue
                per_byte_loss = np.asarray(row["per_byte_loss"], dtype=np.float64)
                document_nll = float(per_byte_loss[score_start:score_end].sum())
                if not math.isfinite(document_nll):
                    continue
                request_id = str(row["request_id"])
                dataset_name = str(row["dataset_name"])
                rows.append(
                    {
                        "run_name": run_name,
                        "document_key": f"{dataset_name}::{request_id}",
                        "dataset_name": dataset_name,
                        "request_id": request_id,
                        "document_nll": document_nll,
                        "document_bytes": int(document_bytes),
                        "document_bpb": document_nll * LOG2E / document_bytes,
                    }
                )
    if not rows:
        raise ValueError(f"No raw-text document rows extracted from {path}")
    return pd.DataFrame.from_records(rows)


def _compact_raw_text_documents(rows: pd.DataFrame, output_dir: str, dtype: np.dtype) -> dict[str, object]:
    run_names = rows["run_name"].astype(str).tolist()
    reference = _document_frame(str(rows.iloc[0]["artifact_uri"]), run_names[0])
    metadata = reference[["document_key", "dataset_name", "request_id", "document_bytes"]].copy()
    document_keys = metadata["document_key"].astype(str).tolist()
    bytes_by_key = dict(zip(metadata["document_key"], metadata["document_bytes"], strict=True))
    nll_matrix = np.empty((len(run_names), len(document_keys)), dtype=dtype)
    bpb_matrix = np.empty_like(nll_matrix)

    for row_index, (_, row) in enumerate(tqdm(list(rows.iterrows()), desc="compact raw text documents")):
        frame = _document_frame(str(row["artifact_uri"]), str(row["run_name"]))
        aligned = frame.set_index("document_key").reindex(document_keys)
        if aligned["document_nll"].isna().any():
            missing = int(aligned["document_nll"].isna().sum())
            raise ValueError(f"{row['run_name']} is missing {missing} raw-text documents")
        observed_bytes = aligned["document_bytes"].astype(int).to_dict()
        for document_key, document_bytes in bytes_by_key.items():
            if observed_bytes[document_key] != int(document_bytes):
                raise ValueError(f"Document byte mismatch for {document_key} in {row['run_name']}")
        nll_values = aligned["document_nll"].to_numpy(dtype=np.float64)
        byte_values = aligned["document_bytes"].to_numpy(dtype=np.float64)
        nll_matrix[row_index] = nll_values.astype(dtype, copy=False)
        bpb_matrix[row_index] = (nll_values * LOG2E / byte_values).astype(dtype, copy=False)

    nll_path = join_path_or_uri(output_dir, RAW_TEXT_DOCUMENT_NLL_MATRIX_FILE)
    bpb_path = join_path_or_uri(output_dir, RAW_TEXT_DOCUMENT_BPB_MATRIX_FILE)
    metadata_path = join_path_or_uri(output_dir, RAW_TEXT_DOCUMENT_METADATA_FILE)
    write_numpy(nll_path, nll_matrix)
    write_numpy(bpb_path, bpb_matrix)
    write_parquet(metadata_path, metadata)
    return {
        "raw_text_document_nll_matrix_path": nll_path,
        "raw_text_document_bpb_matrix_path": bpb_path,
        "raw_text_document_metadata_path": metadata_path,
        "raw_text_document_shape": [int(nll_matrix.shape[0]), int(nll_matrix.shape[1])],
        "raw_text_document_size_bytes": int(nll_matrix.nbytes),
    }


def _request_key_columns(frame: pd.DataFrame) -> list[str]:
    for columns in (
        ["request_id", "metric_prefix"],
        ["request_id", "task_alias", "doc_id", "choice_idx"],
        ["request_id"],
    ):
        if set(columns).issubset(frame.columns):
            return columns
    raise ValueError(f"Cannot infer request key columns from {sorted(frame.columns)}")


def _compact_request_matrix(
    rows: pd.DataFrame,
    *,
    output_dir: str,
    matrix_file: str,
    metadata_file: str,
    progress_label: str,
    dtype: np.dtype,
) -> dict[str, object]:
    run_names = rows["run_name"].astype(str).tolist()
    reference = pd.read_parquet(str(rows.iloc[0]["artifact_uri"]))
    if "nll" not in reference.columns:
        raise ValueError(f"{rows.iloc[0]['artifact_uri']} is missing nll column")
    key_columns = _request_key_columns(reference)
    reference = reference.copy()
    reference["feature_key"] = reference[key_columns].astype(str).agg("::".join, axis=1)
    if reference["feature_key"].duplicated().any():
        raise ValueError(f"Reference request matrix has duplicate keys for {progress_label}")
    feature_keys = reference["feature_key"].astype(str).tolist()
    metadata_columns = [column for column in key_columns if column in reference.columns]
    metadata = reference[["feature_key", *metadata_columns]].copy()
    matrix = np.empty((len(run_names), len(feature_keys)), dtype=dtype)

    for row_index, (_, row) in enumerate(tqdm(list(rows.iterrows()), desc=progress_label)):
        frame = pd.read_parquet(str(row["artifact_uri"]))
        frame = frame.copy()
        frame["feature_key"] = frame[key_columns].astype(str).agg("::".join, axis=1)
        aligned = frame.set_index("feature_key").reindex(feature_keys)
        if aligned["nll"].isna().any():
            missing = int(aligned["nll"].isna().sum())
            raise ValueError(f"{row['run_name']} is missing {missing} request features for {progress_label}")
        matrix[row_index] = aligned["nll"].to_numpy(dtype=np.float64).astype(dtype, copy=False)

    matrix_path = join_path_or_uri(output_dir, matrix_file)
    metadata_path = join_path_or_uri(output_dir, metadata_file)
    write_numpy(matrix_path, matrix)
    write_parquet(metadata_path, metadata)
    return {
        f"{progress_label}_matrix_path": matrix_path,
        f"{progress_label}_metadata_path": metadata_path,
        f"{progress_label}_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        f"{progress_label}_size_bytes": int(matrix.nbytes),
    }


def _compact_token_matrix(token_feature_dir: str, output_dir: str, dtype: np.dtype) -> dict[str, object]:
    selected_tokens_path, shard_manifest_path, source_summary = read_source_paths(token_feature_dir)
    selected = pd.read_parquet(selected_tokens_path)
    token_metadata = selected[["token_key", "dataset_name", "request_id", "token_index", "token_bytes", "hash_value"]]
    token_keys = token_metadata["token_key"].astype(str).tolist()
    manifest = pd.read_csv(shard_manifest_path)
    run_names = manifest["run_name"].astype(str).tolist()
    matrix = np.empty((len(run_names), len(token_keys)), dtype=dtype)
    for row_index, (_, row) in enumerate(tqdm(list(manifest.iterrows()), desc="compact raw text tokens")):
        shard = pd.read_parquet(str(row["token_path"]), columns=["token_key", "token_nll"])
        aligned = shard.set_index("token_key").reindex(token_keys)
        if aligned["token_nll"].isna().any():
            missing = int(aligned["token_nll"].isna().sum())
            raise ValueError(f"{row['run_name']} is missing {missing} selected tokens")
        matrix[row_index] = aligned["token_nll"].to_numpy(dtype=np.float64).astype(dtype, copy=False)

    matrix_path = join_path_or_uri(output_dir, RAW_TEXT_TOKEN_NLL_MATRIX_FILE)
    run_names_path = join_path_or_uri(output_dir, DENSE_RUN_NAMES_FILE)
    metadata_path = join_path_or_uri(output_dir, RAW_TEXT_TOKEN_METADATA_FILE)
    write_numpy(matrix_path, matrix)
    write_csv(run_names_path, pd.DataFrame({"run_name": run_names}))
    write_parquet(metadata_path, token_metadata)
    return {
        "raw_text_token_nll_matrix_path": matrix_path,
        "raw_text_token_metadata_path": metadata_path,
        "raw_text_token_run_names_path": run_names_path,
        "raw_text_token_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "raw_text_token_size_bytes": int(matrix.nbytes),
        "raw_text_token_source_reference_run": source_summary.get("reference_run"),
    }


def compact_mde_vertex_features(config: CompactMdeVertexFeaturesConfig) -> None:
    """Compact vertex-expert feature surfaces into dense matrices."""
    output_dir = config.output_path.rstrip("/")
    dtype = np.dtype(config.dtype)
    feature_index = pd.read_csv(config.feature_index)
    required = {"run_name", "surface", "artifact_name", "artifact_uri", "run_order"}
    missing = sorted(required - set(feature_index.columns))
    if missing:
        raise ValueError(f"Feature index {config.feature_index} is missing columns: {missing}")
    feature_index = feature_index.sort_values(["run_order", "surface", "artifact_name"]).reset_index(drop=True)
    expert_manifest = (
        feature_index.sort_values("run_order")
        .drop_duplicates("run_name")[
            [
                "run_order",
                "run_name",
                "domain_name",
                "is_control",
                "train_tokens",
                "num_train_steps",
                "expected_checkpoint_step",
                "materialized_epochs",
                "checkpoint_path",
            ]
        ]
        .copy()
    )
    expert_manifest_path = join_path_or_uri(output_dir, EXPERT_MANIFEST_FILE)
    write_csv(expert_manifest_path, expert_manifest)

    raw_text_summary = _compact_raw_text_documents(
        _surface_rows(feature_index, TEXT_FEATURE_SURFACE, "scored_documents.parquet"),
        output_dir,
        dtype,
    )
    teacher_summary = _compact_request_matrix(
        _surface_rows(feature_index, TEACHER_FORCED_SURFACE, TEACHER_FORCED_REQUEST_FEATURES_PARQUET),
        output_dir=output_dir,
        matrix_file=TEACHER_FORCED_REQUEST_MATRIX_FILE,
        metadata_file=TEACHER_FORCED_REQUEST_METADATA_FILE,
        progress_label="teacher_forced_request",
        dtype=dtype,
    )
    mcq_summary = _compact_request_matrix(
        _surface_rows(feature_index, MCQ_SURFACE, MCQ_REQUEST_FEATURES_PARQUET),
        output_dir=output_dir,
        matrix_file=MCQ_CHOICE_MATRIX_FILE,
        metadata_file=MCQ_CHOICE_METADATA_FILE,
        progress_label="mcq_choice",
        dtype=dtype,
    )
    token_summary: dict[str, object] = {}
    if config.token_feature_dir:
        token_summary = _compact_token_matrix(config.token_feature_dir, output_dir, dtype)

    summary = {
        "format": "mde_vertex_expert_dense_features_v1",
        "feature_index": config.feature_index,
        "token_feature_dir": config.token_feature_dir,
        "expert_manifest_path": expert_manifest_path,
        "dtype": str(dtype),
        "num_experts": int(len(expert_manifest)),
        "semantics": (
            "Rows follow expert_manifest.csv order. Raw text primary features are NLL substrates; "
            "BPB matrices are diagnostic ratios derived after NLL aggregation."
        ),
        **raw_text_summary,
        **teacher_summary,
        **mcq_summary,
        **token_summary,
    }
    write_json(join_path_or_uri(output_dir, SUMMARY_FILE), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    """Run dense compaction from the command line."""
    args = parse_args()
    compact_mde_vertex_features(
        CompactMdeVertexFeaturesConfig(
            output_path=args.output_dir,
            feature_index=args.feature_index,
            token_feature_dir=args.token_feature_dir,
            dtype=args.dtype,
        )
    )


if __name__ == "__main__":
    main()
