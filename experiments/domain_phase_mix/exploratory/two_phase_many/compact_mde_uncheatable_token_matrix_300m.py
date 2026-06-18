# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "gcsfs", "numpy", "pandas", "pyarrow", "tqdm"]
# ///
"""Compact sharded uncheatable token losses into a dense matrix.

The sharded extraction format is retryable and auditable, but expensive for
iterative modeling because it stores ``(run_name, token_key, token_nll)`` rows
across hundreds of Parquet shards.  This script converts that format into a
dense numeric matrix plus side metadata:

* ``token_nll_matrix.npy`` with shape ``num_runs x num_tokens``;
* ``run_names.csv`` defining row order;
* ``token_metadata.parquet`` defining column order and token metadata;
* ``summary.json`` with provenance and shape information.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURE_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_mde_uncheatable_token_features_sharded_300m_20260530/"
    "pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_mde_uncheatable_token_features_sharded_300m_20260530/"
    "collect_sharded_uncheatable_token_features-676f23"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/mde_uncheatable_token_dense_matrix_300m_20260531"
DENSE_MATRIX_FILE = "token_nll_matrix.npy"
DENSE_RUN_NAMES_FILE = "run_names.csv"
DENSE_TOKEN_METADATA_FILE = "token_metadata.parquet"
SUMMARY_FILE = "summary.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    return parser.parse_args()


def join_path_or_uri(base: str, filename: str) -> str:
    """Join a local directory or URI with one path component."""
    if "://" in base:
        return f"{base.rstrip('/')}/{filename}"
    return str(Path(base) / filename)


def path_exists(path: str) -> bool:
    """Return whether a local path or fsspec URI exists."""
    fs, _, paths = fsspec.get_fs_token_paths(path)
    if len(paths) != 1:
        raise ValueError(f"Expected one path, got {paths}")
    return bool(fs.exists(paths[0]))


def prepare_output_dir(output_dir: str) -> None:
    """Create local output directories; object-store prefixes are implicit."""
    if "://" in output_dir:
        return
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def write_json(path: str, payload: object) -> None:
    """Write JSON to a local path or fsspec URI."""
    with fsspec.open(path, "wt") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_numpy(path: str, array: np.ndarray) -> None:
    """Write a NumPy array to a local path or fsspec URI."""
    with fsspec.open(path, "wb") as handle:
        np.save(handle, array, allow_pickle=False)


def read_source_paths(feature_dir: str) -> tuple[str, str, dict[str, object]]:
    """Resolve selected-token and shard-manifest paths from a feature dir."""
    summary_path = join_path_or_uri(feature_dir, SUMMARY_FILE)
    if not path_exists(summary_path):
        raise FileNotFoundError(f"Missing source summary: {summary_path}")
    with fsspec.open(summary_path, "rt") as handle:
        summary = json.load(handle)
    selected_tokens_path = str(summary.get("selected_tokens_path", join_path_or_uri(feature_dir, "selected_tokens.parquet")))
    shard_manifest_path = str(summary.get("shard_manifest_path", join_path_or_uri(feature_dir, "shard_manifest.csv")))
    return selected_tokens_path, shard_manifest_path, summary


def compact_matrix(*, feature_dir: str, output_dir: str, dtype: np.dtype) -> dict[str, object]:
    """Build and write a dense token-NLL matrix."""
    prepare_output_dir(output_dir)
    selected_tokens_path, shard_manifest_path, source_summary = read_source_paths(feature_dir)
    selected = pd.read_parquet(selected_tokens_path)
    token_metadata = selected[["token_key", "dataset_name", "request_id", "token_index", "token_bytes", "hash_value"]].copy()
    token_metadata["token_key"] = token_metadata["token_key"].astype(str)
    token_keys = token_metadata["token_key"].tolist()

    manifest = pd.read_csv(shard_manifest_path)
    run_names = manifest["run_name"].astype(str).tolist()
    matrix = np.empty((len(run_names), len(token_keys)), dtype=dtype)

    for row_idx, (_, row) in enumerate(tqdm(list(manifest.iterrows()), desc="compact token shards")):
        run_name = str(row["run_name"])
        token_path = str(row["token_path"])
        shard = pd.read_parquet(token_path, columns=["token_key", "token_nll"])
        values = shard.set_index("token_key").reindex(token_keys)["token_nll"].to_numpy(dtype=np.float64)
        if np.isnan(values).any():
            missing = int(np.isnan(values).sum())
            raise ValueError(f"Token shard for run_name={run_name} has {missing} missing selected tokens")
        matrix[row_idx] = values.astype(dtype, copy=False)

    matrix_path = join_path_or_uri(output_dir, DENSE_MATRIX_FILE)
    run_names_path = join_path_or_uri(output_dir, DENSE_RUN_NAMES_FILE)
    token_metadata_path = join_path_or_uri(output_dir, DENSE_TOKEN_METADATA_FILE)
    summary_path = join_path_or_uri(output_dir, SUMMARY_FILE)
    write_numpy(matrix_path, matrix)
    with fsspec.open(run_names_path, "wt") as handle:
        pd.DataFrame({"run_name": run_names}).to_csv(handle, index=False)
    token_metadata.to_parquet(token_metadata_path, index=False)

    summary = {
        "format": "dense_token_nll_matrix_v1",
        "source_feature_dir": feature_dir,
        "source_selected_tokens_path": selected_tokens_path,
        "source_shard_manifest_path": shard_manifest_path,
        "source_reference_run": source_summary.get("reference_run"),
        "source_dataset_prefix": source_summary.get("dataset_prefix"),
        "matrix_path": matrix_path,
        "run_names_path": run_names_path,
        "token_metadata_path": token_metadata_path,
        "dtype": str(matrix.dtype),
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "num_runs": int(matrix.shape[0]),
        "num_tokens": int(matrix.shape[1]),
        "size_bytes": int(matrix.nbytes),
        "semantics": (
            "Rows are run_names.csv order. Columns are token_metadata.parquet order. "
            "Values are checkpoint-observed gold-token negative log-likelihoods on "
            "the deterministic uncheatable token sketch."
        ),
    }
    write_json(summary_path, summary)
    return summary


def main() -> None:
    """Run compaction."""
    args = parse_args()
    summary = compact_matrix(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        dtype=np.dtype(args.dtype),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
