# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a native sparse normalization and PCA reference for GSE81682.

The input is the byte-checked output of prepare_singlecell_qc.py. The original
cell and feature identities, sorting gates, and read-count QC fields are kept.
"""

import argparse
import gc
import gzip
import hashlib
import json
import resource
import time
from decimal import ROUND_HALF_EVEN, Decimal
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse
from scipy.io import mmread
from threadpoolctl import threadpool_limits

from experiments.post_training.bio_tasks.h5ad_verifier import canonical_count_sha256
from experiments.post_training.bio_tasks.real_data import source_catalog

PROTOCOL = Path(__file__).with_suffix(".json")
CELL_FIELDS = (
    "geo_accession",
    "sorting_gate",
    "all_counts",
    "ercc_counts",
    "endogenous_counts",
    "detected_endogenous",
    "keep",
    "matrix_column",
)
FEATURE_FIELDS = ("feature_type", "retained_cell_count", "retained_read_count", "keep", "matrix_row")
ORIGINAL_SOURCE_FILES = ("cells.tsv", "features.tsv", "matrix.mtx.gz")


def file_hash(path: Path) -> dict[str, int | str]:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"sha256": digest, "bytes": path.stat().st_size}


def write_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index_label="id", na_rep="NA", float_format="%.17g", lineterminator="\n")


def reference_rows(frame: pd.DataFrame, integer_fields: tuple[str, ...] = ()) -> dict[str, dict]:
    rows = {}
    for identifier, values in frame.iterrows():
        row = {}
        for name, value in values.items():
            if pd.isna(value):
                row[name] = None
            elif name in integer_fields:
                row[name] = int(value)
            elif isinstance(value, np.generic):
                row[name] = value.item()
            else:
                row[name] = value
        rows[str(identifier)] = row
    return rows


def rounded_three(value: float) -> float:
    return float(Decimal.from_float(value).quantize(Decimal("0.001"), rounding=ROUND_HALF_EVEN))


def pca_admission(
    data: AnnData, scores: np.ndarray, loadings: np.ndarray, eigenvalues: np.ndarray, variance_ratio: np.ndarray
) -> dict[str, float | bool]:
    """Check the exported axes against the same centered feature matrix."""
    cells = data.n_obs
    mean = np.asarray(data.X.mean(axis=0)).ravel()
    projected = np.asarray(data.X @ loadings) - mean @ loadings
    projection_error = float(np.max(np.abs(scores - projected)) / max(1.0, float(np.max(np.abs(scores)))))
    orthonormal_error = float(np.max(np.abs(loadings.T @ loadings - np.eye(loadings.shape[1]))))
    covariance = scores.T @ scores / (cells - 1)
    covariance_error = float(np.max(np.abs(covariance - np.diag(eigenvalues))) / max(1.0, float(eigenvalues[0])))
    squared = np.asarray(data.X.power(2).sum(axis=0)).ravel()
    total_variance = float(np.sum(squared - cells * mean**2) / (cells - 1))
    if total_variance <= 0:
        raise ValueError("Selected-gene total variance is nonpositive")
    ratio_error = float(np.max(np.abs(variance_ratio - eigenvalues / total_variance)))
    threshold = 1e-7
    return {
        "score_projection_relative_error": projection_error,
        "loading_orthonormal_max_error": orthonormal_error,
        "score_covariance_relative_error": covariance_error,
        "explained_variance_fraction_max_error": ratio_error,
        "total_selected_gene_variance": total_variance,
        "tolerance": threshold,
        "passed": max(projection_error, orthonormal_error, covariance_error, ratio_error) <= threshold,
    }


def checked_qc(qc_reference: Path, protocol: dict) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Read the QC-filtered count matrix and preserve its declared ordering."""
    for name, expected in protocol["qc_reference_sha256"].items():
        actual = file_hash(qc_reference / name)["sha256"]
        if actual != expected:
            raise ValueError(f"Changed QC reference artifact: {name}")
    cells = pd.read_csv(qc_reference / "cell_qc.tsv", sep="\t", index_col="id")
    features = pd.read_csv(qc_reference / "feature_qc.tsv", sep="\t", index_col="id")
    if not cells.index.is_unique or not features.index.is_unique:
        raise ValueError("QC identities must be unique")
    if set(CELL_FIELDS) - set(cells) or set(FEATURE_FIELDS) - set(features):
        raise ValueError("QC reference columns are incomplete")
    policy = protocol["qc_policy"]
    cell_keep = (
        (cells.endogenous_counts >= policy["minimum_endogenous_counts"])
        & (cells.detected_endogenous >= policy["minimum_detected_endogenous"])
        & (
            cells.ercc_counts * policy["ercc_fraction_denominator"]
            <= cells.all_counts * policy["ercc_fraction_numerator"]
        )
    )
    feature_keep = (features.feature_type == "endogenous") & (
        features.retained_cell_count >= policy["minimum_retained_cells_per_gene"]
    )
    if not np.array_equal(cells.keep.to_numpy(), cell_keep.to_numpy().astype(int)):
        raise ValueError("QC cell decisions differ from the frozen policy")
    if not np.array_equal(features.keep.to_numpy(), feature_keep.to_numpy().astype(int)):
        raise ValueError("QC feature decisions differ from the frozen policy")
    if not set(cells.sorting_gate) <= {"HSPC", "LT-HSC", "Prog"}:
        raise ValueError("Unexpected sorting gate")
    kept_cells = cells.loc[cell_keep].sort_values("matrix_column")
    kept_features = features.loc[feature_keep].sort_values("matrix_row")
    if not np.array_equal(kept_cells.matrix_column.to_numpy(), np.arange(1, len(kept_cells) + 1)):
        raise ValueError("QC cell matrix columns are not contiguous")
    if not np.array_equal(kept_features.matrix_row.to_numpy(), np.arange(1, len(kept_features) + 1)):
        raise ValueError("QC feature matrix rows are not contiguous")
    if (cells.loc[~cell_keep, "matrix_column"] != 0).any() or (features.loc[~feature_keep, "matrix_row"] != 0).any():
        raise ValueError("Excluded identities have nonzero matrix positions")
    with threadpool_limits(limits=1):
        counts = mmread(qc_reference / "filtered_counts.mtx.gz", spmatrix=True).T.tocsr()
    expected = protocol["expected_qc_matrix"]
    if counts.shape != (expected["cells"], expected["features"]) or counts.nnz != expected["nonzeros"]:
        raise ValueError("QC matrix shape or number of positive entries changed")
    if len(kept_cells) != counts.shape[0] or len(kept_features) != counts.shape[1]:
        raise ValueError("QC identities disagree with matrix dimensions")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts.data <= 0):
        raise ValueError("QC matrix must contain positive integer read counts")
    counts.sum_duplicates()
    counts.sort_indices()
    if int(counts.sum()) != expected["read_counts"]:
        raise ValueError("QC matrix read-count total changed")
    return counts, kept_cells, kept_features


def prepare(qc_reference: Path, output: Path, protocol_path: Path = PROTOCOL) -> None:
    """Emit the count-preserving H5AD, native representation and full tables."""
    started = time.monotonic()
    protocol = json.loads(protocol_path.read_text())
    for name, required in protocol["required_package_versions"].items():
        if version(name) != required:
            raise ValueError(f"Unexpected native package version: {name}")
    output.mkdir(parents=True, exist_ok=False)
    counts, cells, features = checked_qc(qc_reference, protocol)
    cell_ids = cells.index.astype(str)
    feature_ids = features.index.astype(str)
    input_counts_sha256 = canonical_count_sha256(
        cell_ids.tolist(), feature_ids.tolist(), counts.indptr, counts.indices, counts.data
    )
    input_depth = np.asarray(counts.sum(axis=1)).ravel().astype(np.int64)
    input_detected = np.asarray(counts.getnnz(axis=1)).ravel().astype(np.int64)
    prevalence = np.asarray(counts.getnnz(axis=0)).ravel().astype(np.int64)
    gene_read_sums = np.asarray(counts.sum(axis=0)).ravel().astype(np.int64)
    singleton_percentage = 100.0 * float(np.count_nonzero(counts.data == 1)) / counts.nnz

    cell_info = pd.DataFrame(index=cell_ids)
    cell_info["geo_accession"] = cells.geo_accession.to_numpy()
    cell_info["sorting_gate"] = cells.sorting_gate.to_numpy()
    cell_info["original_endogenous_reads"] = cells.endogenous_counts.to_numpy(dtype=np.int64)
    cell_info["input_filtered_reads"] = input_depth
    cell_info["input_detected_genes"] = input_detected
    feature_info = pd.DataFrame(index=feature_ids)
    feature_info["feature_type"] = features.feature_type.to_numpy()
    feature_info["input_prevalence_cells"] = prevalence
    feature_info["input_read_sum"] = gene_read_sums
    source = AnnData(counts, obs=cell_info, var=feature_info)
    source.layers["counts"] = counts.copy()
    source.write_h5ad(output / "input.h5ad", compression="gzip", compression_opts=1)
    del source
    gc.collect()

    minimum_cells = protocol["minimum_cells_per_representation_gene"]
    eligible = prevalence >= minimum_cells
    eligible_ids = feature_ids[eligible]
    eligible_counts = counts[:, eligible].tocsr()
    eligible_counts.sort_indices()
    eligible_counts_sha256 = canonical_count_sha256(
        cell_ids.tolist(), eligible_ids.tolist(), eligible_counts.indptr, eligible_counts.indices, eligible_counts.data
    )
    eligible_depth = np.asarray(eligible_counts.sum(axis=1)).ravel().astype(np.int64)
    if len(eligible_ids) <= protocol["hvg"]["n_top_genes"] or np.any(eligible_depth == 0):
        raise ValueError("No usable eligible-gene normalization universe")
    normalized = AnnData(
        eligible_counts.astype(np.float64), obs=cell_info.copy(), var=feature_info.loc[eligible_ids].copy()
    )
    normalized.layers["counts"] = eligible_counts.copy()
    normalized.obs["eligible_gene_reads"] = eligible_depth
    target = protocol["normalization"]["target_sum"]
    sc.pp.normalize_total(
        normalized,
        target_sum=target,
        exclude_highly_expressed=protocol["normalization"]["exclude_highly_expressed"],
    )
    scaled_totals = np.asarray(normalized.X.sum(axis=1)).ravel()
    if not np.allclose(scaled_totals, target, rtol=1e-10, atol=1e-8):
        raise ValueError("Scanpy normalization did not reach the declared target sum")
    sc.pp.log1p(normalized)
    if not sparse.isspmatrix_csr(normalized.X) or not np.all(np.isfinite(normalized.X.data)):
        raise ValueError("Log-normalized expression must remain finite and sparse")

    hvg_policy = protocol["hvg"]
    sc.pp.highly_variable_genes(
        normalized,
        flavor=hvg_policy["flavor"],
        n_top_genes=hvg_policy["n_top_genes"],
        n_bins=hvg_policy["n_bins"],
        subset=False,
        inplace=True,
    )
    metrics = normalized.var[["means", "dispersions", "dispersions_norm"]].replace([np.inf, -np.inf], np.nan)
    normalized.var[["means", "dispersions", "dispersions_norm"]] = metrics
    ranked = metrics.loc[np.isfinite(metrics.dispersions_norm)].copy()
    ranked["feature_id"] = ranked.index
    ranked = ranked.sort_values(["dispersions_norm", "feature_id"], ascending=[False, True], kind="mergesort")
    if len(ranked) < hvg_policy["n_top_genes"]:
        raise ValueError("Too few genes with finite normalized dispersion")
    selected_set = set(ranked.index[: hvg_policy["n_top_genes"]])
    selected_ids = normalized.var.index[normalized.var.index.isin(selected_set)]
    normalized.var["highly_variable"] = normalized.var.index.isin(selected_ids)
    if int(normalized.var.highly_variable.sum()) != hvg_policy["n_top_genes"]:
        raise ValueError("HVG selection did not retain exactly the declared number")

    pca_policy = protocol["pca"]
    pca_data = normalized[:, selected_ids].copy()
    with threadpool_limits(limits=1):
        sc.tl.pca(
            pca_data,
            n_comps=pca_policy["audit_components"],
            zero_center=pca_policy["zero_center"],
            svd_solver=pca_policy["svd_solver"],
            random_state=pca_policy["random_state"],
            dtype=pca_policy["dtype"],
        )
    scores = np.asarray(pca_data.obsm["X_pca"])
    loadings = np.asarray(pca_data.varm["PCs"])
    eigenvalues = np.asarray(pca_data.uns["pca"]["variance"])
    variance_ratio = np.asarray(pca_data.uns["pca"]["variance_ratio"])
    public_components = pca_policy["n_components"]
    if scores.shape != (len(cells), pca_policy["audit_components"]) or loadings.shape != (
        len(selected_ids),
        pca_policy["audit_components"],
    ):
        raise ValueError("Scanpy PCA shape changed")
    if not all(np.all(np.isfinite(values)) for values in (scores, loadings, eigenvalues, variance_ratio)):
        raise ValueError("Scanpy PCA returned nonfinite values")
    if np.any(eigenvalues <= 0) or np.any(variance_ratio <= 0):
        raise ValueError("Scanpy PCA returned nonpositive variance")
    relative_gaps = (eigenvalues[:-1] - eigenvalues[1:]) / np.maximum(eigenvalues[:-1], 1e-12)
    minimum_gap = float(np.min(relative_gaps[:public_components]))
    pca_checks = pca_admission(pca_data, scores, loadings, eigenvalues, variance_ratio)

    depth = np.log1p(cell_info.original_endogenous_reads.to_numpy(dtype=np.float64))
    if np.std(depth) == 0:
        raise ValueError("Original endogenous depth has no variation")
    correlations = np.array([np.corrcoef(scores[:, i], depth)[0, 1] for i in range(5)])
    if not np.all(np.isfinite(correlations)):
        raise ValueError("PCA depth correlations are nonfinite")

    cell_depth = cell_info.copy()
    cell_depth["eligible_gene_reads"] = eligible_depth
    cell_depth["normalization_factor"] = eligible_depth / target
    cell_depth["log_original_depth"] = depth
    write_table(cell_depth, output / "cell_depth.tsv")
    hvg = pd.DataFrame(index=feature_ids)
    hvg["input_prevalence_cells"] = prevalence
    hvg["input_read_sum"] = gene_read_sums
    hvg["eligible"] = eligible.astype(int)
    hvg["mean"] = metrics.means.reindex(feature_ids)
    hvg["dispersion"] = metrics.dispersions.reindex(feature_ids)
    hvg["normalized_dispersion"] = metrics.dispersions_norm.reindex(feature_ids)
    hvg["rank"] = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index).reindex(feature_ids)
    hvg["selected"] = hvg.index.isin(selected_ids).astype(int)
    write_table(hvg, output / "gene_hvg.tsv")
    pc_names = [f"PC{i}" for i in range(1, public_components + 1)]
    score_table = pd.DataFrame(scores[:, :public_components], index=cell_ids, columns=pc_names)
    score_table.insert(0, "original_endogenous_reads", cell_info.original_endogenous_reads.to_numpy())
    score_table.insert(0, "sorting_gate", cell_info.sorting_gate.to_numpy())
    write_table(score_table, output / "pca_scores.tsv")
    loading_table = pd.DataFrame(loadings[:, :public_components], index=selected_ids, columns=pc_names)
    write_table(loading_table, output / "pca_loadings.tsv")
    variance = pd.DataFrame(
        {"eigenvalue": eigenvalues[:public_components], "variance_fraction": variance_ratio[:public_components]},
        index=pd.Index(pc_names, name="id"),
    )
    write_table(variance, output / "pca_variance.tsv")
    top_loading_index = np.argmax(np.abs(loadings[:, :5]), axis=0)
    diagnostics = pd.DataFrame(
        {
            "top_abs_loading": np.abs(loadings[top_loading_index, np.arange(5)]),
            "top_loading_gene": selected_ids[top_loading_index],
            "depth_correlation": correlations,
            "abs_depth_correlation": np.abs(correlations),
            "n_cells": len(cells),
        },
        index=pd.Index(pc_names[:5], name="id"),
    )
    write_table(diagnostics, output / "diagnostics.tsv")
    normalized.uns["representation_protocol"] = {
        "minimum_cells_per_gene": minimum_cells,
        "normalization": protocol["normalization"],
        "hvg": protocol["hvg"],
        "pca": protocol["pca"],
    }
    normalized.write_h5ad(output / "normalized.h5ad", compression="gzip", compression_opts=1)

    summary = {
        "median_input_reads_per_cell": float(np.median(input_depth)),
        "pct_positive_counts_equal_one": singleton_percentage,
        "median_hvg_prevalence": float(np.median(prevalence[eligible][normalized.var.highly_variable.to_numpy()])),
        "pc1_top_abs_load": rounded_three(float(np.max(np.abs(loadings[:, 0])))),
        "max_top5_depth_corr": rounded_three(float(np.max(np.abs(correlations)))),
    }
    (output / "answer.json").write_text(json.dumps([{"id": "study", **summary}], allow_nan=False) + "\n")
    task_reference = {
        "answer": summary,
        "tables": {
            "cell_depth.tsv": reference_rows(
                cell_depth,
                ("original_endogenous_reads", "input_filtered_reads", "input_detected_genes", "eligible_gene_reads"),
            ),
            "gene_hvg.tsv": reference_rows(
                hvg,
                ("input_prevalence_cells", "input_read_sum", "eligible", "rank", "selected"),
            ),
            "pca_scores.tsv": reference_rows(score_table, ("original_endogenous_reads",)),
            "pca_loadings.tsv": reference_rows(loading_table),
            "pca_variance.tsv": reference_rows(variance),
            "diagnostics.tsv": reference_rows(diagnostics, ("n_cells",)),
        },
        "h5ad": {
            "cell_ids": cell_ids.tolist(),
            "feature_ids": eligible_ids.tolist(),
            "nonzeros": eligible_counts.nnz,
            "counts_sha256": eligible_counts_sha256,
            "eligible_gene_reads": eligible_depth.tolist(),
            "target_sum": float(target),
            "atol": 1e-8,
            "rtol": 1e-7,
            "max_bytes": 512 * 1024 * 1024,
            "max_decoded_bytes": 1024 * 1024 * 1024,
            "obs_metadata": {
                "geo_accession": cell_info.geo_accession.tolist(),
                "sorting_gate": cell_info.sorting_gate.tolist(),
                "original_endogenous_reads": cell_info.original_endogenous_reads.astype(int).tolist(),
                "input_filtered_reads": input_depth.tolist(),
                "input_detected_genes": input_detected.tolist(),
                "eligible_gene_reads": eligible_depth.tolist(),
            },
            "var_metadata": {
                "feature_type": normalized.var.feature_type.tolist(),
                "input_prevalence_cells": normalized.var.input_prevalence_cells.astype(int).tolist(),
                "input_read_sum": normalized.var.input_read_sum.astype(int).tolist(),
            },
        },
        "pca_signs": {
            "score_table": "pca_scores.tsv",
            "loading_table": "pca_loadings.tsv",
            "components": pc_names,
            "correlation_table": "diagnostics.tsv",
            "correlation_column": "depth_correlation",
        },
    }
    with (output / "task-reference.json.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write((json.dumps(task_reference, allow_nan=False, separators=(",", ":")) + "\n").encode())
    source_record = source_catalog()[protocol["source"]]
    input_manifest = {
        "source": protocol["source"],
        "lineage": source_record["lineage"],
        "landing_page": source_record["landing_page"],
        "citation": source_record["citation"],
        "redistribution_review": source_record["redistribution_review"],
        "original_source_file_sha256": {
            name: source_record["file_inputs"][name]["sha256"] for name in ORIGINAL_SOURCE_FILES
        },
        "qc_reference_sha256": protocol["qc_reference_sha256"],
        "protocol": protocol,
        "input_h5ad": file_hash(output / "input.h5ad"),
        "input_counts_sha256": input_counts_sha256,
        "labels": "Original broad sorting gates; no donor, mitochondrial or fine cell-type labels inferred.",
    }
    (output / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2) + "\n")
    artifacts = {path.name: file_hash(path) for path in sorted(output.iterdir()) if path.is_file()}
    admitted = minimum_gap >= pca_policy["minimum_adjacent_relative_eigen_gap"] and pca_checks["passed"]
    reference = {
        "source": protocol["source"],
        "status": "native-reference-passed" if admitted else "pca-spectrum-not-admitted",
        "protocol": protocol,
        "protocol_sha256": file_hash(protocol_path)["sha256"],
        "qc_reference_sha256": protocol["qc_reference_sha256"],
        "package_versions": {
            name: version(name) for name in ("scanpy", "anndata", "numpy", "pandas", "scipy", "scikit-learn", "h5py")
        },
        "summary": summary,
        "input_matrix": {
            "cells": len(cells),
            "features": len(features),
            "nonzeros": counts.nnz,
            "read_counts": int(counts.sum()),
            "input_h5ad_sha256": file_hash(output / "input.h5ad")["sha256"],
            "counts_sha256": input_counts_sha256,
        },
        "eligible_count_matrix": {
            "cells": len(cells),
            "features": int(eligible.sum()),
            "nonzeros": eligible_counts.nnz,
            "read_counts": int(eligible_counts.sum()),
            "counts_sha256": eligible_counts_sha256,
            "eligible_gene_reads": eligible_depth.tolist(),
            "target_sum": target,
        },
        "eligible_features": int(eligible.sum()),
        "selected_hvgs": len(selected_ids),
        "schemas": {
            "input.h5ad": {"cells": len(cells), "features": len(features), "x": "CSR integer read counts"},
            "normalized.h5ad": {
                "cells": len(cells),
                "features": int(eligible.sum()),
                "x": "CSR float64 log1p(CP10k)",
                "layers/counts": "CSR integer read counts",
            },
            "cell_depth.tsv": list(cell_depth.columns),
            "gene_hvg.tsv": list(hvg.columns),
            "pca_scores.tsv": list(score_table.columns),
            "pca_loadings.tsv": list(loading_table.columns),
            "pca_variance.tsv": list(variance.columns),
            "diagnostics.tsv": list(diagnostics.columns),
        },
        "pca_audit": {
            "audit_component_31_eigenvalue": float(eigenvalues[public_components]),
            "adjacent_relative_eigen_gaps_pc1_to_pc31": relative_gaps[:public_components].tolist(),
            "minimum_adjacent_relative_gap": minimum_gap,
            "admitted": admitted,
            "matrix_consistency": pca_checks,
        },
        "artifacts": artifacts,
        "runtime_seconds": time.monotonic() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limits": (
            "Smart-seq2 read counts, not UMIs. PCA is descriptive; broad sorting gates are not "
            "donor or fine cell-type identities. No UMAP, clustering, depth regression or mitochondrial inference."
        ),
    }
    (output / "reference.json").write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")
    if not admitted:
        raise ValueError("PCA reference failed its eigenvalue-gap or matrix-consistency admission check")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    prepare(args.qc_reference, args.output, args.protocol)


if __name__ == "__main__":
    main()
