# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run sparse normalization and one PCA fit from the supplied GSE81682 H5AD."""

import gc
import hashlib
import json
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path


def rounded_three(value: float) -> float:
    return float(Decimal.from_float(value).quantize(Decimal("0.001"), rounding=ROUND_HALF_EVEN))


def write_table(frame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index_label="id", na_rep="NA", float_format="%.17g", lineterminator="\n")


def solve_representation(inputs: Path, output: Path) -> list[dict]:
    """Return the summary while writing all native count and PCA artifacts."""
    # Scanpy is present only in this recipe's pinned native environment.
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    import scanpy as sc  # noqa: PLC0415
    from anndata import AnnData  # noqa: PLC0415
    from scipy import sparse  # noqa: PLC0415
    from threadpoolctl import threadpool_limits  # noqa: PLC0415

    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((inputs / "manifest.json").read_text())
    protocol = manifest["protocol"]
    with (inputs / "input.h5ad").open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != manifest["input_h5ad_sha256"]:
            raise ValueError("Changed observed GSE81682 input H5AD")
    data = sc.read_h5ad(inputs / "input.h5ad")
    counts = data.layers["counts"]
    expected = protocol["expected_qc_matrix"]
    if data.shape != (expected["cells"], expected["features"]) or counts.nnz != expected["nonzeros"]:
        raise ValueError("Input matrix does not match declared QC cohort")
    if not sparse.isspmatrix_csr(data.X) or not sparse.isspmatrix_csr(counts):
        raise ValueError("Input expression and count layer must be CSR")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts.data <= 0):
        raise ValueError("Count layer must contain positive integer reads")
    if not (
        np.array_equal(data.X.indptr, counts.indptr)
        and np.array_equal(data.X.indices, counts.indices)
        and np.array_equal(data.X.data, counts.data)
    ):
        raise ValueError("Input X and count layer must be identical")
    if int(counts.sum()) != expected["read_counts"]:
        raise ValueError("Input read-count total changed")
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError("Input cell and feature IDs must be unique")
    if set(data.obs.sorting_gate) - {"HSPC", "LT-HSC", "Prog"}:
        raise ValueError("Unexpected source sorting gate")

    cell_ids = data.obs_names.astype(str)
    feature_ids = data.var_names.astype(str)
    input_depth = np.asarray(counts.sum(axis=1)).ravel().astype(np.int64)
    input_detected = np.asarray(counts.getnnz(axis=1)).ravel().astype(np.int64)
    prevalence = np.asarray(counts.getnnz(axis=0)).ravel().astype(np.int64)
    gene_read_sums = np.asarray(counts.sum(axis=0)).ravel().astype(np.int64)
    singleton_percentage = 100.0 * float(np.count_nonzero(counts.data == 1)) / counts.nnz
    original_depth = data.obs.original_endogenous_reads.to_numpy(dtype=np.int64)
    cell_info = pd.DataFrame(index=cell_ids)
    cell_info["geo_accession"] = data.obs.geo_accession.to_numpy()
    cell_info["sorting_gate"] = data.obs.sorting_gate.to_numpy()
    cell_info["original_endogenous_reads"] = original_depth
    cell_info["input_filtered_reads"] = input_depth
    cell_info["input_detected_genes"] = input_detected
    feature_info = pd.DataFrame(index=feature_ids)
    feature_info["feature_type"] = data.var.feature_type.to_numpy()
    feature_info["input_prevalence_cells"] = prevalence
    feature_info["input_read_sum"] = gene_read_sums

    eligible = prevalence >= protocol["minimum_cells_per_representation_gene"]
    eligible_ids = feature_ids[eligible]
    eligible_counts = counts[:, eligible].tocsr()
    eligible_depth = np.asarray(eligible_counts.sum(axis=1)).ravel().astype(np.int64)
    if len(eligible_ids) <= protocol["hvg"]["n_top_genes"] or np.any(eligible_depth == 0):
        raise ValueError("No usable eligible-gene normalization universe")
    normalized = AnnData(
        eligible_counts.astype(np.float64), obs=cell_info.copy(), var=feature_info.loc[eligible_ids].copy()
    )
    normalized.layers["counts"] = eligible_counts.copy()
    normalized.obs["eligible_gene_reads"] = eligible_depth
    del data, counts
    gc.collect()
    target = protocol["normalization"]["target_sum"]
    sc.pp.normalize_total(
        normalized,
        target_sum=target,
        exclude_highly_expressed=protocol["normalization"]["exclude_highly_expressed"],
    )
    if not np.allclose(np.asarray(normalized.X.sum(axis=1)).ravel(), target, rtol=1e-10, atol=1e-8):
        raise ValueError("Scanpy normalization changed the declared target")
    sc.pp.log1p(normalized)
    if not sparse.isspmatrix_csr(normalized.X) or not np.all(np.isfinite(normalized.X.data)):
        raise ValueError("Log-normalized X must remain finite and sparse")

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
        raise ValueError("Too few finite native HVG scores")
    selected = set(ranked.index[: hvg_policy["n_top_genes"]])
    selected_ids = normalized.var.index[normalized.var.index.isin(selected)]
    normalized.var["highly_variable"] = normalized.var.index.isin(selected_ids)
    if len(selected_ids) != hvg_policy["n_top_genes"]:
        raise ValueError("HVG selection must have exactly 2000 genes")

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
    if scores.shape != (len(cell_ids), pca_policy["audit_components"]) or loadings.shape != (
        len(selected_ids),
        pca_policy["audit_components"],
    ):
        raise ValueError("Native PCA returned an unexpected shape")
    depth = np.log1p(original_depth.astype(np.float64))
    correlations = np.array([np.corrcoef(scores[:, i], depth)[0, 1] for i in range(5)])
    if not np.all(np.isfinite(correlations)):
        raise ValueError("Depth correlations are nonfinite")

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
    score_table.insert(0, "original_endogenous_reads", original_depth)
    score_table.insert(0, "sorting_gate", cell_info.sorting_gate.to_numpy())
    write_table(score_table, output / "pca_scores.tsv")
    write_table(
        pd.DataFrame(loadings[:, :public_components], index=selected_ids, columns=pc_names),
        output / "pca_loadings.tsv",
    )
    write_table(
        pd.DataFrame(
            {"eigenvalue": eigenvalues[:public_components], "variance_fraction": variance_ratio[:public_components]},
            index=pd.Index(pc_names, name="id"),
        ),
        output / "pca_variance.tsv",
    )
    top_loading_index = np.argmax(np.abs(loadings[:, :5]), axis=0)
    write_table(
        pd.DataFrame(
            {
                "top_abs_loading": np.abs(loadings[top_loading_index, np.arange(5)]),
                "top_loading_gene": selected_ids[top_loading_index],
                "depth_correlation": correlations,
                "abs_depth_correlation": np.abs(correlations),
                "n_cells": len(cell_ids),
            },
            index=pd.Index(pc_names[:5], name="id"),
        ),
        output / "diagnostics.tsv",
    )
    normalized.uns["representation_protocol"] = {
        "minimum_cells_per_gene": protocol["minimum_cells_per_representation_gene"],
        "normalization": protocol["normalization"],
        "hvg": protocol["hvg"],
        "pca": protocol["pca"],
    }
    normalized.write_h5ad(output / "normalized.h5ad", compression="gzip", compression_opts=1)
    return [
        {
            "id": "study",
            "median_input_reads_per_cell": float(np.median(input_depth)),
            "pct_positive_counts_equal_one": singleton_percentage,
            "median_hvg_prevalence": float(np.median(prevalence[eligible][normalized.var.highly_variable.to_numpy()])),
            "pc1_top_abs_load": rounded_three(float(np.max(np.abs(loadings[:, 0])))),
            "max_top5_depth_corr": rounded_three(float(np.max(np.abs(correlations)))),
        }
    ]
