# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference cell QC and sparse-matrix filtering using Scanpy on observed counts."""

import argparse
import gzip
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.io import mmread, mmwrite
from threadpoolctl import threadpool_limits


def prepare(inputs: Path, output: Path) -> None:
    """Retain native package outputs and a complete independently checkable reference."""
    output.mkdir(parents=True, exist_ok=False)
    query = json.loads((inputs / "query.json").read_text())
    features = pd.read_csv(inputs / "features.tsv", sep="\t", index_col="id")
    cells = pd.read_csv(inputs / "cells.tsv", sep="\t", index_col="id")
    with threadpool_limits(limits=1):
        counts = mmread(inputs / "matrix.mtx.gz", spmatrix=True).T.tocsr()
    if counts.shape != (len(cells), len(features)) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("Source matrix and metadata do not agree")
    if not cells.index.is_unique or not features.index.is_unique or np.any(counts.data <= 0):
        raise ValueError("Source counts require distinct identities and positive sparse entries")
    data = AnnData(counts, obs=cells, var=features)
    data.var["ercc"] = data.var.feature_type == "ERCC"
    data.var["endogenous"] = data.var.feature_type == "endogenous"
    sc.pp.calculate_qc_metrics(data, qc_vars=["ercc", "endogenous"], percent_top=None, log1p=False, inplace=True)
    detected = np.asarray(data.X[:, data.var.endogenous.to_numpy()].getnnz(axis=1)).ravel()
    qc = data.obs[
        ["geo_accession", "sorting_gate", "total_counts", "total_counts_ercc", "total_counts_endogenous"]
    ].copy()
    qc.columns = ["geo_accession", "sorting_gate", "all_counts", "ercc_counts", "endogenous_counts"]
    qc["detected_endogenous"] = detected
    qc["ercc_fraction"] = qc.ercc_counts / qc.all_counts
    retained = (
        (qc.endogenous_counts >= query["minimum_endogenous_counts"])
        & (qc.detected_endogenous >= query["minimum_detected_endogenous"])
        & (qc.ercc_counts * query["ercc_fraction_denominator"] <= qc.all_counts * query["ercc_fraction_numerator"])
    )
    qc["keep"] = retained.astype(int)
    qc["matrix_column"] = 0
    qc.loc[retained, "matrix_column"] = np.arange(1, int(retained.sum()) + 1)
    qc.to_csv(output / "cell_qc.tsv", sep="\t", index_label="id", na_rep="NA")
    selected = data[retained, :].copy()
    gene_keep, cell_counts = sc.pp.filter_genes(
        selected, min_cells=query["minimum_retained_cells_per_gene"], inplace=False
    )
    gene_keep &= np.asarray(data.var.endogenous)
    feature_qc = data.var[["feature_type"]].copy()
    feature_qc["retained_cell_count"] = cell_counts.astype(int)
    feature_qc["retained_read_count"] = np.asarray(selected.X.sum(axis=0)).ravel().astype(np.int64)
    feature_qc["keep"] = gene_keep.astype(int)
    feature_qc["matrix_row"] = 0
    feature_qc.loc[gene_keep, "matrix_row"] = np.arange(1, int(gene_keep.sum()) + 1)
    feature_qc.to_csv(output / "feature_qc.tsv", sep="\t", index_label="id", na_rep="NA")
    filtered = selected[:, gene_keep].X.T.tocsr()
    filtered.sort_indices()
    matrix_path = output / "filtered_counts.mtx.gz"
    with matrix_path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as destination:
        with threadpool_limits(limits=1):
            mmwrite(destination, filtered, field="integer", symmetry="general")
    digest = hashlib.sha256()
    for row in range(filtered.shape[0]):
        for offset in range(filtered.indptr[row], filtered.indptr[row + 1]):
            digest.update(f"{row + 1} {filtered.indices[offset] + 1} {filtered.data[offset]}\n".encode())
    summary = {
        "input_cells": len(cells),
        "retained_cells": int(retained.sum()),
        "input_features": len(features),
        "retained_features": int(gene_keep.sum()),
        "output_nonzeros": filtered.nnz,
        "output_read_counts": int(filtered.sum()),
    }
    (output / "answer.json").write_text(json.dumps([{"id": "study", **summary}], allow_nan=False) + "\n")
    record = {
        "source": "GSE81682",
        "query": query,
        "summary": summary,
        "cell_qc": json.loads(qc.to_json(orient="index", double_precision=15)),
        "feature_qc": json.loads(feature_qc.to_json(orient="index")),
        "matrix": {
            "rows": filtered.shape[0],
            "columns": filtered.shape[1],
            "nonzeros": filtered.nnz,
            "sha256": digest.hexdigest(),
        },
        "input_counts": {"nonzeros": data.X.nnz, "total": int(data.X.sum())},
        "qc_distribution": json.loads(
            qc[["endogenous_counts", "detected_endogenous", "ercc_fraction"]].describe().to_json()
        ),
        "gate_retention": json.loads(qc.groupby("sorting_gate")["keep"].agg(["count", "sum"]).to_json(orient="index")),
        "versions": {"scanpy": version("scanpy"), "numpy": np.__version__, "pandas": pd.__version__},
    }
    (output / "reference.json.gz").write_bytes(gzip.compress((json.dumps(record, indent=2) + "\n").encode(), mtime=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.inputs, args.output)


if __name__ == "__main__":
    main()
