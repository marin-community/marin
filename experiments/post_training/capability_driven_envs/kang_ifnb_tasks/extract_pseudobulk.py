# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract compact donor-paired pseudobulk counts from the pinned Kang AnnData artifact."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np

SOURCE_KEY = "conditions/differential_gene_expression.h5ad"
PANEL_GENES = (
    "ACTB",
    "GAPDH",
    "IFI6",
    "IFIT1",
    "IFIT2",
    "IFIT3",
    "IRF7",
    "ISG15",
    "MX1",
    "OAS1",
    "OAS2",
    "STAT1",
)
OUTPUT_COLUMNS = (
    "donor_id",
    "condition",
    "cell_type",
    "n_cells",
    "total_counts",
    "gene",
    "gene_counts",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def donor_id(source_id: str) -> str:
    """Return a stable campaign pseudonym without retaining the source label."""
    value = hashlib.sha256(f"kang-ifnb-pseudobulk-v1\0{source_id}".encode()).hexdigest()[:12]
    return f"donor_{value}"


def _integer_count(value: float, *, field: str) -> int:
    rounded = round(value)
    if not np.isfinite(value) or not np.isclose(value, rounded):
        raise ValueError(f"{field} is not a finite integer count: {value!r}")
    return int(rounded)


def extract(source: Path, expected_sha256: str) -> list[dict[str, str | int]]:
    """Verify and aggregate the cell-level count matrix into compact sufficient statistics."""
    actual_sha256 = sha256_file(source)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"source SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}")

    adata = ad.read_h5ad(source)
    required_obs = {"replicate", "label", "cell_type"}
    missing_obs = required_obs - set(adata.obs.columns)
    if missing_obs:
        raise ValueError(f"source is missing obs columns: {sorted(missing_obs)}")
    missing_genes = set(PANEL_GENES) - set(adata.var_names)
    if missing_genes:
        raise ValueError(f"source is missing panel genes: {sorted(missing_genes)}")
    conditions = set(adata.obs["label"].astype(str))
    if conditions != {"ctrl", "stim"}:
        raise ValueError(f"expected ctrl/stim labels, got {sorted(conditions)}")

    gene_indices = [adata.var_names.get_loc(gene) for gene in PANEL_GENES]
    rows: list[dict[str, str | int]] = []
    grouped = adata.obs.groupby(["replicate", "label", "cell_type"], observed=True, sort=True).indices
    for (replicate, condition, cell_type), indices in grouped.items():
        matrix = adata.X[indices]
        total_counts = _integer_count(float(matrix.sum()), field="total_counts")
        selected = matrix[:, gene_indices]
        gene_sums = np.asarray(selected.sum(axis=0)).reshape(-1)
        for gene, count in zip(PANEL_GENES, gene_sums, strict=True):
            rows.append(
                {
                    "donor_id": donor_id(str(replicate)),
                    "condition": str(condition),
                    "cell_type": str(cell_type),
                    "n_cells": len(indices),
                    "total_counts": total_counts,
                    "gene": gene,
                    "gene_counts": _integer_count(float(count), field=f"gene_counts[{gene}]"),
                }
            )

    rows.sort(key=lambda row: (str(row["donor_id"]), str(row["condition"]), str(row["cell_type"]), str(row["gene"])))
    return rows


def write_output(output: Path, rows: list[dict[str, str | int]], source: Path, source_sha256: str) -> None:
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "pseudobulk_counts.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    combinations = {(row["donor_id"], row["condition"], row["cell_type"]) for row in rows}
    provenance = {
        "format": "kang-ifnb-pseudobulk-v1",
        "status": "unreviewed_do_not_publish",
        "canonical_study": "Kang et al. 2018, GEO GSE96583",
        "convenience_source_key": SOURCE_KEY,
        "source_filename": source.name,
        "source_sha256": source_sha256,
        "output_sha256": sha256_file(csv_path),
        "panel_genes": list(PANEL_GENES),
        "rows": len(rows),
        "donors": len({row["donor_id"] for row in rows}),
        "sample_cell_type_groups": len(combinations),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = extract(args.source, args.expected_sha256)
    write_output(args.output, rows, args.source, args.expected_sha256)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
