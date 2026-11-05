# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract a minimal donor-by-cell-type TRAV1-2 count table from TCR MuData.

This curator-side step is source-specific. It intentionally emits no notebook
conclusions, cell-level records, or original donor identifiers.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mudata as md
import pandas as pd
import scirpy as ir

COLUMNS = ("donor_id", "cell_type_coarse", "n_cells", "n_primary_vj_called", "n_trav1_2")
EXTRACTOR_VERSION = "stephenson-tcr-trav1-2-counts-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(source: Path) -> pd.DataFrame:
    """Return pseudonymous donor-level sufficient statistics."""
    mdata = md.read_h5mu(source, backed="r")
    with ir.get.airr_context(mdata, "v_call"):
        observations = mdata.obs[["gex:patient_id", "gex:cell_type_coarse", "VJ_1_v_call"]].copy()
    observations.columns = ["source_donor_id", "cell_type_coarse", "primary_vj_v_call"]
    if observations[["source_donor_id", "cell_type_coarse"]].isna().any().any():
        raise ValueError("source contains missing donor or coarse cell-type labels")

    source_ids = sorted(observations["source_donor_id"].astype(str).unique())
    donor_map = {source_id: f"donor_{index:03d}" for index, source_id in enumerate(source_ids, start=1)}
    observations["donor_id"] = observations["source_donor_id"].astype(str).map(donor_map)
    observations["has_primary_vj_call"] = observations["primary_vj_v_call"].notna() & (
        observations["primary_vj_v_call"].astype(str).str.len() > 0
    )
    observations["is_trav1_2"] = observations["primary_vj_v_call"].eq("TRAV1-2")

    grouped = (
        observations.groupby(["donor_id", "cell_type_coarse"], observed=True, sort=True)
        .agg(
            n_cells=("is_trav1_2", "size"),
            n_primary_vj_called=("has_primary_vj_call", "sum"),
            n_trav1_2=("is_trav1_2", "sum"),
        )
        .reset_index()
    )
    for column in COLUMNS[2:]:
        grouped[column] = grouped[column].astype(int)
    if not (
        (grouped["n_trav1_2"] <= grouped["n_primary_vj_called"]) & (grouped["n_primary_vj_called"] <= grouped["n_cells"])
    ).all():
        raise ValueError("extracted counts violate TRAV1-2 <= called <= cells")
    return grouped[list(COLUMNS)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    observed_sha256 = sha256_file(args.source)
    if observed_sha256 != args.expected_sha256:
        raise ValueError(f"source SHA-256 {observed_sha256} does not match {args.expected_sha256}")
    args.output.mkdir(parents=True, exist_ok=False)
    counts = extract(args.source)
    counts.to_csv(args.output / "repertoire_counts.csv", index=False)
    provenance = {
        "format": "covid-repertoire-counts-v1",
        "extractor": EXTRACTOR_VERSION,
        "source_sha256": observed_sha256,
        "rows": len(counts),
        "donors": int(counts["donor_id"].nunique()),
        "cell_types": sorted(counts["cell_type_coarse"].astype(str).unique()),
        "row_rule": "all post-QC TCR rows with donor and coarse cell-type labels",
        "primary_chain_field": "VJ_1_v_call from scirpy airr_context(v_call)",
        "source_identifiers_removed": True,
        "release_status": "unreviewed_do_not_publish",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(counts)} rows for {provenance['donors']} donors to {args.output}")


if __name__ == "__main__":
    main()
