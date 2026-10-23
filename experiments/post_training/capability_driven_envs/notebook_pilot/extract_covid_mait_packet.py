# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the compact authoring packet for the COVID MAIT V-gene task.

The full notebook and TCR MuData object are inputs to this curator-side script.
They are not copied into the packet. The solver receives donor-by-cell-type
counts; Taskforge receives those counts, a brief, narrowly scoped private
notebook evidence, and provenance.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mudata as md
import pandas as pd
import scirpy as ir

NOTEBOOK_CELLS = (11, 12, 13)
NOTEBOOK_PATH = "jupyter-book/air_repertoire/clonotype.ipynb"
NOTEBOOK_REVISION = "dd6eb86b30385cc3c5b5ddb2d8b63945b18dc1a5"
SOURCE_ARTIFACT = {
    "key": "air_repertoire/stephenson2021_tcr_qc.h5mu",
    "uid": "A54ugwVUlFz0PV5n0002",
    "lamindb_hash": "ZmNj0JwG7GQbwGvdFgvqVM",
    "size_bytes": 743651887,
    "uri": "s3://lamin-eu-central-1/VPwcjx3CDAa2/.lamindb/A54ugwVUlFz0PV5n0002.h5mu",
}
EXPECTED_FRACTIONS = {
    "CD4": 0.004050,
    "CD8": 0.025099,
    "Lymph_prolif": 0.036866,
    "MAIT": 0.691238,
    "Treg": 0.007282,
    "gdT": 0.018349,
}


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write stable, human-readable JSON."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def notebook_evidence(notebook: Path) -> dict[str, Any]:
    """Extract only the three notebook cells relevant to the task."""
    document = json.loads(notebook.read_text())
    cells = []
    for index in NOTEBOOK_CELLS:
        cell = document["cells"][index]
        outputs = []
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            text = output.get("text") or data.get("text/plain")
            if text:
                outputs.append("".join(text))
        cells.append(
            {
                "cell_index": index,
                "cell_type": cell["cell_type"],
                "source": "".join(cell.get("source", [])),
                "text_outputs": outputs,
            }
        )
    return {
        "notebook_path": NOTEBOOK_PATH,
        "notebook_revision": NOTEBOOK_REVISION,
        "cells": cells,
    }


def extract_counts(source: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract donor-by-cell-type TRAV1-2 counts and private reference summaries."""
    mdata = md.read_h5mu(source, backed="r")
    with ir.get.airr_context(mdata, "v_call"):
        observations = mdata.obs[["gex:patient_id", "gex:cell_type_coarse", "VJ_1_v_call"]].copy()

    observations.columns = ["source_donor_id", "cell_type_coarse", "primary_vj_v_call"]
    if observations[["source_donor_id", "cell_type_coarse"]].isna().any().any():
        raise ValueError("source contains missing donor or coarse cell-type labels")

    donor_ids = sorted(observations["source_donor_id"].astype(str).unique())
    donor_map = {source_id: f"donor_{index:03d}" for index, source_id in enumerate(donor_ids, start=1)}
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
    for column in ("n_cells", "n_primary_vj_called", "n_trav1_2"):
        grouped[column] = grouped[column].astype(int)
    grouped["trav1_2_fraction_all_cells"] = grouped["n_trav1_2"] / grouped["n_cells"]
    grouped["trav1_2_fraction_called"] = grouped["n_trav1_2"] / grouped["n_primary_vj_called"].where(
        grouped["n_primary_vj_called"].ne(0)
    )

    overall = (
        grouped.groupby("cell_type_coarse", observed=True, sort=True)[
            ["n_cells", "n_primary_vj_called", "n_trav1_2"]
        ]
        .sum()
        .reset_index()
    )
    overall["trav1_2_fraction_all_cells"] = overall["n_trav1_2"] / overall["n_cells"]
    overall["trav1_2_fraction_called"] = overall["n_trav1_2"] / overall["n_primary_vj_called"]
    observed_fractions = dict(zip(overall["cell_type_coarse"], overall["trav1_2_fraction_all_cells"], strict=True))
    for label, expected in EXPECTED_FRACTIONS.items():
        observed = float(observed_fractions[label])
        if abs(observed - expected) > 5e-7:
            raise ValueError(f"{label} fraction {observed} does not reproduce notebook value {expected}")

    donor_mait = grouped[grouped["cell_type_coarse"].eq("MAIT")].copy()
    reference = {
        "source_cell_count": int(len(observations)),
        "pseudonymous_donor_count": len(donor_ids),
        "solver_table_rows": int(len(grouped)),
        "overall_by_cell_type": overall.to_dict(orient="records"),
        "mait_donor_fraction_summary": {
            "donors_with_mait_cells": int(len(donor_mait)),
            "median_fraction_all_cells": float(donor_mait["trav1_2_fraction_all_cells"].median()),
            "minimum_fraction_all_cells": float(donor_mait["trav1_2_fraction_all_cells"].min()),
            "maximum_fraction_all_cells": float(donor_mait["trav1_2_fraction_all_cells"].max()),
        },
        "reference_interpretation": (
            "TRAV1-2 is strongly enriched in the transcriptomically labelled MAIT population "
            "(about 69% overall) versus a few percent or less in the other coarse T-cell labels. "
            "This independently supports, but does not prove, the annotation or antigen specificity "
            "of every labelled cell. Missing primary VJ calls and donor heterogeneity must be reported."
        ),
    }
    solver_counts = grouped[
        ["donor_id", "cell_type_coarse", "n_cells", "n_primary_vj_called", "n_trav1_2"]
    ].copy()
    return solver_counts, reference


def packet_brief() -> dict[str, Any]:
    """Return the human-authored task brief supplied to Taskforge."""
    return {
        "seed_id": "covid-mait-gene-usage",
        "notebook_id": "covid-immune-repertoire",
        "split_group": "covid-immune-repertoire",
        "question_family": "annotation_validation",
        "goal": "Evaluate whether receptor V-gene usage independently supports the transcriptomic MAIT-cell annotation.",
        "expected_deliverable": (
            "A reproducible quantitative comparison of TRAV1-2 usage across coarse T-cell labels, "
            "including donor variation, denominator choices, and a calibrated biological conclusion."
        ),
        "avoid": [
            "revealing the approximately 69 percent result in the solver prompt",
            "claiming that one receptor gene proves every cell annotation or antigen specificity",
            "hiding missing primary VJ calls",
            "giving the solver the source notebook or private reference summary",
        ],
        "solver_resource": "solver/v_gene_counts_by_donor_and_cell_type.csv",
        "private_evidence": [
            "private/reference_summary.json",
            "private/notebook_evidence.json",
        ],
        "status": "example_unapproved",
    }


def data_dictionary() -> str:
    """Return the solver-visible table documentation."""
    return """# V-gene count table\n\nEach row is one pseudonymous donor and coarse T-cell label observed in the source cohort.\n\n- `donor_id`: deterministic pseudonym local to this task.\n- `cell_type_coarse`: transcriptome-derived coarse cell label.\n- `n_cells`: cells in this donor/label group, including missing primary VJ calls.\n- `n_primary_vj_called`: cells with a non-missing primary VJ V-gene call.\n- `n_trav1_2`: cells whose primary VJ V-gene call is exactly `TRAV1-2`.\n\nThe table contains sufficient counts for pooled and donor-aware comparisons. Fractions and conclusions are intentionally not supplied. It does not contain the source notebook, receptor sequences, gene-expression matrix, clinical metadata, or original donor identifiers.\n"""


def build_packet(source: Path, notebook: Path, output: Path) -> None:
    """Build a packet without copying either full source input into it."""
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    solver = output / "solver"
    private = output / "private"
    solver.mkdir(parents=True)
    private.mkdir()

    counts, reference = extract_counts(source)
    counts_path = solver / "v_gene_counts_by_donor_and_cell_type.csv"
    counts.to_csv(counts_path, index=False, float_format="%.9g")
    (solver / "README.md").write_text(data_dictionary())
    write_json(output / "brief.json", packet_brief())
    write_json(private / "reference_summary.json", reference)
    write_json(private / "notebook_evidence.json", notebook_evidence(notebook))

    provenance = {
        "source_artifact": {
            **SOURCE_ARTIFACT,
            "sha256": sha256_file(source),
        },
        "source_notebook": {
            "repository": "https://github.com/theislab/single-cell-best-practices",
            "revision": NOTEBOOK_REVISION,
            "path": NOTEBOOK_PATH,
            "sha256": sha256_file(notebook),
            "included_cells": list(NOTEBOOK_CELLS),
        },
        "extraction": {
            "script": Path(__file__).name,
            "script_sha256": sha256_file(Path(__file__)),
            "row_rule": "all rows in the post-QC TCR MuData object with donor and coarse cell-type labels",
            "grouping": ["pseudonymous donor", "coarse cell type"],
            "primary_chain_field": "VJ_1_v_call from scirpy airr_context(v_call)",
            "source_identifiers_removed": True,
        },
        "rights_status": "unreviewed_do_not_publish",
    }
    write_json(output / "provenance.json", provenance)

    files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        files.append(
            {
                "path": str(path.relative_to(output)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "visibility": "solver" if path.is_relative_to(solver) else "author_private",
            }
        )
    write_json(output / "manifest.json", {"packet_id": "covid-mait-gene-usage-v0", "files": files})


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned TCR .h5mu source object.")
    parser.add_argument("--notebook", type=Path, required=True, help="Pinned clonotype.ipynb source.")
    parser.add_argument("--output", type=Path, required=True, help="New packet directory.")
    return parser.parse_args()


def main() -> None:
    """Run the packet extraction."""
    args = parse_args()
    build_packet(args.source, args.notebook, args.output)


if __name__ == "__main__":
    main()
