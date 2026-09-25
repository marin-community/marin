# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-shot source gate for the separately frozen CD14-only gene-length task."""

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import gate_gene_length_expression as base

SOURCE_SHA256 = {
    "counts": "f764402a7cfa05842d7f9a1b004b15e41c3ccedcaa004ba3f4cc9f7820f762af",
    "genes": "8efea9061252bf0e5c021fc6125e576cead0ad5621a337cb655fde6403820f33",
    "cells": "d5fb16e0a7ef1533127569989ec18bda38f41dc4b366bca0e8f2f24fa88acf8a",
    "gtf": "a52356765a41264e17a2076aff1abab703ec3ba239b78f88560756e85c169831",
}
MIN_TOTAL_COUNT = 10
MIN_ELIGIBLE_GENES = 5000


def association(groups: dict, gene_ids: list[str], spans: dict[str, int]) -> dict:
    """Apply the exact total-count filter and report separate CPM prevalence QC."""
    monocyte = {donor: groups[(donor, "monocyte")] for donor in base.DONORS}
    library_sizes = {donor: sum(monocyte[donor]) for donor in base.DONORS}
    if any(total <= 0 for total in library_sizes.values()):
        raise ValueError("Empty CD14 donor pseudobulk")
    x, y = [], []
    filter_counts = Counter()
    for index, gene_id in enumerate(gene_ids):
        if gene_id not in spans:
            continue
        filter_counts["joined_coding"] += 1
        observed_total = sum(monocyte[donor][index] for donor in base.DONORS)
        if observed_total < MIN_TOTAL_COUNT:
            continue
        filter_counts["total_count_pass"] += 1
        filter_counts["eligible"] += 1
        cpm = [1_000_000 * monocyte[donor][index] / library_sizes[donor] for donor in base.DONORS]
        if sum(value >= 1 for value in cpm) >= 2:
            filter_counts["cpm_prevalence_qc"] += 1
        x.append(float(spans[gene_id]))
        y.append(sum(cpm) / len(base.DONORS))
    result = {"filter_counts": dict(filter_counts), "donor_library_sizes": library_sizes}
    if len(x) < MIN_ELIGIBLE_GENES:
        return result
    result["pearson_r"] = base.pearson(x, y)
    result["n_genes"] = len(x)
    return result


def run(output: Path) -> dict:
    working = Path("/tmp/cd14-gene-length-source-gate")
    working.mkdir(exist_ok=False)
    report = {
        "source": "Parse Evercode WT Mini v3 four-donor PBMC",
        "license": "CC BY 4.0",
        "license_url": "https://www.parsebiosciences.com/datasets/performance-of-evercode-wt-mini-v3-in-human-pbmcs/",
        "scope": "bix-22-q4 CD14-only; no bix-22-q1 claim",
        "method_sha256": hashlib.sha256(Path("cd14_method.md").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reused_parser_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        "downloads": {},
        "gates": [],
    }
    try:
        for name in ("counts", "genes", "cells", "gtf"):
            downloaded = base.download(name, working)
            if downloaded["sha256"] != SOURCE_SHA256[name]:
                raise ValueError(f"Changed upstream source digest: {name}")
            report["downloads"][name] = downloaded
        gene_ids, marker_indices = base.read_genes(Path(report["downloads"]["genes"]["path"]))
        cells = base.read_cells(Path(report["downloads"]["cells"]["path"]))
        row_sums, markers = base.marker_counts(Path(report["downloads"]["counts"]["path"]), marker_indices)
        matching = sum(row_sums[index] == int(cell["tscp_count"]) for index, cell in enumerate(cells))
        base.gate(
            report,
            "matrix_metadata_row_order",
            matching,
            matching >= math.ceil(0.99 * base.EXPECTED_CELLS),
            "matching row sums >=99% of cells",
        )
        if not report["gates"][-1]["passed"]:
            return report
        assignments, rows = base.assign_cells(cells, row_sums, markers)
        base.write_label_qc(output, rows)
        counts = Counter((cell["sample"], label) for cell, label in zip(cells, assignments, strict=True))
        group_counts = {
            f"{donor}:{group}": counts[(donor, group)] for donor in base.DONORS for group in base.TARGET_GROUPS
        }
        monocyte_counts = {donor: group_counts[f"{donor}:monocyte"] for donor in base.DONORS}
        base.gate(
            report,
            "cd14_cells_per_donor",
            monocyte_counts,
            min(monocyte_counts.values()) >= 25,
            "each of four CD14 donor groups >=25 cells",
        )
        report["other_group_counts"] = group_counts
        unassigned = sum(label not in base.TARGET_GROUPS for label in assignments) / base.EXPECTED_CELLS
        base.gate(
            report,
            "unassigned_fraction",
            round(unassigned, 6),
            unassigned <= 0.65,
            "other and ambiguous fraction <=0.65",
        )
        report["assignment_reasons"] = dict(Counter(row["reason"] for row in rows))
        spans = base.read_gene_spans(Path(report["downloads"]["gtf"]["path"]))
        overlap = sum(gene_id in spans for gene_id in gene_ids)
        base.gate(
            report,
            "protein_coding_id_overlap",
            overlap,
            overlap >= 15000,
            "at least 15000 canonical protein-coding GTF IDs in Parse table",
        )
        if not all(item["passed"] for item in report["gates"]):
            return report
        groups = base.pseudobulks(Path(report["downloads"]["counts"]["path"]), cells, assignments)
        heldout = [row for row in base.held_out_qc(groups, marker_indices) if row["group"] == "monocyte"]
        report["held_out_cd14"] = heldout
        passing = sum(row["passed"] for row in heldout)
        base.gate(
            report,
            "cd14_held_out_enrichment",
            passing,
            passing >= 3,
            "CD14 CPM enriched >=1.5-fold over each other classified group in >=3 donors",
        )
        if not report["gates"][-1]["passed"]:
            return report
        report["endpoint_preview"] = association(groups, gene_ids, spans)
        eligible = report["endpoint_preview"]["filter_counts"].get("eligible", 0)
        base.gate(
            report,
            "eligible_gene_universe",
            eligible,
            eligible >= MIN_ELIGIBLE_GENES,
            "total monocyte UMI >=10 and protein-coding span join; >=5000 genes",
        )
        return report
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "source_gate.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> None:
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    report = run(output)
    if not all(item["passed"] for item in report["gates"]):
        raise SystemExit("Frozen CD14 source gate failed; no task registration")
    print(
        json.dumps(
            {
                "gate_passed": True,
                "source_hashes": {name: item["sha256"] for name, item in report["downloads"].items()},
                "cd14_cells_per_donor": next(
                    item["observed"] for item in report["gates"] if item["name"] == "cd14_cells_per_donor"
                ),
                "endpoint_preview": report["endpoint_preview"],
            }
        )
    )


if __name__ == "__main__":
    main()
