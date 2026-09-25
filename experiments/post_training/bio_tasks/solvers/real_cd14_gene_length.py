# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Input-reading CD14 pseudobulk and gene-span association oracle."""

import csv
import gzip
import json
import math
from array import array
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

LINEAGE_MARKERS = {
    "T": ("CD3D", "CD3E", "TRAC", "CD2"),
    "B": ("MS4A1", "CD79A", "CD79B", "BANK1"),
    "monocyte": ("LYZ", "FCN1", "S100A8", "S100A9", "CTSS"),
    "NK": ("NKG7", "GNLY", "KLRD1", "FCGR3A"),
}
T_MARKERS = {
    "CD4": ("IL7R", "CCR7", "LTB", "MAL"),
    "CD8": ("CD8A", "GZMK", "CCL5", "GZMA"),
}
TARGET_GROUPS = ("CD4", "CD8", "monocyte", "B")
DONORS = ("Donor_1", "Donor_2", "Donor_3", "Donor_4")
CANONICAL_CHROMOSOMES = {str(index) for index in range(1, 23)} | {"X", "Y"}
EXPECTED_CELLS = 4861
EXPECTED_GENES = 62710
EXPECTED_COORDINATES = 10297684


def source_rows(path: Path) -> Iterator[tuple[int, int, int]]:
    """Stream the observed cell-by-gene MatrixMarket file, including legal zeros."""
    with gzip.open(path, "rt") as handle:
        if handle.readline().split() != ["%%MatrixMarket", "matrix", "coordinate", "integer", "general"]:
            raise ValueError("Unexpected Parse MatrixMarket header")
        while header := handle.readline():
            if not header.startswith("%"):
                break
        if tuple(map(int, header.split())) != (EXPECTED_CELLS, EXPECTED_GENES, EXPECTED_COORDINATES):
            raise ValueError("Unexpected Parse matrix dimensions")
        entries = 0
        for line in handle:
            row, column, value = map(int, line.split())
            if not (1 <= row <= EXPECTED_CELLS and 1 <= column <= EXPECTED_GENES and value >= 0):
                raise ValueError("Invalid observed count coordinate")
            entries += 1
            yield row - 1, column - 1, value
        if entries != EXPECTED_COORDINATES:
            raise ValueError("Incomplete observed count matrix")


def read_genes(path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["gene_id", "gene_name", "genome"]:
            raise ValueError("Unexpected Parse gene columns")
        genes = list(reader)
    if (
        len(genes) != EXPECTED_GENES
        or len({row["gene_id"] for row in genes}) != EXPECTED_GENES
        or any(row["genome"] != "hg38" for row in genes)
    ):
        raise ValueError("Unexpected Parse gene identities or genome")
    names = {marker for panel in (*LINEAGE_MARKERS.values(), *T_MARKERS.values()) for marker in panel}
    names.add("CD14")
    marker_indices = {}
    for marker in names:
        matches = [index for index, gene in enumerate(genes) if gene["gene_name"] == marker]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous marker identity: {marker}")
        marker_indices[marker] = matches[0]
    return genes, marker_indices


def read_cells(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected = [
            "bc_wells",
            "sample",
            "species",
            "gene_count",
            "tscp_count",
            "mread_count",
            "bc1_wind",
            "bc2_wind",
            "bc3_wind",
            "bc1_well",
            "bc2_well",
            "bc3_well",
        ]
        if reader.fieldnames != expected:
            raise ValueError("Unexpected Parse cell columns")
        cells = list(reader)
    if (
        len(cells) != EXPECTED_CELLS
        or len({row["bc_wells"] for row in cells}) != EXPECTED_CELLS
        or {row["sample"] for row in cells} != set(DONORS)
        or any(row["species"] != "hg38" for row in cells)
    ):
        raise ValueError("Unexpected Parse cell identities, donors or genome")
    return cells


def read_spans(path: Path) -> dict[str, int]:
    spans = {}
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["gene_id", "chromosome", "start", "end", "gene_biotype"]:
            raise ValueError("Unexpected Ensembl gene annotation columns")
        for row in reader:
            if row["chromosome"] not in CANONICAL_CHROMOSOMES or row["gene_biotype"] != "protein_coding":
                continue
            gene_id = row["gene_id"].split(".", 1)[0]
            if gene_id in spans:
                raise ValueError(f"Duplicate canonical coding gene: {gene_id}")
            start, end = int(row["start"]), int(row["end"])
            if start <= 0 or end < start:
                raise ValueError("Invalid GTF gene span")
            spans[gene_id] = end - start + 1
    return spans


def marker_data(path: Path, marker_indices: dict[str, int]) -> tuple[array, dict[str, array]]:
    row_sums = array("Q", [0]) * EXPECTED_CELLS
    counts = {marker: array("Q", [0]) * EXPECTED_CELLS for marker in marker_indices}
    markers_by_column = {column: marker for marker, column in marker_indices.items()}
    for row, column, value in source_rows(path):
        row_sums[row] += value
        if column in markers_by_column:
            counts[markers_by_column[column]][row] += value
    return row_sums, counts


def score(markers: tuple[str, ...], row: int, total: int, counts: dict[str, array]) -> float:
    return sum(math.log1p(10000 * counts[marker][row] / total) for marker in markers) / len(markers)


def assign_cells(cells: list[dict[str, str]], row_sums: array, counts: dict[str, array]) -> list[dict]:
    assignments = []
    for index, cell in enumerate(cells):
        total = row_sums[index]
        if total <= 0 or total != int(cell["tscp_count"]):
            raise ValueError("Count matrix and cell metadata disagree in row order")
        lineage = {name: score(panel, index, total, counts) for name, panel in LINEAGE_MARKERS.items()}
        ranked = sorted(lineage, key=lineage.get, reverse=True)
        best, second = ranked[:2]
        label, reason = "ambiguous", "lineage_score"
        if lineage[best] >= 0.55 and lineage[best] - lineage[second] >= 0.20:
            if best == "T":
                if any(counts[marker][index] for marker in ("CD3D", "CD3E", "TRAC")):
                    subtype_scores = {name: score(panel, index, total, counts) for name, panel in T_MARKERS.items()}
                    subtype = max(subtype_scores, key=subtype_scores.get)
                    alternative = "CD8" if subtype == "CD4" else "CD4"
                    anchors = ("IL7R", "CCR7", "MAL") if subtype == "CD4" else ("CD8A", "GZMK")
                    label, reason = "ambiguous_T", "subtype_score"
                    if (
                        subtype_scores[subtype] >= 0.45
                        and subtype_scores[subtype] - subtype_scores[alternative] >= 0.15
                        and any(counts[marker][index] for marker in anchors)
                    ):
                        label, reason = subtype, "assigned"
                else:
                    reason = "missing_T_anchor"
            elif best == "B":
                if counts["MS4A1"][index] or counts["CD79A"][index]:
                    label, reason = "B", "assigned"
                else:
                    reason = "missing_B_anchor"
            elif best == "monocyte":
                if counts["LYZ"][index] and any(counts[marker][index] for marker in ("FCN1", "S100A8", "S100A9")):
                    label, reason = "monocyte", "assigned"
                else:
                    reason = "missing_monocyte_anchor"
            else:
                label, reason = "other", "NK_score"
        assignments.append(
            {
                "id": cell["bc_wells"],
                "donor": cell["sample"],
                "label": label,
                "reason": reason,
                "lineage_score": round(lineage[best], 6),
                "lineage_margin": round(lineage[best] - lineage[second], 6),
            }
        )
    return assignments


def pseudobulks(path: Path, cells: list[dict[str, str]], assignments: list[dict]) -> dict[tuple[str, str], array]:
    groups = {(donor, group): array("Q", [0]) * EXPECTED_GENES for donor in DONORS for group in TARGET_GROUPS}
    targets = [
        groups.get((cell["sample"], assignment["label"])) for cell, assignment in zip(cells, assignments, strict=True)
    ]
    for row, column, value in source_rows(path):
        target = targets[row]
        if target is not None:
            target[column] += value
    return groups


def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def solve_cd14_gene_length(inputs: Path, output: Path) -> list[dict]:
    """Gradeable cell labels, donor pseudobulks, every joined coding gene and Pearson result."""
    from scipy.stats import pearsonr  # noqa: PLC0415

    query = json.loads((inputs / "query.json").read_text())
    if query != {
        "minimum_cd14_cells_per_donor": 25,
        "minimum_total_gene_count": 10,
        "minimum_coding_gene_overlap": 15000,
        "minimum_eligible_genes": 5000,
    }:
        raise ValueError("Unexpected frozen CD14 query")
    genes, marker_indices = read_genes(inputs / "genes.csv")
    cells = read_cells(inputs / "cells.csv")
    row_sums, marker_counts = marker_data(inputs / "matrix.mtx.gz", marker_indices)
    assignments = assign_cells(cells, row_sums, marker_counts)
    write_table(output / "cell_assignments.tsv", assignments)
    groups = pseudobulks(inputs / "matrix.mtx.gz", cells, assignments)
    sizes = Counter((row["donor"], row["label"]) for row in assignments)
    totals = {key: sum(counts) for key, counts in groups.items()}
    donor_rows = []
    marker_index = marker_indices["CD14"]
    for donor in DONORS:
        target = 1_000_000 * groups[(donor, "monocyte")][marker_index] / totals[(donor, "monocyte")]
        comparator = max(
            1_000_000 * groups[(donor, group)][marker_index] / totals[(donor, group)]
            for group in TARGET_GROUPS
            if group != "monocyte"
        )
        donor_rows.append(
            {
                "id": donor,
                "cd14_cells": sizes[(donor, "monocyte")],
                "cd4_cells": sizes[(donor, "CD4")],
                "cd8_cells": sizes[(donor, "CD8")],
                "b_cells": sizes[(donor, "B")],
                "other_ambiguous_cells": sum(sizes[(donor, label)] for label in ("other", "ambiguous", "ambiguous_T")),
                "cd14_total_umi": totals[(donor, "monocyte")],
                "cd14_marker_cpm": target,
                "max_other_cd14_marker_cpm": comparator,
                "held_out_enrichment": (target + 0.1) / (comparator + 0.1),
            }
        )
    if min(row["cd14_cells"] for row in donor_rows) < 25:
        raise ValueError("Too few CD14 cells in a donor")
    if sum(row["other_ambiguous_cells"] for row in donor_rows) / EXPECTED_CELLS > 0.65:
        raise ValueError("Too many unassigned cells")
    if sum(row["held_out_enrichment"] >= 1.5 for row in donor_rows) < 3:
        raise ValueError("CD14 held-out marker gate failed")
    write_table(output / "donor_qc.tsv", donor_rows)
    spans = read_spans(inputs / "annotations.tsv.gz")
    gene_rows = []
    lengths, expressions = [], []
    prevalence_count = 0
    for index, gene in enumerate(genes):
        gene_id = gene["gene_id"].split(".", 1)[0]
        if gene_id not in spans:
            continue
        counts = [groups[(donor, "monocyte")][index] for donor in DONORS]
        cpm = [1_000_000 * count / totals[(donor, "monocyte")] for count, donor in zip(counts, DONORS, strict=True)]
        total_count = sum(counts)
        keep = total_count >= 10
        prevalence = sum(value >= 1 for value in cpm)
        prevalence_count += int(keep and prevalence >= 2)
        mean_cpm = sum(cpm) / len(DONORS)
        gene_rows.append(
            {
                "id": gene_id,
                "gene_name": gene["gene_name"],
                "span_bp": spans[gene_id],
                "total_cd14_count": total_count,
                **{f"count_{donor}": count for donor, count in zip(DONORS, counts, strict=True)},
                **{f"cpm_{donor}": value for donor, value in zip(DONORS, cpm, strict=True)},
                "cpm_prevalence": prevalence,
                "mean_cpm": mean_cpm,
                "eligible": int(keep),
            }
        )
        if keep:
            lengths.append(float(spans[gene_id]))
            expressions.append(mean_cpm)
    if len(gene_rows) < 15000 or len(lengths) < 5000:
        raise ValueError("Insufficient coding overlap or eligible genes")
    write_table(output / "genes.tsv", gene_rows)
    result = pearsonr(lengths, expressions)
    return [
        {
            "id": "CD14",
            "input_cells": len(cells),
            "selected_cells": sum(row["cd14_cells"] for row in donor_rows),
            "donors": len(DONORS),
            "joined_coding_genes": len(gene_rows),
            "eligible_genes": len(lengths),
            "cpm_prevalence_qc": prevalence_count,
            "pearson_r": float(result.statistic),
            "pearson_p": float(result.pvalue),
        }
    ]
