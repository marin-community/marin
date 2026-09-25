# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate a licensed observed PBMC matrix for gene-length association task authoring.

This is a one-shot remote source check. It does not register or build a Harbor task.
The frozen decisions and source rights are recorded in
``gene_length_expression_source_assessment.md``.
"""

import csv
import gzip
import hashlib
import json
import math
import os
import re
import urllib.request
from array import array
from collections import Counter
from pathlib import Path

SOURCES = {
    "counts": ("https://cdn.parsebiosciences.com/pbmc/v3/wt-mini/count_matrix.mtx", 124746086),
    "genes": ("https://cdn.parsebiosciences.com/pbmc/v3/wt-mini/all_genes.csv", 1974562),
    "cells": ("https://cdn.parsebiosciences.com/pbmc/v3/wt-mini/cell_metadata.csv", 266304),
    "gtf": ("https://ftp.ensembl.org/pub/release-111/gtf/homo_sapiens/Homo_sapiens.GRCh38.111.gtf.gz", 54350257),
}
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
HELD_OUT = {"monocyte": "CD14", "B": "CD19", "CD4": "CD4", "CD8": "CD8B"}
TARGET_GROUPS = ("CD4", "CD8", "monocyte", "B")
DONORS = ("Donor_1", "Donor_2", "Donor_3", "Donor_4")
CANONICAL_CHROMOSOMES = {str(index) for index in range(1, 23)} | {"X", "Y"}
GENE_ID = re.compile(r'gene_id "([^"]+)"')
GENE_BIOTYPE = re.compile(r'gene_biotype "([^"]+)"')
EXPECTED_CELLS = 4861
EXPECTED_GENES = 62710
EXPECTED_NONZERO = 10297684


def download(name: str, directory: Path) -> dict:
    """Stream one owner-hosted source and retain its exact digest and byte count."""
    url, expected_bytes = SOURCES[name]
    destination = directory / url.rsplit("/", 1)[-1]
    digest = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(url, timeout=90) as response, destination.open("xb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            total += len(chunk)
            if total > expected_bytes:
                raise ValueError(f"Oversize source: {name}")
    if total != expected_bytes:
        raise ValueError(f"Changed source size: {name}: {total} != {expected_bytes}")
    return {"url": url, "bytes": total, "sha256": digest.hexdigest(), "path": str(destination)}


def read_genes(path: Path) -> tuple[list[str], dict[str, int]]:
    names = []
    identifiers = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["gene_id", "gene_name", "genome"]:
            raise ValueError("Unexpected Parse gene table columns")
        for row in reader:
            if row["genome"] != "hg38":
                raise ValueError("Unexpected Parse genome")
            identifiers.append(row["gene_id"])
            names.append(row["gene_name"])
    if len(names) != EXPECTED_GENES or len(set(identifiers)) != EXPECTED_GENES:
        raise ValueError("Unexpected Parse gene identities")
    requested = set(HELD_OUT.values())
    requested.update(marker for panel in (*LINEAGE_MARKERS.values(), *T_MARKERS.values()) for marker in panel)
    marker_indices = {}
    for marker in requested:
        matches = [index for index, name in enumerate(names) if name == marker]
        if len(matches) != 1:
            raise ValueError(f"Marker {marker} resolves to {len(matches)} genes")
        marker_indices[marker] = matches[0]
    return identifiers, marker_indices


def read_cells(path: Path) -> list[dict]:
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
            raise ValueError("Unexpected Parse cell metadata columns")
        cells = list(reader)
    if len(cells) != EXPECTED_CELLS or len({row["bc_wells"] for row in cells}) != EXPECTED_CELLS:
        raise ValueError("Unexpected Parse cell identities")
    if {row["sample"] for row in cells} != set(DONORS) or any(row["species"] != "hg38" for row in cells):
        raise ValueError("Unexpected Parse donors or genome")
    if any(int(row["tscp_count"]) <= 0 for row in cells):
        raise ValueError("Nonpositive source transcript count")
    return cells


def coordinates(path: Path):
    with path.open() as handle:
        if not handle.readline().startswith("%%MatrixMarket matrix coordinate integer general"):
            raise ValueError("Unexpected MatrixMarket format")
        while header := handle.readline():
            if not header.startswith("%"):
                break
        if tuple(map(int, header.split())) != (EXPECTED_CELLS, EXPECTED_GENES, EXPECTED_NONZERO):
            raise ValueError("Unexpected MatrixMarket shape")
        seen = 0
        for line in handle:
            row, column, value = map(int, line.split())
            # MatrixMarket permits an explicit zero at a coordinate (the Parse
            # file ends with one); it contributes nothing to UMI totals.
            if not (1 <= row <= EXPECTED_CELLS and 1 <= column <= EXPECTED_GENES and value >= 0):
                raise ValueError("Invalid observed MatrixMarket coordinate")
            seen += 1
            yield row - 1, column - 1, value
        if seen != EXPECTED_NONZERO:
            raise ValueError("Unexpected MatrixMarket coordinate count")


def marker_counts(path: Path, marker_indices: dict[str, int]) -> tuple[array, dict[str, array]]:
    row_sums = array("Q", [0]) * EXPECTED_CELLS
    counts = {name: array("Q", [0]) * EXPECTED_CELLS for name in marker_indices}
    by_column = {column: name for name, column in marker_indices.items()}
    for row, column, value in coordinates(path):
        row_sums[row] += value
        if column in by_column:
            counts[by_column[column]][row] += value
    return row_sums, counts


def score(panel: tuple[str, ...], row: int, library: int, counts: dict[str, array]) -> float:
    return sum(math.log1p(10000 * counts[marker][row] / library) for marker in panel) / len(panel)


def assign_cells(cells: list[dict], row_sums: array, counts: dict[str, array]) -> tuple[list[str], list[dict]]:
    assignments = []
    rows = []
    for index, cell in enumerate(cells):
        library = row_sums[index]
        lineage = {name: score(panel, index, library, counts) for name, panel in LINEAGE_MARKERS.items()}
        ranked = sorted(lineage, key=lineage.get, reverse=True)
        best, second = ranked[:2]
        label = "ambiguous"
        reason = "lineage_score"
        if lineage[best] >= 0.55 and lineage[best] - lineage[second] >= 0.20:
            if best == "T":
                if any(counts[marker][index] for marker in ("CD3D", "CD3E", "TRAC")):
                    subtypes = {name: score(panel, index, library, counts) for name, panel in T_MARKERS.items()}
                    subtype = max(subtypes, key=subtypes.get)
                    other = "CD8" if subtype == "CD4" else "CD4"
                    anchors = ("IL7R", "CCR7", "MAL") if subtype == "CD4" else ("CD8A", "GZMK")
                    label, reason = "ambiguous_T", "subtype_score"
                    if (
                        subtypes[subtype] >= 0.45
                        and subtypes[subtype] - subtypes[other] >= 0.15
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
        assignments.append(label)
        rows.append(
            {
                "barcode": cell["bc_wells"],
                "donor": cell["sample"],
                "label": label,
                "reason": reason,
                "lineage_score": round(lineage[best], 6),
                "lineage_margin": round(lineage[best] - lineage[second], 6),
            }
        )
    return assignments, rows


def read_gene_spans(path: Path) -> dict[str, int]:
    lengths = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if fields[2] != "gene" or fields[0] not in CANONICAL_CHROMOSOMES:
                continue
            gene_id = GENE_ID.search(fields[8])
            biotype = GENE_BIOTYPE.search(fields[8])
            if gene_id is None or biotype is None:
                raise ValueError("Malformed Ensembl gene feature")
            if biotype.group(1) != "protein_coding":
                continue
            stable_id = gene_id.group(1).split(".", 1)[0]
            if stable_id in lengths:
                raise ValueError(f"Duplicate protein-coding GTF gene: {stable_id}")
            lengths[stable_id] = int(fields[4]) - int(fields[3]) + 1
    return lengths


def pseudobulks(path: Path, cells: list[dict], assignments: list[str]) -> dict[tuple[str, str], array]:
    groups = {(donor, group): array("Q", [0]) * EXPECTED_GENES for donor in DONORS for group in TARGET_GROUPS}
    targets = [groups.get((cell["sample"], label)) for cell, label in zip(cells, assignments, strict=True)]
    for row, column, value in coordinates(path):
        target = targets[row]
        if target is not None:
            target[column] += value
    return groups


def held_out_qc(groups: dict[tuple[str, str], array], marker_indices: dict[str, int]) -> list[dict]:
    totals = {key: sum(values) for key, values in groups.items()}
    result = []
    for donor in DONORS:
        for group, marker in HELD_OUT.items():
            index = marker_indices[marker]
            target = 1_000_000 * groups[(donor, group)][index] / totals[(donor, group)]
            others = TARGET_GROUPS if group in ("monocyte", "B") else ("CD4", "CD8")
            comparison = max(
                1_000_000 * groups[(donor, other)][index] / totals[(donor, other)] for other in others if other != group
            )
            ratio = (target + 0.1) / (comparison + 0.1)
            result.append(
                {
                    "donor": donor,
                    "group": group,
                    "held_out_marker": marker,
                    "target_cpm": round(target, 6),
                    "max_comparator_cpm": round(comparison, 6),
                    "enrichment": round(ratio, 6),
                    "passed": ratio >= 1.5,
                }
            )
    return result


def pearson(xs: list[float], ys: list[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denominator = math.sqrt(sum(value * value for value in centered_x) * sum(value * value for value in centered_y))
    if denominator == 0:
        raise ValueError("Constant length or expression vector")
    return sum(x * y for x, y in zip(centered_x, centered_y, strict=True)) / denominator


def correlations(groups: dict[tuple[str, str], array], gene_ids: list[str], spans: dict[str, int]) -> dict:
    totals = {key: sum(values) for key, values in groups.items()}
    eligible = []
    for index, gene_id in enumerate(gene_ids):
        if gene_id not in spans:
            continue
        if any(
            sum(1_000_000 * groups[(donor, group)][index] / totals[(donor, group)] >= 1 for donor in DONORS) >= 2
            for group in TARGET_GROUPS
        ):
            eligible.append(index)
    if len(eligible) < 5000:
        raise ValueError(f"Too few eligible coding genes: {len(eligible)}")
    x = [math.log10(spans[gene_ids[index]]) for index in eligible]
    coefficients = {}
    for group in TARGET_GROUPS:
        y = [
            sum(math.log2(1 + 1_000_000 * groups[(donor, group)][index] / totals[(donor, group)]) for donor in DONORS)
            / 4
            for index in eligible
        ]
        coefficients[group] = pearson(x, y)
    ranking = sorted(TARGET_GROUPS, key=lambda group: (abs(coefficients[group]), group))
    return {
        "eligible_genes": len(eligible),
        "pearson_r": coefficients,
        "weakest_absolute_group": ranking[0],
        "ranked_groups": ranking,
        "cd14_monocyte_r": coefficients["monocyte"],
    }


def gate(report: dict, name: str, observed, passed: bool, rule: str) -> None:
    report["gates"].append({"name": name, "observed": observed, "passed": passed, "rule": rule})


def write_label_qc(output: Path, rows: list[dict]) -> None:
    with (output / "label_qc.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path) -> dict:
    working = Path("/tmp/gene-length-source-gate")
    working.mkdir(exist_ok=False)
    report = {
        "source": "Parse Evercode WT Mini v3 four-donor PBMC",
        "license": "CC BY 4.0",
        "license_url": "https://www.parsebiosciences.com/datasets/performance-of-evercode-wt-mini-v3-in-human-pbmcs/",
        "method_sha256": hashlib.sha256(Path("source_assessment.md").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "downloads": {},
        "gates": [],
    }
    try:
        for name in ("counts", "genes", "cells", "gtf"):
            report["downloads"][name] = download(name, working)
        gene_ids, marker_indices = read_genes(Path(report["downloads"]["genes"]["path"]))
        cells = read_cells(Path(report["downloads"]["cells"]["path"]))
        row_sums, markers = marker_counts(Path(report["downloads"]["counts"]["path"]), marker_indices)
        matching = sum(row_sums[index] == int(cell["tscp_count"]) for index, cell in enumerate(cells))
        gate(
            report,
            "matrix_metadata_row_order",
            matching,
            matching >= math.ceil(0.99 * EXPECTED_CELLS),
            "matching row sums >=99% of cells",
        )
        if matching < math.ceil(0.99 * EXPECTED_CELLS):
            return report
        assignments, assignment_rows = assign_cells(cells, row_sums, markers)
        write_label_qc(output, assignment_rows)
        counts = Counter((cell["sample"], label) for cell, label in zip(cells, assignments, strict=True))
        group_counts = {f"{donor}:{group}": counts[(donor, group)] for donor in DONORS for group in TARGET_GROUPS}
        gate(
            report,
            "donor_group_cells",
            group_counts,
            min(group_counts.values()) >= 25,
            "each of 16 donor-group combinations >=25 cells",
        )
        unassigned = sum(label not in TARGET_GROUPS for label in assignments) / EXPECTED_CELLS
        gate(
            report,
            "unassigned_fraction",
            round(unassigned, 6),
            unassigned <= 0.65,
            "other and ambiguous fraction <=0.65",
        )
        report["assignment_reasons"] = dict(Counter(row["reason"] for row in assignment_rows))
        spans = read_gene_spans(Path(report["downloads"]["gtf"]["path"]))
        overlap = sum(gene_id in spans for gene_id in gene_ids)
        gate(
            report,
            "protein_coding_id_overlap",
            overlap,
            overlap >= 15000,
            "at least 15000 unique canonical protein-coding GTF IDs in Parse table",
        )
        groups = pseudobulks(Path(report["downloads"]["counts"]["path"]), cells, assignments)
        if min(group_counts.values()) > 0:
            heldout = held_out_qc(groups, marker_indices)
            report["held_out_markers"] = heldout
            pass_count = {
                group: sum(row["passed"] for row in heldout if row["group"] == group) for group in TARGET_GROUPS
            }
            gate(
                report,
                "held_out_enrichment",
                pass_count,
                min(pass_count.values()) >= 3,
                "each held-out marker enriched >=1.5-fold in >=3 of 4 donors",
            )
        if all(item["passed"] for item in report["gates"]):
            report["endpoint_preview"] = correlations(groups, gene_ids, spans)
            gate(
                report,
                "eligible_gene_universe",
                report["endpoint_preview"]["eligible_genes"],
                True,
                "at least 5000 common eligible coding genes; nonconstant Pearson vectors",
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
        raise SystemExit("Frozen source gate failed; no correlation or task registration")
    print(
        json.dumps(
            {
                "gate_passed": True,
                "source_hashes": {name: item["sha256"] for name, item in report["downloads"].items()},
                "group_counts": next(
                    item["observed"] for item in report["gates"] if item["name"] == "donor_group_cells"
                ),
                "endpoint_preview": report["endpoint_preview"],
            }
        )
    )


if __name__ == "__main__":
    main()
