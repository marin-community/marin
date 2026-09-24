# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream observed single-cell counts through cell QC and feature filtering."""

import csv
import gzip
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.post_training.bio_tasks.solvers.formats import table


def observed_entries(path: Path, rows: int, columns: int) -> Iterator[tuple[int, int, int]]:
    """Read the source's ordered positive integer coordinates without loading a matrix."""
    with gzip.open(path, "rt") as source:
        if next(source).split() != ["%%MatrixMarket", "matrix", "coordinate", "integer", "general"]:
            raise ValueError("Unexpected observed count matrix profile")
        lines = (line for line in source if line.strip() and not line.startswith("%"))
        source_rows, source_columns, nonzeros = map(int, next(lines).split())
        if (source_rows, source_columns) != (rows, columns):
            raise ValueError("Matrix dimensions disagree with cell and feature metadata")
        previous = (0, 0)
        count = 0
        for line in lines:
            row, column, value = map(int, line.split())
            if not (1 <= row <= rows and 1 <= column <= columns and value > 0 and (row, column) > previous):
                raise ValueError("Source must contain distinct ordered positive coordinates")
            previous = (row, column)
            count += 1
            yield row - 1, column - 1, value
        if count != nonzeros:
            raise ValueError("Source matrix entry count does not match its header")


def solve_singlecell(inputs: Path, output: Path) -> list[dict]:
    features = table(inputs / "features.tsv", delimiter="\t")
    cells = table(inputs / "cells.tsv", delimiter="\t")
    query = json.loads((inputs / "query.json").read_text())
    endogenous = [feature["feature_type"] == "endogenous" for feature in features]
    all_counts = [0] * len(cells)
    ercc_counts = [0] * len(cells)
    detected = [0] * len(cells)
    for row, column, value in observed_entries(inputs / "matrix.mtx.gz", len(features), len(cells)):
        all_counts[column] += value
        if endogenous[row]:
            detected[column] += 1
        else:
            if features[row]["feature_type"] != "ERCC":
                raise ValueError("Unrecognized feature class")
            ercc_counts[column] += value
    kept_cells = {}
    qc_rows = []
    for index, cell in enumerate(cells):
        total, ercc = all_counts[index], ercc_counts[index]
        keep = (
            total - ercc >= query["minimum_endogenous_counts"]
            and detected[index] >= query["minimum_detected_endogenous"]
            and ercc * query["ercc_fraction_denominator"] <= total * query["ercc_fraction_numerator"]
        )
        if keep:
            kept_cells[index] = len(kept_cells) + 1
        qc_rows.append(
            {
                "id": cell["id"],
                "geo_accession": cell["geo_accession"],
                "sorting_gate": cell["sorting_gate"],
                "all_counts": total,
                "ercc_counts": ercc,
                "endogenous_counts": total - ercc,
                "detected_endogenous": detected[index],
                "ercc_fraction": ercc / total if total else "NA",
                "keep": int(keep),
                "matrix_column": kept_cells.get(index, 0),
            }
        )
    detected_after_qc = [0] * len(features)
    reads_after_qc = [0] * len(features)
    for row, column, value in observed_entries(inputs / "matrix.mtx.gz", len(features), len(cells)):
        if column in kept_cells:
            detected_after_qc[row] += 1
            reads_after_qc[row] += value
    kept_genes = {}
    feature_rows = []
    for index, feature in enumerate(features):
        keep = endogenous[index] and detected_after_qc[index] >= query["minimum_retained_cells_per_gene"]
        if keep:
            kept_genes[index] = len(kept_genes) + 1
        feature_rows.append(
            {
                "id": feature["id"],
                "feature_type": feature["feature_type"],
                "retained_cell_count": detected_after_qc[index],
                "retained_read_count": reads_after_qc[index],
                "keep": int(keep),
                "matrix_row": kept_genes.get(index, 0),
            }
        )
    for filename, records in (("cell_qc.tsv", qc_rows), ("feature_qc.tsv", feature_rows)):
        with (output / filename).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    nonzeros = 0
    total = 0
    with TemporaryDirectory(prefix="bio-filtered-", dir=output) as temporary:
        body = Path(temporary) / "coordinates.txt"
        with body.open("w") as destination:
            for row, column, value in observed_entries(inputs / "matrix.mtx.gz", len(features), len(cells)):
                if row in kept_genes and column in kept_cells:
                    destination.write(f"{kept_genes[row]} {kept_cells[column]} {value}\n")
                    nonzeros += 1
                    total += value
        with (output / "filtered_counts.mtx.gz").open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as destination:
                destination.write(
                    (
                        "%%MatrixMarket matrix coordinate integer general\n"
                        f"{len(kept_genes)} {len(kept_cells)} {nonzeros}\n"
                    ).encode()
                )
                with body.open("rb") as source:
                    shutil.copyfileobj(source, destination)
    return [
        {
            "id": "study",
            "input_cells": len(cells),
            "retained_cells": len(kept_cells),
            "input_features": len(features),
            "retained_features": len(kept_genes),
            "output_nonzeros": nonzeros,
            "output_read_counts": total,
        }
    ]
