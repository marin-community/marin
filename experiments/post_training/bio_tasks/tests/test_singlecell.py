# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import hashlib
import json

from experiments.post_training.bio_tasks.contract import grade_files
from experiments.post_training.bio_tasks.generators.real_singlecell import singlecell_contract
from experiments.post_training.bio_tasks.solvers.formats import table
from experiments.post_training.bio_tasks.solvers.real_singlecell import solve_singlecell


def test_singlecell_filters_cells_before_genes_and_preserves_count_identities(tmp_path):
    inputs, output = tmp_path / "inputs", tmp_path / "output"
    inputs.mkdir()
    output.mkdir()
    (inputs / "features.tsv").write_text(
        "id\tfeature_type\ng1\tendogenous\ng2\tendogenous\ng3\tendogenous\nERCC-1\tERCC\n"
    )
    (inputs / "cells.tsv").write_text(
        "id\tgeo_accession\tsorting_gate\na\tGSM1\tHSPC\nb\tGSM2\tHSPC\nc\tGSM3\tProg\nd\tGSM4\tProg\n"
    )
    observed = (
        "%%MatrixMarket matrix coordinate integer general\n4 4 11\n"
        "1 1 10\n1 2 2\n1 3 5\n1 4 5\n2 2 8\n2 3 5\n2 4 5\n3 3 1\n4 1 1\n4 2 10\n4 4 1\n"
    )
    (inputs / "matrix.mtx.gz").write_bytes(gzip.compress(observed.encode(), mtime=0))
    query = {
        "minimum_endogenous_counts": 10,
        "minimum_detected_endogenous": 2,
        "ercc_fraction_numerator": 1,
        "ercc_fraction_denominator": 5,
        "minimum_retained_cells_per_gene": 2,
    }
    (inputs / "query.json").write_text(json.dumps(query))
    summary = solve_singlecell(inputs, output)
    assert summary == [
        {
            "id": "study",
            "input_cells": 4,
            "retained_cells": 2,
            "input_features": 4,
            "retained_features": 2,
            "output_nonzeros": 4,
            "output_read_counts": 20,
        }
    ]
    cell_rows = table(output / "cell_qc.tsv", delimiter="\t")
    assert [(row["id"], row["keep"], row["matrix_column"], row["endogenous_counts"]) for row in cell_rows] == [
        ("a", "0", "0", "10"),
        ("b", "0", "0", "10"),
        ("c", "1", "1", "11"),
        ("d", "1", "2", "10"),
    ]
    feature_rows = table(output / "feature_qc.tsv", delimiter="\t")
    assert [(row["id"], row["retained_cell_count"], row["keep"], row["matrix_row"]) for row in feature_rows] == [
        ("g1", "2", "1", "1"),
        ("g2", "2", "1", "2"),
        ("g3", "1", "0", "0"),
        ("ERCC-1", "1", "0", "0"),
    ]
    coordinates = b"1 1 5\n1 2 5\n2 1 5\n2 2 5\n"
    assert gzip.decompress((output / "filtered_counts.mtx.gz").read_bytes()) == (
        b"%%MatrixMarket matrix coordinate integer general\n2 2 4\n" + coordinates
    )
    reference = {
        "summary": {key: value for key, value in summary[0].items() if key != "id"},
        "cell_qc": {
            "a": dict(
                geo_accession="GSM1",
                sorting_gate="HSPC",
                all_counts=11,
                ercc_counts=1,
                endogenous_counts=10,
                detected_endogenous=1,
                ercc_fraction=1 / 11,
                keep=0,
                matrix_column=0,
            ),
            "b": dict(
                geo_accession="GSM2",
                sorting_gate="HSPC",
                all_counts=20,
                ercc_counts=10,
                endogenous_counts=10,
                detected_endogenous=2,
                ercc_fraction=0.5,
                keep=0,
                matrix_column=0,
            ),
            "c": dict(
                geo_accession="GSM3",
                sorting_gate="Prog",
                all_counts=11,
                ercc_counts=0,
                endogenous_counts=11,
                detected_endogenous=3,
                ercc_fraction=0.0,
                keep=1,
                matrix_column=1,
            ),
            "d": dict(
                geo_accession="GSM4",
                sorting_gate="Prog",
                all_counts=11,
                ercc_counts=1,
                endogenous_counts=10,
                detected_endogenous=2,
                ercc_fraction=1 / 11,
                keep=1,
                matrix_column=2,
            ),
        },
        "feature_qc": {
            "g1": dict(feature_type="endogenous", retained_cell_count=2, retained_read_count=10, keep=1, matrix_row=1),
            "g2": dict(feature_type="endogenous", retained_cell_count=2, retained_read_count=10, keep=1, matrix_row=2),
            "g3": dict(feature_type="endogenous", retained_cell_count=1, retained_read_count=1, keep=0, matrix_row=0),
            "ERCC-1": dict(feature_type="ERCC", retained_cell_count=1, retained_read_count=1, keep=0, matrix_row=0),
        },
        "matrix": {"rows": 2, "columns": 2, "nonzeros": 4, "sha256": hashlib.sha256(coordinates).hexdigest()},
    }
    contract = singlecell_contract(reference)
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(contract.model_dump_json())
    answer = output / "answer.json"
    answer.write_text(json.dumps(summary))
    assert grade_files(reference_path, answer).reward == 1
    cells_file = output / "cell_qc.tsv"
    cells_file.write_text(cells_file.read_text().replace("GSM3", "GSM4"))
    assert grade_files(reference_path, answer).reward == 0
