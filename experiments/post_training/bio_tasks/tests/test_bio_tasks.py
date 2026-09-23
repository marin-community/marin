# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import subprocess
import sys
from collections import Counter

import pyarrow.parquet as pq
import pytest
import tomlkit
from tasktrove_verify.grade import Status

from experiments.post_training.bio_tasks.build import build, identity, validate_instance
from experiments.post_training.bio_tasks.contract import Column, Contract, grade_answer
from experiments.post_training.bio_tasks.oracle import (
    solve_counts,
    solve_fractions,
    solve_genotypes,
    solve_images,
    solve_intervals,
    solve_sequence,
    solve_sites,
    solve_taxonomy,
)
from experiments.post_training.bio_tasks.recipes import RECIPES
from experiments.post_training.tasktrove.taskbinary import read_task_binary

BASE_IMAGE = "python:3.12-slim@sha256:" + "0" * 64
TOOL_REF = "12bbd5d45b1b176167ab3cfe905e06004c2f02e3"


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda recipe: recipe.id)
@pytest.mark.parametrize("seed", [0, 1, 20260923])
def test_generated_reference_accepts_independent_solver_and_rejects_scientific_errors(recipe, seed):
    checks = validate_instance(recipe, recipe.generate(seed))
    assert checks["oracle"] == checks["row_permutation"] == 1
    assert all(value == 0 for name, value in checks.items() if name not in {"oracle", "row_permutation"})


def test_wrong_construction_reference_blocks_task_admission():
    recipe = RECIPES[0]
    instance = recipe.generate(3)
    wrong = instance.contract.model_dump()
    wrong["expected"]["tx0"]["sequence"] = "AT"
    with pytest.raises(ValueError, match="validation oracle"):
        validate_instance(recipe, dataclasses.replace(instance, contract=Contract.model_validate(wrong)))


@pytest.fixture
def association_contract():
    return Contract(
        columns={
            "effect": Column(kind="number", description="signed coefficient", unit="log odds", atol=0.01, rtol=0),
            "n": Column(kind="integer", description="included patients", unit="patients"),
        },
        expected={"treated": {"effect": -2.0, "n": 80}, "control": {"effect": 1.0, "n": 58}},
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "reverse_sign",
        "drop_patients",
        "duplicate_id",
        "omit_record",
        "extra_field",
        "boolean_count",
        "string_number",
        "nan",
        "infinity",
        "overflow",
        "outside_tolerance",
    ],
)
def test_contract_rejects_structurally_valid_wrong_answers(association_contract, mutation):
    rows = association_contract.answer()
    row = rows[0]
    if mutation == "reverse_sign":
        row["effect"] = 2
    elif mutation == "drop_patients":
        row["n"] = 58
    elif mutation == "duplicate_id":
        rows.append(rows[0])
    elif mutation == "omit_record":
        rows.pop()
    elif mutation == "extra_field":
        row["extra"] = 0
    elif mutation == "boolean_count":
        row["n"] = True
    elif mutation == "string_number":
        row["effect"] = "-2.0"
    elif mutation == "nan":
        row["effect"] = float("nan")
    elif mutation == "infinity":
        row["effect"] = float("inf")
    elif mutation == "overflow":
        row["effect"] = 10**400
    else:
        row["effect"] = -1.98
    verdict = grade_answer(association_contract, json.dumps(rows))
    assert verdict.status == Status.SCORED
    assert verdict.reward == 0


def test_contract_accepts_order_invariance_and_declared_numeric_tolerance(association_contract):
    rows = list(reversed(association_contract.answer()))
    rows[0]["effect"] += 0.005
    assert grade_answer(association_contract, json.dumps(rows)).reward == 1


def test_duplicate_json_keys_cannot_override_a_wrong_answer(association_contract):
    answer = '[{"id":"treated","effect":2,"effect":-2,"n":80},{"id":"control","effect":1,"n":58}]'
    assert grade_answer(association_contract, answer).reward == 0


def test_donor_join_uses_ids_raw_counts_and_both_inclusion_criteria(tmp_path):
    (tmp_path / "cells.csv").write_text(
        "cell_id,donor,cell_type,qc_pass\na,d1,T,1\nb,d2,T,1\nc,d1,B,1\nd,d1,T,0\ne,d3,B,1\n"
    )
    (tmp_path / "raw_counts.csv").write_text("cell_id,G0\nb,11\na,3\nc,100\ne,1000\nd,10000\n")
    assert solve_counts(tmp_path) == [
        {"id": "d1/G0", "count": 3, "n_cells": 1},
        {"id": "d2/G0", "count": 11, "n_cells": 1},
        {"id": "d3/G0", "count": 0, "n_cells": 0},
    ]


def test_fraction_preserves_patients_with_zero_denominator_and_local_barcode_identity(tmp_path):
    (tmp_path / "specimens.csv").write_text(
        "specimen_id,patient_id,visit,tissue\ns1,p1,baseline,blood\ns2,p2,baseline,blood\n"
        "s3,p1,followup,blood\ns4,p1,baseline,tumor\ns5,p3,baseline,blood\n"
    )
    (tmp_path / "cells.csv").write_text(
        "specimen_id,barcode,cell_type,qc_pass\ns1,c1,T,1\ns1,c2,B,1\ns2,c1,B,1\n" "s3,c1,T,1\ns4,c1,T,1\ns5,c1,T,0\n"
    )
    assert solve_fractions(tmp_path) == [
        {"id": "p1", "numerator": 1, "denominator": 2, "fraction": 0.5},
        {"id": "p2", "numerator": 0, "denominator": 1, "fraction": 0},
        {"id": "p3", "numerator": 0, "denominator": 0, "fraction": None},
    ]


def test_sequence_respects_closed_coordinates_and_reverse_complement(tmp_path):
    (tmp_path / "genome.fa").write_text(">synthetic\nAACG\nTCCA\n")
    (tmp_path / "transcripts.csv").write_text("id,start,end,strand\nfirst,1,1,+\nminus,3,8,-\n")
    assert solve_sequence(tmp_path) == [
        {"id": "first", "sequence": "A", "length": 1},
        {"id": "minus", "sequence": "TGGACG", "length": 6},
    ]


def test_lineage_and_seed_are_preserved_when_recipe_version_changes():
    old = RECIPES[0]
    new = dataclasses.replace(old, version="2")
    for index in range(30):
        first, second = identity(old, 100, index), identity(new, 100, index)
        assert (first.lineage, first.seed) == (second.lineage, second.seed)
        assert first.task_id != second.task_id


def test_bed_overlap_uses_half_open_boundaries_and_chromosome_identity(tmp_path):
    (tmp_path / "features.bed").write_text("chr1\t0\t10\tleft\nchr1\t10\t20\tright\n")
    (tmp_path / "peaks.bed").write_text("chr1\t9\t11\tshort\nchr1\t10\t12\ttouch\nchr2\t0\t20\tdecoy\n")
    (tmp_path / "minimum.txt").write_text("1")
    assert solve_intervals(tmp_path) == [{"id": "left", "peak_count": 1}, {"id": "right", "peak_count": 2}]
    (tmp_path / "minimum.txt").write_text("2")
    assert solve_intervals(tmp_path) == [{"id": "left", "peak_count": 0}, {"id": "right", "peak_count": 1}]


def test_vcf_sample_names_partial_calls_and_haploid_denominator(tmp_path):
    (tmp_path / "samples.csv").write_text("sample_id,include\na,1\nb,1\nc,0\n")
    (tmp_path / "variants.vcf").write_text(
        "##fileformat=VCFv4.3\n##contig=<ID=chr1,length=100>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tc\tb\ta\n"
        "chr1\t1\tv1\tA\tG\t.\t.\t.\tGT:DP\t1/1:99\t1:1\t0|.:50\n"
        "chr1\t2\tv2\tC\tT\t.\t.\t.\tGT\t1/1\t.\t./.\n"
    )
    assert solve_genotypes(tmp_path) == [
        {"id": "v1", "called_alleles": 2, "alt_alleles": 1, "alt_frequency": 0.5},
        {"id": "v2", "called_alleles": 0, "alt_alleles": 0, "alt_frequency": None},
    ]


def test_alignment_sites_distinguish_singletons_from_parsimony_informative_columns(tmp_path):
    (tmp_path / "alignments.csv").write_text("alignment_id,file\nx,x.fa\n")
    (tmp_path / "x.fa").write_text(">a\nAAA\n>b\nAA-\n>c\nAC-\n>d\nCCN\n")
    assert solve_sites(tmp_path) == [{"id": "x", "eligible_sites": 2, "variable_sites": 2, "informative_sites": 1}]


def test_taxonomy_clades_overlap_and_include_unclassified_in_denominator(tmp_path):
    (tmp_path / "taxonomy.csv").write_text("taxid,parent\n3,2\n1,\n2,1\n")
    (tmp_path / "assignments.csv").write_text("fragment_id,taxid\na,3\nb,2\nc,0\nd,0\n")
    assert solve_taxonomy(tmp_path) == [
        {"id": "3", "direct": 1, "clade": 1, "fraction": 0.25},
        {"id": "1", "direct": 0, "clade": 2, "fraction": 0.5},
        {"id": "2", "direct": 1, "clade": 2, "fraction": 0.5},
    ]


def test_image_labels_keep_disconnected_pixels_and_use_physical_pixel_centers(tmp_path):
    (tmp_path / "images.json").write_text(
        json.dumps(
            [
                {
                    "id": "x",
                    "mask": [[1, 0], [0, 1]],
                    "intensity": [[2, 99], [99, 6]],
                    "spacing_x": 0.5,
                    "spacing_y": 2,
                }
            ]
        )
    )
    assert solve_images(tmp_path) == [
        {"id": "x/1", "pixels": 2, "area": 2, "centroid_x": 0.5, "centroid_y": 2, "mean_intensity": 4}
    ]


def test_corpus_roundtrip_separates_oracles_and_grades_packaged_answers(tmp_path):
    output = tmp_path / "corpus"
    manifest = build(output, 3, 20260923, BASE_IMAGE, TOOL_REF)
    rows = pq.read_table(output / "tasks" / "part-00000.parquet").to_pylist()
    assert manifest["tasks"] == len(rows) == 3 * len(RECIPES)
    assert Counter(row["template_id"] for row in rows) == {f"{recipe.id}-v{recipe.version}": 3 for recipe in RECIPES}
    assert manifest["counts"]["train"] == len(rows)
    assert {path.name for path in (output / "harbor").iterdir()} == {"train"}
    for row in rows:
        files = read_task_binary(row["task_binary"])
        config = tomlkit.parse(files.text("task.toml"))
        assert config["metadata"]["split"] == "train"
        assert {f"format:{profile}" for profile in config["metadata"]["input_formats"]}.issubset(row["tags"])
        assert "train" in row["tags"] and not {"dev", "test"}.intersection(row["tags"])
        assert config["environment"]["allow_internet"] is False
        assert config["verifier"]["environment_mode"] == "separate"
        assert config["artifacts"] == ["/app/answer.json"]
        assert not files.under("solution/")
        assert set(files.under("environment/")) == {
            "environment/Dockerfile",
            *[p for p in files.files if p.startswith("environment/inputs/")],
        }
        assert "solution/oracle.py" in read_task_binary(row["solution_binary"]).files
        task_dir = output / "harbor" / str(config["metadata"]["split"]) / row["path"]
        answer = tmp_path / "answer.json"
        contract = Contract.model_validate_json(files.text("tests/reference.json"))
        answer.write_text(json.dumps(contract.answer()))
        logs = tmp_path / "verifier"
        command = [
            sys.executable,
            "-I",
            str(task_dir / "tests" / "contract.py"),
            "--answer",
            str(answer),
            "--logs",
            str(logs),
            "--reference",
            str(task_dir / "tests" / "reference.json"),
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=30)
        assert json.loads((logs / "reward.json").read_text()) == {"reward": 1.0}
        # A reference defect is unscored and removes the previous valid reward.
        (task_dir / "tests" / "reference.json").write_text("{}")
        subprocess.run(command, check=True, capture_output=True, timeout=30)
        assert json.loads((logs / "verdict.json").read_text())["status"] == "invalid_task"
        assert not (logs / "reward.json").exists()


def test_rebuilding_same_recipe_seeds_produces_identical_archives(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    build(left, 1, 3, BASE_IMAGE, TOOL_REF)
    build(right, 1, 3, BASE_IMAGE, TOOL_REF)
    assert (left / "ledger.jsonl").read_bytes() == (right / "ledger.jsonl").read_bytes()
    left_rows = pq.read_table(left / "tasks" / "part-00000.parquet").to_pylist()
    right_rows = pq.read_table(right / "tasks" / "part-00000.parquet").to_pylist()
    assert left_rows == right_rows
