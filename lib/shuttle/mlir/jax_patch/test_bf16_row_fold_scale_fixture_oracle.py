# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the ordinary-JAX BF16 row Fold fixture corpus."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from verify_bf16_row_fold_scale_fixture_oracle import verify_fixture_corpus

FIXTURE_DIRECTORY = Path(__file__).resolve().parents[1] / "test" / "Inputs"
GENERATOR = FIXTURE_DIRECTORY / "regenerate-jax-bf16-row-fold-scale-fixtures.py"
ORACLE = Path(__file__).with_name("jax-0.10.1-bf16-row-fold-scale-fixture-oracle.json")


def _copied_corpus(tmp_path: Path) -> tuple[Path, Path, dict]:
    fixture_directory = tmp_path / "Inputs"
    fixture_directory.mkdir()
    oracle = json.loads(ORACLE.read_text())
    for case in oracle["cases"]:
        source = FIXTURE_DIRECTORY / case["filename"]
        shutil.copy2(source, fixture_directory / source.name)
    oracle_path = tmp_path / ORACLE.name
    oracle_path.write_text(json.dumps(oracle, indent=2, sort_keys=True) + "\n")
    return fixture_directory, oracle_path, oracle


def _verify_oracle_mutation(tmp_path: Path, mutate, diagnostic: str) -> None:
    fixture_directory, oracle_path, oracle = _copied_corpus(tmp_path)
    mutate(oracle)
    oracle_path.write_text(json.dumps(oracle, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match=diagnostic):
        verify_fixture_corpus(fixture_directory, oracle_path, GENERATOR)


def test_checked_corpus_matches_generator_and_independent_oracle(tmp_path):
    verify_fixture_corpus(FIXTURE_DIRECTORY, ORACLE, GENERATOR)
    output_directory = tmp_path / "generated"
    result = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--output-dir",
            str(output_directory),
            "--write",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    for case in json.loads(ORACLE.read_text())["cases"]:
        expected = (FIXTURE_DIRECTORY / case["filename"]).read_text()
        assert (output_directory / case["filename"]).read_text() == expected


def test_oracle_rejects_normalized_graph_drift(tmp_path):
    def mutate(oracle):
        oracle["cases"][0]["hashes"]["hook_normalized_sha256"] = "0" * 64

    _verify_oracle_mutation(tmp_path, mutate, "hook_normalized_sha256")


def test_oracle_rejects_operation_inventory_drift(tmp_path):
    def mutate(oracle):
        sequence = oracle["cases"][0]["hook_boundary"]["top_level_operation_sequence"]
        sequence[0] = "stablehlo.exponential"

    _verify_oracle_mutation(tmp_path, mutate, "hook_boundary.top_level_operation_sequence")


def test_oracle_rejects_nested_combiner_provenance_drift(tmp_path):
    def mutate(oracle):
        reducer = oracle["cases"][0]["hook_boundary"]["reducers"][0]
        reducer["operations"][0]["result_refs"][0] = [0, 999, 0, 0]

    _verify_oracle_mutation(tmp_path, mutate, "hook_boundary.reducers")


def test_oracle_rejects_nested_terminator_provenance_drift(tmp_path):
    def mutate(oracle):
        reducer = oracle["cases"][0]["hook_boundary"]["reducers"][0]
        reducer["operations"][1]["operation_ref"] = [0, 999, 1]

    _verify_oracle_mutation(tmp_path, mutate, "hook_boundary.reducers")


def test_oracle_rejects_output_rewiring(tmp_path):
    fixture_directory, oracle_path, oracle = _copied_corpus(tmp_path)
    composed = next(case for case in oracle["cases"] if case["boundary"] == "composed" and case["shape"]["rows"] == 7)
    path = fixture_directory / composed["filename"]
    text = path.read_text()
    return_line = next(line for line in text.splitlines() if line.lstrip().startswith("return "))
    operands, suffix = return_line.strip().removeprefix("return ").split(" : ", 1)
    values = operands.split(", ")
    values[0], values[1] = values[1], values[0]
    path.write_text(text.replace(return_line, "    return " + ", ".join(values) + " : " + suffix, 1))

    with pytest.raises(ValueError, match="output_anchors"):
        verify_fixture_corpus(fixture_directory, oracle_path, GENERATOR)


def test_oracle_rejects_toolchain_provenance_drift(tmp_path):
    def mutate(oracle):
        oracle["provenance"]["xla_revision"] = "0" * 40

    _verify_oracle_mutation(tmp_path, mutate, "toolchain provenance drift")


def test_oracle_rejects_added_acceptance_claim(tmp_path):
    def mutate(oracle):
        oracle["accepted"] = True

    _verify_oracle_mutation(tmp_path, mutate, "schema drift")


def test_oracle_rejects_non_jax_generator_dependency(tmp_path):
    fixture_directory, oracle_path, _ = _copied_corpus(tmp_path)
    generator = tmp_path / GENERATOR.name
    generator.write_text(GENERATOR.read_text() + "\nimport tile_lifetime.tensor_program\n")

    with pytest.raises(ValueError, match="ordinary-JAX fixture boundary"):
        verify_fixture_corpus(fixture_directory, oracle_path, generator)
