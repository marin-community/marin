# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the ordinary-JAX composed BF16 fixture corpus."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from verify_bf16_composed_fixture_oracle import verify_fixture_corpus

FIXTURE_DIRECTORY = Path(__file__).resolve().parents[1] / "test" / "Inputs"
GENERATOR = FIXTURE_DIRECTORY / "regenerate-jax-bf16-composed-fixtures.py"
ORACLE = Path(__file__).with_name("jax-0.10.1-bf16-composed-fixture-oracle.json")


def _fake_normalizer(tmp_path: Path, oracle: dict) -> Path:
    mapping = {}
    for case in oracle["cases"]:
        hashes = case["hashes"]
        mapping[hashes["raw_sha256"]] = hashes["raw_normalized_sha256"]
        mapping[hashes["hook_sha256"]] = hashes["hook_normalized_sha256"]
    normalizer = tmp_path / "shuttle-test-opt"
    normalizer.write_text(
        "#!/usr/bin/env python3\n"
        "import hashlib\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"mapping = {mapping!r}\n"
        "digest = hashlib.sha256(Path(sys.argv[-1]).read_bytes()).hexdigest().upper()\n"
        "print(mapping.get(digest, '0' * 64))\n"
    )
    normalizer.chmod(0o755)
    return normalizer


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


def _verify_copy(tmp_path: Path, mutate_oracle, diagnostic: str) -> None:
    fixture_directory, oracle_path, oracle = _copied_corpus(tmp_path)
    mutate_oracle(oracle)
    oracle_path.write_text(json.dumps(oracle, indent=2, sort_keys=True) + "\n")
    normalizer = _fake_normalizer(tmp_path, json.loads(ORACLE.read_text()))
    with pytest.raises(ValueError, match=diagnostic):
        verify_fixture_corpus(fixture_directory, oracle_path, GENERATOR, normalizer)


def test_checked_corpus_matches_generator_and_independent_oracle(tmp_path):
    oracle = json.loads(ORACLE.read_text())
    normalizer = _fake_normalizer(tmp_path, oracle)

    verify_fixture_corpus(FIXTURE_DIRECTORY, ORACLE, GENERATOR, normalizer)
    result = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--normalizer",
            str(normalizer),
            "--output-dir",
            str(FIXTURE_DIRECTORY),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_oracle_rejects_operation_inventory_drift(tmp_path):
    def mutate(oracle):
        inventory = oracle["cases"][0]["hook_boundary"]["operation_inventory"]
        inventory["stablehlo.multiply"] += 1

    _verify_copy(tmp_path, mutate, "hook_boundary.operation_inventory")


def test_oracle_rejects_cast_boundary_drift(tmp_path):
    def mutate(oracle):
        boundary = oracle["cases"][0]["hook_boundary"]["cast_boundaries"][0]
        boundary["kind"] = "f32_to_bf16_unspecified_rounding"

    _verify_copy(tmp_path, mutate, "hook_boundary.cast_boundaries")


def test_oracle_rejects_output_rewiring(tmp_path):
    fixture_directory, oracle_path, oracle = _copied_corpus(tmp_path)
    path = fixture_directory / oracle["cases"][0]["filename"]
    text = path.read_text()
    match = re.search(r"(?m)^    return (%[0-9]+), (%[0-9]+),", text)
    assert match
    replacement = f"    return {match.group(2)}, {match.group(1)},"
    path.write_text(text[: match.start()] + replacement + text[match.end() :])
    normalizer = _fake_normalizer(tmp_path, oracle)

    with pytest.raises(ValueError, match=r"hook_boundary\.output_anchors"):
        verify_fixture_corpus(fixture_directory, oracle_path, GENERATOR, normalizer)


def test_oracle_rejects_declared_output_drift(tmp_path):
    fixture_directory, oracle_path, oracle = _copied_corpus(tmp_path)
    path = fixture_directory / oracle["cases"][0]["filename"]
    path.write_text(path.read_text().replace("// Outputs: forward=", "// Outputs: primal=", 1))
    normalizer = _fake_normalizer(tmp_path, oracle)

    with pytest.raises(ValueError, match="Outputs"):
        verify_fixture_corpus(fixture_directory, oracle_path, GENERATOR, normalizer)


def test_oracle_rejects_case_identity_drift(tmp_path):
    fixture_directory, oracle_path, oracle = _copied_corpus(tmp_path)
    path = fixture_directory / oracle["cases"][0]["filename"]
    path.write_text(path.read_text().replace(oracle["cases"][0]["case_id"], "contract_map_0000000000000000", 1))
    normalizer = _fake_normalizer(tmp_path, oracle)

    with pytest.raises(ValueError, match="Case ID"):
        verify_fixture_corpus(fixture_directory, oracle_path, GENERATOR, normalizer)


def test_oracle_rejects_toolchain_provenance_drift(tmp_path):
    def mutate(oracle):
        oracle["provenance"]["xla"] = "0" * 40

    _verify_copy(tmp_path, mutate, "provenance drift")


def test_oracle_rejects_normalized_hash_drift(tmp_path):
    def mutate(oracle):
        oracle["cases"][0]["hashes"]["hook_normalized_sha256"] = "0" * 64

    _verify_copy(tmp_path, mutate, "hashes.hook_normalized_sha256")


def test_oracle_rejects_generator_source_drift(tmp_path):
    fixture_directory, oracle_path, oracle = _copied_corpus(tmp_path)
    generator = tmp_path / GENERATOR.name
    generator.write_text(GENERATOR.read_text() + "\n# source drift\n")
    normalizer = _fake_normalizer(tmp_path, oracle)

    with pytest.raises(ValueError, match="provenance drift"):
        verify_fixture_corpus(fixture_directory, oracle_path, generator, normalizer)


def test_oracle_rejects_non_jax_generator_dependency(tmp_path):
    fixture_directory, oracle_path, oracle = _copied_corpus(tmp_path)
    generator = tmp_path / GENERATOR.name
    generator.write_text(GENERATOR.read_text() + "\nimport tile_lifetime.tensor_program\n")
    normalizer = _fake_normalizer(tmp_path, oracle)

    with pytest.raises(ValueError, match="ordinary-JAX fixture boundary"):
        verify_fixture_corpus(fixture_directory, oracle_path, generator, normalizer)
