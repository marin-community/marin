# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check acceptance fingerprints against the independently audited fixtures."""

import argparse
import re
from pathlib import Path

from acceptance_contract import FORWARD_EXPECTATION, VJP_EXPECTATION, FixtureExpectation

FINGERPRINT_PATTERN = re.compile(r"(?m)^// Normalized StableHLO SHA-256: ([0-9A-F]{64})$")
FIXTURE_FILENAMES = {
    "forward": "jax-0.10.1-tanh-dot-forward.mlir",
    "vjp": "jax-0.10.1-tanh-dot-vjp.mlir",
}
EXPECTATIONS = (FORWARD_EXPECTATION, VJP_EXPECTATION)


def audited_fingerprint(fixture_path: Path) -> str:
    matches = FINGERPRINT_PATTERN.findall(fixture_path.read_text())
    if len(matches) != 1:
        raise ValueError(f"{fixture_path}: expected one normalized StableHLO audit fingerprint")
    return matches[0].lower()


def fixture_path(fixture_directory: Path, expectation: FixtureExpectation) -> Path:
    return fixture_directory / FIXTURE_FILENAMES[expectation.name]


def verify_oracles(fixture_directory: Path) -> None:
    mismatches = []
    for expectation in EXPECTATIONS:
        path = fixture_path(fixture_directory, expectation)
        audited = audited_fingerprint(path)
        if audited != expectation.final_normalized_fingerprint:
            mismatches.append(f"{path.name}: acceptance={expectation.final_normalized_fingerprint}, fixture={audited}")
    if mismatches:
        raise ValueError("acceptance fixture fingerprint drift:\n" + "\n".join(mismatches))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture-directory",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "test" / "Inputs",
    )
    arguments = parser.parse_args()
    verify_oracles(arguments.fixture_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
