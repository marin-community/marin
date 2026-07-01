# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Emit uv/pytest arguments for a unified-unit.yaml leg as GitHub Actions step outputs.

Usage:
    python infra/ci/prepare.py <scope> [--tests "path1 path2 ..."]

Outputs (to $GITHUB_OUTPUT or stdout):
    package     uv package name (e.g. marin-core)
    extras      --extra flags (e.g. "--extra cpu --extra dedup"), may be empty
    test_paths  space-separated test paths, or the full suite dir if none given
"""

import argparse
import os

SYNC_PACKAGE: dict[str, str] = {
    "rigging": "marin-rigging",
    "finelog": "marin-finelog",
    "haliax": "marin-haliax",
    "iris": "marin-iris",
    "fray": "marin-fray",
    "levanter": "marin-levanter",
    "zephyr": "marin-zephyr",
    "marin": "marin-core",
}

# levanter's torch_test extra is intentionally omitted: torch/TPU legs stay in levanter-unit.yaml.
SYNC_EXTRAS: dict[str, list[str]] = {
    "marin": ["cpu", "dedup"],
}

TEST_DIR: dict[str, str] = {
    **{s: f"lib/{s}/tests" for s in SYNC_PACKAGE if s != "marin"},
    "marin": "tests",
}


def emit_outputs(outputs: dict[str, str]) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT", "")
    if github_output:
        with open(github_output, "a") as f:
            for k, v in outputs.items():
                f.write(f"{k}={v}\n")
    else:
        for k, v in outputs.items():
            print(f"{k}={v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit uv/pytest args for a CI leg.")
    parser.add_argument("scope", choices=list(SYNC_PACKAGE))
    parser.add_argument("--tests", default="", help="Space-separated test file paths")
    args = parser.parse_args()

    extras = " ".join(f"--extra {e}" for e in SYNC_EXTRAS.get(args.scope, []))
    test_paths = args.tests.strip() or TEST_DIR[args.scope]

    emit_outputs({"package": SYNC_PACKAGE[args.scope], "extras": extras, "test_paths": test_paths})


if __name__ == "__main__":
    main()
