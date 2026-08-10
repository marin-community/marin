# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage generic H100 Contract/Map evidence without allocating a GPU."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

_GPU_EXECUTION_REFUSAL = (
    "H100 execution is disabled for the architecture-nonconforming staging harness; "
    "SOURCE_ORDERED, FAST, and resource collectors require review"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shuttle-revision", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument(
        "--execute-gpu",
        action="store_true",
        help="Reserved launch gate; currently fails before importing JAX or querying a device.",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.execute_gpu:
        raise RuntimeError(_GPU_EXECUTION_REFUSAL)

    # tile_lifetime.__init__ imports JAX, so module loading belongs after the package-independent launch gate.
    benchmark_module = importlib.import_module("tile_lifetime.h100_contract_map_benchmark")
    result = benchmark_module.staging_manifest(shuttle_revision=args.shuttle_revision)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
