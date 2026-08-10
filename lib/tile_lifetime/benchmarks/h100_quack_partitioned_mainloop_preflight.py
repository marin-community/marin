# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preflight the pinned QuACK partitioned-mainloop extension on an H100 host.

This script does not allocate or benchmark a GPU. It deliberately fails until
the external QuACK checkout exposes the complete one-launch executor, rather
than silently benchmarking the stock single-RHS GEMM or a split-GEMM fallback.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from pathlib import Path

from tile_lifetime.quack_partitioned_gemm_adapter import QUACK_PARTITION_ADAPTER_BASE_REVISION
from tile_lifetime.quack_partitioned_mainloop import (
    QUACK_PARTITIONED_SM90_PATCH_SHA256,
    audit_quack_partitioned_extension_patch,
)

_REQUIRED_EXECUTOR_SYMBOL = "PartitionedGemmSm90"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quack-checkout", type=Path, required=True)
    parser.add_argument(
        "--patch",
        type=Path,
        default=Path(__file__).parents[1] / "backends/h100/quack_partitioned_sm90.patch",
    )
    args = parser.parse_args()

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.quack_checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    patch_audit = audit_quack_partitioned_extension_patch(args.patch)
    if revision != QUACK_PARTITION_ADAPTER_BASE_REVISION:
        raise RuntimeError(f"QuACK checkout is {revision}; expected {QUACK_PARTITION_ADAPTER_BASE_REVISION}")
    if not patch_audit.clean:
        raise RuntimeError(f"QuACK extension patch failed its static audit: {patch_audit}")

    module = importlib.import_module("quack.partitioned_sm90")
    helper_symbols = tuple(symbol for symbol in patch_audit.required_symbols if hasattr(module, symbol))
    has_executor = hasattr(module, _REQUIRED_EXECUTOR_SYMBOL)
    report = {
        "quack_revision": revision,
        "patch_sha256": patch_audit.sha256,
        "expected_patch_sha256": QUACK_PARTITIONED_SM90_PATCH_SHA256,
        "imported_helper_symbols": helper_symbols,
        "required_executor_symbol": _REQUIRED_EXECUTOR_SYMBOL,
        "has_executor": has_executor,
        "gpu_allocated": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not has_executor:
        raise RuntimeError(
            "the helper patch imports, but the one-launch segmented-RHS producer/epilogue "
            "executor is not implemented; GPU correctness/performance is intentionally blocked"
        )


if __name__ == "__main__":
    main()
