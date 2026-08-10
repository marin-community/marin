# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preflight the pinned QuACK partitioned-mainloop extension on an H100 host.

This script does not launch or benchmark a GPU. It imports the patched generic
executor and a generated authoring module, rather than silently accepting the
stock single-RHS GEMM or a split-GEMM fallback.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from tile_lifetime.quack_partitioned_gemm_adapter import QUACK_PARTITION_ADAPTER_BASE_REVISION
from tile_lifetime.quack_partitioned_mainloop import (
    QUACK_PARTITIONED_SM90_PATCH_SHA256,
    audit_quack_partitioned_extension_patch,
    generate_quack_partitioned_mainloop,
)
from tile_lifetime.xla_partitioned_contract_map import plan_attached_partitioned_contract_maps

_REQUIRED_EXECUTOR_SYMBOL = "PartitionedGemmSm90"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quack-checkout", type=Path, required=True)
    parser.add_argument("--hlo", type=Path, required=True)
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
    hlo = gzip.decompress(args.hlo.read_bytes()).decode() if args.hlo.suffix == ".gz" else args.hlo.read_text()
    semantic_plan = plan_attached_partitioned_contract_maps(
        hlo, target_prefix="shuttle.generic.partitioned_contract_map.preflight"
    )
    if len(semantic_plan.families) != 1:
        raise RuntimeError(f"expected one partitioned Contract family, found {len(semantic_plan.families)}")
    generated = generate_quack_partitioned_mainloop(semantic_plan.families[0].program)
    with tempfile.TemporaryDirectory(prefix="shuttle-partitioned-sm90-") as directory:
        module_path = Path(directory) / f"{generated.module_name}.py"
        module_path.write_text(generated.source)
        spec = importlib.util.spec_from_file_location(generated.module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load generated module spec from {module_path}")
        generated_module = importlib.util.module_from_spec(spec)
        sys.modules[generated.module_name] = generated_module
        spec.loader.exec_module(generated_module)
    has_generated_run = hasattr(generated_module, "run")
    report = {
        "quack_revision": revision,
        "patch_sha256": patch_audit.sha256,
        "expected_patch_sha256": QUACK_PARTITIONED_SM90_PATCH_SHA256,
        "imported_helper_symbols": helper_symbols,
        "required_executor_symbol": _REQUIRED_EXECUTOR_SYMBOL,
        "has_executor": has_executor,
        "generated_module": generated.module_name,
        "generated_source_sha256": generated.source_digest,
        "generated_rhs_mma_ns": generated.rhs_mma_ns,
        "generated_output_count": generated.output_count,
        "has_generated_run": has_generated_run,
        "gpu_allocated": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not has_executor:
        raise RuntimeError(
            "the helper patch imports, but the one-launch segmented-RHS producer/epilogue "
            "executor is not implemented; GPU correctness/performance is intentionally blocked"
        )
    if not has_generated_run:
        raise RuntimeError("generated authoring module did not expose run")


if __name__ == "__main__":
    main()
