#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "jax==0.11.0",
#   "jaxlib==0.11.0",
# ]
# ///

"""Capture and audit the HLO observed by a JAX pre-scheduler transform."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib

_CUSTOM_CALL_TARGET = re.compile(r'custom_call_target="([^"]+)"')
_FORBIDDEN_SEMANTIC_TARGET_FRAGMENTS = (
    "deepep",
    "flash_attention",
    "flashattention",
    "fa4",
    "gdn",
    "mok",
    "sonic",
)
_PASS_NAME = "shuttle_pre_scheduler_inspection_v1"


def _instruction_inventory(module: Any) -> dict[str, int]:
    inventory: Counter[str] = Counter()
    for computation in module.computations():
        for instruction in computation.instructions():
            opcode = str(instruction.opcode).removeprefix("HloOpcode.k")
            inventory[opcode] += 1
    return dict(sorted(inventory.items()))


def capture_pre_scheduler_compile(
    compile_fn: Callable[[], Any],
    *,
    frontend_stablehlo: str,
    frontend_name: str,
    artifact_directory: Path,
    module_name: str,
    platform: str,
) -> dict[str, Any]:
    """Run a no-op PRE_SCHEDULER callback and persist its exact input HLO."""
    hlo = importlib.import_module("jaxlib._hlo")
    xla = importlib.import_module("jax.extend.xla")
    frontend_custom_calls = tuple(sorted(set(_CUSTOM_CALL_TARGET.findall(frontend_stablehlo))))
    if frontend_custom_calls:
        raise RuntimeError(f"frontend StableHLO contains custom calls: {frontend_custom_calls}")

    jax.config.update("jax_enable_compilation_cache", False)
    captures: list[tuple[bytes, Any]] = []

    def capture(serialized_module: bytes) -> None:
        module = hlo.HloModule.from_serialized_hlo_module_proto(serialized_module)
        captures.append((serialized_module, module))

    xla.register_hlo_module_transformation(
        capture,
        name=_PASS_NAME,
        stage=xla.PipelineStage.PRE_SCHEDULER,
        platforms=platform,
    )
    compile_start = time.monotonic()
    try:
        compile_fn()
    finally:
        xla.clear_hlo_module_transformation(
            _PASS_NAME,
            stage=xla.PipelineStage.PRE_SCHEDULER,
            platforms=platform,
        )
    compile_duration = time.monotonic() - compile_start

    matching_captures = [capture for capture in captures if capture[1].name == module_name]
    if len(matching_captures) != 1:
        captured_names = [module.name for _, module in captures]
        raise RuntimeError(f"expected one {module_name!r} pre-scheduler module, got {captured_names}")
    serialized_module, module = matching_captures[0]
    hlo_text = module.to_string()
    custom_calls = tuple(sorted(set(_CUSTOM_CALL_TARGET.findall(hlo_text))))
    forbidden_custom_calls = tuple(
        target
        for target in custom_calls
        if any(fragment in target.lower() for fragment in _FORBIDDEN_SEMANTIC_TARGET_FRAGMENTS)
    )
    if forbidden_custom_calls:
        raise RuntimeError(f"pre-scheduler HLO contains semantic custom calls: {forbidden_custom_calls}")

    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "pre-scheduler-hlo.pb").write_bytes(serialized_module)
    (artifact_directory / "pre-scheduler-hlo.txt.gz").write_bytes(gzip.compress(hlo_text.encode(), mtime=0))
    device = jax.devices(platform)[0]
    summary = {
        "kind": "jax_hlo_module_transformation_pre_scheduler_probe",
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "platform": platform,
        "device_kind": device.device_kind,
        "pipeline_stage": xla.PipelineStage.PRE_SCHEDULER.name,
        "callback_return": None,
        "compilation_cache_enabled": False,
        "compile_duration_seconds": compile_duration,
        "frontend_stablehlo": frontend_name,
        "frontend_stablehlo_sha256": hashlib.sha256(frontend_stablehlo.encode()).hexdigest(),
        "frontend_custom_call_targets": frontend_custom_calls,
        "captured_module_name": module.name,
        "captured_module_count": len(captures),
        "captured_proto_bytes": len(serialized_module),
        "captured_proto_sha256": hashlib.sha256(serialized_module).hexdigest(),
        "captured_hlo_text_characters": len(hlo_text),
        "instruction_inventory": _instruction_inventory(module),
        "custom_call_targets": custom_calls,
        "forbidden_semantic_custom_call_targets": forbidden_custom_calls,
        "clean_semantic_boundary": not forbidden_custom_calls,
        "no_op_transform_completed": True,
    }
    (artifact_directory / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--platform", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    def probe(x: jax.Array, weight: jax.Array) -> jax.Array:
        return jnp.sum(jnp.tanh(x @ weight), dtype=jnp.float32)

    lowered = jax.jit(probe).lower(
        jax.ShapeDtypeStruct((4, 8), jnp.float32),
        jax.ShapeDtypeStruct((8, 16), jnp.float32),
    )
    frontend_stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    summary = capture_pre_scheduler_compile(
        lowered.compile,
        frontend_stablehlo=frontend_stablehlo,
        frontend_name="generated trivial Contract+Map+Fold smoke",
        artifact_directory=args.artifact_directory,
        module_name="jit_probe",
        platform=args.platform,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
