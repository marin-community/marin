#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Emit the generated CUDA families used by the fixed-capacity MoE graph."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tile_lifetime.distributed_expert_jax_module import (
    DistributedExpertJaxModuleConfig,
    build_natural_router_relation,
    plan_distributed_expert_jax_module,
)
from tile_lifetime.xla_routed_shared_map_training_ffi import plan_routed_shared_map_training_typed_ffi

_ROOT = Path(__file__).parents[1]
_NATURAL_HLO = _ROOT / "benchmarks/artifacts/xla_grug_routed_combined_gpu_gb200_v0/original-gpu-pre-scheduler-hlo.txt.gz"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--architecture", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    keys = jax.random.split(jax.random.key(41), 2)
    source = jax.random.normal(keys[0], (8, 32), dtype=jnp.bfloat16) / 5
    router = jax.random.normal(keys[1], (32, 8), dtype=jnp.bfloat16) / 5
    relation = build_natural_router_relation(
        source,
        router,
        route_slots=2,
        destination_rank_by_item=np.arange(8, dtype=np.int32) // 2,
        destination_local_item_by_item=np.arange(8, dtype=np.int32) % 2,
        destination_capacity=5,
    )
    templates = plan_routed_shared_map_training_typed_ffi(gzip.decompress(_NATURAL_HLO.read_bytes()).decode())
    plan = plan_distributed_expert_jax_module(
        relation,
        config=DistributedExpertJaxModuleConfig(source_items_per_rank=2, hidden=32, intermediate=32),
        input_adjoint_template=templates.recovered_input_adjoint,
        weight_gradient_templates=templates.weight_gradients,
        source_fold_template=templates.source_fold,
        target_prefix="shuttle.distributed_expert_cpu",
    )
    generated_families = (
        ("relation_edge_map_fold", plan.composition.ranks[0].edge_reverse),
        ("segmented_input_adjoint", plan.handlers.input_adjoint),
        ("group_batched_w13_weight_adjoint", plan.handlers.weight_gradients[0]),
        ("group_batched_w2_weight_adjoint", plan.handlers.weight_gradients[1]),
        ("deterministic_source_fold", plan.handlers.source_fold),
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    handlers = []
    for family, generated in generated_families:
        source_name = f"{generated.handler_symbol}.cu"
        source_path = args.output_directory / source_name
        source_path.write_text(generated.source + "\n")
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if digest != hashlib.sha256((generated.source + "\n").encode()).hexdigest():
            raise RuntimeError(f"failed to preserve generated source for {family}")
        lowered = generated.source.lower()
        if any(token in lowered for token in ("torch", "pybind", "at::tensor", "deep_ep", "mok")):
            raise RuntimeError(f"generated family {family} contains an excluded dependency")
        handlers.append(
            {
                "family": family,
                "target": generated.target,
                "handler_symbol": generated.handler_symbol,
                "semantic_digest": generated.semantic_digest,
                "source": source_name,
                "source_sha256": digest,
            }
        )
    manifest = {
        "kind": "distributed_expert_jax_cuda_compile_register_preflight",
        "source_revision": args.source_revision,
        "architecture": args.architecture,
        "runtime_dependencies": ["JAX/XLA typed FFI", "CUDA runtime", "cuBLAS"],
        "device_query_or_execution": False,
        "handlers": handlers,
    }
    manifest_path = args.output_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
