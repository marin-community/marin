#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the fixed-capacity JAX-owned distributed reverse boundary on CPU."""

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tile_lifetime import DType, ExpertParallelConfig, NumericalPolicy, compile_stablehlo_expert_parallel_region
from tile_lifetime.distributed_expert_jax_module import (
    DistributedExpertJaxModuleConfig,
    audit_handler_module_stablehlo,
    build_natural_router_relation,
    evaluate_decomposed_training_reference,
    evaluate_natural_jax_training,
    jax_payload_all_to_all,
    lower_handler_module_stablehlo,
    plan_distributed_expert_jax_module,
)
from tile_lifetime.expert_parallel_training import derive_expert_parallel_training_plan
from tile_lifetime.xla_routed_shared_map_training_ffi import plan_routed_shared_map_training_typed_ffi

_ROOT = Path(__file__).parents[1]
_NATURAL_HLO = _ROOT / "benchmarks/artifacts/xla_grug_routed_combined_gpu_gb200_v0/original-gpu-pre-scheduler-hlo.txt.gz"
_PRIMARY_FIXTURE = _ROOT / "tests/fixtures/stablehlo/moe_primary_t2048_h7168_i3072_e384_k6_v1_14_1.mlir.bc.b64"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def _training_plan():
    forward = compile_stablehlo_expert_parallel_region(
        base64.b64decode(_PRIMARY_FIXTURE.read_text()),
        input_names=(
            "x",
            "router_weight",
            "shared_gate_weight",
            "shared_up_weight",
            "shared_down_weight",
            "routed_gate_weight",
            "routed_up_weight",
            "routed_down_weight",
        ),
        gemm_accumulation_dtype=DType.FP32,
        config=ExpertParallelConfig(expert_parallel_size=4),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    return derive_expert_parallel_training_plan(forward)


def _payload_roundtrip():
    devices = np.asarray(jax.devices("cpu"), dtype=object)
    if devices.size != 4:
        raise RuntimeError("set XLA_FLAGS=--xla_force_host_platform_device_count=4 before importing JAX")
    mesh = Mesh(devices, ("expert",))

    def local(payload):
        local_payload = payload[0]
        received = jax_payload_all_to_all(local_payload, axis_name="expert")
        returned = jax_payload_all_to_all(received, axis_name="expert")
        return returned[None]

    roundtrip = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=P("expert", None, None, None),
        out_specs=P("expert", None, None, None),
        check_vma=False,
    )
    payload = jnp.arange(4 * 4 * 2 * 32, dtype=jnp.int32).reshape(4, 4, 2, 32)
    output = jax.jit(roundtrip)(payload)
    stablehlo = str(jax.jit(roundtrip).lower(payload).compiler_ir(dialect="stablehlo"))
    return np.asarray(payload), np.asarray(output), stablehlo


def main() -> None:
    args = _parse_args()
    keys = jax.random.split(jax.random.key(41), 7)
    source = jax.random.normal(keys[0], (8, 32), dtype=jnp.bfloat16) / 5
    router = jax.random.normal(keys[1], (32, 8), dtype=jnp.bfloat16) / 5
    gate = jax.random.normal(keys[2], (8, 32, 32), dtype=jnp.bfloat16) / 5
    up = jax.random.normal(keys[3], (8, 32, 32), dtype=jnp.bfloat16) / 5
    down = jax.random.normal(keys[4], (8, 32, 32), dtype=jnp.bfloat16) / 5
    output_cotangent = jax.random.normal(keys[5], (8, 32), dtype=jnp.bfloat16) / 5
    relation = build_natural_router_relation(
        source,
        router,
        route_slots=2,
        destination_rank_by_item=np.arange(8, dtype=np.int32) // 2,
        destination_local_item_by_item=np.arange(8, dtype=np.int32) % 2,
        destination_capacity=5,
    )
    template_hlo = gzip.decompress(_NATURAL_HLO.read_bytes()).decode()
    templates = plan_routed_shared_map_training_typed_ffi(template_hlo)
    plan = plan_distributed_expert_jax_module(
        relation,
        config=DistributedExpertJaxModuleConfig(source_items_per_rank=2, hidden=32, intermediate=32),
        input_adjoint_template=templates.recovered_input_adjoint,
        weight_gradient_templates=templates.weight_gradients,
        source_fold_template=templates.source_fold,
        target_prefix="shuttle.distributed_expert_cpu",
    )
    natural = evaluate_natural_jax_training(
        source,
        router,
        gate,
        up,
        down,
        output_cotangent,
        route_slots=2,
    )
    first = evaluate_decomposed_training_reference(
        plan,
        _training_plan(),
        source,
        router,
        gate,
        up,
        down,
        output_cotangent,
    )
    second = evaluate_decomposed_training_reference(
        plan,
        _training_plan(),
        source,
        router,
        gate,
        up,
        down,
        output_cotangent,
    )
    maximum_errors = {}
    mean_errors = {}
    deterministic = {}
    for name in natural.__dataclass_fields__:
        expected = np.asarray(getattr(natural, name), dtype=np.float32)
        actual = np.asarray(getattr(first, name), dtype=np.float32)
        repeated = np.asarray(getattr(second, name), dtype=np.float32)
        error = np.abs(actual - expected)
        maximum_errors[name] = float(np.max(error))
        mean_errors[name] = float(np.mean(error))
        deterministic[name] = bool(np.array_equal(actual, repeated))
    handler_hlo = lower_handler_module_stablehlo(plan)
    handler_occurrences = audit_handler_module_stablehlo(plan, handler_hlo)
    sent_payload, returned_payload, collective_hlo = _payload_roundtrip()
    summary = {
        "kind": "fixed_capacity_distributed_expert_jax_module_cpu",
        "devices": [device.device_kind for device in jax.devices("cpu")],
        "natural_frontend": "ordinary JAX router/top-k/normalized weights and selected expert tensor algebra",
        "ad_owner": plan.ad_owner,
        "runtime_dependencies": list(plan.runtime_dependencies),
        "relation": {
            "sources": relation.source_item_count,
            "route_slots": relation.route_slots,
            "destinations": relation.destination_count,
            "ranks": relation.destination_rank_count,
            "destination_rows": relation.destination_row_count,
            "exchange_rows": int(relation.exchange_source_item.size),
            "capacity_per_destination": plan.destination_capacity,
        },
        "handler_occurrences": handler_occurrences,
        "handler_custom_call_count": handler_hlo.count("stablehlo.custom_call"),
        "router_vjp_dot_count": handler_hlo.count("stablehlo.dot_general"),
        "collective_all_to_all_count": collective_hlo.count("stablehlo.all_to_all"),
        "collective_roundtrip_exact": bool(np.array_equal(sent_payload, returned_payload)),
        "collective_boundaries": [boundary.__dict__ for boundary in plan.collectives],
        "input_adjoint_weight_abi": plan.input_adjoint_weight_abi.__dict__,
        "maximum_absolute_error": maximum_errors,
        "mean_absolute_error": mean_errors,
        "deterministic": deterministic,
        "torch_free": "torch" not in handler_hlo.lower(),
    }
    if max(maximum_errors.values()) >= 1e-3:
        raise RuntimeError(f"natural/decomposed error exceeds BF16 policy: {maximum_errors}")
    if not all(deterministic.values()):
        raise RuntimeError(f"decomposed reverse is not deterministic: {deterministic}")
    if summary["collective_all_to_all_count"] != 2 or not summary["collective_roundtrip_exact"]:
        raise RuntimeError("JAX payload transport did not lower and round-trip exactly")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "handler-stablehlo.mlir").write_text(handler_hlo)
    (args.output_directory / "collective-stablehlo.mlir").write_text(collective_hlo)
    (args.output_directory / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    checksum_lines = []
    for path in sorted(args.output_directory.iterdir()):
        if path.name == "SHA256SUMS":
            continue
        checksum_lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (args.output_directory / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
