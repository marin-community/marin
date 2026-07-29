# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify D-2 Muon layout numerics and compiled sharding on four GB200 GPUs."""

import argparse
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from jax import shard_map
from jax.sharding import AxisType, Mesh, NamedSharding, reshard
from jax.sharding import PartitionSpec as P
from levanter.optim.grugmuon import (
    _newtonschulz_4d_distributed,
    _newtonschulz_batched_syrk,
    _newtonschulz_padded_stack_sharded,
    _zeropower_via_newtonschulz_local,
)

AXIS_NAMES = ("replica_dcn", "data", "expert", "model")
MESH_SHAPES = {
    "data1-expert4": (1, 1, 4, 1),
    "data2-expert2": (1, 2, 2, 1),
}
NS_STEPS = (0, 2, 5)
EPSILON = 1e-8
COEFFICIENT_TYPE = "quintic"
MOMENTUM = 0.95
LEARNING_RATE = 0.01
TRAJECTORY_STEPS = 5


@dataclass(frozen=True)
class QualificationCase:
    name: str
    shape: tuple[int, ...]
    sharding: P
    is_expert: bool
    is_w_down: bool


class StructureAudit(TypedDict):
    expert_merge_count: int
    padded_reshard_specs: list[str]
    replicated_padded_outbound_count: int


CASES = (
    QualificationCase("expert_gate_up", (4, 4, 5120, 1280), P(None, "expert", "data", "model"), True, False),
    QualificationCase("expert_down", (4, 4, 1280, 5120), P(None, "expert", "model", "data"), True, True),
    QualificationCase("nonexpert_tall", (3, 5120, 1280), P(None, "data", "model"), False, False),
    QualificationCase("nonexpert_wide", (3, 1280, 5120), P(None, "model", "data"), False, False),
    QualificationCase("nonexpert_square", (3, 5120, 5120), P(None, "data", "model"), False, False),
)


def _reference_4d_pre_reconciliation(
    path: tuple[object, ...],
    x: jax.Array,
    steps: int,
    eps: float,
    coefficient_type: str,
) -> jax.Array:
    """The 4D implementation at f53f781ce, frozen as the comparison reference."""
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return x
    mesh_shape_items = [(name, size) for name, size in mesh.shape.items() if size > 1]
    if not mesh_shape_items:
        return x

    layers, expert_count, d, last = x.shape
    merged = layers * expert_count
    best_axes: tuple[str, ...] = ()
    best_shards = 0
    for mask in range(1, 1 << len(mesh_shape_items)):
        subset = [mesh_shape_items[i] for i in range(len(mesh_shape_items)) if mask & (1 << i)]
        prod = math.prod(size for _, size in subset)
        if merged % prod == 0 and prod > best_shards:
            best_axes = tuple(name for name, _ in subset)
            best_shards = prod
    if not best_axes:
        raise ValueError(f"No mesh-axis subset divides merged expert stack {merged}")

    is_w_down = any(getattr(entry, "name", None) == "w_down" for entry in path)
    trailing = ("model", "data") if is_w_down else ("data", "model")
    intermediate_3d_spec = P(None, *trailing)
    orig_4d_spec = P(None, "expert", *trailing)
    target_3d_spec = P(best_axes[0], None, None) if len(best_axes) == 1 else P(best_axes, None, None)

    x_bf16 = x.astype(jnp.bfloat16)
    expert_axis_size = int(mesh.shape.get("expert", 1))
    expert_spec_3d = P("expert", None, None)
    use_expert_merge = expert_axis_size > 1 and "expert" in best_axes and merged % expert_axis_size == 0
    if use_expert_merge:
        x_swapped = jnp.swapaxes(x_bf16, 0, 1)
        x_expert = jax.lax.reshape(x_swapped, (merged, d, last), out_sharding=expert_spec_3d)
        x_distributed = x_expert if target_3d_spec == expert_spec_3d else reshard(x_expert, target_3d_spec)
    else:
        x_flat = jax.lax.reshape(x_bf16, (merged, d, last), out_sharding=intermediate_3d_spec)
        x_distributed = reshard(x_flat, target_3d_spec)

    if os.environ.get("SCALE_MUON_SYRK") == "1":
        updated_distributed = shard_map(
            lambda stack: _newtonschulz_batched_syrk(stack, steps, eps, coefficient_type),
            mesh=mesh,
            in_specs=target_3d_spec,
            out_specs=target_3d_spec,
            check_vma=False,
        )(x_distributed)
    else:

        def local_ns(matrix):
            return _zeropower_via_newtonschulz_local(matrix, steps, eps, coefficient_type)

        updated_distributed = jax.vmap(local_ns)(x_distributed)

    if use_expert_merge:
        swapped_4d_spec = P("expert", None, None, None)
        updated_expert = (
            updated_distributed if target_3d_spec == expert_spec_3d else reshard(updated_distributed, expert_spec_3d)
        )
        updated_swapped = jax.lax.reshape(
            updated_expert,
            (expert_count, layers, d, last),
            out_sharding=swapped_4d_spec,
        )
        updated_bf16 = jnp.swapaxes(updated_swapped, 0, 1)
    else:
        updated_flat = reshard(updated_distributed, intermediate_3d_spec)
        updated_bf16 = jax.lax.reshape(
            updated_flat,
            (layers, expert_count, d, last),
            out_sharding=orig_4d_spec,
        )
    return updated_bf16.astype(x.dtype)


def _reference_padded_pre_reconciliation(
    x: jax.Array,
    steps: int,
    eps: float,
    coefficient_type: str,
    target_sharding: NamedSharding,
) -> jax.Array:
    """The padded-stack implementation at f53f781ce, including its replicated exit."""
    assert x.ndim == 3

    def local(matrix):
        return _zeropower_via_newtonschulz_local(matrix, steps, eps, coefficient_type)

    mesh = jax.sharding.get_abstract_mesh()
    axes = [(name, size) for name, size in mesh.shape.items() if size > 1]
    batch_axis = tuple(name for name, _ in axes)
    batch_shards = math.prod(size for _, size in axes)
    layers = x.shape[0]
    pad = (-layers) % batch_shards

    padded = jnp.pad(x, ((0, pad), (0, 0), (0, 0))) if pad else x
    target = P(batch_axis[0], None, None) if len(batch_axis) == 1 else P(batch_axis, None, None)
    distributed = reshard(padded, target)
    updated = jax.vmap(local)(distributed)
    updated = reshard(updated, P(None, None, None))
    updated = updated[:layers] if pad else updated
    return reshard(updated, target_sharding)


def _functions(
    case: QualificationCase,
    steps: int,
    target_sharding: NamedSharding,
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    if case.is_expert:
        path_name = "w_down" if case.is_w_down else "w_gate"
        path = (jax.tree_util.GetAttrKey(path_name),)

        def current(x):
            return _newtonschulz_4d_distributed(
                path,
                x,
                steps=steps,
                eps=EPSILON,
                coefficient_type=COEFFICIENT_TYPE,
            )

        def reference(x):
            return _reference_4d_pre_reconciliation(
                path,
                x,
                steps=steps,
                eps=EPSILON,
                coefficient_type=COEFFICIENT_TYPE,
            )

        return current, reference

    def current(x):
        return _newtonschulz_padded_stack_sharded(
            x,
            steps=steps,
            eps=EPSILON,
            coefficient_type=COEFFICIENT_TYPE,
            target_sharding=target_sharding,
        )

    def reference(x):
        return _reference_padded_pre_reconciliation(
            x,
            steps=steps,
            eps=EPSILON,
            coefficient_type=COEFFICIENT_TYPE,
            target_sharding=target_sharding,
        )

    return current, reference


def _difference_metrics(actual: jax.Array, expected: jax.Array) -> dict[str, jax.Array]:
    actual_f32 = actual.astype(jnp.float32)
    expected_f32 = expected.astype(jnp.float32)
    diff = actual_f32 - expected_f32
    actual_norm = jnp.sqrt(jnp.sum(jnp.square(actual_f32)))
    expected_norm = jnp.sqrt(jnp.sum(jnp.square(expected_f32)))
    diff_norm = jnp.sqrt(jnp.sum(jnp.square(diff)))
    return {
        "max_abs": jnp.max(jnp.abs(diff)),
        "mean_abs": jnp.mean(jnp.abs(diff)),
        "relative_l2": diff_norm / jnp.maximum(expected_norm, jnp.finfo(jnp.float32).tiny),
        "cosine": (
            jnp.sum(actual_f32 * expected_f32)
            / jnp.maximum(
                actual_norm * expected_norm,
                jnp.finfo(jnp.float32).tiny,
            )
        ),
        "exact_fraction": jnp.mean(actual == expected),
        "finite": jnp.all(jnp.isfinite(actual)) & jnp.all(jnp.isfinite(expected)),
    }


def _random_array(case: QualificationCase, sharding: NamedSharding, seed: int) -> jax.Array:
    make_array = jax.jit(
        lambda: jax.random.normal(jax.random.key(seed), case.shape, dtype=jnp.float32) * 0.02,
        out_shardings=sharding,
    )
    return make_array()


def _host_metrics(metrics: dict[str, jax.Array]) -> dict[str, float | bool]:
    return {
        key: bool(value) if value.dtype == jnp.bool_ else float(value) for key, value in jax.device_get(metrics).items()
    }


def _comparison(
    current: Callable[[jax.Array], jax.Array],
    reference: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array], dict[str, jax.Array]]:
    def compare(value):
        return _difference_metrics(current(value), reference(value))

    return jax.jit(compare)


def _paired_step(
    current_fn: Callable[[jax.Array], jax.Array],
    reference_fn: Callable[[jax.Array], jax.Array],
    scale: float,
) -> Callable[..., tuple[jax.Array, ...]]:
    def paired_step(current_param, reference_param, current_momentum, reference_momentum):
        current_gradient = current_param
        reference_gradient = reference_param
        current_momentum = MOMENTUM * current_momentum + current_gradient
        reference_momentum = MOMENTUM * reference_momentum + reference_gradient
        current_nesterov = MOMENTUM * current_momentum + current_gradient
        reference_nesterov = MOMENTUM * reference_momentum + reference_gradient
        current_update = current_fn(current_nesterov) * scale
        reference_update = reference_fn(reference_nesterov) * scale
        current_param = current_param - LEARNING_RATE * current_update
        reference_param = reference_param - LEARNING_RATE * reference_update
        current_loss = 0.5 * jnp.mean(jnp.square(current_param.astype(jnp.float32)))
        reference_loss = 0.5 * jnp.mean(jnp.square(reference_param.astype(jnp.float32)))
        return (
            current_param,
            reference_param,
            current_momentum,
            reference_momentum,
            current_loss,
            reference_loss,
            _difference_metrics(current_update, reference_update),
        )

    return jax.jit(paired_step)


def _run_point_comparisons(mesh: Mesh) -> dict[str, dict[str, dict[str, float | bool]]]:
    all_results: dict[str, dict[str, dict[str, float | bool]]] = {}
    for case_index, case in enumerate(CASES):
        sharding = NamedSharding(mesh, case.sharding)
        x = _random_array(case, sharding, case_index)
        case_results: dict[str, dict[str, float | bool]] = {}
        for steps in NS_STEPS:
            current, reference = _functions(case, steps, sharding)
            compare = _comparison(current, reference)
            metrics = _host_metrics(compare(x))
            case_results[str(steps)] = metrics
            print("D2_NUMERICAL " + json.dumps({"case": case.name, "ns_steps": steps, **metrics}), flush=True)
        all_results[case.name] = case_results
        del x
        jax.clear_caches()
    return all_results


def _run_loss_trajectories(mesh: Mesh) -> dict[str, dict[str, object]]:
    trajectories: dict[str, dict[str, object]] = {}
    for case_index, case in enumerate(CASES):
        sharding = NamedSharding(mesh, case.sharding)
        current_fn, reference_fn = _functions(case, 5, sharding)
        initial = _random_array(case, sharding, 100 + case_index)
        current_param = initial
        reference_param = initial
        current_momentum = jax.jit(jnp.zeros_like, out_shardings=sharding)(initial)
        reference_momentum = current_momentum
        fan_in, fan_out = case.shape[-2:]
        scale = math.sqrt(max(1.0, fan_out / fan_in))
        paired_step = _paired_step(current_fn, reference_fn, scale)
        current_losses: list[float] = []
        reference_losses: list[float] = []
        maximum_relative_loss_difference = 0.0
        for step in range(TRAJECTORY_STEPS):
            (
                current_param,
                reference_param,
                current_momentum,
                reference_momentum,
                current_loss,
                reference_loss,
                update_metrics,
            ) = paired_step(current_param, reference_param, current_momentum, reference_momentum)
            current_loss_host = float(current_loss)
            reference_loss_host = float(reference_loss)
            relative_loss_difference = abs(current_loss_host - reference_loss_host) / max(
                abs(reference_loss_host),
                np.finfo(np.float32).tiny,
            )
            maximum_relative_loss_difference = max(maximum_relative_loss_difference, relative_loss_difference)
            current_losses.append(current_loss_host)
            reference_losses.append(reference_loss_host)
            print(
                "D2_LOSS "
                + json.dumps(
                    {
                        "case": case.name,
                        "step": step,
                        "current": current_loss_host,
                        "reference": reference_loss_host,
                        "relative_difference": relative_loss_difference,
                        "update": _host_metrics(update_metrics),
                    }
                ),
                flush=True,
            )
        trajectories[case.name] = {
            "current": current_losses,
            "reference": reference_losses,
            "maximum_relative_difference": maximum_relative_loss_difference,
        }
        del initial, current_param, reference_param, current_momentum, reference_momentum
        jax.clear_caches()
    return trajectories


def _structure_audit(mesh: Mesh) -> StructureAudit:
    expert_case = CASES[0]
    expert_sharding = NamedSharding(mesh.abstract_mesh, expert_case.sharding)
    expert_input = jax.ShapeDtypeStruct(expert_case.shape, jnp.float32, sharding=expert_sharding)
    expert_path = (jax.tree_util.GetAttrKey("w_gate"),)

    def expert_apply(x):
        return _newtonschulz_4d_distributed(
            expert_path,
            x,
            steps=2,
            eps=EPSILON,
            coefficient_type=COEFFICIENT_TYPE,
        )

    padded_case = CASES[2]
    padded_sharding = NamedSharding(mesh.abstract_mesh, padded_case.sharding)
    padded_input = jax.ShapeDtypeStruct(padded_case.shape, jnp.float32, sharding=padded_sharding)

    def padded_apply(x):
        return _newtonschulz_padded_stack_sharded(
            x,
            steps=2,
            eps=EPSILON,
            coefficient_type=COEFFICIENT_TYPE,
            target_sharding=padded_sharding,
        )

    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        expert_jaxpr = jax.make_jaxpr(expert_apply)(expert_input)
        padded_jaxpr = jax.make_jaxpr(padded_apply)(padded_input)

    merged_shape = (expert_case.shape[0] * expert_case.shape[1], *expert_case.shape[2:])
    merge_count = sum(
        equation.primitive.name == "reshape" and equation.outvars[0].aval.shape == merged_shape
        for equation in expert_jaxpr.jaxpr.eqns
    )
    batch_shards = math.prod(size for size in mesh.shape.values() if size > 1)
    padded_layers = math.ceil(padded_case.shape[0] / batch_shards) * batch_shards
    padded_shape = (padded_layers, *padded_case.shape[1:])
    padded_specs = [
        repr(equation.params["dst_sharding"].spec)
        for equation in padded_jaxpr.jaxpr.eqns
        if equation.primitive.name == "reshard" and equation.invars[0].aval.shape == padded_shape
    ]
    audit = {
        "expert_merge_count": merge_count,
        "padded_reshard_specs": padded_specs,
        "replicated_padded_outbound_count": sum(spec == repr(P(None, None, None)) for spec in padded_specs),
    }
    print("D2_STRUCTURE " + json.dumps(audit), flush=True)
    return audit


def _run_compile_smoke(mesh: Mesh) -> None:
    audit = _structure_audit(mesh)
    if audit["expert_merge_count"] != 0:
        raise AssertionError(f"4D expert path merged L and E: {audit}")
    batch_axes = tuple(name for name, size in mesh.shape.items() if size > 1)
    expected_inbound = [repr(P(batch_axes[0], None, None))]
    if len(batch_axes) > 1:
        expected_inbound.append(repr(P(batch_axes, None, None)))
    if audit["padded_reshard_specs"][: len(expected_inbound)] != expected_inbound:
        raise AssertionError(f"Padded inbound path did not use the expected reshard sequence: {audit}")
    if audit["replicated_padded_outbound_count"] != 0:
        raise AssertionError(f"Padded outbound path replicated the full padded stack: {audit}")

    for case_index, case in enumerate((CASES[0], CASES[2])):
        sharding = NamedSharding(mesh, case.sharding)
        x = _random_array(case, sharding, 200 + case_index)
        current, _ = _functions(case, 2, sharding)
        compiled = jax.jit(current).lower(x).compile()
        result = compiled(x)
        result.block_until_ready()
        print(
            "D2_COMPILE "
            + json.dumps(
                {
                    "case": case.name,
                    "syrk": os.environ.get("SCALE_MUON_SYRK") == "1",
                    "shape": result.shape,
                    "sharding": repr(result.sharding.spec),
                    "finite": bool(jnp.all(jnp.isfinite(result))),
                }
            ),
            flush=True,
        )
        del x, result, compiled
        jax.clear_caches()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("numerical", "compile"), required=True)
    parser.add_argument("--mesh", choices=tuple(MESH_SHAPES), required=True)
    parser.add_argument("--syrk", choices=("0", "1"), required=True)
    args = parser.parse_args()
    os.environ["SCALE_MUON_SYRK"] = args.syrk

    devices = jax.devices()
    if len(devices) != 4:
        raise ValueError(f"D-2 qualification requires exactly four visible GPUs, got {len(devices)}")
    if any(device.platform != "gpu" for device in devices):
        raise ValueError(f"D-2 qualification requires GPUs, got {devices}")

    mesh = Mesh(
        np.asarray(devices).reshape(MESH_SHAPES[args.mesh]),
        AXIS_NAMES,
        axis_types=(AxisType.Explicit,) * len(AXIS_NAMES),
    )
    print(
        "D2_ENV "
        + json.dumps(
            {
                "jax_version": jax.__version__,
                "devices": [str(device) for device in devices],
                "mesh": dict(mesh.shape),
                "mode": args.mode,
                "syrk": args.syrk == "1",
            }
        ),
        flush=True,
    )
    with jax.set_mesh(mesh):
        if args.mode == "numerical":
            point_results = _run_point_comparisons(mesh)
            trajectories = _run_loss_trajectories(mesh)
            print(
                "D2_NUMERICAL_SUMMARY " + json.dumps({"point_results": point_results, "trajectories": trajectories}),
                flush=True,
            )
        else:
            _run_compile_smoke(mesh)


if __name__ == "__main__":
    main()
