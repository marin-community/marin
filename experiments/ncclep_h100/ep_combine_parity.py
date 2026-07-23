# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Isolate BF16 numerical differences in TE NCCL_EP dispatch and combine."""

import importlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

EP_SIZE = 8
TOKENS_PER_RANK = 16_384
GLOBAL_TOKENS = TOKENS_PER_RANK * EP_SIZE
HIDDEN_DIM = 2_560
NUM_EXPERTS = 64
LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE
CAPACITY_FACTOR = 1.25
MAX_TOP_K = 4
RECV_CAPACITY_PER_RANK = int(CAPACITY_FACTOR * TOKENS_PER_RANK * MAX_TOP_K)
DISPATCH_ALIGNMENT = 16
PROBE_FEATURES = 17
SEED = 12_345
TOP_K_ENV = "NCCLEP_DIAGNOSTIC_TOP_K"
BF16_RTOL = 0.1
BF16_ATOL = 2e-4
SUMMARY_EVENT = "ncclep_h100_combine_parity"
REFERENCE_FP32 = "fp32_accumulate_then_bf16"
REFERENCE_FORWARD = "bf16_forward_route_order"
REFERENCE_REVERSE = "bf16_reverse_route_order"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    top_k: int
    expert_scaled: bool
    combine_input_dtype: str = "bfloat16"


def case_specs() -> tuple[CaseSpec, ...]:
    """Return the fixed diagnostic matrix in execution order."""
    return (
        CaseSpec("topk1_identity", top_k=1, expert_scaled=False),
        CaseSpec("topk1_expert_scaled_identity", top_k=1, expert_scaled=True),
        CaseSpec("topk4_identity", top_k=4, expert_scaled=False),
        CaseSpec("topk4_expert_scaled_identity", top_k=4, expert_scaled=True),
        CaseSpec(
            "topk4_expert_scaled_identity_fp32_combine_input",
            top_k=4,
            expert_scaled=True,
            combine_input_dtype="float32",
        ),
    )


def case_specs_for_top_k(top_k: int) -> tuple[CaseSpec, ...]:
    """Return the cases for one process-wide TE top-k configuration."""
    selected = tuple(spec for spec in case_specs() if spec.top_k == top_k)
    if not selected:
        raise ValueError(f"top_k must be 1 or 4, got {top_k}")
    return selected


def balanced_route_table(top_k: int) -> np.ndarray:
    """Build a fixed, exactly balanced route table."""
    if top_k not in (1, 4):
        raise ValueError(f"top_k must be 1 or 4, got {top_k}")
    assignments = np.arange(GLOBAL_TOKENS * top_k, dtype=np.int64)
    return (assignments % NUM_EXPERTS).reshape(GLOBAL_TOKENS, top_k).astype(np.int32)


def route_weights(top_k: int) -> np.ndarray:
    """Return exact binary-fraction route weights."""
    if top_k == 1:
        return np.ones((GLOBAL_TOKENS, 1), dtype=np.float32)
    if top_k == 4:
        return np.broadcast_to(
            np.asarray([0.5, 0.25, 0.125, 0.125], dtype=np.float32),
            (GLOBAL_TOKENS, 4),
        ).copy()
    raise ValueError(f"top_k must be 1 or 4, got {top_k}")


def expert_scales() -> np.ndarray:
    """Return distinct, exactly representable BF16 expert scales."""
    expert_ids = np.arange(NUM_EXPERTS, dtype=np.int32)
    return (0.5 + ((expert_ids * 17) % NUM_EXPERTS) / NUM_EXPERTS).astype(np.float32)


def route_capacity_report(routes: np.ndarray) -> dict[str, Any]:
    """Validate that a route table is balanced and fits the fixed TE capacity."""
    if routes.ndim != 2 or routes.shape[0] != GLOBAL_TOKENS or routes.shape[1] not in (1, 4):
        raise ValueError(f"unexpected route shape {routes.shape}")
    if routes.dtype != np.int32:
        raise TypeError(f"routes must be int32, got {routes.dtype}")
    if int(routes.min()) < 0 or int(routes.max()) >= NUM_EXPERTS:
        raise ValueError(f"route IDs must be in [0, {NUM_EXPERTS})")

    counts = np.bincount(routes.reshape(-1), minlength=NUM_EXPERTS)
    aligned = (counts + DISPATCH_ALIGNMENT - 1) // DISPATCH_ALIGNMENT * DISPATCH_ALIGNMENT
    destinations = aligned.reshape(EP_SIZE, LOCAL_EXPERTS).sum(axis=1)
    if np.any(destinations > RECV_CAPACITY_PER_RANK):
        raise ValueError(f"routing exceeds receive capacity {RECV_CAPACITY_PER_RANK}: {destinations.tolist()}")
    return {
        "top_k": int(routes.shape[1]),
        "expert_count_min": int(counts.min()),
        "expert_count_max": int(counts.max()),
        "aligned_destination_counts": destinations.tolist(),
        "recv_capacity_per_rank": RECV_CAPACITY_PER_RANK,
        "dispatch_alignment": DISPATCH_ALIGNMENT,
        "validated_before_dispatch": True,
    }


def expected_dispatch_fingerprints(routes: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, ...]:
    """Compute exact token-membership and route-weight fingerprints per expert."""
    if routes.shape != weights.shape:
        raise ValueError(f"routes and weights must have the same shape, got {routes.shape} and {weights.shape}")
    token_ids = np.arange(GLOBAL_TOKENS, dtype=np.uint32)
    probes = ((token_ids[:, None] >> np.arange(PROBE_FEATURES, dtype=np.uint32)) & 1).astype(np.float32)
    counts = np.zeros((NUM_EXPERTS,), dtype=np.int32)
    token_sums = np.zeros((NUM_EXPERTS, PROBE_FEATURES), dtype=np.float32)
    weighted_token_sums = np.zeros((NUM_EXPERTS, PROBE_FEATURES), dtype=np.float32)
    for route_slot in range(routes.shape[1]):
        expert_ids = routes[:, route_slot]
        np.add.at(counts, expert_ids, 1)
        np.add.at(token_sums, expert_ids, probes)
        np.add.at(weighted_token_sums, expert_ids, probes * weights[:, route_slot, None])
    return counts, token_sums, weighted_token_sums


def distinct_route_contributions(routes: np.ndarray, weights: np.ndarray, scales: np.ndarray) -> bool:
    """Return whether every top-k4 token has distinct scaled route coefficients."""
    if routes.shape[1] != 4:
        raise ValueError("distinct route contribution check requires top-k4")
    coefficients = weights * scales[routes]
    ordered = np.sort(coefficients, axis=1)
    return bool(np.all(np.diff(ordered, axis=1) != 0))


def _assert_topology(jax: Any) -> tuple[int, int]:
    rank = jax.process_index()
    world = jax.process_count()
    expected_rank = int(os.environ.get("IRIS_MULTIGPU_PROCESS_INDEX", "-1"))
    expected_world = int(os.environ.get("IRIS_MULTIGPU_PROCESS_COUNT", "-1"))
    local_devices = jax.local_devices()
    errors = []
    if world != EP_SIZE:
        errors.append(f"JAX process_count={world}, expected {EP_SIZE}")
    if expected_world != EP_SIZE:
        errors.append(f"IRIS_MULTIGPU_PROCESS_COUNT={expected_world}, expected {EP_SIZE}")
    if rank != expected_rank:
        errors.append(f"JAX process_index={rank}, supervisor rank={expected_rank}")
    if len(local_devices) != 1:
        errors.append(f"rank {rank} sees {len(local_devices)} local devices, expected 1")
    if len(jax.devices()) != EP_SIZE:
        errors.append(f"global device_count={len(jax.devices())}, expected {EP_SIZE}")
    if local_devices and (local_devices[0].platform != "gpu" or "H100" not in local_devices[0].device_kind):
        errors.append(f"rank {rank} device={local_devices[0]}, expected one H100")
    if errors:
        raise RuntimeError("process topology mismatch: " + "; ".join(errors))
    return rank, world


def _make_array(
    jax: Any,
    sharding: Any,
    shape: tuple[int, ...],
    generator: Any,
) -> Any:
    def callback(index: tuple[slice, ...]) -> np.ndarray:
        local_shape = tuple((part.stop or shape[axis]) - (part.start or 0) for axis, part in enumerate(index))
        return generator(index, local_shape)

    return jax.make_array_from_callback(shape, sharding, callback)


def _make_tokens(jax: Any, jnp: Any, batch_sharding: Any) -> Any:
    def generate(index: tuple[slice, ...], shape: tuple[int, ...]) -> np.ndarray:
        start = index[0].start or 0
        values = np.random.default_rng(SEED + start).standard_normal(shape, dtype=np.float32)
        token_ids = np.arange(start, start + shape[0], dtype=np.uint32)
        values[:, :PROBE_FEATURES] = ((token_ids[:, None] >> np.arange(PROBE_FEATURES, dtype=np.uint32)) & 1).astype(
            np.float32
        )
        return values.astype(jnp.bfloat16)

    return _make_array(jax, batch_sharding, (GLOBAL_TOKENS, HIDDEN_DIM), generate)


def _make_case_inputs(
    jax: Any,
    jnp: Any,
    batch_sharding: Any,
    expert_sharding: Any,
    top_k: int,
) -> tuple[Any, ...]:
    routes = balanced_route_table(top_k)
    weights = route_weights(top_k)
    expected_counts, expected_probes, expected_weighted_probes = expected_dispatch_fingerprints(routes, weights)

    def batch_from_host(values: np.ndarray) -> Any:
        return _make_array(
            jax,
            batch_sharding,
            values.shape,
            lambda index, shape: values[index[0].start or 0 : index[0].stop or values.shape[0]].reshape(shape),
        )

    def expert_from_host(values: np.ndarray, dtype: Any) -> Any:
        return _make_array(
            jax,
            expert_sharding,
            values.shape,
            lambda index, shape: values[index[0].start or 0 : index[0].stop or values.shape[0]]
            .reshape(shape)
            .astype(dtype),
        )

    return (
        batch_from_host(routes),
        batch_from_host(weights),
        expert_from_host(expert_scales(), jnp.bfloat16),
        expert_from_host(expected_counts, jnp.int32),
        expert_from_host(expected_probes, jnp.float32),
        expert_from_host(expected_weighted_probes, jnp.float32),
    )


def _ordered_bf16_keys(values: Any, jax: Any, jnp: Any) -> Any:
    bits = jax.lax.bitcast_convert_type(values.astype(jnp.bfloat16), jnp.uint16).astype(jnp.int32)
    magnitude = jnp.bitwise_and(bits, 0x7FFF)
    negative = jnp.bitwise_and(bits, 0x8000) != 0
    return jnp.where(negative, 0x8000 - magnitude, 0x8000 + bits)


def strict_metrics(candidate: Any, reference: Any, jax: Any, jnp: Any) -> dict[str, Any]:
    """Return fixed-tolerance error and BF16 ULP metrics."""
    candidate_f32 = candidate.astype(jnp.float32)
    reference_f32 = reference.astype(jnp.float32)
    difference = jnp.abs(candidate_f32 - reference_f32)
    tolerance = BF16_ATOL + BF16_RTOL * jnp.abs(reference_f32)
    mismatch = difference > tolerance
    error_l2 = jnp.sqrt(jnp.sum(jnp.square(difference), dtype=jnp.float32))
    reference_l2 = jnp.sqrt(jnp.sum(jnp.square(reference_f32), dtype=jnp.float32))
    ulp = jnp.abs(_ordered_bf16_keys(candidate, jax, jnp) - _ordered_bf16_keys(reference, jax, jnp))
    element_count = candidate.size
    return {
        "rtol": jnp.asarray(BF16_RTOL, dtype=jnp.float32),
        "atol": jnp.asarray(BF16_ATOL, dtype=jnp.float32),
        "allclose": jnp.logical_not(jnp.any(mismatch)),
        "mismatch_count": jnp.sum(mismatch, dtype=jnp.int32),
        "element_count": jnp.asarray(element_count, dtype=jnp.int32),
        "mismatch_fraction": jnp.sum(mismatch, dtype=jnp.float32) / element_count,
        "max_abs": jnp.max(difference),
        "mean_abs": jnp.mean(difference),
        "relative_l2_error": jnp.where(
            reference_l2 != 0,
            error_l2 / reference_l2,
            jnp.where(error_l2 == 0, 0.0, jnp.inf),
        ),
        "candidate_finite": jnp.all(jnp.isfinite(candidate_f32)),
        "reference_finite": jnp.all(jnp.isfinite(reference_f32)),
        "absolute_error_histogram": {
            "0": jnp.sum(difference == 0, dtype=jnp.int32),
            "(0,0.0009765625]": jnp.sum(
                (difference > 0) & (difference <= 0.0009765625),
                dtype=jnp.int32,
            ),
            "(0.0009765625,0.001953125]": jnp.sum(
                (difference > 0.0009765625) & (difference <= 0.001953125),
                dtype=jnp.int32,
            ),
            "(0.001953125,0.00390625]": jnp.sum(
                (difference > 0.001953125) & (difference <= 0.00390625),
                dtype=jnp.int32,
            ),
            "(0.00390625,0.0078125]": jnp.sum(
                (difference > 0.00390625) & (difference <= 0.0078125),
                dtype=jnp.int32,
            ),
            ">0.0078125": jnp.sum(difference > 0.0078125, dtype=jnp.int32),
        },
        "ulp": {
            "max": jnp.max(ulp),
            "mean": jnp.mean(ulp.astype(jnp.float32)),
            "histogram": {
                "0": jnp.sum(ulp == 0, dtype=jnp.int32),
                "1": jnp.sum(ulp == 1, dtype=jnp.int32),
                "2": jnp.sum(ulp == 2, dtype=jnp.int32),
                "3-4": jnp.sum((ulp >= 3) & (ulp <= 4), dtype=jnp.int32),
                "5-8": jnp.sum((ulp >= 5) & (ulp <= 8), dtype=jnp.int32),
                "9-16": jnp.sum((ulp >= 9) & (ulp <= 16), dtype=jnp.int32),
                ">16": jnp.sum(ulp > 16, dtype=jnp.int32),
            },
        },
    }


def _exact_metrics(candidate: Any, reference: Any, jnp: Any) -> dict[str, Any]:
    difference = jnp.abs(candidate.astype(jnp.float32) - reference.astype(jnp.float32))
    mismatch = candidate != reference
    return {
        "exact": jnp.logical_not(jnp.any(mismatch)),
        "mismatch_count": jnp.sum(mismatch, dtype=jnp.int32),
        "element_count": jnp.asarray(candidate.size, dtype=jnp.int32),
        "max_abs": jnp.max(difference),
        "mean_abs": jnp.mean(difference),
        "candidate_finite": jnp.all(jnp.isfinite(candidate)),
        "reference_finite": jnp.all(jnp.isfinite(reference)),
    }


def _references(
    tokens: Any,
    routes: Any,
    weights: Any,
    scales: Any,
    *,
    route_sharding: Any,
    top_k: int,
    expert_scaled: bool,
    jax: Any,
    jnp: Any,
) -> dict[str, Any]:
    selected_scales = (
        scales.at[routes].get(out_sharding=route_sharding)
        if expert_scaled
        else jnp.ones_like(weights, dtype=jnp.bfloat16)
    )
    coefficients_f32 = weights.astype(jnp.float32) * selected_scales.astype(jnp.float32)
    analytic = (tokens.astype(jnp.float32) * jnp.sum(coefficients_f32, axis=1, dtype=jnp.float32)[:, None]).astype(
        jnp.bfloat16
    )

    def accumulate(route_order: tuple[int, ...]) -> Any:
        output = jnp.zeros_like(tokens)
        for route_slot in route_order:
            expert_output = tokens
            if expert_scaled:
                expert_output = expert_output * selected_scales[:, route_slot, None]
            contribution = expert_output * weights[:, route_slot, None].astype(jnp.bfloat16)
            output = output + contribution
        return output

    return {
        REFERENCE_FP32: analytic,
        REFERENCE_FORWARD: accumulate(tuple(range(top_k))),
        REFERENCE_REVERSE: accumulate(tuple(reversed(range(top_k)))),
    }


def _compiled_case(
    jax: Any,
    jnp: Any,
    mesh: Any,
    *,
    spec: CaseSpec,
    layer_config: Any,
    ep_dispatch: Any,
    ep_combine: Any,
) -> Any:
    P = jax.sharding.PartitionSpec
    batch_spec = P(("replica_dcn", "data", "expert"), None)
    lead = P(batch_spec[0], None, None)
    lead2 = P(batch_spec[0], None)
    expert_vector_spec = P("expert")
    expert_probe_spec = P("expert", None)

    def local_transform(
        recv_tokens: Any,
        recv_weights: Any,
        token_counts: Any,
        local_scales: Any,
    ) -> tuple[Any, ...]:
        flat_tokens = recv_tokens.reshape(recv_tokens.shape[-2], recv_tokens.shape[-1])
        flat_weights = recv_weights.reshape(-1)
        counts = token_counts.reshape(-1).astype(jnp.int32)
        valid_count = jnp.sum(counts, dtype=jnp.int32)
        padded_counts = counts.at[-1].add(flat_tokens.shape[0] - valid_count)
        group_ids = jnp.repeat(
            jnp.arange(LOCAL_EXPERTS, dtype=jnp.int32),
            padded_counts,
            total_repeat_length=flat_tokens.shape[0],
        )
        valid = jnp.arange(flat_tokens.shape[0], dtype=jnp.int32) < valid_count
        probes = jnp.where(valid[:, None], flat_tokens[:, :PROBE_FEATURES].astype(jnp.float32), 0.0)
        observed_probe_sums = jax.ops.segment_sum(probes, group_ids, num_segments=LOCAL_EXPERTS)
        observed_weighted_probe_sums = jax.ops.segment_sum(
            probes * flat_weights[:, None],
            group_ids,
            num_segments=LOCAL_EXPERTS,
        )

        expert_output = flat_tokens
        if spec.expert_scaled:
            row_scales = jnp.repeat(
                local_scales,
                padded_counts,
                total_repeat_length=flat_tokens.shape[0],
            )
            expert_output = expert_output * row_scales[:, None]
        weighted_bf16 = expert_output * flat_weights[:, None].astype(jnp.bfloat16)
        weighted_bf16 = jnp.where(valid[:, None], weighted_bf16, jnp.zeros((), dtype=jnp.bfloat16))
        weighted = weighted_bf16
        if spec.combine_input_dtype == "float32":
            weighted = weighted.astype(jnp.float32)
        return (
            weighted.reshape((*recv_tokens.shape[:-1], HIDDEN_DIM)),
            observed_probe_sums,
            observed_weighted_probe_sums,
            counts,
        )

    def body(
        tokens: Any,
        routes: Any,
        weights: Any,
        scales: Any,
    ) -> tuple[Any, ...]:
        recv_tokens, recv_weights, handle_memory, token_counts = ep_dispatch(
            layer_config,
            routes.astype(jnp.int32),
            tokens,
            weights.astype(jnp.float32),
            RECV_CAPACITY_PER_RANK,
        )
        recv_tokens = jax.lax.with_sharding_constraint(recv_tokens, lead)
        recv_weights = jax.lax.with_sharding_constraint(recv_weights, lead2)
        token_counts = jax.lax.with_sharding_constraint(token_counts, lead2)
        transform = jax.shard_map(
            local_transform,
            mesh=jax.sharding.get_abstract_mesh(),
            in_specs=(lead, lead2, lead2, expert_vector_spec),
            out_specs=(lead, expert_probe_spec, expert_probe_spec, expert_vector_spec),
            check_vma=False,
        )
        weighted, observed_probes, observed_weighted_probes, observed_counts = transform(
            recv_tokens,
            recv_weights,
            token_counts,
            scales,
        )
        weighted = jax.lax.with_sharding_constraint(weighted, lead)
        output = ep_combine(
            layer_config,
            handle_memory,
            token_counts,
            weighted,
            tuple(tokens.shape[:-1]),
        )
        return output, observed_probes, observed_weighted_probes, observed_counts

    round_trip = jax.sharding.auto_axes(
        body,
        axes=tuple(mesh.axis_names),
        out_sharding=(batch_spec, expert_probe_spec, expert_probe_spec, expert_vector_spec),
    )

    def evaluate(
        tokens: Any,
        routes: Any,
        weights: Any,
        scales: Any,
        expected_counts: Any,
        expected_probes: Any,
        expected_weighted_probes: Any,
    ) -> dict[str, Any]:
        output, observed_probes, observed_weighted_probes, observed_counts = round_trip(
            tokens,
            routes,
            weights,
            scales,
        )
        references = _references(
            tokens,
            routes,
            weights,
            scales,
            route_sharding=lead2,
            top_k=spec.top_k,
            expert_scaled=spec.expert_scaled,
            jax=jax,
            jnp=jnp,
        )
        reference_metrics = {name: strict_metrics(output, reference, jax, jnp) for name, reference in references.items()}
        dispatch = {
            "counts": _exact_metrics(observed_counts, expected_counts, jnp),
            "token_bit_sums": _exact_metrics(observed_probes, expected_probes, jnp),
            "routing_weighted_token_bit_sums": _exact_metrics(
                observed_weighted_probes,
                expected_weighted_probes,
                jnp,
            ),
        }
        return {
            "output_finite": jnp.all(jnp.isfinite(output)),
            "dispatch": dispatch,
            "references": reference_metrics,
        }

    return jax.jit(evaluate)


def _python_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _python_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_python_value(item) for item in value]
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _case_passes(case: dict[str, Any], reference: str = REFERENCE_FORWARD) -> bool:
    return (
        case.get("status") == "completed"
        and bool(case["output_finite"])
        and bool(case["dispatch"]["passed"])
        and bool(case["references"][reference]["allclose"])
    )


def attribute_results(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Attribute the discrepancy as narrowly as the completed cases permit."""
    completed = [case for case in cases.values() if case.get("status") == "completed"]
    dispatch_passed = bool(completed) and all(bool(case["dispatch"]["passed"]) for case in completed)
    topk1_identity = _case_passes(cases["topk1_identity"])
    topk1_scaled = _case_passes(cases["topk1_expert_scaled_identity"])
    topk4_identity = _case_passes(cases["topk4_identity"])
    topk4_scaled = _case_passes(cases["topk4_expert_scaled_identity"])
    topk1_completed = all(
        cases[name].get("status") == "completed" for name in ("topk1_identity", "topk1_expert_scaled_identity")
    )
    topk4_completed = all(
        cases[name].get("status") == "completed" for name in ("topk4_identity", "topk4_expert_scaled_identity")
    )

    scaled_metrics = cases["topk4_expert_scaled_identity"].get("references", {})
    closest_reference = None
    if scaled_metrics:
        closest_reference = min(
            scaled_metrics,
            key=lambda name: (
                float(scaled_metrics[name]["relative_l2_error"]),
                int(scaled_metrics[name]["mismatch_count"]),
            ),
        )

    if not dispatch_passed:
        attribution = "dispatch_membership_or_route_weight_transport"
    elif topk1_completed and not topk4_completed:
        attribution = "topk1_completed_topk4_not_run_in_this_process"
    elif topk4_completed and not topk1_completed:
        attribution = "topk4_completed_topk1_not_run_in_this_process"
    elif not topk1_identity:
        attribution = "single_route_weight_application_or_combine"
    elif not topk1_scaled:
        attribution = "expert_transform_or_single_route_weight_application"
    elif not topk4_identity or not topk4_scaled:
        attribution = "multi_route_weight_application_or_combine_accumulation_order"
    else:
        attribution = "not_reproduced_by_dispatch_weighted_identity_combine"

    fp32_case = cases["topk4_expert_scaled_identity_fp32_combine_input"]
    return {
        "most_specific_attribution": attribution,
        "dispatch_fingerprints_exact": dispatch_passed,
        "topk1_identity_strict_parity": topk1_identity,
        "topk1_expert_scaled_identity_strict_parity": topk1_scaled,
        "topk4_identity_strict_parity": topk4_identity,
        "topk4_expert_scaled_identity_strict_parity": topk4_scaled,
        "topk4_scaled_closest_reference": closest_reference,
        "fp32_combine_input": {
            "status": fp32_case.get("status"),
            "strict_parity": _case_passes(fp32_case) if fp32_case.get("status") == "completed" else None,
        },
        "expert_gemm": {
            "status": "not_exercised",
            "interpretation": (
                "a reproduced discrepancy excludes expert GEMM; a clean result leaves GEMM or its interaction in scope"
            ),
        },
        "limits": [
            (
                "exact dispatch fingerprints validate counts, token identity bits, and route-weight attachment, "
                "not every hidden element"
            ),
            "a closest BF16 reference is evidence about accumulation order only when its error is materially smaller",
        ],
    }


def _dispatch_passed(dispatch: dict[str, Any]) -> bool:
    return all(bool(metrics["exact"]) for metrics in dispatch.values())


def run_isolation() -> int:
    # TE registers NCCL_EP FFI handlers during import, before initialize_jax
    # creates the CUDA client.
    transformer_engine = importlib.import_module("transformer_engine")
    te_ep = importlib.import_module("transformer_engine.jax.ep")
    te_sharding = importlib.import_module("transformer_engine.jax.sharding")
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    jmu = importlib.import_module("jax.experimental.multihost_utils")
    initialize_jax = importlib.import_module("iris.runtime.jax_init").initialize_jax
    compact_grug_mesh = importlib.import_module("levanter.grug.sharding").compact_grug_mesh

    initialize_jax()
    rank, world = _assert_topology(jax)
    selected_top_k = int(os.environ[TOP_K_ENV])
    selected_specs = case_specs_for_top_k(selected_top_k)
    mesh = compact_grug_mesh(expert_axis_size=EP_SIZE, replica_axis_size=1)
    P = jax.sharding.PartitionSpec
    batch_sharding = jax.sharding.NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
    expert_sharding = jax.sharding.NamedSharding(mesh, P("expert"))
    MeshResource = te_sharding.MeshResource
    global_shard_guard = te_sharding.global_shard_guard

    routes_by_top_k = {selected_top_k: balanced_route_table(selected_top_k)}
    routing = {str(top_k): route_capacity_report(routes) for top_k, routes in routes_by_top_k.items()}
    topk4_distinct = (
        distinct_route_contributions(
            routes_by_top_k[4],
            route_weights(4),
            expert_scales(),
        )
        if selected_top_k == 4
        else None
    )
    if selected_top_k == 4 and not topk4_distinct:
        raise RuntimeError("top-k4 scaled route contributions must be distinct")

    cases: dict[str, dict[str, Any]] = {
        spec.name: {
            "top_k": spec.top_k,
            "expert_transform": "per_expert_scaled_identity" if spec.expert_scaled else "identity",
            "combine_input_dtype": spec.combine_input_dtype,
            "status": "not_run_in_this_process",
        }
        for spec in case_specs()
    }
    with jax.set_mesh(mesh), global_shard_guard(MeshResource(dp_resource="data", ep_resource="expert")):
        te_ep.ep_bootstrap(
            world_size=world,
            rank=rank,
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=TOKENS_PER_RANK,
            recv_capacity_per_rank=RECV_CAPACITY_PER_RANK,
            hidden_dim=HIDDEN_DIM,
        )
        tokens = _make_tokens(jax, jnp, batch_sharding)
        inputs_by_top_k = {
            selected_top_k: _make_case_inputs(
                jax,
                jnp,
                batch_sharding,
                expert_sharding,
                selected_top_k,
            )
        }

        for spec in selected_specs:
            layer_config = te_ep.EpLayerConfig(
                top_k=spec.top_k,
                dispatch_output_per_expert_alignment=DISPATCH_ALIGNMENT,
            )
            evaluator = _compiled_case(
                jax,
                jnp,
                mesh,
                spec=spec,
                layer_config=layer_config,
                ep_dispatch=te_ep.ep_dispatch,
                ep_combine=te_ep.ep_combine,
            )
            case_inputs = (tokens, *inputs_by_top_k[spec.top_k])
            try:
                result = jax.block_until_ready(evaluator(*case_inputs))
                case = _python_value(jax.device_get(result))
                case["status"] = "completed"
                case["dispatch"]["passed"] = _dispatch_passed(case["dispatch"])
            except Exception as error:
                if spec.combine_input_dtype != "float32":
                    raise
                case = {
                    "status": "unsupported",
                    "error_type": type(error).__name__,
                    "error": str(error)[:2_000],
                    "interpretation": "TE NCCL_EP did not accept FP32 combine input; BF16 tolerances were not changed",
                }
            support = int(case["status"] == "completed")
            support_by_rank = np.asarray(
                jmu.process_allgather(np.asarray([support], dtype=np.int32), tiled=False)
            ).reshape(-1)
            if not np.all(support_by_rank == support_by_rank[0]):
                raise RuntimeError(f"inconsistent case support across ranks for {spec.name}: {support_by_rank.tolist()}")
            cases[spec.name] = {
                "top_k": spec.top_k,
                "expert_transform": "per_expert_scaled_identity" if spec.expert_scaled else "identity",
                "combine_input_dtype": spec.combine_input_dtype,
                **case,
            }
            jmu.sync_global_devices(f"ncclep-combine-parity-{spec.name}")

    summary = {
        "event": SUMMARY_EVENT,
        "schema_version": 1,
        "status": "completed",
        "purpose": "numerical_attribution_only",
        "prior_full_mlp_evidence": {
            "shape": "e64/top-k4/d2560/i1280/16384_tokens_per_rank",
            "output_strict_mismatch_fraction": 0.00203554,
            "output_relative_l2_error": 0.002962,
            "output_max_abs": 0.0078125,
            "loss_and_all_gradients_strict_parity": True,
            "te_value_and_grad_speedup_over_marin_ring": 1.4525,
        },
        "shape": {
            "topology": "one_node_8_processes_x_1_h100",
            "ep": EP_SIZE,
            "tokens_per_rank": TOKENS_PER_RANK,
            "global_tokens": GLOBAL_TOKENS,
            "hidden_dim": HIDDEN_DIM,
            "num_experts": NUM_EXPERTS,
            "local_experts": LOCAL_EXPERTS,
            "top_k_values": [selected_top_k],
            "token_dtype": "bfloat16",
            "routing_weight_dtype": "float32",
            "capacity_factor": CAPACITY_FACTOR,
            "seed": SEED,
            "probe_features": PROBE_FEATURES,
        },
        "runtime": {
            "rank_count": world,
            "local_devices_per_rank": 1,
            "gpu": str(jax.local_devices()[0].device_kind),
            "jax_version": jax.__version__,
            "transformer_engine_version": transformer_engine.__version__,
            "te_sha": os.environ["NCCLEP_TE_SHA"],
            "nccl_runtime_version": os.environ["NCCLEP_NCCL_RUNTIME_VERSION"],
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "xla_preallocation_fraction": float(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]),
        },
        "strict_tolerances": {
            "rtol": BF16_RTOL,
            "atol": BF16_ATOL,
            "source": "experiments/grug/moe/benchmark_ep_ring.py strict BF16 parity",
            "changed": False,
        },
        "routing_capacity": routing,
        "topk4_scaled_route_contributions_distinct": topk4_distinct,
        "cases": cases,
        "attribution": attribute_results(cases),
        "full_mlp_checkpoint": {
            "status": "not_run",
            "reason": (
                "the transport-only cases remove expert GEMM and directly test dispatch, weight application, and combine"
            ),
        },
        "promotion_decision": None,
    }
    if rank == 0:
        print(json.dumps(summary, sort_keys=True), flush=True)
    jmu.sync_global_devices("ncclep-combine-parity-summary-emitted")
    return 0


def main() -> int:
    try:
        return run_isolation()
    except Exception as error:
        if os.environ.get("IRIS_MULTIGPU_PROCESS_INDEX", "0") == "0":
            print(
                json.dumps(
                    {
                        "event": SUMMARY_EVENT,
                        "schema_version": 1,
                        "status": "error",
                        "selected_top_k": os.environ.get(TOP_K_ENV),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        raise


if __name__ == "__main__":
    sys.exit(main())
