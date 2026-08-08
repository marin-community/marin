# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched MiniMax Sparse Attention oracle for Shuttle SM100 benchmarks.

This module is oracle-only. It must never be imported by an accepted Shuttle
execution path. It exposes the pinned MSA implementation as a payload callable
whose timed boundary starts from q2k block indices and includes:

    q2k -> k2q CSR and physical schedule -> sparse QK/Fold/PV -> combine

``compare_matched_boundaries`` measures that payload boundary and a natural
full-route boundary. The latter invokes the same caller-supplied route function
inside both the Shuttle and MSA operations. Thus both implementations see the
same route semantics while paying the route cost independently.

The module intentionally imports Torch and MSA lazily. Its validation and
counterbalancing logic can be tested on a CPU host without either dependency.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import statistics
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

MINIMAX_MSA_COMMIT = "80434d7f67877c6570ca19cac444b84bc9855dac"
MINIMAX_MSA_CUTLASS_COMMIT = "eb61c911471867a5fd2466bfd8f29306cea6ebf8"
MINIMAX_MSA_CUTLASS_PATH = "python/fmha_sm100/cutlass"
MINIMAX_MSA_REPOSITORY = "https://github.com/MiniMax-AI/MSA"


@dataclass(frozen=True)
class MsaOracleWorkload:
    """Tensor inputs and explicit numerical choices for one MSA payload."""

    query: Any
    key_pages: Any
    value_pages: Any
    query_segment_lengths: Any
    key_value_segment_lengths: Any
    key_value_page_indices: Any
    query_offsets: Any
    block_size: int
    top_k: int
    softmax_scale: float
    partial_dtype: Any
    usable_sm_count: int = -1


@dataclass(frozen=True)
class CorrectnessTolerance:
    """Required pointwise error limits for a matched comparison."""

    maximum_absolute_error: float
    mean_absolute_error: float


@dataclass(frozen=True)
class MsaModules:
    """Pinned MSA functions used by the oracle adapter."""

    plan: Callable[..., dict[str, Any]]
    build_k2q_csr: Callable[..., tuple[Any, ...]]
    sparse_attention: Callable[..., Any]
    build_page_table: Callable[..., Any]


@dataclass(frozen=True)
class TensorCapture:
    """Host numerical values and a hash of the original tensor representation."""

    numerical: np.ndarray
    sha256: str


def _git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def minimax_msa_source_record(msa_root: Path, *, allow_dirty: bool = False) -> dict[str, Any]:
    """Validate the pinned MSA checkout and return reproducibility metadata."""
    root = msa_root.resolve()
    revision = _git_output(root, "rev-parse", "HEAD")
    if revision != MINIMAX_MSA_COMMIT:
        raise ValueError(f"MSA checkout is {revision}; expected {MINIMAX_MSA_COMMIT}")

    cutlass_revision = _git_output(root, "rev-parse", f"HEAD:{MINIMAX_MSA_CUTLASS_PATH}")
    if cutlass_revision != MINIMAX_MSA_CUTLASS_COMMIT:
        raise ValueError(f"MSA CUTLASS gitlink is {cutlass_revision}; expected {MINIMAX_MSA_CUTLASS_COMMIT}")

    modifications = _git_output(root, "status", "--short")
    if modifications and not allow_dirty:
        raise ValueError("MSA checkout has local modifications; pass allow_dirty=True to record and use them")

    return {
        "repository": MINIMAX_MSA_REPOSITORY,
        "root": str(root),
        "commit": revision,
        "cutlass_path": MINIMAX_MSA_CUTLASS_PATH,
        "cutlass_commit": cutlass_revision,
        "local_modifications": modifications.splitlines(),
    }


def _module_under_root(module: ModuleType, root: Path) -> bool:
    module_path = getattr(module, "__file__", None)
    if module_path is None:
        return False
    try:
        Path(module_path).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def load_minimax_msa_modules(msa_root: Path) -> MsaModules:
    """Load the oracle functions from the pinned MSA source tree."""
    python_root = (msa_root / "python").resolve()
    python_root_string = str(python_root)
    if python_root_string not in sys.path:
        sys.path.insert(0, python_root_string)

    adapter = importlib.import_module("fmha_sm100.sparse_fmha_adapter")
    if not _module_under_root(adapter, msa_root):
        raise RuntimeError(f"loaded MSA adapter from {adapter.__file__}, not {msa_root}")
    return MsaModules(
        plan=adapter.sparse_fmha_plan,
        build_k2q_csr=adapter.build_k2q_csr,
        sparse_attention=adapter.sparse_atten_func,
        build_page_table=adapter._build_page_table,
    )


def validate_q2k_indices(
    q2k_indices: np.ndarray,
    *,
    key_value_heads: int,
    total_queries: int,
    top_k: int,
    maximum_key_value_blocks: int,
) -> None:
    """Validate MSA's canonical ``[H_kv, total_q, top_k]`` relation layout."""
    expected_shape = (key_value_heads, total_queries, top_k)
    if q2k_indices.shape != expected_shape:
        raise ValueError(f"q2k indices must have shape {expected_shape}, got {q2k_indices.shape}")
    if q2k_indices.dtype != np.int32:
        raise TypeError(f"q2k indices must use int32, got {q2k_indices.dtype}")
    if maximum_key_value_blocks <= 0:
        raise ValueError("maximum key-value block count must be positive")

    for head_index, query_index in np.ndindex(q2k_indices.shape[:2]):
        row = q2k_indices[head_index, query_index]
        padded = row == -1
        first_padding = int(np.argmax(padded)) if padded.any() else row.size
        if not padded[first_padding:].all():
            raise ValueError("q2k -1 padding must form a contiguous tail")
        selected = row[:first_padding]
        if selected.size == 0:
            raise ValueError(f"q2k row ({head_index}, {query_index}) selects no KV block")
        if np.any(selected < 0) or np.any(selected >= maximum_key_value_blocks):
            raise ValueError(f"q2k row ({head_index}, {query_index}) contains an out-of-range block")
        if np.any(selected[1:] <= selected[:-1]):
            raise ValueError(f"q2k row ({head_index}, {query_index}) must be strictly ascending before padding")


def tensor_numpy(value: Any) -> np.ndarray:
    """Copy a NumPy or Torch tensor into a contiguous host array."""
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value).copy()
    detached = value.detach().contiguous().cpu()
    try:
        host = detached.numpy()
    except TypeError:
        # NumPy does not represent torch.bfloat16. Converting to FP32 preserves
        # every BF16 value exactly for numerical comparison.
        host = detached.float().numpy()
    return np.ascontiguousarray(host).copy()


def _framed_sha256(dtype: str, shape: tuple[int, ...], payload: bytes) -> str:
    header = json.dumps(
        {"dtype": dtype, "shape": shape},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(payload)
    return digest.hexdigest()


def framed_array_sha256(value: np.ndarray) -> str:
    """Hash dtype, shape, and bytes so equal payload bytes cannot hide a shape mismatch."""
    contiguous = np.ascontiguousarray(value)
    return _framed_sha256(str(contiguous.dtype), contiguous.shape, contiguous.tobytes(order="C"))


def tensor_sha256(value: Any) -> str:
    """Hash a NumPy or Torch tensor without changing its dtype representation."""
    if isinstance(value, np.ndarray):
        return framed_array_sha256(value)
    detached = value.detach().contiguous()
    byte_payload = detached.view(importlib.import_module("torch").uint8).cpu().numpy().tobytes(order="C")
    return _framed_sha256(str(detached.dtype), tuple(detached.shape), byte_payload)


def alternating_launch_orders(labels: tuple[str, str], rounds: int) -> list[tuple[str, str]]:
    """Return alternating A/B and B/A launch orders."""
    if rounds < 0:
        raise ValueError("round count must be non-negative")
    return [labels if index % 2 == 0 else tuple(reversed(labels)) for index in range(rounds)]


def measure_counterbalanced_pair(
    generated: Callable[[], Any],
    oracle: Callable[[], Any],
    *,
    measure_one: Callable[[Callable[[], Any]], float],
    synchronize: Callable[[], None],
    warmups: int,
    repeats: int,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Measure a matched pair while alternating which implementation runs first."""
    if repeats <= 0 or repeats % 2:
        raise ValueError("counterbalanced measurement requires a positive even repeat count")
    if warmups < 0:
        raise ValueError("warmup count must be non-negative")

    labels = ("generated_shuttle", "matched_msa_oracle")
    operations = dict(zip(labels, (generated, oracle), strict=True))
    warmup_orders = alternating_launch_orders(labels, warmups)
    for order in warmup_orders:
        for label in order:
            operations[label]()
    synchronize()

    samples = {label: [] for label in labels}
    sample_orders = alternating_launch_orders(labels, repeats)
    for order in sample_orders:
        for label in order:
            samples[label].append(float(measure_one(operations[label])))

    return samples, {
        "name": "alternating_counterbalanced_pairs",
        "variants": labels,
        "warmup_launch_orders": warmup_orders,
        "sample_launch_orders": sample_orders,
        "sample_pairing": "samples at the same index form one Shuttle/MSA pair",
        "timing": "one independent CUDA-event interval per operation",
    }


def cuda_event_measure_one(operation: Callable[[], Any]) -> float:
    """Measure one operation with CUDA events, synchronizing only its end event."""
    torch = importlib.import_module("torch")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _sample_record(samples: list[float]) -> dict[str, Any]:
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
    }


def _numerical_record(generated: np.ndarray, oracle: np.ndarray) -> dict[str, float]:
    if generated.shape != oracle.shape:
        raise ValueError(f"generated and oracle outputs differ in shape: {generated.shape} != {oracle.shape}")
    generated_fp32 = generated.astype(np.float32)
    oracle_fp32 = oracle.astype(np.float32)
    difference = np.abs(generated_fp32 - oracle_fp32)
    return {
        "maximum_absolute_error": float(difference.max(initial=0.0)),
        "mean_absolute_error": float(difference.mean()) if difference.size else 0.0,
    }


def _capture(operation: Callable[[], Any], synchronize: Callable[[], None]) -> TensorCapture:
    result = operation()
    synchronize()
    return TensorCapture(numerical=tensor_numpy(result), sha256=tensor_sha256(result))


def _correctness_and_determinism(
    generated: Callable[[], Any],
    oracle: Callable[[], Any],
    semantic_reference: Callable[[], Any],
    *,
    correctness_projection: Callable[[Any], Any],
    synchronize: Callable[[], None],
    tolerance: CorrectnessTolerance,
) -> dict[str, Any]:
    generated_result = generated()
    synchronize()
    generated_first = TensorCapture(
        numerical=tensor_numpy(correctness_projection(generated_result)),
        sha256=tensor_sha256(generated_result),
    )
    oracle_result = oracle()
    synchronize()
    oracle_first = TensorCapture(
        numerical=tensor_numpy(correctness_projection(oracle_result)),
        sha256=tensor_sha256(oracle_result),
    )
    reference = _capture(semantic_reference, synchronize)
    generated_second = _capture(generated, synchronize)
    oracle_second = _capture(oracle, synchronize)
    generated_vs_msa = _numerical_record(generated_first.numerical, oracle_first.numerical)
    generated_vs_reference = _numerical_record(generated_first.numerical, reference.numerical)
    msa_vs_reference = _numerical_record(oracle_first.numerical, reference.numerical)
    for label, numerical in (
        ("generated/semantic reference", generated_vs_reference),
        ("MSA/semantic reference", msa_vs_reference),
    ):
        if numerical["maximum_absolute_error"] > tolerance.maximum_absolute_error:
            raise ValueError(
                f"{label} maximum absolute error exceeds the declared tolerance: "
                f"{numerical['maximum_absolute_error']} > {tolerance.maximum_absolute_error}"
            )
        if numerical["mean_absolute_error"] > tolerance.mean_absolute_error:
            raise ValueError(
                f"{label} mean absolute error exceeds the declared tolerance: "
                f"{numerical['mean_absolute_error']} > {tolerance.mean_absolute_error}"
            )

    generated_hash = generated_first.sha256
    oracle_hash = oracle_first.sha256
    return {
        "generated_vs_msa": generated_vs_msa,
        "generated_vs_semantic_reference": generated_vs_reference,
        "msa_vs_semantic_reference": msa_vs_reference,
        "declared_tolerance": {
            "maximum_absolute_error": tolerance.maximum_absolute_error,
            "mean_absolute_error": tolerance.mean_absolute_error,
        },
        "generated_output_sha256": generated_hash,
        "msa_output_sha256": oracle_hash,
        "generated_repeat_sha256": generated_second.sha256,
        "msa_repeat_sha256": oracle_second.sha256,
        "semantic_reference_sha256": reference.sha256,
        "generated_deterministic": generated_hash == generated_second.sha256,
        "msa_deterministic": oracle_hash == oracle_second.sha256,
    }


def build_minimax_msa_payload(
    msa_root: Path,
    workload: MsaOracleWorkload,
) -> tuple[Callable[[Any], Any], dict[str, Any]]:
    """Build the MSA q2k-to-output oracle callable.

    Shape-only planning and the static paged-KV table happen before timing. Every
    payload call rebuilds the k2q CSR and physical work schedule from q2k.
    """
    source = minimax_msa_source_record(msa_root)
    modules = load_minimax_msa_modules(msa_root)

    query = workload.query
    key = workload.key_pages
    value = workload.value_pages
    if query.ndim != 3 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("MSA expects query [T,Hq,D] and paged K/V [P,Hkv,B,D]")
    if key.shape != value.shape:
        raise ValueError("MSA key and value page tensors must have identical shapes")
    if query.shape[-1] != 128 or key.shape[-1] != 128:
        raise ValueError("pinned MSA sparse prefill supports head dimension 128")
    if key.shape[2] != workload.block_size:
        raise ValueError("KV page size must equal the selected block size")
    if query.shape[1] % key.shape[1]:
        raise ValueError("query head count must be divisible by KV head count")
    if workload.top_k not in {4, 8, 16, 32}:
        raise ValueError("pinned MSA sparse prefill supports top-k 4, 8, 16, or 32")
    if not math.isfinite(workload.softmax_scale) or workload.softmax_scale <= 0.0:
        raise ValueError("softmax scale must be finite and positive")

    plan = modules.plan(
        workload.query_segment_lengths,
        workload.key_value_segment_lengths,
        int(query.shape[1]),
        num_kv_heads=int(key.shape[1]),
        qo_offset=workload.query_offsets,
        page_size=workload.block_size,
        kv_block_num=workload.top_k,
        causal=True,
        usable_SM_count=workload.usable_sm_count,
    )
    if workload.usable_sm_count > 0:
        raise ValueError("matched oracle currently requires usable_sm_count=-1 so CSR building returns its schedule")

    page_table = modules.build_page_table(
        workload.key_value_page_indices,
        plan["kv_segment_lens"],
        workload.block_size,
        plan["batch"],
    )
    seqused_k = plan["seqused_k"]

    def payload(q2k_indices: Any) -> Any:
        k2q_row_ptr, k2q_q_indices, schedule = modules.build_k2q_csr(
            q2k_indices,
            plan["cu_seqlens_q"],
            plan["cu_seqlens_k"],
            workload.block_size,
            total_k=plan["total_k"],
            max_seqlen_k=plan["max_seqlen_k"],
            max_seqlen_q=plan["max_seqlen_q"],
            total_rows=plan["total_rows"],
            qhead_per_kv=plan["qhead_per_kv"],
            return_schedule=True,
        )
        return modules.sparse_attention(
            query,
            key,
            value,
            k2q_row_ptr,
            k2q_q_indices,
            workload.top_k,
            cu_seqlens_q=plan["cu_seqlens_q"],
            cu_seqlens_k=plan["cu_seqlens_k"],
            max_seqlen_q=plan["max_seqlen_q"],
            max_seqlen_k=plan["max_seqlen_k"],
            blk_kv=workload.block_size,
            causal=True,
            softmax_scale=workload.softmax_scale,
            partial_dtype=workload.partial_dtype,
            return_softmax_lse=False,
            page_table=page_table,
            seqused_k=seqused_k,
            schedule=schedule,
            usable_SM_count=-1,
        )

    manifest = {
        "implementation": "pinned MiniMax MSA SM100 expert oracle",
        "source": source,
        "interface": "q2k [H_kv,total_q,top_k] -> output [total_q,H_q,128]",
        "included_per_payload_call": (
            "q2k-to-k2q CSR",
            "physical sparse schedule construction",
            "KV-outer sparse QK/normalized-exp/PV",
            "deterministic two-phase partial combine",
        ),
        "excluded_from_payload_call": (
            "route/index projection",
            "token score Contract",
            "block-max Fold",
            "top-k Selection",
            "static shape plan",
            "static page table",
        ),
        "q2k_contract": {
            "key_value_heads": int(key.shape[1]),
            "total_queries": int(query.shape[0]),
            "top_k": workload.top_k,
            "maximum_key_value_blocks": int(
                np.ceil(tensor_numpy(plan["kv_segment_lens"]).astype(np.float64) / workload.block_size).max()
            ),
        },
        "numerics": {
            "query_dtype": str(query.dtype),
            "key_dtype": str(key.dtype),
            "value_dtype": str(value.dtype),
            "partial_dtype": str(workload.partial_dtype),
            "softmax_scale": workload.softmax_scale,
        },
        "oracle_only": True,
    }
    return payload, manifest


def compare_matched_boundaries(
    *,
    generated_payload: Callable[[Any], Any],
    msa_payload: Callable[[Any], Any],
    precomputed_q2k: Any,
    common_route: Callable[[], Any],
    semantic_reference: Callable[[Any], Any],
    tolerance: CorrectnessTolerance,
    oracle_manifest: dict[str, Any],
    warmups: int,
    repeats: int,
    correctness_projection: Callable[[Any], Any] | None = None,
    measure_one: Callable[[Callable[[], Any]], float] = cuda_event_measure_one,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Compare Shuttle and MSA at identical payload and natural-route boundaries."""
    if synchronize is None:
        torch = importlib.import_module("torch")
        synchronize = torch.cuda.synchronize
    projection = correctness_projection if correctness_projection is not None else lambda output: output

    precomputed_route = tensor_numpy(precomputed_q2k)
    precomputed_hash = framed_array_sha256(precomputed_route)

    def generated_payload_operation() -> Any:
        return generated_payload(precomputed_q2k)

    def msa_payload_operation() -> Any:
        return msa_payload(precomputed_q2k)

    q2k_contract = oracle_manifest.get("q2k_contract")
    if q2k_contract is not None:
        validate_q2k_indices(precomputed_route, **q2k_contract)

    first_natural_route = _capture(common_route, synchronize)
    second_natural_route = _capture(common_route, synchronize)
    first_natural_hash = first_natural_route.sha256
    second_natural_hash = second_natural_route.sha256
    if first_natural_hash != second_natural_hash:
        raise ValueError("common route callable is not deterministic across repeated execution")
    if q2k_contract is not None:
        validate_q2k_indices(first_natural_route.numerical, **q2k_contract)

    def generated_full_operation() -> Any:
        return generated_payload(common_route())

    def msa_full_operation() -> Any:
        return msa_payload(common_route())

    payload_correctness = _correctness_and_determinism(
        generated_payload_operation,
        msa_payload_operation,
        lambda: semantic_reference(precomputed_q2k),
        correctness_projection=projection,
        synchronize=synchronize,
        tolerance=tolerance,
    )
    full_correctness = _correctness_and_determinism(
        generated_full_operation,
        msa_full_operation,
        lambda: semantic_reference(common_route()),
        correctness_projection=projection,
        synchronize=synchronize,
        tolerance=tolerance,
    )
    payload_samples, payload_protocol = measure_counterbalanced_pair(
        generated_payload_operation,
        msa_payload_operation,
        measure_one=measure_one,
        synchronize=synchronize,
        warmups=warmups,
        repeats=repeats,
    )
    full_samples, full_protocol = measure_counterbalanced_pair(
        generated_full_operation,
        msa_full_operation,
        measure_one=measure_one,
        synchronize=synchronize,
        warmups=warmups,
        repeats=repeats,
    )

    def boundary_record(
        samples: dict[str, list[float]],
        protocol: dict[str, Any],
        correctness: dict[str, Any],
    ) -> dict[str, Any]:
        generated = _sample_record(samples["generated_shuttle"])
        oracle = _sample_record(samples["matched_msa_oracle"])
        return {
            "generated_shuttle": generated,
            "matched_msa_oracle": oracle,
            "generated_to_oracle_ratio": generated["median_ms"] / oracle["median_ms"],
            "measurement_protocol": protocol,
            "correctness": correctness,
        }

    return {
        "benchmark": "shuttle_vs_minimax_msa_sm100",
        "oracle_manifest": oracle_manifest,
        "relation": {
            "precomputed_q2k_sha256": precomputed_hash,
            "natural_q2k_sha256": first_natural_hash,
            "natural_route_deterministic": first_natural_hash == second_natural_hash,
            "payload_uses_identical_object": True,
            "full_route_callable_shared": True,
        },
        "boundaries": {
            "payload": (
                boundary_record(payload_samples, payload_protocol, payload_correctness)
                | {
                    "included": oracle_manifest["included_per_payload_call"],
                    "excluded": oracle_manifest["excluded_from_payload_call"],
                }
            ),
            "natural_full_route": (
                boundary_record(full_samples, full_protocol, full_correctness)
                | {
                    "included": ("common natural route callable",) + oracle_manifest["included_per_payload_call"],
                    "excluded": ("main QKV projection", "output projection"),
                }
            ),
        },
        "acceptance_boundary": "natural_full_route",
        "raw_samples_preserved": True,
    }
