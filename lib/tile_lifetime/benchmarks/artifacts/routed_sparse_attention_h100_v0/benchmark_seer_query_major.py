# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark a pinned Triton query-major sparse-attention oracle on H100."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import torch

KERNEL_PATH = Path(__file__).parent / "seer_attn" / "kernels" / "block_sparse_attn.py"
KERNEL_SPEC = importlib.util.spec_from_file_location("seer_block_sparse_attn", KERNEL_PATH)
assert KERNEL_SPEC is not None and KERNEL_SPEC.loader is not None
KERNEL_MODULE = importlib.util.module_from_spec(KERNEL_SPEC)
KERNEL_SPEC.loader.exec_module(KERNEL_MODULE)
block_sparse_triton_fn = KERNEL_MODULE.block_sparse_triton_fn


QUERY_HEADS = 32
KV_HEADS = 8
HEAD_DIMENSION = 128
QUERY_BLOCK_SIZE = 128
KV_BLOCK_SIZE = 128
TOP_K = 8


def deterministic_relation(sequence_length: int, top_k: int) -> torch.Tensor:
    """Return a causal block-shared relation with the current block included."""
    block_count = sequence_length // QUERY_BLOCK_SIZE
    relation = torch.zeros((block_count, block_count), dtype=torch.bool)
    for query_block in range(block_count):
        available = query_block + 1
        count = min(top_k, available)
        # Stable, approximately even coverage of history plus the current block.
        selected = torch.linspace(0, query_block, count).round().to(torch.int64).unique(sorted=True)
        relation[query_block, selected] = True
        relation[query_block, query_block] = True
    return relation


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    relation: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the prerecorded block relation with source-ordered FP32 softmax."""
    _, query_heads, sequence_length, _ = query.shape
    output = torch.empty_like(query)
    group_size = query_heads // key.shape[1]
    scale = 1.0 / math.sqrt(HEAD_DIMENSION)
    for query_block in range(sequence_length // QUERY_BLOCK_SIZE):
        query_start = query_block * QUERY_BLOCK_SIZE
        query_end = query_start + QUERY_BLOCK_SIZE
        selected_blocks = relation[query_block].nonzero(as_tuple=False).flatten()
        token_indices = torch.cat(
            [
                torch.arange(
                    int(block) * KV_BLOCK_SIZE,
                    (int(block) + 1) * KV_BLOCK_SIZE,
                    device=query.device,
                )
                for block in selected_blocks
            ]
        )
        for query_head in range(query_heads):
            kv_head = query_head // group_size
            q = query[0, query_head, query_start:query_end].float()
            k = key[0, kv_head, token_indices].float()
            v = value[0, kv_head, token_indices].float()
            scores = q @ k.transpose(0, 1) * scale
            causal = (
                token_indices[None, :]
                <= torch.arange(
                    query_start,
                    query_end,
                    device=query.device,
                )[:, None]
            )
            probabilities = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
            output[0, query_head, query_start:query_end] = (probabilities @ v).to(query.dtype)
    return output


def timed_samples(operation, warmups: int, repeats: int) -> list[float]:
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def tensor_hash(value: torch.Tensor) -> str:
    as_bytes = value.detach().contiguous().view(torch.uint16).cpu().numpy().tobytes()
    return hashlib.sha256(as_bytes).hexdigest()


def gpu_metadata() -> dict[str, str]:
    fields = "name,driver_version,clocks.current.sm,clocks.current.memory,clocks.max.sm,power.limit"
    values = (
        subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", "-i", "0"],
            text=True,
        )
        .strip()
        .split(", ")
    )
    return dict(zip(fields.split(","), values, strict=True))


def run(sequence_length: int, correctness: bool, warmups: int, repeats: int) -> dict[str, object]:
    torch.manual_seed(20260807)
    query = torch.randn(
        (1, QUERY_HEADS, sequence_length, HEAD_DIMENSION),
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        (1, KV_HEADS, sequence_length, HEAD_DIMENSION),
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    relation_started = time.perf_counter()
    relation = deterministic_relation(sequence_length, TOP_K)
    relation_plan_ms = (time.perf_counter() - relation_started) * 1_000
    relation_hash = hashlib.sha256(relation.numpy().tobytes()).hexdigest()
    mask = relation[None, None].expand(1, QUERY_HEADS, -1, -1).contiguous().cuda()

    # The oracle has no GQA indexing. Expand only at this external adapter boundary.
    group_size = QUERY_HEADS // KV_HEADS
    expand_start = torch.cuda.Event(enable_timing=True)
    expand_end = torch.cuda.Event(enable_timing=True)
    expand_start.record()
    expanded_key = key.repeat_interleave(group_size, dim=1).contiguous()
    expanded_value = value.repeat_interleave(group_size, dim=1).contiguous()
    expand_end.record()
    expand_end.synchronize()
    gqa_expansion_ms = float(expand_start.elapsed_time(expand_end))

    def operation():
        return block_sparse_triton_fn(
            query,
            expanded_key,
            expanded_value,
            mask,
            1.0 / math.sqrt(HEAD_DIMENSION),
            BLOCK_M=QUERY_BLOCK_SIZE,
            BLOCK_N=KV_BLOCK_SIZE,
            layout="bhsd",
        )

    compiled_at = time.time()
    actual = operation()
    torch.cuda.synchronize()
    compile_and_first_run_seconds = time.time() - compiled_at

    errors = None
    if correctness:
        expected = reference_attention(query, key, value, relation)
        absolute = (actual.float() - expected.float()).abs()
        errors = {
            "max_absolute": float(absolute.max()),
            "mean_absolute": float(absolute.mean()),
            "p99_absolute": float(torch.quantile(absolute, 0.99)),
            "allclose_atol_0.03_rtol_0.03": bool(torch.allclose(actual.float(), expected.float(), atol=0.03, rtol=0.03)),
        }

    samples = timed_samples(operation, warmups, repeats)

    def dense_operation():
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            enable_gqa=True,
        )

    dense_samples = timed_samples(dense_operation, warmups, repeats)
    edge_count = int(relation.sum())
    selected_work_flops = 4 * QUERY_HEADS * edge_count * QUERY_BLOCK_SIZE * KV_BLOCK_SIZE * HEAD_DIMENSION
    return {
        "sequence_length": sequence_length,
        "query_heads": QUERY_HEADS,
        "kv_heads": KV_HEADS,
        "head_dimension": HEAD_DIMENSION,
        "query_block_size": QUERY_BLOCK_SIZE,
        "kv_block_size": KV_BLOCK_SIZE,
        "top_k": TOP_K,
        "causal": True,
        "dtype": "bfloat16",
        "adapter": "repeat_interleave K/V from 8 to 32 heads at oracle boundary",
        "relation_hash": relation_hash,
        "relation_edge_count": edge_count,
        "relation_plan_ms": relation_plan_ms,
        "selected_qk_pv_flops": selected_work_flops,
        "selected_qk_pv_tflops_at_median": selected_work_flops / statistics.median(samples) / 1.0e9,
        "gqa_expansion": {
            "excluded_from_kernel_timing": True,
            "latency_ms": gqa_expansion_ms,
            "additional_bytes": int(2 * (QUERY_HEADS - KV_HEADS) * sequence_length * HEAD_DIMENSION * 2),
        },
        "compile_and_first_run_seconds": compile_and_first_run_seconds,
        "output_hash": tensor_hash(actual),
        "correctness": errors,
        "latency_ms": {
            "samples": samples,
            "median": statistics.median(samples),
            "mean": statistics.fmean(samples),
            "minimum": min(samples),
            "maximum": max(samples),
        },
        "dense_torch_sdpa_latency_ms": {
            "samples": dense_samples,
            "median": statistics.median(dense_samples),
            "mean": statistics.fmean(dense_samples),
            "minimum": min(dense_samples),
            "maximum": max(dense_samples),
            "note": "PyTorch scaled_dot_product_attention causal GQA; no score tensor materialized by script",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    result = {
        "oracle": "microsoft/SeerAttention Triton query-major block-sparse kernel",
        "oracle_revision": "aba03e3f2caefd0ccd21e576670aa830b748c84e",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": __import__("triton").__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_metadata": gpu_metadata(),
        "structural_caveat": (
            "Seer iterates every causal KV block and tests the dense binary mask in-loop; "
            "metadata traversal is O(dense causal block count), not O(selected edges)."
        ),
        "result": run(args.sequence_length, args.correctness, args.warmups, args.repeats),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
