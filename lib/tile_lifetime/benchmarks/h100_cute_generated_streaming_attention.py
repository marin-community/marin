# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the SM90 skeleton instantiated from Contract/Map/Fold semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
from backends.h100.cute_streaming_emitter import compile_h100_streaming_program

from shuttle.ir import DType
from tile_lifetime.ir import ScaledDotProductAttentionOp, TensorGraph
from tile_lifetime.streaming_attention import (
    StreamingTileSchedule,
    apply_causal_score_mask,
    apply_tanh_softcap,
    build_attention_tensor_program,
    derive_streaming_attention,
    scaled_score_map,
    streaming_attention_from_semantic_operation,
)

OFFICIAL_FA3_ORACLE_REVISION = "3fa810570e17bb4354155bdb71d826eca6079208"
FA4_CUTE_PACKAGE = "flash-attn-4==4.0.0b16"
FA4_BASE_SHA256 = "dec41cc35c28ee122c9808238dd97c482edfe22c2817697c2df44e5dfa46a222"
FA4_SM90_SHA256 = "4dcf8ecabc518888aad8a677de2279348b03bc278ccdb78f356f453fedb5f3f4"
FA3_MEDIAN_MS = {2048: 0.0672, 4096: 0.2100}


def _samples(function, *, warmups: int, repeats: int) -> list[float]:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _sampled_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    softcap: float | None,
    rows: tuple[int, ...],
) -> torch.Tensor:
    group_size = query.shape[2] // key.shape[2]
    head_mapping = torch.arange(query.shape[2], device=query.device) // group_size
    outputs = []
    for row in rows:
        query_row = query[0, row].float()
        key_prefix = key[0, : row + 1, head_mapping].float()
        value_prefix = value[0, : row + 1, head_mapping].float()
        scores = torch.einsum("hd,khd->hk", query_row, key_prefix) * scale
        if softcap is not None:
            scores = softcap * torch.tanh(scores / softcap)
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hk,khd->hd", probabilities, value_prefix))
    return torch.stack(outputs).bfloat16()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hardware() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,power.limit,clocks.sm,clocks.mem",
        "--format=csv,noheader",
    ]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, choices=tuple(FA3_MEDIAN_MS), required=True)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--softcap", type=float)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()

    batch_size = 1
    query_heads = 32
    key_value_heads = 8
    head_dimension = 128
    sequence_length = args.sequence_length
    scale = args.scale if args.scale is not None else 1.0 / math.sqrt(head_dimension)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    query = torch.randn(
        batch_size,
        sequence_length,
        query_heads,
        head_dimension,
        dtype=torch.bfloat16,
        device=device,
    )
    key = torch.randn(
        batch_size,
        sequence_length,
        key_value_heads,
        head_dimension,
        dtype=torch.bfloat16,
        device=device,
    )
    value = torch.randn_like(key)
    output = torch.empty_like(query)
    log_sum_exp = torch.empty(
        batch_size,
        query_heads,
        sequence_length,
        dtype=torch.float32,
        device=device,
    )

    semantic_graph = TensorGraph()
    semantic_query = semantic_graph.input("query", shape=tuple(query.shape), dtype=DType.BF16)
    semantic_key = semantic_graph.input("key", shape=tuple(key.shape), dtype=DType.BF16)
    semantic_value = semantic_graph.input("value", shape=tuple(value.shape), dtype=DType.BF16)
    semantic_graph.scaled_dot_product_attention(
        semantic_query,
        semantic_key,
        semantic_value,
        name="attention.output",
        scale=scale,
        causal=True,
        accumulation_dtype=DType.FP32,
        source_location="recovered StableHLO attention",
    )
    semantic_operation = semantic_graph.operations[0]
    assert isinstance(semantic_operation, ScaledDotProductAttentionOp)
    schedule = StreamingTileSchedule(query_tile_size=128, key_value_tile_size=128, pipeline_depth=2)
    if args.softcap is None:
        program = streaming_attention_from_semantic_operation(semantic_operation, schedule=schedule)
    else:
        score_map = apply_causal_score_mask(scaled_score_map(scale))
        score_map = apply_tanh_softcap(score_map, args.softcap)
        source = build_attention_tensor_program(
            batch_size=batch_size,
            query_length=sequence_length,
            key_length=sequence_length,
            query_heads=query_heads,
            key_value_heads=key_value_heads,
            key_dimension=head_dimension,
            value_dimension=head_dimension,
            score_map=score_map,
            input_dtype=DType.BF16,
        )
        program = derive_streaming_attention(source, schedule=schedule)
    compile_start = time.perf_counter()
    compiled = compile_h100_streaming_program(
        program,
        query=query,
        key=key,
        value=value,
        output=output,
        log_sum_exp=log_sum_exp,
    )
    compile_seconds = time.perf_counter() - compile_start

    def execute() -> None:
        compiled(query, key, value, output, log_sum_exp)

    samples = _samples(execute, warmups=args.warmups, repeats=args.repeats)
    rows = tuple(dict.fromkeys((0, 1, 127, 128, sequence_length // 2, sequence_length - 1)))
    reference = _sampled_reference(
        query,
        key,
        value,
        scale=scale,
        softcap=args.softcap,
        rows=rows,
    )
    selected = output[0, list(rows)]
    difference = (selected.float() - reference.float()).abs()
    hashes = []
    for _ in range(3):
        execute()
        torch.cuda.synchronize()
        hashes.append(_sha256_tensor(output))

    repository_root = Path(__file__).resolve().parents[3]
    backend_root = repository_root / "lib/tile_lifetime/backends/h100"
    median_ms = statistics.median(samples)
    record = {
        "name": "shuttle_generated_sm90_streaming_contract_fold",
        "semantic_source": "Contract -> scalar Map -> max/sum Fold -> Contract -> normalize Map",
        "semantic_frontend": (
            "recovered ScaledDotProductAttentionOp -> Contract/Map/Fold bridge"
            if args.softcap is None
            else "generic scalar score-map mutation -> Contract/Map/Fold"
        ),
        "uses_official_attention_interface": False,
        "shape": {
            "batch": batch_size,
            "sequence": sequence_length,
            "query_heads": query_heads,
            "key_value_heads": key_value_heads,
            "head_dimension": head_dimension,
        },
        "score_map": asdict(compiled.score_map),
        "gqa_head_group_size": compiled.head_group_size,
        "schedule": asdict(compiled.schedule),
        "compile_seconds": compile_seconds,
        "warmups": args.warmups,
        "samples_ms": samples,
        "median_ms": median_ms,
        "minimum_ms": min(samples),
        "pinned_fa3_oracle_ms": FA3_MEDIAN_MS[sequence_length],
        "ratio_to_pinned_fa3": median_ms / FA3_MEDIAN_MS[sequence_length],
        "sampled_rows": rows,
        "sampled_maximum_absolute_error": difference.max().item(),
        "sampled_mean_absolute_error": difference.mean().item(),
        "deterministic_output_hashes": hashes,
        "deterministic": len(set(hashes)) == 1,
        "hardware": _hardware(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "official_fa3_oracle_revision": OFFICIAL_FA3_ORACLE_REVISION,
        "extraction_package": FA4_CUTE_PACKAGE,
        "extraction_source_sha256": {
            "flash_fwd.py": FA4_BASE_SHA256,
            "flash_fwd_sm90.py": FA4_SM90_SHA256,
        },
        "shuttle_backend_sha256": {
            "cute_streaming_base.py": _sha256_file(backend_root / "cute_streaming_base.py"),
            "cute_streaming_sm90.py": _sha256_file(backend_root / "cute_streaming_sm90.py"),
            "cute_streaming_emitter.py": _sha256_file(backend_root / "cute_streaming_emitter.py"),
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
