# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen inputs and pure helpers for the GrugMoE inference preflight.

The live launcher is ``scripts/iris/grugmoe_inference_preflight.py``. This
module deliberately has no JAX, torch, or vLLM imports so its contract can be
checked on a laptop before reserving a GB200.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MARIN_BASE_SHA = "75bf2437035cf731d1a4bd71266229dfcdda9478"
VLLM_SHA = "afb26719464d5957e695bde478ae93a160b11d14"
TRAINING_REFERENCE_SHA = "fd3e9bc5b428633027f944be7fdf1136567db028"
PINNED_REFERENCE_URL = "https://github.com/marin-community/marin/issues/7201#issuecomment-5093392733"
SNOWBALL_EXPORT = "s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b/" "step-42150/hf-bf16-vllm/d819cbc63780bd86/"
ARTIFACT_ROOT = "s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture"

DUMMY_SEED = 1234
DTYPE = "bfloat16"
KV_CACHE_DTYPE = "bfloat16"
GPU_MEMORY_UTILIZATION = 0.90
KV_BLOCK_SIZE = 16
ROOT_COUNT = 18
BRANCHES_PER_ROOT = 8
BRANCH_COUNT = ROOT_COUNT * BRANCHES_PER_ROOT


@dataclass(frozen=True)
class ModelCase:
    name: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    max_model_len: int
    sliding_window: int
    data_parallel_size: int

    def __post_init__(self) -> None:
        positive = (
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.num_experts,
            self.num_experts_per_tok,
            self.moe_intermediate_size,
            self.max_model_len,
            self.sliding_window,
            self.data_parallel_size,
        )
        if any(value <= 0 for value in positive):
            raise ValueError(f"{self.name}: dimensions and topology must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(f"{self.name}: hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(f"{self.name}: query heads must be divisible by KV heads")
        if self.num_experts_per_tok >= self.num_experts:
            raise ValueError(f"{self.name}: QB routing requires top-k < expert count")
        if self.shared_expert_intermediate_size < 0:
            raise ValueError(f"{self.name}: shared expert width cannot be negative")
        if self.data_parallel_size not in {1, 4, 8, 16}:
            raise ValueError(f"{self.name}: unsupported preflight DP/EP size")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def node_count(self) -> int:
        return max(1, math.ceil(self.data_parallel_size / 4))

    def hf_config(self) -> dict[str, Any]:
        return {
            "architectures": ["GrugMoeForCausalLM"],
            "model_type": "grug_moe",
            "vocab_size": 256,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_model_len,
            "sliding_window": self.sliding_window,
            "rms_norm_eps": 1e-5,
            "initializer_range": 0.02,
            "rope_theta": 10_000.0,
            "tie_word_embeddings": False,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_intermediate_size": self.moe_intermediate_size,
            # Two half-width shared experts are represented by concatenating
            # their intermediate axes into this one fused shared MLP.
            "shared_expert_intermediate_size": self.shared_expert_intermediate_size,
            "qk_mult": 1.3,
            "disable_pko": True,
            "disable_long_rope": True,
            "grugmoe_attention_mode": "production",
            "grug_moe_artifact_schema_version": 1,
            "torch_dtype": DTYPE,
        }


CASES: dict[str, ModelCase] = {
    "tiny": ModelCase(
        name="tiny",
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        max_model_len=2048,
        sliding_window=512,
        data_parallel_size=1,
    ),
    "one-node-ep4": ModelCase(
        name="one-node-ep4",
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_experts=128,
        num_experts_per_tok=4,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
        max_model_len=4096,
        sliding_window=512,
        data_parallel_size=4,
    ),
    "reference-ep8": ModelCase(
        name="reference-ep8",
        hidden_size=6144,
        num_hidden_layers=48,
        num_attention_heads=48,
        num_key_value_heads=12,
        num_experts=128,
        num_experts_per_tok=4,
        moe_intermediate_size=3072,
        shared_expert_intermediate_size=3072,
        max_model_len=65_536,
        sliding_window=512,
        data_parallel_size=8,
    ),
    "granular-ep16": ModelCase(
        name="granular-ep16",
        hidden_size=6144,
        num_hidden_layers=48,
        num_attention_heads=48,
        num_key_value_heads=12,
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=1536,
        shared_expert_intermediate_size=3072,
        max_model_len=65_536,
        sliding_window=512,
        data_parallel_size=16,
    ),
}


def layer_types(num_hidden_layers: int, *, global_interval: int = 4) -> list[str]:
    """Return the layer schedule enforced by the pinned serving fork."""
    if num_hidden_layers <= 0 or global_interval <= 0:
        raise ValueError("layer count and global interval must be positive")
    return [
        (
            "full_attention"
            if (layer_index + 1) % global_interval == 0 or layer_index == num_hidden_layers - 1
            else "sliding_attention"
        )
        for layer_index in range(num_hidden_layers)
    ]


def predict_kv_bytes(
    *,
    sequence_length: int,
    local_layers: int,
    global_layers: int,
    local_kv_heads: int,
    global_kv_heads: int,
    head_dim: int,
    sliding_window: int,
    bytes_per_element: int = 2,
) -> int:
    """Predict steady-state K+V bytes for one sequence, before block rounding."""
    if (
        min(
            sequence_length,
            local_layers,
            global_layers,
            local_kv_heads,
            global_kv_heads,
            head_dim,
            sliding_window,
            bytes_per_element,
        )
        < 0
    ):
        raise ValueError("KV inputs cannot be negative")
    local_tokens = min(sequence_length, sliding_window)
    elements = (
        2 * head_dim * (local_layers * local_kv_heads * local_tokens + global_layers * global_kv_heads * sequence_length)
    )
    return elements * bytes_per_element


def deterministic_workload(
    *,
    max_prefix_tokens: int,
    roots: int = ROOT_COUNT,
    branches_per_root: int = BRANCHES_PER_ROOT,
    seed: int = DUMMY_SEED,
) -> dict[str, Any]:
    """Build the frozen append-style token workload without a tokenizer."""
    if roots <= 0 or branches_per_root <= 0:
        raise ValueError("roots and branches_per_root must be positive")
    if max_prefix_tokens < 513:
        raise ValueError("max_prefix_tokens must cross the 512-token window")
    rng = random.Random(seed)
    lengths = sorted(
        {
            KV_BLOCK_SIZE + 1,
            513,
            min(2048, max_prefix_tokens),
            min(8192, max_prefix_tokens),
            max_prefix_tokens,
        }
    )
    root_records: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for root in range(roots):
        prefix_length = lengths[root % len(lengths)]
        prefix = [1]
        prefix.extend(rng.randrange(3, 255) for _ in range(prefix_length - 1))
        # Change the first complete cache block. vLLM includes the parent block
        # hash when hashing each later block, so this prevents a mutated prompt
        # from accidentally reusing the unchanged beginning of a long prefix.
        mutated_prefix = [*prefix]
        mutated_prefix[1] = (mutated_prefix[1] % 251) + 3
        root_records.append(
            {
                "root": root,
                "prefix_token_ids": prefix,
                "mutated_prefix_token_ids": mutated_prefix,
            }
        )
        for branch in range(branches_per_root):
            append = [2, 3 + root, 32 + branch, 200 + ((root + branch) % 50)]
            requests.append(
                {
                    "request_id": f"root-{root:02d}-branch-{branch:02d}",
                    "root": root,
                    "branch": branch,
                    "prefix_token_count": len(prefix),
                    "append_token_ids": append,
                    "max_tokens": 4,
                }
            )
    return {
        "schema_version": 1,
        "seed": seed,
        "root_count": roots,
        "branches_per_root": branches_per_root,
        "request_count": len(requests),
        "lengths": lengths,
        "roots": root_records,
        "requests": requests,
    }


def materialize_prompt(workload: dict[str, Any], request: dict[str, Any], *, mutated: bool = False) -> list[int]:
    """Resolve a compact branch record into the token IDs sent to vLLM."""
    root_index = int(request["root"])
    roots = workload["roots"]
    if not 0 <= root_index < len(roots):
        raise ValueError(f"request root {root_index} is outside the workload")
    root = roots[root_index]
    if int(root["root"]) != root_index:
        raise ValueError(f"workload root index mismatch at {root_index}")
    prefix_key = "mutated_prefix_token_ids" if mutated else "prefix_token_ids"
    return [*root[prefix_key], *request["append_token_ids"]]


def parse_prometheus(text: str) -> dict[str, float]:
    """Collapse scalar Prometheus samples by metric name.

    The preflight starts one model, so summing labeled ranks is the useful
    global value for counters. Histogram buckets are intentionally retained by
    their base metric name only when callers ask for them explicitly.
    """
    values: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        sample, separator, raw_value = line.rpartition(" ")
        if not separator:
            continue
        metric = sample.split("{", 1)[0]
        try:
            value = float(raw_value)
        except ValueError:
            continue
        values[metric] = values.get(metric, 0.0) + value
    return values


def metric_delta(before: dict[str, float], after: dict[str, float], metric: str) -> float:
    # prometheus_client exposes Counters with a ``_total`` suffix even when
    # vLLM constructs them with the unsuffixed name.
    def value(samples: dict[str, float]) -> float:
        if metric in samples:
            return samples[metric]
        return samples.get(f"{metric}_total", 0.0)

    return value(after) - value(before)


def decode_routed_experts(encoded: str) -> Any:
    """Decode the fork's base64-encoded NumPy routed-expert response."""
    import numpy as np  # noqa: PLC0415

    return np.load(io.BytesIO(base64.b64decode(encoded)), allow_pickle=False)


def routing_histogram(routed_experts: Any, *, num_experts: int) -> list[int]:
    import numpy as np  # noqa: PLC0415

    values = np.asarray(routed_experts, dtype=np.int64).reshape(-1)
    if values.size and (values.min() < 0 or values.max() >= num_experts):
        raise ValueError("routed expert id is outside the configured expert range")
    return np.bincount(values, minlength=num_experts).astype(int).tolist()


def expert_parallel_rank_histogram(expert_histogram: list[int], *, ep_size: int) -> list[int]:
    """Map a linear-placement expert histogram to its owning EP ranks."""
    if ep_size <= 0:
        raise ValueError("EP size must be positive")
    if not expert_histogram or len(expert_histogram) % ep_size:
        raise ValueError("expert count must be nonzero and divisible by EP size")
    experts_per_rank = len(expert_histogram) // ep_size
    return [sum(expert_histogram[rank * experts_per_rank : (rank + 1) * experts_per_rank]) for rank in range(ep_size)]


def deterministic_balanced_routing_fixture(*, num_experts: int, top_k: int, ep_size: int) -> dict[str, Any]:
    """Build a routing control where every expert and EP rank receive equal work.

    This tests the histogram/placement instrumentation. It is not a claim that
    a live model's router is balanced.
    """
    if num_experts <= 0 or not 0 < top_k < num_experts:
        raise ValueError("balanced routing requires 0 < top_k < num_experts")
    if num_experts % ep_size:
        raise ValueError("expert count must be divisible by EP size")
    assignments = [[(token + offset) % num_experts for offset in range(top_k)] for token in range(num_experts)]
    expert_histogram = routing_histogram(assignments, num_experts=num_experts)
    rank_histogram = expert_parallel_rank_histogram(expert_histogram, ep_size=ep_size)
    return {
        "kind": "instrumentation-control",
        "tokens": num_experts,
        "assignments_sha256": hashlib.sha256(json.dumps(assignments, separators=(",", ":")).encode()).hexdigest(),
        "expert_histogram": expert_histogram,
        "ep_rank_histogram": rank_histogram,
        "all_experts_equal": len(set(expert_histogram)) == 1,
        "all_ep_ranks_equal": len(set(rank_histogram)) == 1,
    }


def frozen_manifest(
    case: ModelCase,
    *,
    run_id: str,
    git_sha: str,
    model_source: str = "dummy",
) -> dict[str, Any]:
    if model_source not in {"dummy", "snowball"}:
        raise ValueError(f"unknown model source: {model_source}")
    model_path = "staged-config.json" if model_source == "dummy" else SNOWBALL_EXPORT
    load_format = "dummy" if model_source == "dummy" else "runai_streamer"
    return {
        "schema_version": 1,
        "run_id": run_id,
        "case": asdict(case),
        "git_sha": git_sha,
        "frozen_refs": {
            "marin_base_sha": MARIN_BASE_SHA,
            "vllm_sha": VLLM_SHA,
            "training_reference_sha": TRAINING_REFERENCE_SHA,
            "reference_url": PINNED_REFERENCE_URL,
            "snowball_export": SNOWBALL_EXPORT,
        },
        "runtime": {
            "model_source": model_source,
            "model_path": model_path,
            "load_format": load_format,
            "seed": DUMMY_SEED,
            "dtype": DTYPE,
            "kv_cache_dtype": KV_CACHE_DTYPE,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "prefix_caching": True,
            "chunked_prefill": True,
            "pipeline_parallel_size": 1,
            "tensor_parallel_size": 1,
            "data_parallel_size": case.data_parallel_size,
            "expert_parallel_size": case.data_parallel_size,
        },
        "artifact_prefix": f"{ARTIFACT_ROOT}/{case.name}/{run_id}/",
    }


def write_case(output_dir: Path, *, case: ModelCase, run_id: str, git_sha: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(case.hf_config(), indent=2, sort_keys=True) + "\n")
    workload = deterministic_workload(max_prefix_tokens=min(case.max_model_len - 8, 65_536))
    (output_dir / "workload.json").write_text(json.dumps(workload, indent=2, sort_keys=True) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(frozen_manifest(case, run_id=run_id, git_sha=git_sha), indent=2, sort_keys=True) + "\n"
    )


__all__ = [
    "ARTIFACT_ROOT",
    "BRANCH_COUNT",
    "CASES",
    "DUMMY_SEED",
    "KV_BLOCK_SIZE",
    "MARIN_BASE_SHA",
    "SNOWBALL_EXPORT",
    "VLLM_SHA",
    "ModelCase",
    "decode_routed_experts",
    "deterministic_balanced_routing_fixture",
    "deterministic_workload",
    "expert_parallel_rank_histogram",
    "frozen_manifest",
    "layer_types",
    "materialize_prompt",
    "metric_delta",
    "parse_prometheus",
    "predict_kv_bytes",
    "routing_histogram",
    "write_case",
]
