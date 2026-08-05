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
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

MARIN_BASE_SHA = "75bf2437035cf731d1a4bd71266229dfcdda9478"
VLLM_SHA = "06af5cff3b97723356ec590b9ecf635b7690bd40"
TRAINING_REFERENCE_SHA = "fd3e9bc5b428633027f944be7fdf1136567db028"
PINNED_REFERENCE_URL = "https://github.com/marin-community/marin/issues/7201#issuecomment-5093392733"
SNOWBALL_EXPORT = "s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b/step-42150/hf-bf16-vllm/d819cbc63780bd86/"
ARTIFACT_ROOT = "s3://marin-us-east-02a/marin/users/romain/moe-inference-architecture"
FROZEN_FIXTURE_PATH = "tests/cluster/vllm/resources/grug_exact_reference"

DUMMY_SEED = 1234
IDENTITY_CHAT_TOKENS = tuple(f"{token_id:02x}" for token_id in range(256))
DTYPE = "bfloat16"
KV_CACHE_DTYPE = "bfloat16"
GPU_MEMORY_UTILIZATION = 0.90
KV_BLOCK_SIZE = 16
ROOT_COUNT = 18
BRANCHES_PER_ROOT = 8
BRANCH_COUNT = ROOT_COUNT * BRANCHES_PER_ROOT
ACCEPTANCE_HISTORY_LENGTHS = (10_240, 30_720, 62_464)
ACCEPTANCE_APPEND_TOKENS = 1_024
ACCEPTANCE_RESPONSE_TOKENS = 2_048
ACCEPTANCE_FINAL_LENGTHS = (13_312, 33_792, 65_536)
TRAJECTORY_INITIAL_HISTORY_LENGTHS = (10_240, 30_720, 53_248)
TRAJECTORY_TURNS = 4
CAPACITY_HISTORY_TOKENS = 121_856
CAPACITY_APPEND_TOKENS = 1_024
CAPACITY_RESPONSE_TOKENS = 8_192
CAPACITY_FINAL_TOKENS = 131_072
SITES = ("k", "v", "attn", "mlp")


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
    local_kv_heads: int
    global_kv_heads: int
    global_every: int
    rope_fraction: float
    rope_fused: bool
    gated_norm: bool
    attn_gate: bool
    xsa: bool
    qb_routing: bool
    legacy_input_output_gated_norm: bool
    mtp_depth: int
    over_encoding_vocab_size: int
    num_shared_experts: int
    sconv: bool
    sconv_kernel: int
    sconv_sites: tuple[str, ...]

    def __post_init__(self) -> None:
        positive = (
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.local_kv_heads,
            self.global_kv_heads,
            self.num_experts,
            self.num_experts_per_tok,
            self.moe_intermediate_size,
            self.shared_expert_intermediate_size,
            self.max_model_len,
            self.sliding_window,
            self.data_parallel_size,
            self.global_every,
            self.num_shared_experts,
            self.sconv_kernel,
        )
        if any(value <= 0 for value in positive):
            raise ValueError(f"{self.name}: dimensions and topology must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(f"{self.name}: hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(f"{self.name}: query heads must be divisible by KV heads")
        if self.num_key_value_heads != max(self.local_kv_heads, self.global_kv_heads):
            raise ValueError(f"{self.name}: stored KV heads must equal the larger logical KV-head count")
        if self.num_key_value_heads % self.local_kv_heads or self.num_key_value_heads % self.global_kv_heads:
            raise ValueError(f"{self.name}: stored KV heads must be divisible by each logical KV-head count")
        if self.num_experts_per_tok >= self.num_experts:
            raise ValueError(f"{self.name}: QB routing requires top-k < expert count")
        if self.shared_expert_intermediate_size % self.num_shared_experts:
            raise ValueError(f"{self.name}: shared expert width must divide across the shared experts")
        if not 0 < self.rope_fraction <= 1:
            raise ValueError(f"{self.name}: rope_fraction must be in (0, 1]")
        if set(self.sconv_sites) - set(SITES):
            raise ValueError(f"{self.name}: unknown SConv site")
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
            "local_kv_heads": self.local_kv_heads,
            "global_kv_heads": self.global_kv_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_model_len,
            "sliding_window": self.sliding_window,
            "global_every": self.global_every,
            "rms_norm_eps": 1e-5,
            "initializer_range": 0.02,
            "rope_theta": 10_000.0,
            "rope_fraction": self.rope_fraction,
            "rope_fused": self.rope_fused,
            "tie_word_embeddings": False,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_intermediate_size": self.moe_intermediate_size,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_size,
            "num_shared_experts": self.num_shared_experts,
            "qk_mult": 1.3,
            "qk_mult_long_scale": 1.0,
            "disable_pko": True,
            "disable_long_rope": True,
            "gated_norm": self.gated_norm,
            "attn_gate": self.attn_gate,
            "xsa": self.xsa,
            "qb_routing": self.qb_routing,
            "legacy_input_output_gated_norm": self.legacy_input_output_gated_norm,
            "mtp_depth": self.mtp_depth,
            "mtp_dense": True,
            "over_encoding_vocab_size": self.over_encoding_vocab_size,
            "sconv": self.sconv,
            "sconv_kernel": self.sconv_kernel,
            "sconv_sites": list(self.sconv_sites),
            "grugmoe_attention_mode": "production",
            "grug_moe_artifact_schema_version": 1,
            "torch_dtype": DTYPE,
        }


def _case(
    name: str,
    *,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    local_kv_heads: int,
    global_kv_heads: int,
    num_experts: int,
    num_experts_per_tok: int,
    moe_intermediate_size: int,
    shared_expert_intermediate_size: int,
    max_model_len: int,
    sliding_window: int,
    data_parallel_size: int,
    global_every: int = 6,
    exact_blocks: bool = True,
    sconv: bool = True,
) -> ModelCase:
    """Construct a case while keeping the exact custom-block switches together."""
    return ModelCase(
        name=name,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        local_kv_heads=local_kv_heads,
        global_kv_heads=global_kv_heads,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        max_model_len=max_model_len,
        sliding_window=sliding_window,
        data_parallel_size=data_parallel_size,
        global_every=global_every,
        rope_fraction=0.5 if exact_blocks else 1.0,
        rope_fused=exact_blocks,
        gated_norm=True,
        attn_gate=True,
        xsa=True,
        qb_routing=True,
        legacy_input_output_gated_norm=not exact_blocks,
        mtp_depth=1 if exact_blocks else 0,
        over_encoding_vocab_size=0,
        num_shared_experts=2 if exact_blocks else 1,
        sconv=sconv,
        sconv_kernel=4,
        sconv_sites=SITES,
    )


CASES: dict[str, ModelCase] = {
    # Laptop/unit-test scale, but every exact model path is enabled.
    "tiny": _case(
        "tiny",
        hidden_size=32,
        num_hidden_layers=7,
        num_attention_heads=4,
        num_key_value_heads=2,
        local_kv_heads=2,
        global_kv_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=32,
        max_model_len=1024,
        sliding_window=512,
        data_parallel_size=1,
    ),
    # One-node exact path gate.
    "one-node-ep4": _case(
        "one-node-ep4",
        hidden_size=512,
        num_hidden_layers=7,
        num_attention_heads=8,
        num_key_value_heads=4,
        local_kv_heads=4,
        global_kv_heads=2,
        num_experts=128,
        num_experts_per_tok=4,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=512,
        max_model_len=4096,
        sliding_window=512,
        data_parallel_size=4,
    ),
    # The old approximation remains a non-ranking launcher control.
    "legacy-control-ep4": _case(
        "legacy-control-ep4",
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        local_kv_heads=4,
        global_kv_heads=4,
        num_experts=128,
        num_experts_per_tok=4,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
        max_model_len=4096,
        sliding_window=512,
        data_parallel_size=4,
        global_every=4,
        exact_blocks=False,
        sconv=False,
    ),
    # Reuses the heterogeneous-KV and sliding-window implementations at the
    # other P0 values without pretending to be a performance comparison.
    "kv2-window2048-ep4": _case(
        "kv2-window2048-ep4",
        hidden_size=512,
        num_hidden_layers=7,
        num_attention_heads=8,
        num_key_value_heads=4,
        local_kv_heads=4,
        global_kv_heads=1,
        num_experts=128,
        num_experts_per_tok=4,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=512,
        max_model_len=4096,
        sliding_window=2048,
        data_parallel_size=4,
    ),
    "reference-ep8": _case(
        "reference-ep8",
        hidden_size=6144,
        num_hidden_layers=48,
        num_attention_heads=48,
        num_key_value_heads=12,
        local_kv_heads=12,
        global_kv_heads=6,
        num_experts=128,
        num_experts_per_tok=4,
        moe_intermediate_size=3072,
        shared_expert_intermediate_size=6144,
        max_model_len=65_536,
        sliding_window=512,
        data_parallel_size=8,
    ),
    "granular-ep16": _case(
        "granular-ep16",
        hidden_size=512,
        num_hidden_layers=7,
        num_attention_heads=8,
        num_key_value_heads=4,
        local_kv_heads=4,
        global_kv_heads=2,
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=512,
        max_model_len=4096,
        sliding_window=512,
        data_parallel_size=16,
    ),
    "exact-reference-ep16": _case(
        "exact-reference-ep16",
        hidden_size=6144,
        num_hidden_layers=48,
        num_attention_heads=48,
        num_key_value_heads=12,
        local_kv_heads=12,
        global_kv_heads=6,
        num_experts=128,
        num_experts_per_tok=4,
        moe_intermediate_size=3072,
        shared_expert_intermediate_size=6144,
        max_model_len=65_536,
        sliding_window=512,
        data_parallel_size=16,
    ),
    "window1024-ep16": _case(
        "window1024-ep16",
        hidden_size=6144,
        num_hidden_layers=48,
        num_attention_heads=48,
        num_key_value_heads=12,
        local_kv_heads=12,
        global_kv_heads=6,
        num_experts=128,
        num_experts_per_tok=4,
        moe_intermediate_size=3072,
        shared_expert_intermediate_size=6144,
        max_model_len=65_536,
        sliding_window=1024,
        data_parallel_size=16,
    ),
    "window2048-ep16": _case(
        "window2048-ep16",
        hidden_size=6144,
        num_hidden_layers=48,
        num_attention_heads=48,
        num_key_value_heads=12,
        local_kv_heads=12,
        global_kv_heads=6,
        num_experts=128,
        num_experts_per_tok=4,
        moe_intermediate_size=3072,
        shared_expert_intermediate_size=6144,
        max_model_len=65_536,
        sliding_window=2048,
        data_parallel_size=16,
    ),
    "global-every4-ep16": _case(
        "global-every4-ep16",
        hidden_size=6144,
        num_hidden_layers=48,
        num_attention_heads=48,
        num_key_value_heads=12,
        local_kv_heads=12,
        global_kv_heads=6,
        num_experts=128,
        num_experts_per_tok=4,
        moe_intermediate_size=3072,
        shared_expert_intermediate_size=6144,
        max_model_len=65_536,
        sliding_window=512,
        data_parallel_size=16,
        global_every=4,
    ),
}

# The finalist validation changes only the advertised context ceiling. This
# lets the 65K trajectory and 131K capacity attempt run on the same fresh
# server without introducing a serving or sharding change.
for _base_name in (
    "exact-reference-ep16",
    "window1024-ep16",
    "window2048-ep16",
    "global-every4-ep16",
):
    _base_case = CASES[_base_name]
    _extended_name = f"{_base_name.removesuffix('-ep16')}-131k-ep16"
    CASES[_extended_name] = replace(_base_case, name=_extended_name, max_model_len=CAPACITY_FINAL_TOKENS)

P0_SMOKE_CASES: dict[str, tuple[str, ...]] = {
    "uniform-kv_every4_sconv-off": ("legacy-control-ep4",),
    "heterogeneous-kv_every6_sconv-on": ("one-node-ep4", "reference-ep8"),
    "global-kv-2_window-2048": ("kv2-window2048-ep4",),
    "top8-256_ep16": ("granular-ep16",),
    "exact-ep16": ("exact-reference-ep16",),
}

REQUIRED_ACCEPTANCE_CHECKS = (
    "placement",
    "all_rank_health",
    "correctness",
    "duration",
    "token_count",
    "repeatability",
    "artifact_readback",
)


def aggregate_preflight_status(components: dict[str, Any]) -> dict[str, Any]:
    """Return the literal conjunction used for the top-level live result."""
    missing = [name for name in REQUIRED_ACCEPTANCE_CHECKS if name not in components]
    if missing:
        raise ValueError(f"missing required acceptance checks: {missing}")
    checks: dict[str, bool] = {}
    for name in REQUIRED_ACCEPTANCE_CHECKS:
        component = components[name]
        if isinstance(component, bool):
            checks[name] = component
        elif isinstance(component, dict) and isinstance(component.get("passed"), bool):
            checks[name] = bool(component["passed"])
        else:
            raise TypeError(f"{name} must be a bool or a mapping with boolean 'passed'")
    passed = all(checks.values())
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checks": checks,
    }


def layer_types(num_hidden_layers: int, *, global_interval: int = 6) -> list[str]:
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


def hybrid_kv_cache_hit_alignment(case: ModelCase) -> int:
    """Return vLLM's token alignment for hybrid prefix-cache hits.

    The pinned fork gives every KV-cache group the largest physical page size
    by increasing smaller groups' token block sizes. Its scheduler then uses
    the least common multiple of those adjusted block sizes.

    Page sizes are measured in elements here because every group uses the same
    dtype, so bytes per element cancel from all ratios.
    """
    group_pages = [
        ("local-attention", KV_BLOCK_SIZE, 2 * KV_BLOCK_SIZE * case.local_kv_heads * case.head_dim),
        ("global-attention", KV_BLOCK_SIZE, 2 * KV_BLOCK_SIZE * case.global_kv_heads * case.head_dim),
    ]
    if case.sconv:
        for site in case.sconv_sites:
            stream_width = case.num_key_value_heads * case.head_dim if site in {"k", "v"} else case.hidden_size
            group_pages.append((f"sconv-{site}", case.sconv_kernel, case.sconv_kernel * stream_width))

    max_page_elements = max(page_elements for _, _, page_elements in group_pages)
    adjusted_block_sizes: list[int] = []
    for group_name, block_size, page_elements in group_pages:
        if max_page_elements % page_elements:
            raise ValueError(
                f"{case.name}: {group_name} page size {page_elements} does not divide "
                f"the largest hybrid page size {max_page_elements}"
            )
        adjusted_block_sizes.append(block_size * (max_page_elements // page_elements))
    return math.lcm(*adjusted_block_sizes)


def deterministic_workload(
    *,
    max_prefix_tokens: int = ACCEPTANCE_HISTORY_LENGTHS[-1],
    roots: int = ROOT_COUNT,
    branches_per_root: int = BRANCHES_PER_ROOT,
    seed: int = DUMMY_SEED,
) -> dict[str, Any]:
    """Build the exact 18-root, 144-branch acceptance workload."""
    if roots <= 0 or branches_per_root <= 0:
        raise ValueError("roots and branches_per_root must be positive")
    if roots != ROOT_COUNT or branches_per_root != BRANCHES_PER_ROOT:
        raise ValueError("the acceptance workload is frozen at 18 roots and 8 branches")
    if max_prefix_tokens != ACCEPTANCE_HISTORY_LENGTHS[-1]:
        raise ValueError(
            f"the acceptance workload's longest cached history is frozen at {ACCEPTANCE_HISTORY_LENGTHS[-1]}"
        )
    rng = random.Random(seed)
    root_records: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for root in range(roots):
        cohort = root // 6
        prefix_length = ACCEPTANCE_HISTORY_LENGTHS[cohort]
        prefix = [1]
        prefix.extend(rng.randrange(3, 255) for _ in range(prefix_length - 1))
        root_records.append(
            {
                "root": root,
                "cohort": ("short", "medium", "long")[cohort],
                "prefix_token_ids": prefix,
            }
        )
        for branch in range(branches_per_root):
            append_rng = random.Random((seed << 16) + root * branches_per_root + branch)
            append = [
                3 + ((root * 17 + branch * 29 + position + append_rng.randrange(251)) % 252)
                for position in range(ACCEPTANCE_APPEND_TOKENS)
            ]
            requests.append(
                {
                    "request_id": f"root-{root:02d}-branch-{branch:02d}",
                    "root": root,
                    "branch": branch,
                    "cohort": ("short", "medium", "long")[cohort],
                    "prefix_token_count": len(prefix),
                    "append_token_count": len(append),
                    "append_token_ids": append,
                    "max_tokens": ACCEPTANCE_RESPONSE_TOKENS,
                    "final_token_count": len(prefix) + len(append) + ACCEPTANCE_RESPONSE_TOKENS,
                }
            )
    return {
        "schema_version": 2,
        "kind": "exact-reference-acceptance",
        "seed": seed,
        "root_count": roots,
        "branches_per_root": branches_per_root,
        "request_count": len(requests),
        "history_lengths": list(ACCEPTANCE_HISTORY_LENGTHS),
        "append_tokens": ACCEPTANCE_APPEND_TOKENS,
        "response_tokens": ACCEPTANCE_RESPONSE_TOKENS,
        "final_lengths": list(ACCEPTANCE_FINAL_LENGTHS),
        "roots": root_records,
        "requests": requests,
    }


def deterministic_trajectory_workload(*, seed: int = DUMMY_SEED) -> dict[str, Any]:
    """Build the frozen four-turn, 18-root, 144-branch 65K trajectory."""
    rng = random.Random(seed)
    roots: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for root in range(ROOT_COUNT):
        cohort_index = root // 6
        cohort = ("short", "medium", "long")[cohort_index]
        history_length = TRAJECTORY_INITIAL_HISTORY_LENGTHS[cohort_index]
        prefix = [1, *(rng.randrange(3, 255) for _ in range(history_length - 1))]
        roots.append(
            {
                "root": root,
                "cohort": cohort,
                "prefix_token_ids": prefix,
            }
        )
        for branch in range(BRANCHES_PER_ROOT):
            turns: list[dict[str, Any]] = []
            carried_tokens = history_length
            for turn in range(TRAJECTORY_TURNS):
                append_rng = random.Random(
                    (seed << 20) + root * BRANCHES_PER_ROOT * TRAJECTORY_TURNS + branch * TRAJECTORY_TURNS + turn
                )
                append = [
                    3 + ((root * 17 + branch * 29 + turn * 43 + position + append_rng.randrange(251)) % 252)
                    for position in range(ACCEPTANCE_APPEND_TOKENS)
                ]
                prompt_tokens = carried_tokens + len(append)
                final_tokens = prompt_tokens + ACCEPTANCE_RESPONSE_TOKENS
                turns.append(
                    {
                        "turn": turn + 1,
                        "append_token_ids": append,
                        "append_token_count": len(append),
                        "prompt_token_count": prompt_tokens,
                        "max_tokens": ACCEPTANCE_RESPONSE_TOKENS,
                        "final_token_count": final_tokens,
                    }
                )
                carried_tokens = final_tokens
            requests.append(
                {
                    "request_id": f"trajectory-root-{root:02d}-branch-{branch:02d}",
                    "root": root,
                    "branch": branch,
                    "cohort": cohort,
                    "initial_history_tokens": history_length,
                    "turns": turns,
                    "final_token_count": carried_tokens,
                }
            )
    return {
        "schema_version": 1,
        "kind": "four-turn-trajectory-65k",
        "seed": seed,
        "root_count": ROOT_COUNT,
        "branches_per_root": BRANCHES_PER_ROOT,
        "request_count": len(requests),
        "turn_count": TRAJECTORY_TURNS,
        "initial_history_lengths": list(TRAJECTORY_INITIAL_HISTORY_LENGTHS),
        "append_tokens_per_turn": ACCEPTANCE_APPEND_TOKENS,
        "response_tokens_per_turn": ACCEPTANCE_RESPONSE_TOKENS,
        "final_lengths": [
            length + TRAJECTORY_TURNS * (ACCEPTANCE_APPEND_TOKENS + ACCEPTANCE_RESPONSE_TOKENS)
            for length in TRAJECTORY_INITIAL_HISTORY_LENGTHS
        ],
        "roots": roots,
        "requests": requests,
    }


def deterministic_capacity_stress_workload(*, seed: int = DUMMY_SEED) -> dict[str, Any]:
    """Build six roots and 48 branches that end at exactly 131,072 tokens."""
    rng = random.Random(seed ^ 0x131072)
    roots: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    root_count = 6
    for root in range(root_count):
        prefix = [1, *(rng.randrange(3, 255) for _ in range(CAPACITY_HISTORY_TOKENS - 1))]
        roots.append({"root": root, "prefix_token_ids": prefix})
        for branch in range(BRANCHES_PER_ROOT):
            append_rng = random.Random((seed << 16) + root * BRANCHES_PER_ROOT + branch)
            append = [
                3 + ((root * 17 + branch * 29 + position + append_rng.randrange(251)) % 252)
                for position in range(CAPACITY_APPEND_TOKENS)
            ]
            requests.append(
                {
                    "request_id": f"capacity-root-{root:02d}-branch-{branch:02d}",
                    "root": root,
                    "branch": branch,
                    "prefix_token_count": CAPACITY_HISTORY_TOKENS,
                    "append_token_ids": append,
                    "append_token_count": CAPACITY_APPEND_TOKENS,
                    "prompt_token_count": CAPACITY_HISTORY_TOKENS + CAPACITY_APPEND_TOKENS,
                    "max_tokens": CAPACITY_RESPONSE_TOKENS,
                    "final_token_count": CAPACITY_FINAL_TOKENS,
                }
            )
    return {
        "schema_version": 1,
        "kind": "capacity-stress-131k",
        "seed": seed,
        "root_count": root_count,
        "branches_per_root": BRANCHES_PER_ROOT,
        "request_count": len(requests),
        "history_tokens": CAPACITY_HISTORY_TOKENS,
        "append_tokens": CAPACITY_APPEND_TOKENS,
        "response_tokens": CAPACITY_RESPONSE_TOKENS,
        "final_tokens": CAPACITY_FINAL_TOKENS,
        "roots": roots,
        "requests": requests,
    }


def deterministic_boundary_workload(case: ModelCase, *, seed: int = DUMMY_SEED) -> dict[str, Any]:
    """Build the two exact cold/reuse probes without the large load fixture."""
    cache_hit_alignment = hybrid_kv_cache_hit_alignment(case)
    if cache_hit_alignment >= 512 or 512 % cache_hit_alignment:
        raise ValueError(
            f"{case.name}: cache-hit alignment {cache_hit_alignment} cannot probe the frozen 512-token window"
        )
    prefix_lengths = (cache_hit_alignment + 1, 513)
    rng = random.Random(seed)
    roots: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for root, prefix_length in enumerate(prefix_lengths):
        prefix = [1, *(rng.randrange(3, 255) for _ in range(prefix_length - 1))]
        mutated_prefix = [*prefix]
        mutated_prefix[1] = (mutated_prefix[1] % 251) + 3
        roots.append(
            {
                "root": root,
                "prefix_token_ids": prefix,
                "mutated_prefix_token_ids": mutated_prefix,
            }
        )
        requests.append(
            {
                "request_id": f"boundary-{prefix_length}",
                "root": root,
                "branch": 0,
                "prefix_token_count": prefix_length,
                "append_token_ids": [3 + root],
                "max_tokens": 4,
            }
        )
    return {
        "schema_version": 3,
        "kind": "exact-reference-boundaries",
        "seed": seed,
        "base_attention_block_size": KV_BLOCK_SIZE,
        "cache_hit_alignment": cache_hit_alignment,
        "root_count": len(roots),
        "branches_per_root": 1,
        "request_count": len(requests),
        "lengths": list(prefix_lengths),
        "roots": roots,
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
    if prefix_key not in root:
        if mutated:
            raise ValueError("this workload has no mutated-prefix control")
        raise ValueError("workload root omits prefix_token_ids")
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
    if model_source not in {"dummy", "fixture", "snowball"}:
        raise ValueError(f"unknown model source: {model_source}")
    model_path = {
        "dummy": "staged-config.json",
        "fixture": FROZEN_FIXTURE_PATH,
        "snowball": SNOWBALL_EXPORT,
    }[model_source]
    load_format = {
        "dummy": "dummy",
        "fixture": "safetensors",
        "snowball": "runai_streamer",
    }[model_source]
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
    # The benchmark normally sends token IDs directly and starts vLLM with
    # ``--skip-tokenizer-init``.  The matched MarinSkyRL carrier check must use
    # ``/v1/chat/completions`` instead.  This tiny tokenizer makes the chat
    # renderer an identity map over the exact frozen 0..255 model vocabulary.
    # Two-digit hexadecimal keeps a 65K prompt below vLLM's conservative
    # four-characters-per-token frontend guard: ``01 25 09`` becomes token IDs
    # ``[1, 37, 9]``. It adds no role markers, BOS token, EOS token, or suffix.
    vocabulary = {token: token_id for token_id, token in enumerate(IDENTITY_CHAT_TOKENS)}
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None,
        "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": False},
        "model": {
            "type": "WordLevel",
            "vocab": vocabulary,
            "unk_token": IDENTITY_CHAT_TOKENS[0],
        },
    }
    chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    (output_dir / "tokenizer.json").write_text(json.dumps(tokenizer, sort_keys=True) + "\n")
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": case.max_model_len,
                "unk_token": IDENTITY_CHAT_TOKENS[0],
                "chat_template": chat_template,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps({"unk_token": IDENTITY_CHAT_TOKENS[0]}, sort_keys=True) + "\n"
    )
    workload = deterministic_workload()
    (output_dir / "workload.json").write_text(json.dumps(workload, indent=2, sort_keys=True) + "\n")
    boundary_workload = deterministic_boundary_workload(case)
    (output_dir / "correctness-workload.json").write_text(json.dumps(boundary_workload, indent=2, sort_keys=True) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(frozen_manifest(case, run_id=run_id, git_sha=git_sha), indent=2, sort_keys=True) + "\n"
    )


__all__ = [
    "ARTIFACT_ROOT",
    "BRANCH_COUNT",
    "CASES",
    "DUMMY_SEED",
    "FROZEN_FIXTURE_PATH",
    "IDENTITY_CHAT_TOKENS",
    "KV_BLOCK_SIZE",
    "MARIN_BASE_SHA",
    "P0_SMOKE_CASES",
    "SNOWBALL_EXPORT",
    "VLLM_SHA",
    "ModelCase",
    "aggregate_preflight_status",
    "decode_routed_experts",
    "deterministic_balanced_routing_fixture",
    "deterministic_boundary_workload",
    "deterministic_workload",
    "expert_parallel_rank_histogram",
    "frozen_manifest",
    "hybrid_kv_cache_hit_alignment",
    "layer_types",
    "materialize_prompt",
    "metric_delta",
    "parse_prometheus",
    "predict_kv_bytes",
    "routing_histogram",
    "write_case",
]
