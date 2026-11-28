# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Megatron-DCP -> Levanter-canonical parameter mapping for the RL checkpoints.

The SkyRL grug Megatron model stores FP32 master weights as full stacked tensors
under ``chained_<i>.optimizer.state.fp32_param.<megatron path>``, with the HF
export orientation ([out, in] projections, fused gate+up and fused q|k|v rows).
This module emits, per Levanter canonical tensor, the read entry the measurement
needs: the DCP key, the Megatron global shape, which axes swap (transpose), and
the fused-axis range for qkv / gate+up fusion.

Orientation for square projections (w_o, expert matrices) is ambiguous from
shapes alone; the measurement resolves it empirically on the <=3-update anchor
pairs (a mis-oriented square tensor shows a ~100% delta on a 1-update pair).
"""

from __future__ import annotations

import json
import re

import numpy as np

# Levanter canonical tensor shapes (stacked leading dim = layer), used by the
# HF side for shape-driven orientation checks and by the mapping builder.
LEVANTER_SHAPES = {
    "token_embed": ([128256, 6144], False),
    "output_proj": ([6144, 128256], True),
    "embed_norm/weight": ([6144], False),
    "final_norm/weight": ([6144], False),
    "embed_gated_norm/w_down": ([6144, 128], True),
    "embed_gated_norm/w_up": ([128, 6144], False),
    "final_gated_norm/w_down": ([6144, 128], True),
    "final_gated_norm/w_up": ([128, 6144], False),
    "stacked_blocks/stacked/rms_attn/weight": ([48, 6144], False),
    "stacked_blocks/stacked/rms_mlp/weight": ([48, 6144], False),
    "stacked_blocks/stacked/attn/w_q": ([48, 6144, 6144], True),
    "stacked_blocks/stacked/attn/w_k": ([48, 6144, 1536], True),
    "stacked_blocks/stacked/attn/w_v": ([48, 6144, 1536], True),
    "stacked_blocks/stacked/attn/w_o": ([48, 6144, 6144], True),
    "stacked_blocks/stacked/attn/attn_gate": ([48, 6144, 48], False),
    "stacked_blocks/stacked/attn/sconv_k/weight": ([48, 4, 1536], False),
    "stacked_blocks/stacked/sconv_attn/weight": ([48, 4, 6144], False),
    "stacked_blocks/stacked/sconv_mlp/weight": ([48, 4, 6144], False),
    "stacked_blocks/stacked/attn_gated_norm/w_down": ([48, 6144, 128], True),
    "stacked_blocks/stacked/attn_gated_norm/w_up": ([48, 128, 6144], False),
    "stacked_blocks/stacked/mlp_gated_norm/w_down": ([48, 6144, 128], True),
    "stacked_blocks/stacked/mlp_gated_norm/w_up": ([48, 128, 6144], False),
    "stacked_blocks/stacked/mlp/router": ([48, 6144, 384], True),
    "stacked_blocks/stacked/mlp/router_bias": ([48, 384], False),
    "stacked_blocks/stacked/mlp/latent_norm/weight": ([48, 3072], False),
    "stacked_blocks/stacked/mlp/w_latent_down": ([48, 6144, 3072], True),
    "stacked_blocks/stacked/mlp/w_latent_up": ([48, 3072, 6144], True),
    "stacked_blocks/stacked/shared/0/w_gate": ([48, 6144, 3072], True),
    "stacked_blocks/stacked/shared/0/w_up": ([48, 6144, 3072], True),
    "stacked_blocks/stacked/shared/0/w_down": ([48, 3072, 6144], True),
    "stacked_blocks/stacked/shared/1/w_gate": ([48, 6144, 3072], True),
    "stacked_blocks/stacked/shared/1/w_up": ([48, 6144, 3072], True),
    "stacked_blocks/stacked/shared/1/w_down": ([48, 3072, 6144], True),
    "stacked_blocks/stacked/mlp/expert_mlp/w_gate": ([48, 384, 3072, 3072], True),
    "stacked_blocks/stacked/mlp/expert_mlp/w_up": ([48, 384, 3072, 3072], True),
    "stacked_blocks/stacked/mlp/expert_mlp/w_down": ([48, 384, 3072, 3072], True),
}

# Megatron key fragments per canonical tensor. {N} placeholders are resolved
# against the checkpoint's own key inventory (shared experts carry an index).
# fused: (megatron axis, start, stop) selects this tensor out of a fused axis.
MEGATRON_LAYOUT = {
    "token_embed": ("embedding.word_embeddings.weight", False, None),
    "output_proj": ("output_layer.weight", True, None),
    "embed_norm/weight": ("embed_norm.norm.weight", False, None),
    "final_norm/weight": ("decoder.final_layernorm.norm.weight", False, None),
    "embed_gated_norm/w_down": ("embed_norm.down_proj.weight", True, None),
    "embed_gated_norm/w_up": ("embed_norm.up_proj.weight", True, None),
    "final_gated_norm/w_down": ("decoder.final_layernorm.down_proj.weight", True, None),
    "final_gated_norm/w_up": ("decoder.final_layernorm.up_proj.weight", True, None),
    "stacked_blocks/stacked/rms_attn/weight": ("decoder.layers.input_layernorm.norm.weight", False, None),
    "stacked_blocks/stacked/rms_mlp/weight": ("decoder.layers.pre_mlp_layernorm.norm.weight", False, None),
    "stacked_blocks/stacked/attn/w_q": (
        "decoder.layers.self_attention.linear_qkv.weight",
        True,
        {"dim": 1, "layout": "qkv", "part": "q", "lev_extent": 6144},
    ),
    "stacked_blocks/stacked/attn/w_k": (
        "decoder.layers.self_attention.linear_qkv.weight",
        True,
        {"dim": 1, "layout": "qkv", "part": "k", "lev_extent": 1536},
    ),
    "stacked_blocks/stacked/attn/w_v": (
        "decoder.layers.self_attention.linear_qkv.weight",
        True,
        {"dim": 1, "layout": "qkv", "part": "v", "lev_extent": 1536},
    ),
    "stacked_blocks/stacked/attn/w_o": ("decoder.layers.self_attention.linear_proj.weight", True, None),
    "stacked_blocks/stacked/attn/attn_gate": ("decoder.layers.self_attention.attn_gate.weight", True, None),
    "stacked_blocks/stacked/attn/sconv_k/weight": ("decoder.layers.self_attention.sconv_k.weight", False, None),
    "stacked_blocks/stacked/sconv_attn/weight": ("decoder.layers.self_attention.sconv_attn.weight", False, None),
    "stacked_blocks/stacked/sconv_mlp/weight": ("decoder.layers.sconv_mlp.weight", False, None),
    "stacked_blocks/stacked/attn_gated_norm/w_down": ("decoder.layers.input_layernorm.down_proj.weight", True, None),
    "stacked_blocks/stacked/attn_gated_norm/w_up": ("decoder.layers.input_layernorm.up_proj.weight", True, None),
    "stacked_blocks/stacked/mlp_gated_norm/w_down": ("decoder.layers.pre_mlp_layernorm.down_proj.weight", True, None),
    "stacked_blocks/stacked/mlp_gated_norm/w_up": ("decoder.layers.pre_mlp_layernorm.up_proj.weight", True, None),
    "stacked_blocks/stacked/mlp/router": ("decoder.layers.mlp.router.weight", True, None),
    "stacked_blocks/stacked/mlp/router_bias": ("decoder.layers.mlp.router.expert_bias", False, None),
    "stacked_blocks/stacked/mlp/latent_norm/weight": ("decoder.layers.mlp.latent_norm.weight", False, None),
    "stacked_blocks/stacked/mlp/w_latent_down": ("decoder.layers.mlp.fc1_latent_proj.weight", True, None),
    "stacked_blocks/stacked/mlp/w_latent_up": ("decoder.layers.mlp.fc2_latent_proj.weight", True, None),
    "stacked_blocks/stacked/shared/{N}/w_gate": (
        "decoder.layers.mlp.shared_experts.experts.{N}.linear_fc1.weight",
        True,
        (1, 0, 3072),
    ),
    "stacked_blocks/stacked/shared/{N}/w_up": (
        "decoder.layers.mlp.shared_experts.experts.{N}.linear_fc1.weight",
        True,
        (1, 3072, 6144),
    ),
    "stacked_blocks/stacked/shared/{N}/w_down": (
        "decoder.layers.mlp.shared_experts.experts.{N}.linear_fc2.weight",
        True,
        None,
    ),
    "stacked_blocks/stacked/mlp/expert_mlp/w_gate": (
        "decoder.layers.mlp.experts.experts.linear_fc1.weight",
        True,
        (2, 0, 3072),
    ),
    "stacked_blocks/stacked/mlp/expert_mlp/w_up": (
        "decoder.layers.mlp.experts.experts.linear_fc1.weight",
        True,
        (2, 3072, 6144),
    ),
    "stacked_blocks/stacked/mlp/expert_mlp/w_down": ("decoder.layers.mlp.experts.experts.linear_fc2.weight", True, None),
}


def _find_key(entries_by_key: dict, fragment: str):
    """Find the fp32_param key ending with ``fragment``; returns (key, size) or None."""
    for key, size in entries_by_key.items():
        if key.startswith("chained_") and ".optimizer.state.fp32_param." in key and key.endswith("." + fragment):
            return key, size
    return None


def build_mapping(entries: list[dict]) -> tuple[dict, list[str]]:
    """Emit the mapping for one checkpoint from its inventory entries.

    Returns ({canonical_name: entry}, notes). Entries:
      {"key": dcp_key, "shape": [...], "transpose": bool,
       "fused": {"dim": int, "start": int, "stop": int}
                | {"dim": int, "layout": "qkv", "part": "q"|"k"|"v"}
                | None}
    """
    notes: list[str] = []
    by_key = {row["key"]: row["size"] for row in entries}
    prefix = "chained_0.optimizer.state.fp32_param."
    mapping: dict[str, dict] = {}
    for canonical, (fragment, transpose, fused) in MEGATRON_LAYOUT.items():
        fused_spec = None
        if fused is not None:
            fused_spec = (
                dict(fused) if isinstance(fused, dict) else {"dim": fused[0], "start": fused[1], "stop": fused[2]}
            )
        if "{N}" in canonical:
            for n in (0, 1):
                found = _find_key(by_key, fragment.replace("{N}", str(n)))
                name = canonical.replace("{N}", str(n))
                if found is None:
                    notes.append(f"missing: {name} ({fragment})")
                    continue
                key, size = found
                mapping[name] = {
                    "key": key,
                    "shape": size,
                    "transpose": transpose,
                    "fused": fused_spec,
                }
            continue
        found = _find_key(by_key, fragment)
        if found is None:
            # Fall back to a prefix-relative match for keys without the chained_ wrapper.
            direct = by_key.get(prefix + fragment)
            found = (prefix + fragment, direct) if direct else None
        if found is None:
            notes.append(f"missing: {canonical} ({fragment})")
            continue
        key, size = found
        mapping[canonical] = {
            "key": key,
            "shape": size,
            "transpose": transpose,
            "fused": fused_spec,
        }
    notes.append(f"mapped {len(mapping)} canonical tensors")
    return mapping, notes


def qkv_row_index(part: str, num_groups: int = 12, heads_per_group: int = 4, head_dim: int = 128):
    """Megatron interleaved qkv row order: per group [q heads | k | v]."""
    group_rows = (heads_per_group + 2) * head_dim
    offset = {"q": 0, "k": heads_per_group * head_dim, "v": (heads_per_group + 1) * head_dim}[part]
    width = heads_per_group * head_dim if part == "q" else head_dim
    return np.concatenate(
        [np.arange(g * group_rows + offset, g * group_rows + offset + width) for g in range(num_groups)]
    )


def preview(entries: list[dict], limit: int = 8) -> dict:
    """Compact preview of an inventory: family stats plus example keys."""
    families: dict[str, dict] = {}
    for row in entries:
        family = re.sub(r"\d+", "N", row["key"])
        stat = families.setdefault(family, {"count": 0, "sizes": set(), "examples": []})
        stat["count"] += 1
        stat["sizes"].add(tuple(row["size"]))
        if len(stat["examples"]) < limit:
            stat["examples"].append({"key": row["key"], "n_chunks": row["n_chunks"]})
    return json.loads(
        json.dumps(
            {
                family: {"count": s["count"], "sizes": [list(x) for x in s["sizes"]][:3], "examples": s["examples"]}
                for family, s in sorted(families.items())
            },
            default=str,
        )
    )
