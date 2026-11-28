# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the Megatron-DCP -> Levanter-canonical parameter mapping for RL checkpoints.

The RL trainer initializes its FP32 master weights from the BF16 HF export of the
SFT checkpoint, so Megatron-side tensors keep the HF export orientation
([out, in] for projections). Mapping entries therefore carry ``transpose=True``
for the tensors that HF_MAP marks as transposed, and the measurement slices in
Levanter orientation after the transpose.

The mapping is shape- and name-driven; ``build_mapping`` reports anything it
cannot place so the heuristics can be refined without touching checkpoints.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

# Levanter canonical name -> (role regex hints, Levanter shape, transpose-from-HF)
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

# Match order matters: expert patterns before shared/router patterns.
ROLE_PATTERNS: list[tuple[str, str, re.Pattern]] = [
    # (canonical-name regex anchor, subgroup, key regex over the DCP key)
    (
        "stacked_blocks/stacked/mlp/expert_mlp/w_gate",
        "expert",
        re.compile(r"gate_proj|w_gate|fc1|linear_fc1|gate", re.I),
    ),
    ("stacked_blocks/stacked/mlp/expert_mlp/w_up", "expert", re.compile(r"up_proj|w_up|linear_fc1[2]?|up", re.I)),
    ("stacked_blocks/stacked/mlp/expert_mlp/w_down", "expert", re.compile(r"down_proj|w_down|linear_fc2|down", re.I)),
]


@dataclass
class MapResult:
    mapping: dict = field(default_factory=dict)
    unmapped: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _is_master_weight(row: dict) -> bool:
    key = row["key"]
    dtype = row["dtype"]
    return "float32" in dtype and ("fp32_from_fp16" in key or "optimizer" in key)


def build_mapping(entries: list[dict], layers_per_stage: int | None = None) -> MapResult:
    """Map DCP entries to Levanter canonical names.

    ``entries`` is the compact inventory from MegatronInventory.entries().
    Returns the mapping plus diagnostics; unmapped families are listed, not fatal.
    """
    result = MapResult()
    masters = [row for row in entries if _is_master_weight(row)]
    result.notes.append(f"entries={len(entries)} master_candidates={len(masters)}")
    if not masters:
        result.unmapped.append({"reason": "no fp32 master candidates found", "sample": entries[:10]})
        return result

    # Family grouping: strip digits to see the key structure.
    families: dict[str, list[dict]] = defaultdict(list)
    for row in masters:
        families[re.sub(r"\d+", "N", row["key"])].append(row)
    result.notes.append(
        "families: "
        + json.dumps(
            {
                name: {"count": len(rows), "size": rows[0]["size"], "dtype": rows[0]["dtype"]}
                for name, rows in sorted(families.items())[:80]
            },
            indent=1,
            default=str,
        )[:4000]
    )

    # The real mapping pass runs after the key layout is known; for now attach
    # every family to the preview so the layout can be finalized offline.
    result.unmapped.append({"reason": "mapping pass pending layout confirmation"})
    return result


def preview(entries: list[dict], limit: int = 12) -> dict:
    """Compact preview of an inventory: family stats plus example keys."""
    families: dict[str, dict] = {}
    for row in entries:
        family = re.sub(r"\d+", "N", row["key"])
        stat = families.setdefault(family, {"count": 0, "dtypes": set(), "sizes": set(), "examples": []})
        stat["count"] += 1
        stat["dtypes"].add(row["dtype"])
        stat["sizes"].add(tuple(row["size"]))
        if len(stat["examples"]) < limit:
            chunk_files = sorted({c["file"] for c in row["chunks"]})[:4]
            offsets = [c["offsets"] for c in row["chunks"][:3]]
            stat["examples"].append({"key": row["key"], "files": chunk_files, "offsets": offsets})
    out = {}
    for family, stat in families.items():
        out[family] = {
            "count": stat["count"],
            "dtypes": sorted(stat["dtypes"]),
            "sizes": [list(s) for s in stat["sizes"]][:4],
            "examples": stat["examples"],
        }
    return out
