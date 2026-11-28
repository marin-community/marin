# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint registry for the Mochi per-update weight-movement measurement.

Every URI below is read-only. Outputs go only under
``s3://marin-us-east-08a/marin/users/benfeuer/mochi-update-magnitude/``.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field

import fsspec

OUTPUT_PREFIX = "s3://marin-us-east-08a/marin/users/benfeuer/mochi-update-magnitude"

# ---------------------------------------------------------------------------
# Checkpoint formats
# ---------------------------------------------------------------------------

LEVANTER = "levanter_ocdbt"  # Grug/Levanter native OCDBT+zarr3, FP32 params
MEGATRON = "megatron_dcp"  # SkyRL native torch distributed checkpoint (policy/)
SAFETENSORS = "safetensors_hf"  # BF16 Hugging Face export

TMP_SKYRL = "s3://marin-us-east-08a/tmp/ttl=14d/skyrl/marin/users/benfeuer/mochi-long-loop"
LOOP = "s3://marin-us-east-08a/marin/users/benfeuer/mochi-long-loop"


@dataclass(frozen=True)
class Checkpoint:
    """One readable checkpoint side."""

    id: str
    uri: str
    kind: str
    step: int
    note: str = ""


@dataclass(frozen=True)
class Pair:
    """Two checkpoints a known number of optimizer updates apart."""

    id: str
    side_a: str  # Checkpoint.id
    side_b: str
    updates: int
    stage: str  # "rl" or "sft"
    config: str  # optimizer-config label for the report
    note: str = ""


SFT_BASE = Checkpoint(
    id="hero-base-393491",
    uri="s3://marin-us-east-08a/marin/users/benfeuer/mochi-sft-initializers/hero-base393491-native/step-393491",
    kind=LEVANTER,
    step=393491,
    note="verified regional copy of the hero initializer (metadata_sha256 07df6c2f); FP32 native",
)

SFT_HERO_FINAL = Checkpoint(
    id="hero-sft-400407-export",
    uri="s3://marin-us-east-08a/marin/models/mochi-sft/hero-sft-fullab-step393k-v2/step-400407/2af3f828aa-cpu-a1",
    kind=SAFETENSORS,
    step=400407,
    note="hero SFT final; BF16 HF export only (no FP32 copy in region)",
)

SFT_EXP3_FINAL = Checkpoint(
    id="exp3-v5-native-2100",
    uri=f"{LOOP}/long-loop-mochi-sft-exp3-v5-bankadam-128/sft/2026.11.24.128/checkpoints/step-2100",
    kind=LEVANTER,
    step=2100,
    note="exp3-v5-bankadam SFT final; FP32 native; initializer = hero base 393491",
)

SFT_EXP3_EXPORT = Checkpoint(
    id="v5-sft2100-export",
    uri=f"{LOOP}/long-loop-mochi-sft-exp3-v5-bankadam-128/sft/2026.11.24.128/exports/step-2100",
    kind=SAFETENSORS,
    step=2100,
    note="BF16 export of exp3 step 2100; the campaign RL initializer",
)


def _rl_ckpt(ckpt_id: str, uri: str, step: int, note: str) -> Checkpoint:
    return Checkpoint(id=ckpt_id, uri=uri, kind=MEGATRON, step=step, note=note)


RL_533_STEP3 = _rl_ckpt(
    "rl-533-step3",
    f"{TMP_SKYRL}/rl-data-campaign/v3-mix-007/2026.11.26.533/checkpoints/global_step_3/policy",
    3,
    "campaign row v3-mix-007; exact .131 config; init v5-sft2100",
)
RL_131_33 = _rl_ckpt(
    "rl-131-step33",
    f"{LOOP}/long-loop/mochi-exp4-dspark-rl/2026.11.25.131/checkpoints/global_step_33/policy",
    33,
    "original .131 run",
)
RL_131_34 = _rl_ckpt(
    "rl-131-step34",
    f"{LOOP}/long-loop/mochi-exp4-dspark-rl/2026.11.25.131/checkpoints/global_step_34/policy",
    34,
    "original .131 run",
)
RL_129_24 = _rl_ckpt(
    "rl-129-step24",
    f"{LOOP}/long-loop/mochi-v5-exp3-rl/2026.11.25.129/checkpoints/global_step_24/policy",
    24,
    ".129 run; .131 optimizer config",
)
RL_129_25 = _rl_ckpt(
    "rl-129-step25",
    f"{LOOP}/long-loop/mochi-v5-exp3-rl/2026.11.25.129/checkpoints/global_step_25/policy",
    25,
    ".129 run; .131 optimizer config",
)
RL_717_25 = _rl_ckpt(
    "rl-717-step25",
    f"{TMP_SKYRL}/rl-data-campaign/v2-mix-009/2026.11.27.717/checkpoints/global_step_25/policy",
    25,
    "campaign row v2-mix-009 (.131 config)",
)
RL_717_30 = _rl_ckpt(
    "rl-717-step30",
    f"{TMP_SKYRL}/rl-data-campaign/v2-mix-009/2026.11.27.717/checkpoints/global_step_30/policy",
    30,
    "campaign row v2-mix-009 (.131 config)",
)

# Live campaign rows roll their checkpoints (keep-2 retention); the .719 step-20
# and .721 step-10/5 pins were pruned by that retention. Resolve whatever
# consecutive cadence pair each live row still holds at job start instead.
LIVE_ROWS = (
    ("rl-719", "v2-mix-033/2026.11.27.719"),
    ("rl-721", "v2-mix-019/2026.11.27.721"),
)


def live_checkpoint_root(row: str) -> str:
    return f"{TMP_SKYRL}/rl-data-campaign/{row}/checkpoints"


def resolve_live_pairs() -> list[Pair]:
    """Pin the two newest retained checkpoints of each live row as a pair."""

    pairs: list[Pair] = []
    for base, row in LIVE_ROWS:
        root = live_checkpoint_root(row)
        try:
            fs, _ = fsspec.url_to_fs(root)
            steps = sorted(
                int(entry.rsplit("global_step_", 1)[-1])
                for entry in fs.ls(root, detail=False)
                if "global_step_" in entry
            )
        except (FileNotFoundError, OSError):
            continue
        if len(steps) < 2:
            continue
        a, b = steps[-2], steps[-1]
        for step in (a, b):
            ckpt_id = f"{base}-step{step}"
            CHECKPOINTS.setdefault(
                ckpt_id,
                _rl_ckpt(
                    ckpt_id,
                    posixpath.join(root, f"global_step_{step}", "policy"),
                    step,
                    f"live campaign row {row} (.131 config)",
                ),
            )
        pairs.append(
            Pair(
                f"{base}-live",
                f"{base}-step{a}",
                f"{base}-step{b}",
                b - a,
                "rl",
                CONFIG_131,
                f"live rolling pair resolved at job start: steps {a} -> {b}",
            )
        )
    return pairs


RL_403_1 = _rl_ckpt(
    "rl-403-step1",
    f"{LOOP}/long-loop/anchor-mochi-acceptance/2026.11.20.403/checkpoints/global_step_1/policy",
    1,
    "anchor acceptance run from SFT 400407; max_grad_norm 0.0",
)
RL_403_2 = _rl_ckpt(
    "rl-403-step2",
    f"{LOOP}/long-loop/anchor-mochi-acceptance/2026.11.20.403/checkpoints/global_step_2/policy",
    2,
    "anchor acceptance run from SFT 400407; max_grad_norm 0.0",
)
RL_123_9 = _rl_ckpt(
    "rl-123-step9",
    f"{LOOP}/long-loop/mochi-throughput-train-track1/2026.11.23.123/checkpoints/global_step_9/policy",
    9,
    "track1 run from SFT 400407; max_grad_norm 0.0",
)
RL_123_22 = _rl_ckpt(
    "rl-123-step22",
    f"{LOOP}/long-loop/mochi-throughput-train-track1/2026.11.23.123/checkpoints/global_step_22/policy",
    22,
    "track1 run from SFT 400407; max_grad_norm 0.0",
)

CHECKPOINTS: dict[str, Checkpoint] = {
    c.id: c
    for c in (
        SFT_BASE,
        SFT_HERO_FINAL,
        SFT_EXP3_FINAL,
        SFT_EXP3_EXPORT,
        RL_533_STEP3,
        RL_131_33,
        RL_131_34,
        RL_129_24,
        RL_129_25,
        RL_717_25,
        RL_717_30,
        RL_403_1,
        RL_403_2,
        RL_123_9,
        RL_123_22,
    )
}

# .131 optimizer config (campaign): MuonH lr 1e-5, Adam 2.31e-6, constant, clip 0.5.
CONFIG_131 = "131: MuonH 1e-5 / Adam 2.31e-6 / constant / clip 0.5"
# .123/.403 track/anchor arms differ only in the clip.
CONFIG_131_NOCLIP = "131-noclip: MuonH 1e-5 / Adam 2.31e-6 / constant / clip 0.0"

PAIRS: list[Pair] = [
    # --- RL, .131 config -----------------------------------------------------
    Pair(
        "rl-533-0to3",
        SFT_EXP3_EXPORT.id,
        RL_533_STEP3.id,
        3,
        "rl",
        CONFIG_131,
        "closest available 'RL step 3' anchor (initializer is the campaign's SFT export)",
    ),
    Pair("rl-131-33to34", RL_131_33.id, RL_131_34.id, 1, "rl", CONFIG_131, "single update, original .131 run"),
    Pair("rl-129-24to25", RL_129_24.id, RL_129_25.id, 1, "rl", CONFIG_131, "single update"),
    Pair("rl-717-25to30", RL_717_25.id, RL_717_30.id, 5, "rl", CONFIG_131, "campaign row"),
    # --- RL anchored at the hero SFT final -----------------------------------
    Pair(
        "rl-400407-to-403s1",
        SFT_HERO_FINAL.id,
        RL_403_1.id,
        1,
        "rl",
        CONFIG_131_NOCLIP,
        "brief's 'SFT 400407 -> RL step 3' request: step-3 checkpoints from a 400407-initialized "
        "run are not retained; nearest anchors are step 1/2 (.403) and step 9 (.123)",
    ),
    Pair("rl-400407-to-403s2", SFT_HERO_FINAL.id, RL_403_2.id, 2, "rl", CONFIG_131_NOCLIP, ""),
    Pair("rl-400407-to-123s9", SFT_HERO_FINAL.id, RL_123_9.id, 9, "rl", CONFIG_131_NOCLIP, ""),
    Pair("rl-123-9to22", RL_123_9.id, RL_123_22.id, 13, "rl", CONFIG_131_NOCLIP, ""),
    # --- SFT ------------------------------------------------------------------
    Pair(
        "sft-exp3-0to2100",
        SFT_BASE.id,
        SFT_EXP3_FINAL.id,
        2100,
        "sft",
        "sft exp3-v5-bankadam: MuonH 1.4031e-5 (2D/3D) / Adam 3.2378e-6 (banks+rest) / linear decay tail 210",
        "clean FP32 pair; the SFT stage that produced the campaign initializer",
    ),
    Pair(
        "sft-hero-0to6916",
        SFT_BASE.id,
        SFT_HERO_FINAL.id,
        6916,
        "sft",
        "sft hero: MuonH 3.2905e-5 (2D/3D/4D) / Adam 3.2905e-5 / clip 1.0",
        "mixed precision: side B is a BF16 export; BF16 rounding floor ~0.11% RMS per element",
    ),
    # --- calibration ----------------------------------------------------------
    Pair(
        "calib-exp3-native-vs-export",
        SFT_EXP3_FINAL.id,
        SFT_EXP3_EXPORT.id,
        0,
        "sft",
        "same step, FP32 native vs its BF16 export",
        "measures pure BF16 export rounding",
    ),
]

# ---------------------------------------------------------------------------
# Optimizer-config table (for the report)
# ---------------------------------------------------------------------------

OPTIMIZER_CONFIGS = {
    "hero-sft-400407": {
        "muon_lr": 3.2905e-5,
        "adam_lr": 3.2905e-5,
        "max_grad_norm": 1.0,
        "schedule": "warmup 0, decay 692/6916 (linear to 0.1x), 6,916 updates",
        "muonh_param_ndims": "(2, 3, 4) - expert banks are MuonH",
        "source": "experiments/post_training/mochi_sft.py",
    },
    "exp3-v5-bankadam-128": {
        "muon_lr": 1.4030739143362721e-05,
        "adam_lr": 3.2378366391493846e-06,
        "max_grad_norm": 1.0,
        "schedule": "warmup 5, linear decay tail 210 of 2,100 updates, min ratio 0.1",
        "muonh_param_ndims": "(2, 3) - expert banks are Adam",
        "source": "s3://.../mochi-sft-preparation/bankadam128-20261124/reference-dafc25df.json",
    },
    "rl-131": {
        "muon_lr": 1.0e-5,
        "adam_lr": 2.307692307692308e-06,
        "max_grad_norm": 0.5,
        "schedule": "constant_with_warmup, warmup 0, 50 steps",
        "muonh_param_ndims": "Megatron MuonH per grug_megatron grouping",
        "source": "experiments/post_training/long_loop/campaign/mochi131.yaml",
    },
}

# ---------------------------------------------------------------------------
# Fixed sample (applied identically to both sides of every pair)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """The fixed tensor sample. Record this with every result."""

    layers: tuple[int, ...] = (0, 8, 16, 24, 32, 40, 47)
    experts: tuple[int, ...] = (0, 1, 2, 3, 4, 5)  # one contiguous bank group on both OCDBT layouts
    expert_layer_groups: tuple[tuple[int, int], ...] = ((0, 6), (24, 6), (42, 6))  # (start, count)
    # Expert-matrix area slab [dim2 start : start+384) x [dim3 start : start+768) - the
    # 1/64 area imposed by the exp3 checkpoint's chunk shape [48, 6, 384, 768].
    expert_area: tuple[int, int, int, int] = (0, 384, 0, 768)
    attention_rows: int = 48  # rows 0..47 of w_q / w_k / w_v, cols 0..47 of w_o
    embed_row_bands: tuple[tuple[int, int], ...] = ((0, 768), (64000, 768), (127488, 768))
    router_layers: tuple[int, ...] = (0, 1, 16, 17, 32, 33, 46, 47)
    norm_layers: tuple[int, ...] = tuple(range(48))  # norm vectors are tiny; read fully

    def describe(self) -> str:
        return (
            f"layers every 8th + last {list(self.layers)}; experts 0..5 (contiguous group); expert area slab "
            "dim2 0..383 x dim3 0..767 (1/64 of each 3072x3072 expert matrix, imposed by the exp3 "
            "checkpoint chunk shape); expert layer groups 0-5, 24-29, 42-47; attention rows/cols "
            "0..47; embedding row bands 0-767, 64000-64767, 127488-128255; router layers "
            f"{list(self.router_layers)}; norm vectors read in full"
        )


SAMPLE = Sample()

# ---------------------------------------------------------------------------
# Parameter groups (Levanter/Grug canonical names, leading stacked dim = layer)
# ---------------------------------------------------------------------------

GROUPS: dict[str, dict[str, list[str]]] = {
    # group -> {subgroup -> Levanter tensor-name templates}
    "attention": {
        "w_q": ["stacked_blocks/stacked/attn/w_q"],
        "w_k": ["stacked_blocks/stacked/attn/w_k"],
        "w_v": ["stacked_blocks/stacked/attn/w_v"],
        "w_o": ["stacked_blocks/stacked/attn/w_o"],
    },
    "expert_banks": {
        "w_gate": ["stacked_blocks/stacked/mlp/expert_mlp/w_gate"],
        "w_up": ["stacked_blocks/stacked/mlp/expert_mlp/w_up"],
        "w_down": ["stacked_blocks/stacked/mlp/expert_mlp/w_down"],
    },
    "shared_dense_mlp": {
        "shared_experts": [
            "stacked_blocks/stacked/shared/0/w_gate",
            "stacked_blocks/stacked/shared/0/w_up",
            "stacked_blocks/stacked/shared/0/w_down",
            "stacked_blocks/stacked/shared/1/w_gate",
            "stacked_blocks/stacked/shared/1/w_up",
            "stacked_blocks/stacked/shared/1/w_down",
        ],
        "dense_latent": [
            "stacked_blocks/stacked/mlp/w_latent_down",
            "stacked_blocks/stacked/mlp/w_latent_up",
        ],
    },
    "router": {
        "router": ["stacked_blocks/stacked/mlp/router"],
        "router_bias": ["stacked_blocks/stacked/mlp/router_bias"],
    },
    "norm_scales": {
        "rms": [
            "stacked_blocks/stacked/rms_attn/weight",
            "stacked_blocks/stacked/rms_mlp/weight",
            "stacked_blocks/stacked/mlp/latent_norm/weight",
            "embed_norm/weight",
            "final_norm/weight",
        ],
        "gated_norm_projections": [
            "stacked_blocks/stacked/attn_gated_norm/w_down",
            "stacked_blocks/stacked/attn_gated_norm/w_up",
            "stacked_blocks/stacked/mlp_gated_norm/w_down",
            "stacked_blocks/stacked/mlp_gated_norm/w_up",
            "embed_gated_norm/w_down",
            "embed_gated_norm/w_up",
            "final_gated_norm/w_down",
            "final_gated_norm/w_up",
        ],
    },
    "embed_output": {
        "token_embed": ["token_embed"],
        "output_proj": ["output_proj"],
    },
    "attention_conv_gate": {
        "conv_gate": [
            "stacked_blocks/stacked/attn/attn_gate",
            "stacked_blocks/stacked/attn/sconv_k/weight",
            "stacked_blocks/stacked/sconv_attn/weight",
            "stacked_blocks/stacked/sconv_mlp/weight",
        ],
    },
}

# Stacked (layer-leading) tensors are sampled at SAMPLE.layers; scalars/norms read fully.
FULL_READ_TENSORS = {
    "stacked_blocks/stacked/rms_attn/weight",
    "stacked_blocks/stacked/rms_mlp/weight",
    "stacked_blocks/stacked/mlp/latent_norm/weight",
    "stacked_blocks/stacked/mlp/router_bias",
    "embed_norm/weight",
    "final_norm/weight",
}

OUTPUT_TENSORS = {"token_embed", "output_proj"}


@dataclass
class PairPlan:
    """Resolved read plan for one pair."""

    pair: Pair
    a: Checkpoint
    b: Checkpoint
    requests: list[dict] = field(default_factory=list)  # per-reader slice requests


def pairs_for(stage: str | None = None, pair_ids: list[str] | None = None) -> list[Pair]:
    selected = list(PAIRS)
    if stage in (None, "rl"):
        selected.extend(resolve_live_pairs())
    if stage:
        selected = [p for p in selected if p.stage == stage]
    if pair_ids:
        by_id = {p.id: p for p in selected}
        selected = [by_id[pid] for pid in pair_ids]
    return selected
