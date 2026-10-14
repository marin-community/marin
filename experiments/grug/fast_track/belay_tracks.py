# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Marin Belay baseline tracks (marin#9451): Vanilla and Basic MoE, one run per H100 ladder rung.

- **Vanilla**: a barebones Qwen3-style dense transformer trained with AdamW at 20 tokens per active parameter. Full
  causal GQA attention on every layer (KV heads = heads / 2) with full RoPE and QK-norm, SwiGLU MLP 3 d wide,
  RMSNorm; no short convs, gated norms, XSA or attention output gate. AdamW with Levanter's defaults (warmup 1%,
  cosine to 10%, weight decay 0.1, betas 0.9 / 0.95, clip 1.0) and DeepSeek LLM's compute-optimal learning rate
  0.3118 C^-0.125 (C = training FLOPs): 2.3e-3 at d512 to 1.25e-3 at d1280.
- **Basic MoE**: the fast_track baseline MoE (QB routing, sliding-window / global GQA) with 128 experts top-4, two
  shared experts, expert width and latent d/2, MuonH, at 60 tokens per active parameter.

    uv run python -m experiments.grug.fast_track.belay_tracks --track vanilla --size d512 --submit \\
        --run-id belay-vanilla-d512 --target-cluster cw-us-east-02a --priority production

Arguments after the track's own are passed to ``experiments.grug.fast_track.launch`` and override it.
"""

import sys
from enum import StrEnum

from experiments.grug.fast_track import launch

VERSION = "2026.09.25"


class Track(StrEnum):
    VANILLA = "vanilla"
    BASIC_MOE = "moe"


# Per rung: (attention heads, Vanilla steps at 20 tokens / active param, Basic MoE steps at 60 tokens / active param),
# with the rung's default batch (128 sequences for d512 / d768, 256 for d1024 / d1280) of 4,096 tokens.
RUNGS = {"d512": (4, 720, 1971), "d768": (6, 2161, 5547), "d1024": (8, 2881, 7334), "d1280": (10, 5251, 13053)}

VANILLA_MODEL_SETTINGS = (
    "global_every=1",
    "sconv=false",
    "sublayer_gated_norm=false",
    "embed_gated_norm=false",
    "final_gated_norm=false",
    "rope_global_layers=true",
    "rope_full_head=true",
    "attn_xsa=false",
    "attn_output_gate=false",
    "qk_mult=1.0",
)
BASIC_MOE_MODEL_SETTINGS = ("num_experts=128", "num_experts_per_token=4")


def track_args(track: Track, size: str) -> list[str]:
    """Launcher arguments for ``track`` at the H100 ladder rung ``size``."""
    if size not in RUNGS:
        raise ValueError(f"size must be one of {list(RUNGS)}, got {size!r}")
    heads, vanilla_steps, moe_steps = RUNGS[size]
    args = ["--size", size, "--recipe", "baseline", "--version", VERSION, "--save-checkpoints"]
    if track is Track.VANILLA:
        kv = heads // 2
        settings = (*VANILLA_MODEL_SETTINGS, f"num_kv_heads={kv}", f"local_kv_heads={kv}", f"global_kv_heads={kv}")
        args += ["--dense", "--optimizer", "adamw", "--num-steps", str(vanilla_steps)]
    else:
        settings = BASIC_MOE_MODEL_SETTINGS
        args += ["--num-steps", str(moe_steps)]
    for setting in settings:
        args += ["--model-set", setting]
    return args


def main(argv: list[str]) -> None:
    if "--track" not in argv or "--size" not in argv:
        raise SystemExit("usage: belay_tracks --track <vanilla|moe> --size <d512|d768|d1024|d1280> [launch args...]")
    rest = list(argv)
    values = {}
    for flag in ("--track", "--size"):
        i = rest.index(flag)
        values[flag] = rest[i + 1]
        del rest[i : i + 2]
    launch.main.main(args=[*track_args(Track(values["--track"]), values["--size"]), *rest])


if __name__ == "__main__":
    main(sys.argv[1:])
