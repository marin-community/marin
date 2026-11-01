# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The S7 ladder: the final stage of the Hero -> Belay MFU ladder, run at the H100 ladder rungs (bf16).

The baseline recipe's ladder model (384 experts, top-8, two shared experts, latent MoE at half width, qk_mult 1.3,
SConv) with KDA local layers, MLA global layers (512-dim KV latent, Q/K norms, no QK-norm) with the Inkling
relative-position bias, Block AttnRes over 8 blocks, and ungated ReLU^2 routed and shared experts of width 0.75 d.
Every rung trains 207 tokens per active parameter (5.456T tokens over the S7 hero's 26.333B active params).

Two ladders share the model:

- ``s7l``: the document-shuffled v16384 cache (pinned) and the default (linear) LR schedule (``s7l-*`` runs,
  marin_belay_v1_fp8quality @ 6a3a5fe57a).
- ``s7n``: the sequence-shuffled v16384 cache (the launcher default), a cosine LR schedule and MuonH decay power 0.7
  (``s7n-*`` runs, marin_belay_v1_fp8quality @ a5a8647840).

Rungs: ``d512``, ``d768``, ``d1024`` (one 8xH100 node each) and ``d1280L16`` (16 layers, batch 1024, four 8xH100
nodes with experts sharded within each node and data parallel across nodes).

    uv run python -m experiments.grug.fast_track.belay_s7 --ladder s7n --size d1280L16 --submit \\
        --run-id my-d1280 --target-cluster cw-rno2a --priority batch \\
        --job-env MARIN_PREFIX=s3://marin-us-east-02a/marin --job-env MARIN_TEMP_PREFIX=s3://marin-us-east-02a

Arguments after the ladder's own are passed to ``experiments.grug.fast_track.launch`` and override it.
"""

import sys
from enum import StrEnum

from experiments.grug.fast_track import launch

VERSION = "2026.10.20"


class Ladder(StrEnum):
    S7L = "s7l"
    S7N = "s7n"


MODEL_SETTINGS = (
    "moe_implementation=ragged_all_to_all",
    "gpu_attention=gpu_pallas_fa4",
    "sconv_impl=triton_gpu",
    "moe_ungated_relu2=true",
    "moe_ungated_kernel=true",
    "shared_ungated_relu2=true",
    "attn_res=true",
    "attn_res_num_blocks=8",
    "attn_res_plain_sum=false",
    "local_mixer=kda",
    "kda_chunk_size=32",
    "kda_onchip_decay_gate=true",
    "kda_state_block_v=64",
    "kda_state_grads=pallas",
    "mla=true",
    "inkling_relpos=true",
    "mla_kv_latent_dim=512",
    "qk_norm=false",
    "mla_q_norm=true",
    "mla_k_norm=true",
    "rel_extent=512",
)
LADDER_ARGS = (
    "--single-process",
    "--ragged-transport",
    "one_shot",
    "--max-retries",
    "3",
    "--temporary-checkpoints-only",
)
# s7l predates the sequence-shuffled default, so it pins the document-shuffled cache.
S7L_ARGS = ("--train-cache", launch.V16384_CACHE_DIR)
S7N_ARGS = (
    "--opt-set",
    "lr_schedule=cosine",
    "--opt-set",
    "muonh_decay_power=0.7",
)


# Rung -> (launcher size, steps at 207 tokens per active param, extra launcher args).
RUNGS: dict[str, tuple[str, int, tuple[str, ...]]] = {
    "d512": ("d512", 10767, ()),
    "d768": ("d768", 30389, ()),
    "d1024": ("d1024", 39210, ()),
    "d1280L16": (
        "d1280",
        20023,
        (
            "--nodes",
            "4",
            "--expert-axis-size",
            "8",
            "--batch-size",
            "1024",
            "--loader-buffer-batches",
            "32",
            "--model-set",
            "num_layers=16",
        ),
    ),
}


def recipe_args(ladder: Ladder, rung: str) -> list[str]:
    """Launcher arguments for the S7 ``ladder`` at ``rung``."""
    if rung not in RUNGS:
        raise ValueError(f"rung must be one of {list(RUNGS)}, got {rung!r}")
    size, steps, rung_args = RUNGS[rung]
    hidden = int(size.removeprefix("d"))
    width = 3 * hidden // 4
    args = ["--size", size, "--recipe", "baseline", "--version", VERSION, *LADDER_ARGS, "--num-steps", str(steps)]
    args += rung_args
    settings = (
        *MODEL_SETTINGS,
        f"intermediate_dim={width}",
        f"shared_expert_intermediate_dim={width}",
        f"mla_num_heads={hidden // 128}",
    )
    for setting in settings:
        args += ["--model-set", setting]
    args += S7N_ARGS if ladder is Ladder.S7N else S7L_ARGS
    return args


def _pop_option(argv: list[str], name: str) -> tuple[str, list[str]]:
    if name not in argv:
        raise SystemExit(f"usage: belay_s7 --ladder <s7l|s7n> --size <{'|'.join(RUNGS)}> [launch args...]")
    i = argv.index(name)
    return argv[i + 1], argv[:i] + argv[i + 2 :]


def main(argv: list[str]) -> None:
    ladder, argv = _pop_option(argv, "--ladder")
    rung, argv = _pop_option(argv, "--size")
    args = [*recipe_args(Ladder(ladder), rung), *argv]
    # --submit re-runs the launcher on the cluster from sys.argv, so it must hold the expanded launcher arguments.
    sys.argv = [sys.argv[0], *args]
    launch.main.main(args=args)


if __name__ == "__main__":
    main(sys.argv[1:])
