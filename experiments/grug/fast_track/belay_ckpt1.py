# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Belay Ckpt 1: the Marin Belay recipe validated on one GB200 rack (marin#9451, 2026-10-14).

KDA local layers, MLA global layers with Inkling relative position, Block AttnRes, latent MoE with 384 ungated ReLU^2
experts (top-10, routed and shared width 0.75 d, one shared expert), gated value embeddings, MuonH + AdEMAMix.

The ladder runs every rung at 171 tokens per active parameter, the hero budget's ratio (300k GB200-hours at 26.9%
11-rack MFU = 4.51T tokens for 26.33B active params); steps are that token count over the rung's batch.

    uv run python -m experiments.grug.fast_track.belay_ckpt1 --size d512 --submit --run-id my-d512 \\
        --target-cluster cw-us-east-02a --priority production

``--size gb200-d6144`` gives the one-rack throughput configuration (8 steps by default; 14.59 s/step, 28.9% MFU).
Arguments after the recipe's own are passed to ``experiments.grug.fast_track.launch`` and override it.
"""

import sys

from experiments.grug.fast_track import launch

TOKENS_PER_ACTIVE_PARAM = 171
VERSION = "2026.09.25"

MODEL_SETTINGS = (
    "qk_mult=4.0",
    "learnable_qk_mult=true",
    "embed_gated_norm=false",
    "final_gated_norm=false",
    "logit_soft_cap=10",
    "mla_key_offset=true",
    "mla_k_norm=true",
    "mla_q_norm=true",
    "moe_ungated_relu2=true",
    "moe_ungated_kernel=true",
    "shared_ungated_relu2=true",
    "value_embeds=gated",
    "num_experts=384",
    "num_experts_per_token=10",
    "num_shared_experts=1",
    "init_std_mult_gates=2",
    "init_std_mult_beta=1",
    "router_combine=sqrt_softplus_renorm",
    "kda_chunk_size=32",
    "kda_onchip_decay_gate=true",
    "moe_implementation=ragged_all_to_all",
    "moe_drop_renorm=false",
    "capacity_factor=1.05",
)
OPTIMIZER_SETTINGS = (
    "muon_pre_norm=in",
    "muonh_decay_power=0.7",
    "adam_ademamix_alpha=5.0",
    "adam_ademamix_beta3=0.98",
    "lr_schedule=cosine",
    "router_lr_mult=0.5",
    "kda_beta_group=adam",
)

# H100 ladder rungs: (attention heads, steps at TOKENS_PER_ACTIVE_PARAM with the rung's default batch).
LADDER_RUNGS = {"d512": (4, 8821), "d768": (6, 24995), "d1024": (8, 32323), "d1280": (10, 57578)}
LADDER_ARGS = ("--single-process", "--ragged-transport", "one_shot", "--save-checkpoints")
LADDER_MODEL_SETTINGS = ("kda_state_grads=pallas",)

GB200_SIZE = "gb200-d6144"
# The XLA scheduler's memory slop must stay at 90: higher values grow the peak past what NCCL leaves free.
GB200_XLA_FLAGS = (
    "--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_command_buffer= "
    "--xla_gpu_memory_limit_slop_factor=90"
)
GB200_ARGS = (
    "--ragged-transport",
    "device",
    "--master-param-mode",
    "device",
    "--num-steps",
    "8",
    "--no-eval",
    "--max-retries",
    "3",
    "--job-env",
    f"XLA_FLAGS={GB200_XLA_FLAGS}",
    "--job-env",
    "XLA_PYTHON_CLIENT_MEM_FRACTION=0.78",
)
GB200_MODEL_SETTINGS = (
    "mla_num_heads=48",
    "intermediate_dim=4608",
    "shared_expert_intermediate_dim=4608",
    "moe_shared_placement=overlap",
    "scan_save_routed_output=true",
    "attn_res_fused_source_grads=true",
    "gpu_attention=gpu_pallas_fa4",
    "kda_state_block_v=64",
    "kda_state_grads=xla",
    "scan_layers=true",
    "scan_offload_partials=true",
    "value_embed_fsdp=true",
    "fp8_recipe=mxfp8",
    "moe_mxfp8_dispatch=true",
    "mxfp8_fused_grad_quant=true",
)


def recipe_args(size: str) -> list[str]:
    """Launcher arguments for Belay Ckpt 1 at ``size`` (an H100 ladder rung or ``gb200-d6144``)."""
    if size == GB200_SIZE:
        extra_args, extra_settings = GB200_ARGS, GB200_MODEL_SETTINGS
    elif size in LADDER_RUNGS:
        heads, steps = LADDER_RUNGS[size]
        width = 3 * int(size.removeprefix("d")) // 4
        extra_args = (*LADDER_ARGS, "--num-steps", str(steps))
        extra_settings = (
            *LADDER_MODEL_SETTINGS,
            f"mla_num_heads={heads}",
            f"intermediate_dim={width}",
            f"shared_expert_intermediate_dim={width}",
        )
    else:
        raise ValueError(f"size must be one of {[*LADDER_RUNGS, GB200_SIZE]}, got {size!r}")
    args = ["--size", size, "--recipe", "kma", "--version", VERSION, *extra_args]
    for setting in (*MODEL_SETTINGS, *extra_settings):
        args += ["--model-set", setting]
    for setting in OPTIMIZER_SETTINGS:
        args += ["--opt-set", setting]
    return args


def main(argv: list[str]) -> None:
    if "--size" not in argv:
        raise SystemExit("usage: belay_ckpt1 --size <d512|d768|d1024|d1280|gb200-d6144> [launch args...]")
    i = argv.index("--size")
    size, rest = argv[i + 1], argv[:i] + argv[i + 2 :]
    args = [*recipe_args(size), *rest]
    # --submit re-runs the launcher on the cluster from sys.argv, so it must hold the expanded launcher arguments.
    sys.argv = [sys.argv[0], *args]
    launch.main.main(args=args)


if __name__ == "__main__":
    main(sys.argv[1:])
