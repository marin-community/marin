# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paired d512 boundary-operator arms on the candidate-9 fast_track recipe.

The control matches the finished ``lc1-b102-stack4`` runs in #9451. The only
model difference between an arm and its control is ``boundary_alpha``. Runs
are pinned to the same 16k BPE data, Paloma eval, batch, steps, and H100 region.
"""

import argparse
import os

MODEL_SETTINGS = (
    "intermediate_dim=384",
    "shared_expert_intermediate_dim=768",
    "num_shared_experts=1",
    "num_experts=512",
    "qk_mult=4.0",
    "value_embeds=gated",
    "router_combine=sqrt_softplus_renorm",
    "embed_gated_norm=false",
    "final_gated_norm=false",
    "mla_k_norm=true",
    "mla_q_norm=true",
    "mla_key_offset=true",
    "logit_soft_cap=10",
    "init_std_mult_gates=2",
    "kda_write_gate=true",
    "kda_erase_gate=true",
    "shared_ungated_relu2=true",
    "moe_ungated_kernel=true",
    "moe_ungated_relu2=true",
    "moe_drop_renorm=true",
    "learnable_qk_mult=true",
    "second_embed=true",
    "embed2_rows=524288",
    "bigram_gate=true",
    "bigram_gate_rank=16",
    "embed2_fsdp=true",
    "second_embed_bigram=true",
)

OPTIMIZER_SETTINGS = (
    "lr_schedule=cosine",
    "beta1=0.8",
    "beta2=0.98",
    "embed2_lr_mult=10",
    "muon_pre_norm=in",
    "muonh_decay_power=0.7",
    "muon_bimaxwell=true",
    "adam_ademamix_alpha=5",
    "adam_ademamix_beta3=0.98",
    "router_lr_mult=0.5",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--jax-port", type=int, default=None)
    args = parser.parse_args()
    if args.submit and args.jax_port is None:
        parser.error("--jax-port is required with --submit so concurrent gangs use distinct ports")

    command = [
        "uv",
        "run",
        "fast-track",
        "--run-id",
        args.run_id,
        "--size",
        "d512",
        "--recipe",
        "kma",
        "--batch-size",
        "128",
        "--num-steps",
        "2817",
        "--seed",
        str(args.seed),
        "--version",
        "2026.09.29",
        "--ema-beta",
        "0.995",
        "--ema-last-steps",
        "1000",
        "--ema-blend",
        "0",
        "--ema-blend",
        "0.5",
        "--ema-blend",
        "0.75",
        "--target-cluster",
        "cw-us-east-02a",
    ]
    for setting in MODEL_SETTINGS:
        command.extend(("--model-set", setting))
    if args.alpha is not None:
        command.extend(("--model-set", f"boundary_alpha={args.alpha}"))
    for setting in OPTIMIZER_SETTINGS:
        command.extend(("--opt-set", setting))
    if args.submit:
        command.extend(("--job-env", f"IRIS_PORT_JAX={args.jax_port}", "--submit"))
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
