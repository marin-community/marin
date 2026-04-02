# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct Grug MoE bring-up near 116B-A16B on v4-1024."""

from __future__ import annotations

import argparse
import dataclasses
import os

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.grug.moe.launch import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugRunConfig, GrugTrainerConfig, _run_grug_local

_BASE_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=4864,
    intermediate_dim=2560,
    shared_expert_intermediate_dim=8064,
    dense_intermediate_dim=9728,
    num_experts=64,
    num_experts_per_token=4,
    num_layers=47,
    num_heads=38,
    num_kv_heads=2,
    head_dim=None,
    max_seq_len=4096,
    initializer_std=0.5 / (4864**0.5),
    qk_mult=1.3,
)
_MOE_IMPL_CHOICES = ("auto", "flip", "ring", "ragged_all_to_all")


def _resolve_run_id(default_run_id: str) -> str:
    return os.environ.get("GRUG_RUN_ID", default_run_id)


def _resolve_output_path(run_id: str) -> str:
    return os.environ.get("GRUG_OUTPUT_PATH", f"gs://marin-us-central2/grug/{run_id}")


def _resolve_moe_implementation(mode: str, base_model: GrugModelConfig) -> str | None:
    current = base_model.moe_implementation or "ring"
    if mode == "auto":
        return base_model.moe_implementation
    if mode == "flip":
        return "ragged_all_to_all" if current != "ragged_all_to_all" else "ring"
    if mode in ("ring", "ragged_all_to_all"):
        return mode
    raise ValueError(f"Unsupported moe implementation mode: {mode!r}")


def _effective_moe_implementation(model: GrugModelConfig) -> str:
    return model.moe_implementation or "ring"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--expert-axis-size", type=int, default=32)
    parser.add_argument("--moe-implementation", type=str, choices=_MOE_IMPL_CHOICES, default="ragged_all_to_all")
    parser.add_argument("--moe-capacity-factor", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--profiler-start-step", type=int, default=9)
    parser.add_argument("--profiler-num-steps", type=int, default=3)
    parser.add_argument("--intermediate-dim", type=int, default=_BASE_MODEL.intermediate_dim)
    parser.add_argument(
        "--shared-expert-intermediate-dim",
        type=int,
        default=_BASE_MODEL.shared_expert_intermediate_dim,
    )
    parser.add_argument(
        "--offload-moe-activations",
        action=argparse.BooleanOptionalAction,
        default=_BASE_MODEL.offload_moe_activations,
        help="Offload named MoE remat activations to host memory.",
    )
    parser.add_argument(
        "--offload-moe-layer-inputs",
        action=argparse.BooleanOptionalAction,
        default=_BASE_MODEL.offload_moe_layer_inputs,
        help="Offload per-block layer inputs under remat.",
    )
    parser.add_argument(
        "--offload-src-memory-kind",
        type=str,
        default=_BASE_MODEL.offload_src_memory_kind,
        help="Source memory kind used by JAX remat offload policy.",
    )
    parser.add_argument(
        "--offload-dst-memory-kind",
        type=str,
        default=_BASE_MODEL.offload_dst_memory_kind,
        help="Destination memory kind used by JAX remat offload policy.",
    )
    parser.add_argument("--run-suffix", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = dataclasses.replace(
        _BASE_MODEL,
        intermediate_dim=args.intermediate_dim,
        shared_expert_intermediate_dim=args.shared_expert_intermediate_dim,
        moe_implementation=_resolve_moe_implementation(args.moe_implementation, _BASE_MODEL),
        moe_capacity_factor=args.moe_capacity_factor,
        offload_moe_activations=args.offload_moe_activations,
        offload_moe_layer_inputs=args.offload_moe_layer_inputs,
        offload_src_memory_kind=args.offload_src_memory_kind,
        offload_dst_memory_kind=args.offload_dst_memory_kind,
    )
    moe_implementation = _effective_moe_implementation(model)

    if model.num_experts % args.expert_axis_size != 0:
        raise ValueError(
            f"num_experts={model.num_experts} must be divisible by expert_axis_size={args.expert_axis_size}"
        )

    run_stem = (
        "grug-moe-116b-a16b-v4-1024"
        f"-h{model.hidden_dim}"
        f"-l{model.num_layers}"
        f"-e{model.num_experts}"
        f"-k{model.num_experts_per_token}"
        f"-ix{model.intermediate_dim}"
        f"-sx{model.shared_expert_intermediate_dim}"
        f"-bs{args.batch_size}"
        f"-ep{args.expert_axis_size}"
        f"-mi{moe_implementation}"
    )
    if args.run_suffix:
        run_stem = f"{run_stem}-{args.run_suffix}"
    if model.offload_moe_activations:
        run_stem = f"{run_stem}-offloadmoe"
    if model.offload_moe_layer_inputs:
        run_stem = f"{run_stem}-offloadlayerin"

    run_id = _resolve_run_id(run_stem)
    output_path = _resolve_output_path(run_id)

    trainer = TrainerConfig(
        id=run_id,
        seed=0,
        train_batch_size=args.batch_size,
        num_train_steps=args.steps,
        profiler=ProfilerConfig(
            enabled=True,
            start_step=args.profiler_start_step,
            num_steps=args.profiler_num_steps,
            perfetto_link=False,
        ),
        mp=jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=(
                [
                    "grug",
                    "moe",
                    "v4-1024",
                    "116b-a16b",
                    "target-1t-tokens",
                    f"ix{model.intermediate_dim}",
                    f"sx{model.shared_expert_intermediate_dim}",
                    f"bs{args.batch_size}",
                    f"ep{args.expert_axis_size}",
                    f"moeimpl-{moe_implementation}",
                    f"cf{model.moe_capacity_factor}",
                    "agent-generated",
                    "direct",
                ]
                + (["offload-moe-activations"] if model.offload_moe_activations else [])
                + (["offload-moe-layer-inputs"] if model.offload_moe_layer_inputs else [])
            ),
            group=(
                "grug-moe-116b-a16b-v4-1024"
                f"-e{model.num_experts}-k{model.num_experts_per_token}"
                f"-ix{model.intermediate_dim}-sx{model.shared_expert_intermediate_dim}"
                f"-bs{args.batch_size}-ep{args.expert_axis_size}"
                f"-mi{moe_implementation}"
            ),
            name=run_id,
        ),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1, "expert": args.expert_axis_size}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(output_path, "checkpoints"),
            append_run_id_to_base_path=False,
            save_interval=None,
            keep=[],
        ),
    )

    run_config = GrugRunConfig(
        model=model,
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        resources=ResourceConfig.with_tpu("v4-1024"),
        optimizer=AdamConfig(
            learning_rate=1e-4,
            weight_decay=0.1,
            lr_schedule="constant",
            warmup=0,
        ),
        trainer=GrugTrainerConfig(
            trainer=trainer,
            z_loss_weight=1e-4,
            ema_beta=None,
            log_every=1,
        ),
        eval=None,
    )
    _run_grug_local(run_config)


if __name__ == "__main__":
    main()
