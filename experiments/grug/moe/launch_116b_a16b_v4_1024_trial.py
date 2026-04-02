# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Short Grug MoE bring-up near 116B-A16B on v4-1024."""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from dataclasses import dataclass, field

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.launch import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION, _resolve_run_id, _resolve_tracker
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugRunConfig, GrugTrainerConfig, run_grug

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
    use_array_stacked_blocks=True,
)
_MOE_IMPL_CHOICES = ("auto", "flip", "ring", "ragged_all_to_all")


@dataclass(frozen=True)
class GrugMoe116bTrialConfig:
    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str
    expert_axis_size: int
    profiler_start_step: int
    profiler_num_steps: int
    watch_interval: int
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)


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


def _parse_expert_axis_sizes(expert_axis_sizes: str, fallback: int) -> list[int]:
    if not expert_axis_sizes.strip():
        return [fallback]
    parsed: list[int] = []
    for token in expert_axis_sizes.split(","):
        token = token.strip()
        if not token:
            continue
        parsed.append(int(token))
    if not parsed:
        raise ValueError("--expert-axis-sizes must include at least one integer")
    return list(dict.fromkeys(parsed))


def run_grug_moe_116b_trial(config: GrugMoe116bTrialConfig) -> None:
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=ProfilerConfig(
            enabled=True,
            start_step=config.profiler_start_step,
            num_steps=config.profiler_num_steps,
            perfetto_link=False,
        ),
        watch=WatchConfig(interval=config.watch_interval),
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1, "expert": config.expert_axis_size}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            append_run_id_to_base_path=False,
            save_interval=None,
            keep=[],
        ),
    )

    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=dataclasses.replace(config.grug_trainer, trainer=trainer),
        eval=None,
    )
    run_grug(run_config)


def _build_run_stem(
    *,
    model: GrugModelConfig,
    batch_size: int,
    expert_axis_size: int,
    moe_implementation: str,
    run_suffix: str,
) -> str:
    stem = (
        "grug-moe-116b-a16b-v4-1024"
        f"-h{model.hidden_dim}"
        f"-l{model.num_layers}"
        f"-e{model.num_experts}"
        f"-k{model.num_experts_per_token}"
        f"-ix{model.intermediate_dim}"
        f"-sx{model.shared_expert_intermediate_dim}"
        f"-bs{batch_size}"
        f"-ep{expert_axis_size}"
        f"-mi{moe_implementation}"
    )
    if run_suffix:
        stem = f"{stem}-{run_suffix}"
    return stem


def _append_suffix(base: str, suffix: str) -> str:
    return f"{base}-{suffix}" if base else suffix


def build_step(
    *,
    batch_size: int,
    expert_axis_size: int,
    moe_implementation_mode: str,
    moe_capacity_factor: float,
    steps: int,
    profiler_start_step: int,
    profiler_num_steps: int,
    watch_interval: int,
    hidden_dim: int,
    dense_intermediate_dim: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    initializer_std: float,
    intermediate_dim: int,
    shared_expert_intermediate_dim: int,
    save_moe_activations: bool,
    save_moe_layer_inputs: bool,
    offload_moe_activations: bool,
    offload_moe_layer_inputs: bool,
    offload_opt_state: bool,
    opt_state_memory_kind: str,
    offload_src_memory_kind: str,
    offload_dst_memory_kind: str,
    run_suffix: str,
) -> ExecutorStep:
    resolved_moe_implementation = _resolve_moe_implementation(moe_implementation_mode, _BASE_MODEL)
    model = dataclasses.replace(
        _BASE_MODEL,
        hidden_dim=hidden_dim,
        dense_intermediate_dim=dense_intermediate_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        initializer_std=initializer_std,
        intermediate_dim=intermediate_dim,
        shared_expert_intermediate_dim=shared_expert_intermediate_dim,
        moe_implementation=resolved_moe_implementation,
        moe_capacity_factor=moe_capacity_factor,
        save_moe_activations=save_moe_activations,
        save_moe_layer_inputs=save_moe_layer_inputs,
        offload_moe_activations=offload_moe_activations,
        offload_moe_layer_inputs=offload_moe_layer_inputs,
        offload_src_memory_kind=offload_src_memory_kind,
        offload_dst_memory_kind=offload_dst_memory_kind,
    )
    moe_implementation = _effective_moe_implementation(model)

    resolved_run_suffix = run_suffix
    if model.save_moe_activations:
        resolved_run_suffix = _append_suffix(resolved_run_suffix, "savemoe")
    if model.save_moe_layer_inputs:
        resolved_run_suffix = _append_suffix(resolved_run_suffix, "savelayerin")
    if model.offload_moe_activations:
        resolved_run_suffix = _append_suffix(resolved_run_suffix, "offloadmoe")
    if model.offload_moe_layer_inputs:
        resolved_run_suffix = _append_suffix(resolved_run_suffix, "offloadlayerin")
    if offload_opt_state:
        resolved_run_suffix = _append_suffix(resolved_run_suffix, "offloadopt")

    if model.num_experts % expert_axis_size != 0:
        raise ValueError(f"num_experts={model.num_experts} must be divisible by expert_axis_size={expert_axis_size}")

    run_stem = _build_run_stem(
        model=model,
        batch_size=batch_size,
        expert_axis_size=expert_axis_size,
        moe_implementation=moe_implementation,
        run_suffix=resolved_run_suffix,
    )
    run_id = _resolve_run_id(run_stem)

    return ExecutorStep(
        name=f"grug/{run_stem}",
        fn=run_grug_moe_116b_trial,
        config=GrugMoe116bTrialConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v4-1024")),
            steps=versioned(steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            expert_axis_size=versioned(expert_axis_size),
            profiler_start_step=versioned(profiler_start_step),
            profiler_num_steps=versioned(profiler_num_steps),
            watch_interval=versioned(watch_interval),
            tracker=WandbConfig(
                project="marin",
                tags=[
                    "grug",
                    "moe",
                    "v4-1024",
                    "116b-a16b",
                    "target-1t-tokens",
                    f"ix{model.intermediate_dim}",
                    f"sx{model.shared_expert_intermediate_dim}",
                    f"bs{batch_size}",
                    f"ep{expert_axis_size}",
                    f"moeimpl-{moe_implementation}",
                    "agent-generated",
                ]
                + (["arraystacked"] if model.use_array_stacked_blocks else [])
                + (["save-moe-activations"] if model.save_moe_activations else [])
                + (["save-moe-layer-inputs"] if model.save_moe_layer_inputs else [])
                + (["offload-moe-activations"] if model.offload_moe_activations else [])
                + (["offload-moe-layer-inputs"] if model.offload_moe_layer_inputs else [])
                + (["offload-opt-state"] if offload_opt_state else []),
                group=(
                    "grug-moe-116b-a16b-v4-1024"
                    f"-e{model.num_experts}-k{model.num_experts_per_token}"
                    f"-ix{model.intermediate_dim}-sx{model.shared_expert_intermediate_dim}"
                    f"-bs{batch_size}-ep{expert_axis_size}"
                    f"-mi{moe_implementation}"
                ),
                name=None,
            ),
            optimizer=versioned(
                AdamConfig(
                    learning_rate=1e-4,
                    weight_decay=0.1,
                    lr_schedule="constant",
                    warmup=0,
                )
            ),
            grug_trainer=versioned(
                GrugTrainerConfig(
                    z_loss_weight=1e-4,
                    ema_beta=None,
                    log_every=1,
                    opt_state_memory_kind=(opt_state_memory_kind if offload_opt_state else None),
                )
            ),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--expert-axis-size", type=int, default=32)
    parser.add_argument(
        "--expert-axis-sizes",
        type=str,
        default="",
        help="Comma-separated EP axis sizes to sweep. Overrides --expert-axis-size when set.",
    )
    parser.add_argument("--moe-implementation", type=str, choices=_MOE_IMPL_CHOICES, default="ragged_all_to_all")
    parser.add_argument("--moe-capacity-factor", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--profiler-start-step", type=int, default=9)
    parser.add_argument("--profiler-num-steps", type=int, default=3)
    parser.add_argument(
        "--watch-interval",
        type=int,
        default=0,
        help="Training watch-stats interval; 0 disables watch-stats to avoid extra JVP compile pressure.",
    )
    parser.add_argument("--hidden-dim", type=int, default=_BASE_MODEL.hidden_dim)
    parser.add_argument("--dense-intermediate-dim", type=int, default=_BASE_MODEL.dense_intermediate_dim)
    parser.add_argument("--num-layers", type=int, default=_BASE_MODEL.num_layers)
    parser.add_argument("--num-heads", type=int, default=_BASE_MODEL.num_heads)
    parser.add_argument("--num-kv-heads", type=int, default=_BASE_MODEL.num_kv_heads)
    parser.add_argument(
        "--initializer-std",
        type=float,
        default=None,
        help="If unset, defaults to 0.5/sqrt(hidden_dim).",
    )
    parser.add_argument("--intermediate-dim", type=int, default=_BASE_MODEL.intermediate_dim)
    parser.add_argument("--shared-expert-intermediate-dim", type=int, default=_BASE_MODEL.shared_expert_intermediate_dim)
    parser.add_argument(
        "--save-moe-activations",
        action=argparse.BooleanOptionalAction,
        default=_BASE_MODEL.save_moe_activations,
        help="Save named MoE remat activations (no offload) to avoid recomputing dispatch/experts.",
    )
    parser.add_argument(
        "--save-moe-layer-inputs",
        action=argparse.BooleanOptionalAction,
        default=_BASE_MODEL.save_moe_layer_inputs,
        help="Save per-block layer inputs under remat (no offload).",
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
        "--offload-opt-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Move optimizer state to host memory between steps.",
    )
    parser.add_argument(
        "--opt-state-memory-kind",
        type=str,
        default="pinned_host",
        help="Target memory kind for optimizer-state offload.",
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
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


def main() -> None:
    args = parse_args()
    initializer_std = args.initializer_std if args.initializer_std is not None else 0.5 / (args.hidden_dim**0.5)
    expert_axis_sizes = _parse_expert_axis_sizes(args.expert_axis_sizes, args.expert_axis_size)
    steps = [
        build_step(
            batch_size=args.batch_size,
            expert_axis_size=ep_size,
            moe_implementation_mode=args.moe_implementation,
            moe_capacity_factor=args.moe_capacity_factor,
            steps=args.steps,
            profiler_start_step=args.profiler_start_step,
            profiler_num_steps=args.profiler_num_steps,
            watch_interval=args.watch_interval,
            hidden_dim=args.hidden_dim,
            dense_intermediate_dim=args.dense_intermediate_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            initializer_std=initializer_std,
            intermediate_dim=args.intermediate_dim,
            shared_expert_intermediate_dim=args.shared_expert_intermediate_dim,
            save_moe_activations=args.save_moe_activations,
            save_moe_layer_inputs=args.save_moe_layer_inputs,
            offload_moe_activations=args.offload_moe_activations,
            offload_moe_layer_inputs=args.offload_moe_layer_inputs,
            offload_opt_state=args.offload_opt_state,
            opt_state_memory_kind=args.opt_state_memory_kind,
            offload_src_memory_kind=args.offload_src_memory_kind,
            offload_dst_memory_kind=args.offload_dst_memory_kind,
            run_suffix=args.run_suffix,
        )
        for ep_size in expert_axis_sizes
    ]

    executor_main(
        steps=steps,
        description=(
            "Short Grug MoE bring-up near 116B-A16B on v4-1024 "
            f"(h{args.hidden_dim}/l{args.num_layers}/heads{args.num_heads}/kv{args.num_kv_heads}, "
            f"ix={args.intermediate_dim}, sx={args.shared_expert_intermediate_dim}, "
            f"watch_interval={args.watch_interval}, "
            f"save_moe_activations={args.save_moe_activations}, "
            f"save_moe_layer_inputs={args.save_moe_layer_inputs}, "
            f"offload_moe_activations={args.offload_moe_activations}, "
            f"offload_moe_layer_inputs={args.offload_moe_layer_inputs}, "
            f"offload_opt_state={args.offload_opt_state}, "
            f"cf={args.moe_capacity_factor}, bs={args.batch_size}, ep={expert_axis_sizes}, "
            f"moe_impl_mode={args.moe_implementation}; "
            "long-run target ~1T tokens)."
        ),
    )


if __name__ == "__main__":
    main()
