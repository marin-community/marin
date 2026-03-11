# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Short Iris-friendly Grug MoE EP sweep with per-run profiling on v5p-64."""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
import jax.numpy as jnp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.kernels.pallas.fused_cross_entropy_loss import BlockSizes
from levanter.kernels.pallas.fused_cross_entropy_loss.tuned_block_sizes import infer_block_sizes_with_tuned_match
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.data import qwen3_moe_perf_mix, qwen3_moe_perf_mix_block_shuffle
from experiments.grug.moe.launch import _resolve_run_id, _resolve_tracker
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug

QWEN3_32B_A4B_V5P_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=2048,
    intermediate_dim=768,
    shared_expert_intermediate_dim=2048,
    num_experts=128,
    num_experts_per_token=8,
    num_layers=48,
    num_heads=32,
    num_kv_heads=4,
    head_dim=128,
    max_seq_len=4096,
    layer_norm_eps=1e-6,
    router_z_loss_coef=0.001,
)


def _short_remat_slug(block_remat: str) -> str:
    return {
        "full": "rfull",
        "off": "roff",
        "save_moe_input": "rmei",
        "save_moe_hidden": "rmeh",
        "save_moe_output": "rmeo",
        "save_moe_inputs_outputs": "rmeio",
        "save_moe": "rme",
        "save_mlp_inputs": "rsmi",
        "save_mlp_outputs": "rsmo",
        "save_mlp": "rsm",
        "offload_moe_input": "romei",
        "offload_moe_hidden": "romeh",
        "offload_moe_output": "romeo",
        "offload_moe_inputs_outputs": "romeio",
        "offload_moe": "rome",
        "offload_mlp_inputs": "romi",
        "offload_mlp_outputs": "romo",
        "offload_mlp": "rom",
    }[block_remat]


def _cross_entropy_impls(spec: str) -> tuple[str, ...] | None:
    if spec == "auto":
        return None
    return tuple(part.strip() for part in spec.split(",") if part.strip())


def _cross_entropy_slug(spec: str) -> str:
    return {
        "auto": "",
        "xla": "-cex",
        "reference": "-cer",
        "pallas_tpu": "-cept",
        "pallas_tpu,xla": "-ceptx",
    }[spec]


def _cross_entropy_block_slug(block_sizes: BlockSizes | None) -> str:
    if block_sizes is None:
        return ""
    return f"-v{block_sizes.v_block_size}"


def _compact_loader_slug(*, loader_prefetch_size: int, loader_max_buffered_batches: int) -> str:
    if loader_prefetch_size == 32 and loader_max_buffered_batches == 64:
        return ""
    return f"-p{loader_prefetch_size}-q{loader_max_buffered_batches}"


def _resolve_cross_entropy_block_sizes(
    *,
    model: GrugModelConfig,
    batch_size: int,
    cross_entropy_implementation: str,
    cross_entropy_v_block_divisor: int,
) -> BlockSizes | None:
    if cross_entropy_v_block_divisor == 1 or "pallas_tpu" not in cross_entropy_implementation:
        return None
    if cross_entropy_v_block_divisor <= 0:
        raise ValueError("cross_entropy_v_block_divisor must be positive")

    local_ce_batch = (batch_size * model.max_seq_len) // 32
    inferred, _ = infer_block_sizes_with_tuned_match(
        local_ce_batch,
        model.hidden_dim,
        model.vocab_size,
        dtype=jnp.float32,
    )
    reduced_v = max(128, inferred.v_block_size // cross_entropy_v_block_divisor)
    reduced_v = max(128, (reduced_v // 128) * 128)
    return BlockSizes(
        b_block_size=inferred.b_block_size,
        h_block_size=inferred.h_block_size,
        v_block_size=reduced_v,
    )


@dataclass(frozen=True)
class GrugMoeV5pEpProfileConfig:
    """Last-mile config for short v5p-64 EP profile runs."""

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
    block_shuffle: bool
    synthetic_data: bool
    loader_prefetch_size: int
    loader_max_buffered_batches: int | None
    profiler_start_step: int
    profiler_num_steps: int
    cross_entropy_implementation: tuple[str, ...] | None
    cross_entropy_block_sizes: BlockSizes | None
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = None


def run_grug_moe_v5p_ep_profile(config: GrugMoeV5pEpProfileConfig) -> None:
    """Map EP sweep knobs onto the standard grug train loop."""

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
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1, "expert": config.expert_axis_size}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=30),
            keep=[{"every": 1000}],
        ),
    )

    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer)
    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug(run_config)


def build_step(
    *,
    expert_axis_size: int,
    batch_size: int,
    capacity_factor: float,
    block_remat: str,
    report_capacity_overflow: bool,
    num_experts: int,
    num_experts_per_token: int,
    intermediate_dim: int | None,
    match_activated_params: bool,
    shared_expert_intermediate_dim: int,
    match_total_active_flops: bool,
    block_shuffle: bool,
    synthetic_data: bool,
    loader_prefetch_size: int,
    loader_max_buffered_batches: int,
    steps: int,
    profiler_start_step: int,
    profiler_num_steps: int,
    cross_entropy_implementation: str,
    cross_entropy_v_block_divisor: int,
    hidden_dim: int | None = None,
    num_layers: int | None = None,
    num_heads: int | None = None,
    num_kv_heads: int | None = None,
    run_suffix: str = "",
) -> ExecutorStep:
    """Build one EP profile step with fixed workload and profiler window."""

    suffix = f"-{run_suffix}" if run_suffix else ""
    cf_slug = str(capacity_factor).replace(".", "p")
    resolved_hidden_dim = hidden_dim or QWEN3_32B_A4B_V5P_MODEL.hidden_dim
    resolved_num_layers = num_layers or QWEN3_32B_A4B_V5P_MODEL.num_layers
    resolved_num_heads = num_heads or QWEN3_32B_A4B_V5P_MODEL.num_heads
    resolved_num_kv_heads = num_kv_heads or QWEN3_32B_A4B_V5P_MODEL.num_kv_heads

    expert_intermediate_dim = intermediate_dim or QWEN3_32B_A4B_V5P_MODEL.intermediate_dim
    if match_activated_params and intermediate_dim is None:
        base_topk = QWEN3_32B_A4B_V5P_MODEL.num_experts_per_token
        if base_topk % num_experts_per_token != 0:
            raise ValueError("match_activated_params requires num_experts_per_token to divide the baseline top-k")
        expert_intermediate_dim = QWEN3_32B_A4B_V5P_MODEL.intermediate_dim * base_topk // num_experts_per_token
    baseline_active_dim = (
        QWEN3_32B_A4B_V5P_MODEL.shared_expert_intermediate_dim + num_experts_per_token * expert_intermediate_dim
    )
    if match_total_active_flops:
        remaining_routed_dim = baseline_active_dim - shared_expert_intermediate_dim
        if remaining_routed_dim <= 0:
            raise ValueError("shared_expert_intermediate_dim must be smaller than the matched active FFN budget")
        if remaining_routed_dim % num_experts_per_token != 0:
            raise ValueError(
                "match_total_active_flops requires "
                "(shared_baseline + topk * routed_baseline - shared_expert_intermediate_dim) "
                "to be divisible by num_experts_per_token"
            )
        expert_intermediate_dim = remaining_routed_dim // num_experts_per_token
    remat_slug = f"-{_short_remat_slug(block_remat)}" if block_remat != QWEN3_32B_A4B_V5P_MODEL.block_remat else ""
    ce_slug = _cross_entropy_slug(cross_entropy_implementation)
    ce_block_sizes = _resolve_cross_entropy_block_sizes(
        model=dataclasses.replace(
            QWEN3_32B_A4B_V5P_MODEL,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            intermediate_dim=expert_intermediate_dim,
            shared_expert_intermediate_dim=shared_expert_intermediate_dim,
        ),
        batch_size=batch_size,
        cross_entropy_implementation=cross_entropy_implementation,
        cross_entropy_v_block_divisor=cross_entropy_v_block_divisor,
    )
    ce_block_slug = _cross_entropy_block_slug(ce_block_sizes)
    overflow_slug = "-ovf" if report_capacity_overflow else ""
    experts_slug = f"-e{num_experts}" if num_experts != QWEN3_32B_A4B_V5P_MODEL.num_experts else ""
    topk_slug = (
        f"-k{num_experts_per_token}" if num_experts_per_token != QWEN3_32B_A4B_V5P_MODEL.num_experts_per_token else ""
    )
    matched_slug = "-ma" if match_activated_params else ""
    intermediate_slug = (
        f"-ix{expert_intermediate_dim}" if expert_intermediate_dim != QWEN3_32B_A4B_V5P_MODEL.intermediate_dim else ""
    )
    shared_slug = (
        f"-sx{shared_expert_intermediate_dim}"
        if shared_expert_intermediate_dim != QWEN3_32B_A4B_V5P_MODEL.shared_expert_intermediate_dim
        else ""
    )
    hidden_slug = f"-h{resolved_hidden_dim}" if resolved_hidden_dim != QWEN3_32B_A4B_V5P_MODEL.hidden_dim else ""
    layers_slug = f"-l{resolved_num_layers}" if resolved_num_layers != QWEN3_32B_A4B_V5P_MODEL.num_layers else ""
    total_flops_slug = "-mf" if match_total_active_flops else ""
    shuffle_slug = "-blk" if block_shuffle else ""
    synthetic_slug = "-syn" if synthetic_data else ""
    loader_slug = _compact_loader_slug(
        loader_prefetch_size=loader_prefetch_size, loader_max_buffered_batches=loader_max_buffered_batches
    )
    run_stem = (
        f"gq32-v5p64-b{batch_size}-e{expert_axis_size}-c{cf_slug}"
        f"{experts_slug}{topk_slug}{matched_slug}{intermediate_slug}{shared_slug}{hidden_slug}{layers_slug}{total_flops_slug}"
        f"{remat_slug}{ce_slug}{overflow_slug}"
        f"{ce_block_slug}{loader_slug}{shuffle_slug}{synthetic_slug}-p{suffix}"
    )
    resolved_run_id = _resolve_run_id(run_stem)

    return ExecutorStep(
        name=(
            f"grug/gq32-v5p64-b{batch_size}-e{expert_axis_size}-c{cf_slug}"
            f"{experts_slug}{topk_slug}{matched_slug}{intermediate_slug}{shared_slug}{hidden_slug}{layers_slug}{total_flops_slug}"
            f"{remat_slug}{ce_slug}{overflow_slug}"
            f"{ce_block_slug}{loader_slug}{shuffle_slug}{synthetic_slug}-p"
        ),
        fn=run_grug_moe_v5p_ep_profile,
        config=GrugMoeV5pEpProfileConfig(
            model=versioned(
                dataclasses.replace(
                    QWEN3_32B_A4B_V5P_MODEL,
                    hidden_dim=resolved_hidden_dim,
                    capacity_factor=capacity_factor,
                    block_remat=block_remat,
                    report_capacity_overflow=report_capacity_overflow,
                    cross_entropy_implementation=_cross_entropy_impls(cross_entropy_implementation),
                    cross_entropy_block_sizes=ce_block_sizes,
                    num_experts=num_experts,
                    num_experts_per_token=num_experts_per_token,
                    intermediate_dim=expert_intermediate_dim,
                    shared_expert_intermediate_dim=shared_expert_intermediate_dim,
                    num_layers=resolved_num_layers,
                    num_heads=resolved_num_heads,
                    num_kv_heads=resolved_num_kv_heads,
                )
            ),
            data=qwen3_moe_perf_mix_block_shuffle() if block_shuffle else qwen3_moe_perf_mix(),
            output_path=this_output_path(),
            run_id=resolved_run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-64")),
            steps=versioned(steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            expert_axis_size=versioned(expert_axis_size),
            block_shuffle=versioned(block_shuffle),
            synthetic_data=versioned(synthetic_data),
            loader_prefetch_size=versioned(loader_prefetch_size),
            loader_max_buffered_batches=versioned(loader_max_buffered_batches),
            profiler_start_step=versioned(profiler_start_step),
            profiler_num_steps=versioned(profiler_num_steps),
            cross_entropy_implementation=versioned(_cross_entropy_impls(cross_entropy_implementation)),
            cross_entropy_block_sizes=versioned(ce_block_sizes),
            tracker=WandbConfig(
                project="marin",
                tags=[
                    "grug",
                    "moe",
                    "perf",
                    "qwen3-shape",
                    "v5p-64",
                    f"bs{batch_size}",
                    f"ep{expert_axis_size}",
                    f"cf{cf_slug}",
                    f"experts{num_experts}",
                    f"topk{num_experts_per_token}",
                    *(["matched-activated-params"] if match_activated_params else []),
                    f"expert-dim{expert_intermediate_dim}",
                    f"shared-expert-dim{shared_expert_intermediate_dim}",
                    *(["matched-total-active-flops"] if match_total_active_flops else []),
                    f"remat-{block_remat}",
                    *(["cross-entropy-auto"] if cross_entropy_implementation == "auto" else []),
                    *(
                        [f"cross-entropy-{cross_entropy_implementation.replace(',', '-')}"]
                        if cross_entropy_implementation != "auto"
                        else []
                    ),
                    *([f"cross-entropy-v{ce_block_sizes.v_block_size}"] if ce_block_sizes is not None else []),
                    *(["capacity-overflow-logging"] if report_capacity_overflow else []),
                    f"pf{loader_prefetch_size}",
                    f"buf{loader_max_buffered_batches}",
                    *(["block-shuffle"] if block_shuffle else []),
                    *(["synthetic-data"] if synthetic_data else []),
                    "profile",
                    "iris",
                    *([run_suffix] if run_suffix else []),
                ],
                group=(
                    f"gq32-v5p64-b{batch_size}-c{cf_slug}"
                    f"{experts_slug}{topk_slug}{matched_slug}{intermediate_slug}{shared_slug}{hidden_slug}{layers_slug}{total_flops_slug}"
                    f"{remat_slug}{ce_slug}{overflow_slug}"
                    f"{ce_block_slug}{loader_slug}{shuffle_slug}{synthetic_slug}"
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
                    synthetic_data=synthetic_data,
                    loader_prefetch_size=loader_prefetch_size,
                    loader_max_buffered_batches=loader_max_buffered_batches,
                    z_loss_weight=1e-4,
                    ema_beta=None,
                    log_every=1,
                )
            ),
            eval=None,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-axis-size", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--batch-size", type=int, default=320)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument(
        "--block-remat",
        choices=(
            "full",
            "off",
            "save_moe_input",
            "save_moe_hidden",
            "save_moe_output",
            "save_moe_inputs_outputs",
            "save_moe",
            "save_mlp_inputs",
            "save_mlp_outputs",
            "save_mlp",
            "offload_moe_input",
            "offload_moe_hidden",
            "offload_moe_output",
            "offload_moe_inputs_outputs",
            "offload_moe",
            "offload_mlp_inputs",
            "offload_mlp_outputs",
            "offload_mlp",
        ),
        default=QWEN3_32B_A4B_V5P_MODEL.block_remat,
    )
    parser.add_argument("--report-capacity-overflow", action="store_true")
    parser.add_argument("--num-experts", type=int, default=QWEN3_32B_A4B_V5P_MODEL.num_experts)
    parser.add_argument(
        "--num-experts-per-token",
        type=int,
        default=QWEN3_32B_A4B_V5P_MODEL.num_experts_per_token,
    )
    parser.add_argument("--intermediate-dim", type=int, default=None)
    parser.add_argument("--match-activated-params", action="store_true")
    parser.add_argument(
        "--shared-expert-intermediate-dim",
        type=int,
        default=QWEN3_32B_A4B_V5P_MODEL.shared_expert_intermediate_dim,
    )
    parser.add_argument("--match-total-active-flops", action="store_true")
    parser.add_argument("--block-shuffle", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--loader-prefetch-size", type=int, default=32)
    parser.add_argument("--loader-max-buffered-batches", type=int, default=64)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--profiler-start-step", type=int, default=8)
    parser.add_argument("--profiler-num-steps", type=int, default=5)
    parser.add_argument(
        "--cross-entropy-implementation",
        choices=("auto", "xla", "reference", "pallas_tpu", "pallas_tpu,xla"),
        default="auto",
    )
    parser.add_argument("--cross-entropy-v-block-divisor", type=int, default=1)
    parser.add_argument("--run-suffix", type=str, default="")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


def main() -> None:
    args = parse_args()
    step = build_step(
        expert_axis_size=args.expert_axis_size,
        batch_size=args.batch_size,
        capacity_factor=args.capacity_factor,
        block_remat=args.block_remat,
        report_capacity_overflow=args.report_capacity_overflow,
        num_experts=args.num_experts,
        num_experts_per_token=args.num_experts_per_token,
        intermediate_dim=args.intermediate_dim,
        match_activated_params=args.match_activated_params,
        shared_expert_intermediate_dim=args.shared_expert_intermediate_dim,
        match_total_active_flops=args.match_total_active_flops,
        block_shuffle=args.block_shuffle,
        synthetic_data=args.synthetic_data,
        loader_prefetch_size=args.loader_prefetch_size,
        loader_max_buffered_batches=args.loader_max_buffered_batches,
        steps=args.steps,
        profiler_start_step=args.profiler_start_step,
        profiler_num_steps=args.profiler_num_steps,
        cross_entropy_implementation=args.cross_entropy_implementation,
        cross_entropy_v_block_divisor=args.cross_entropy_v_block_divisor,
        run_suffix=args.run_suffix,
    )
    executor_main(
        steps=[step],
        description=(
            "Short Grug MoE EP profile run on a Qwen3-inspired ~32B-A4B shape using v5p-64 "
            f"at batch size {args.batch_size}, expert axis size {args.expert_axis_size}, "
            f"{args.num_experts} total experts, top-{args.num_experts_per_token} routing, "
            f"and shared expert dim {args.shared_expert_intermediate_dim}."
        ),
    )


if __name__ == "__main__":
    main()
