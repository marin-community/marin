# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the Grug MoE EP trainer on synthetic tokens from one node, without Fray or Iris.

The counterpart of ``experiments/grug/snowball_synthetic`` for the Grug code path: the same Snowball
shape, batch, and portable kernels (reference or blockwise XLA attention, XLA ``ragged_dot``), but the
model, loop, and expert-parallel dispatch are this directory's, not Levanter's ``SnowballConfig``.
Synthetic mode repeats one random batch, so the loss is not a correctness check here; step time,
tokens/s, MFU, and memory are what it measures. Checkpoints are never written.

Example, 8 GPUs, full shape, expert parallelism over all 8, profile of steps 10-12::

    RAGGED_DOT_IMPL=xla python experiments/grug/moe_hero_ep/synthetic_benchmark.py \\
        --size full --steps 20 --expert-axis 8 --profile-steps 3
"""

import argparse
import dataclasses
import logging
import time
from pathlib import Path

import jax
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig, XprofUploadConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.distributed import DistributedConfig
from levanter.grug.attention import PORTABLE_ATTENTION_IMPLEMENTATIONS
from levanter.optim.config import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig

from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.train import (
    GrugRunConfig,
    GrugTrainerConfig,
    TrainingDataMode,
    _compute_flops,
    _run_grug_local,
    grug_trainer_mesh_config,
)

logger = logging.getLogger("grug_synthetic_benchmark")

# Snowball (67B-A2B, June recipe) shape, as in levanter.models.snowball.SnowballConfig, at the benchmark's
# 4096-token sequence length rather than the model's 65536-token maximum.
FULL = GrugModelConfig(
    vocab_size=128256,
    hidden_dim=2560,
    intermediate_dim=1280,
    shared_expert_intermediate_dim=2560,
    num_shared_experts=1,
    num_experts=256,
    num_experts_per_token=4,
    num_layers=26,
    num_heads=20,
    num_kv_heads=5,
    head_dim=128,
    max_seq_len=4096,
    sliding_window=2048,
    global_every=4,
)
TINY = dataclasses.replace(
    FULL,
    vocab_size=128,
    hidden_dim=64,
    intermediate_dim=64,
    shared_expert_intermediate_dim=64,
    num_experts=16,
    num_layers=5,
    num_heads=8,
    num_kv_heads=4,
    head_dim=16,
    max_seq_len=32,
    sliding_window=4,
)
PRESETS = {
    "tiny": (TINY, "f32"),
    "medium": (dataclasses.replace(FULL, num_layers=4, max_seq_len=1024), "p=f32,c=bfloat16"),
    "full": (FULL, "p=f32,c=bfloat16"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", choices=sorted(PRESETS), required=True)
    parser.add_argument("--layers", type=int, help="Override the preset layer count.")
    parser.add_argument("--seq-len", type=int, help="Override the preset sequence length.")
    parser.add_argument("--batch-size", type=int, help="Global batch in sequences (default: one per device).")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--mp", help="jmp policy override, e.g. 'p=f32,c=float16' (default: preset).")
    parser.add_argument("--expert-axis", type=int, default=1, help="Expert-parallel mesh axis size (1 = FSDP only).")
    parser.add_argument("--attention", choices=PORTABLE_ATTENTION_IMPLEMENTATIONS, default="reference")
    parser.add_argument("--moe-impl", default="ring", help="Expert-parallel MoE backend (default: ring).")
    parser.add_argument("--profile-steps", type=int, default=0, help="Profile this many steps (0 disables).")
    parser.add_argument("--profile-start", type=int, default=10, help="First profiled step.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/grug-synthetic"))
    parser.add_argument("--run-id", help="Run id under --log-dir (default: size and timestamp).")
    parser.add_argument("--compilation-cache-dir", help="Persistent JAX compilation cache directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, default_mp = PRESETS[args.size]
    model = dataclasses.replace(model, attention_implementation=args.attention, moe_implementation=args.moe_impl)
    if args.layers is not None:
        model = dataclasses.replace(model, num_layers=args.layers)
    if args.seq_len is not None:
        model = dataclasses.replace(model, max_seq_len=args.seq_len)
    batch_size = jax.device_count() if args.batch_size is None else args.batch_size
    run_id = args.run_id or f"grug-{args.size}-{time.strftime('%Y%m%d-%H%M%S')}"

    trainer = TrainerConfig(
        id=run_id,
        log_dir=args.log_dir,
        mesh=grug_trainer_mesh_config(1),
        use_explicit_mesh_axes=True,
        require_accelerator=False,
        mp=jmp.get_policy(args.mp or default_mp),
        train_batch_size=batch_size,
        num_train_steps=args.steps,
        tracker=JsonLoggerConfig(),
        checkpointer=CheckpointerConfig(base_path=str(args.log_dir / run_id / "checkpoints")),
        distributed=DistributedConfig(initialize_jax_distributed=False),
        profiler=ProfilerConfig(
            enabled=args.profile_steps > 0,
            start_step=args.profile_start,
            num_steps=args.profile_steps,
            upload=XprofUploadConfig(enabled=False),
        ),
        jax_compilation_cache_dir=args.compilation_cache_dir,
        log_jaxprs=False,
        log_xla_hlo=False,
    )
    config = GrugRunConfig(
        model=model,
        data=LmDataConfig(tokenizer="passthrough", vocab_size=model.vocab_size),
        resources=ResourceConfig(),
        optimizer=AdamConfig(learning_rate=args.learning_rate, warmup=0.0),
        trainer=GrugTrainerConfig(
            trainer=trainer,
            training_data_mode=TrainingDataMode.SYNTHETIC,
            expert_axis_size=args.expert_axis,
            replica_axis_size=1,
        ),
        eval=None,
        stop_after_steps=args.steps,
    )
    flops_per_example, _ = _compute_flops(model_config=model)
    logger.info(
        "size=%s layers=%d hidden=%d experts=%d topk=%d batch=%d seq_len=%d mp=%s expert_axis=%d attention=%s moe=%s "
        "flops_per_example=%.4e",
        args.size,
        model.num_layers,
        model.hidden_dim,
        model.num_experts,
        model.num_experts_per_token,
        batch_size,
        model.max_seq_len,
        trainer.mp,
        args.expert_axis,
        args.attention,
        args.moe_impl,
        flops_per_example,
    )
    _run_grug_local(config)


if __name__ == "__main__":
    main()
