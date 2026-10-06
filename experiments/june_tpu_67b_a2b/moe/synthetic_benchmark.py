# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the June Snowball Grug trainer on synthetic tokens from one node, without Fray or Iris.

This is the Grug-code counterpart of ``experiments/grug/snowball_synthetic``: the June 67B-A2B recipe's own
model (``experiments/june_tpu_67b_a2b/moe/model.py``) and training loop, on the same shape, batch and portable
kernels (reference or ``xla_flash`` attention, XLA ``ragged_dot``), so the two code paths can be compared on
hardware that has no CUDA or TPU kernels. Every step sees fresh uniform-random tokens with no repeats, so the
loss cannot fall below ln(vocab_size) and a lower value means something leaks the targets. No checkpoint is
written.

Example, 8 GPUs, full 67B shape, expert parallelism over all 8, profile of steps 10-12::

    RAGGED_DOT_IMPL=xla python experiments/june_tpu_67b_a2b/moe/synthetic_benchmark.py \\
        --size full --steps 20 --expert-axis 8 --profile-steps 3
"""

import argparse
import dataclasses
import logging
import time
from pathlib import Path
from typing import get_args

import jax
import jmp
import numpy as np
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig, XprofUploadConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.grug.attention import GrugAttentionImplementation
from levanter.optim.config import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig

from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, RematMode
from experiments.june_tpu_67b_a2b.moe.train import (
    GrugRunConfig,
    GrugTrainerConfig,
    _compute_flops,
    _run_grug_local,
)

logger = logging.getLogger("june_synthetic_benchmark")

SNOWBALL_HIDDEN_DIM = 2560
# The production window (half of the 4096-token training context), which Levanter's SnowballConfig pins as well.
# The width heuristic would otherwise derive seq_len // 2 and give the medium preset a shorter window than Snowball's.
SNOWBALL_SLIDING_WINDOW = 2048


def snowball_model(seq_len: int) -> GrugModelConfig:
    """The June 67B-A2B production model at ``seq_len``, as the cooldown launch builds it, minus its
    long-context YaRN temperature scaling and with the production sliding window at every ``seq_len``."""
    base = MoeMuonHHeuristic().build_model_config(SNOWBALL_HIDDEN_DIM, seq_len=seq_len)
    return dataclasses.replace(
        base,
        sliding_window=SNOWBALL_SLIDING_WINDOW,
        disable_pko=True,
        disable_long_rope=True,
        use_array_stacked_blocks=True,
    )


def preset(size: str, seq_len: int | None) -> tuple[GrugModelConfig, str]:
    if size == "full":
        return snowball_model(seq_len or 4096), "p=f32,c=bfloat16"
    if size == "medium":
        return dataclasses.replace(snowball_model(seq_len or 1024), num_layers=4), "p=f32,c=bfloat16"
    tiny = dataclasses.replace(
        snowball_model(seq_len or 32),
        vocab_size=128,
        hidden_dim=64,
        intermediate_dim=64,
        shared_expert_intermediate_dim=64,
        num_experts=16,
        num_layers=5,
        num_heads=8,
        num_kv_heads=4,
        head_dim=16,
        sliding_window=4,
    )
    return tiny, "f32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", choices=["tiny", "medium", "full"], required=True)
    parser.add_argument("--layers", type=int, help="Override the preset layer count.")
    parser.add_argument("--seq-len", type=int, help="Override the preset sequence length.")
    parser.add_argument("--batch-size", type=int, help="Global batch in sequences (default: one per device).")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--mp", help="jmp policy override, e.g. 'p=f32,c=float16' (default: preset).")
    parser.add_argument("--expert-axis", type=int, default=1, help="Expert-parallel mesh axis size (1 = FSDP only).")
    parser.add_argument("--attention", choices=get_args(GrugAttentionImplementation), default="reference")
    parser.add_argument("--moe-impl", default="ring", help="MoE dispatch backend (default: ring).")
    parser.add_argument("--profile-steps", type=int, default=0, help="Profile this many steps (0 disables).")
    parser.add_argument("--profile-start", type=int, default=10, help="First profiled step.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/june-synthetic"))
    parser.add_argument("--run-id", help="Run id under --log-dir (default: size and timestamp).")
    parser.add_argument("--compilation-cache-dir", help="Persistent JAX compilation cache directory.")
    parser.add_argument("--remat-mode", choices=get_args(RematMode), default="recompute_all")
    parser.add_argument(
        "--watch-interval", type=int, default=10, help="Steps between gradient watch steps (0 disables)."
    )
    return parser.parse_args()


def synthetic_examples(count: int, seq_len: int, vocab_size: int, seed: int = 0) -> list[GrugLmExample]:
    rng = np.random.default_rng(seed)
    loss_weight = np.ones(seq_len, dtype=np.float32)
    loss_weight[-1] = 0
    return [
        GrugLmExample(tokens=rng.integers(0, vocab_size, seq_len, dtype=np.int32), loss_weight=loss_weight)
        for _ in range(count)
    ]


def report_memory() -> None:
    for device in jax.devices():
        stats = device.memory_stats()
        if stats is None:
            return
        logger.info(
            "%s: peak %.1f GiB of %.1f GiB", device, stats["peak_bytes_in_use"] / 2**30, stats["bytes_limit"] / 2**30
        )


def main() -> None:
    args = parse_args()
    model, default_mp = preset(args.size, args.seq_len)
    model = dataclasses.replace(
        model, attention_implementation=args.attention, moe_implementation=args.moe_impl, remat_mode=args.remat_mode
    )
    if args.layers is not None:
        model = dataclasses.replace(model, num_layers=args.layers)
    batch_size = jax.device_count() if args.batch_size is None else args.batch_size
    if batch_size == 1:
        # jnp.roll in the next-token loss slices the (1, seq_len) token grid to (1, 1), and under an explicit mesh JAX
        # drops the sharding of that slice, then rejects concatenating it with the (1, seq_len - 1) remainder.
        raise ValueError("a global batch of 1 fails in the loss under explicit mesh axes; pass --batch-size 2 or more")
    run_id = args.run_id or f"june-{args.size}-{time.strftime('%Y%m%d-%H%M%S')}"

    trainer = TrainerConfig(
        id=run_id,
        log_dir=args.log_dir,
        use_explicit_mesh_axes=True,
        require_accelerator=False,
        mp=jmp.get_policy(args.mp or default_mp),
        train_batch_size=batch_size,
        num_train_steps=args.steps,
        tracker=JsonLoggerConfig(),
        checkpointer=CheckpointerConfig(base_path=str(args.log_dir / run_id / "checkpoints")),
        load_checkpoint=False,  # never resume a reused --run-id; every benchmark starts from step 0
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
        watch=WatchConfig(interval=args.watch_interval),
    )
    # One fresh example per sequence, and a mixture block exactly as long as the dataset so nothing repeats.
    examples = synthetic_examples(args.steps * batch_size, model.max_seq_len, model.vocab_size)
    data = LmDataConfig(
        components={"synthetic": DirectDatasetComponent(datasets={"train": ListAsyncDataset(examples)})},
        tokenizer="passthrough",
        vocab_size=model.vocab_size,
        mixture_block_size=len(examples),
    )
    config = GrugRunConfig(
        model=model,
        data=data,
        resources=ResourceConfig(),
        optimizer=AdamConfig(learning_rate=args.learning_rate, warmup=0.0),
        trainer=GrugTrainerConfig(
            trainer=trainer, expert_axis_size=args.expert_axis, replica_axis_size=1, save_checkpoints=False
        ),
        eval=None,
    )
    flops_per_example, _ = _compute_flops(model_config=model)
    logger.info(
        "size=%s layers=%d hidden=%d experts=%d topk=%d heads=%d kv=%d batch=%d seq_len=%d window=%d mp=%s "
        "expert_axis=%d attention=%s moe=%s remat=%s watch_interval=%d flops_per_example=%.4e",
        args.size,
        model.num_layers,
        model.hidden_dim,
        model.num_experts,
        model.num_experts_per_token,
        model.num_heads,
        model.num_kv_heads,
        batch_size,
        model.max_seq_len,
        model.sliding_window,
        trainer.mp,
        args.expert_axis,
        args.attention,
        args.moe_impl,
        args.remat_mode,
        args.watch_interval,
        flops_per_example,
    )
    _run_grug_local(config)
    report_memory()


if __name__ == "__main__":
    main()
