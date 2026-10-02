# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train Snowball on synthetic tokens through the Levanter trainer and report step throughput.

A single-process, one-node harness (for example one Slurm job) that measures Snowball's real training
step: mixed precision, Adam, the model's per-layer remat, and the trainer's own throughput and MFU
logging. It uses the portable code paths only: reference attention, and XLA ragged_dot when
``RAGGED_DOT_IMPL=xla`` is set, which AMD GPUs need because the default kernels are CUDA-only.

Every step sees fresh uniform-random tokens and no example repeats, so no model can get below a loss of
ln(vocab_size) (11.76 at full vocab). A loss that drops below that floor means something is leaking the
targets. Checkpoints are never written. The trainer's forced final save of the 67B model plus Adam state would
be about 1 TB, so the default hooks are replaced with the logging and profiler hooks only.

The default batch is one sequence per device. On a single device pass ``--batch-size 2``: a global batch of 1
fails in the next-token loss (see ``main``).

Example, 8 GPUs, full 67B shape, one 4096-token sequence per GPU, profile of steps 10-12::

    RAGGED_DOT_IMPL=xla python experiments/grug/snowball_synthetic/train.py \\
        --size full --steps 20 --profile-steps 3
"""

import argparse
import dataclasses
import logging
import statistics
import time
from pathlib import Path

import jax
import jax.random as jrandom
import jmp
import levanter.trainer
import numpy as np
from haliax import Axis
from levanter import callbacks
from levanter.callbacks._metrics import aggregate_device_flops, compute_instant_throughput
from levanter.callbacks.profiler import ProfilerConfig, XprofUploadConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import NamedLmDataset
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.models.snowball import SnowballConfig, num_long_attention_layers
from levanter.optim.config import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import StepInfo, Trainer, TrainerConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.mesh import MeshConfig

logger = logging.getLogger("snowball_synthetic")

TINY = SnowballConfig(
    vocab_size=128,
    hidden_dim=64,
    intermediate_dim=64,
    shared_expert_intermediate_dim=64,
    num_experts=16,
    num_experts_per_token=4,
    num_layers=5,
    num_heads=8,
    num_kv_heads=4,
    head_dim=16,
    max_seq_len=32,
    sliding_window=4,
)
FULL = SnowballConfig()  # 26 layers, hidden 2560, 256 experts


@dataclasses.dataclass(frozen=True)
class Preset:
    model: SnowballConfig
    seq_len: int
    mp: str


PRESETS = {
    "tiny": Preset(TINY, 32, "f32"),
    # Full width and expert count, 4 layers: layers 0-2 short, layer 3 long.
    "medium": Preset(dataclasses.replace(FULL, num_layers=4), 1024, "p=f32,c=bfloat16"),
    "full": Preset(FULL, 4096, "p=f32,c=bfloat16"),
}
# Steps 0 and 1 are excluded from the summary: step 0 compiles, and the trainer runs per-step hooks from step 2.
FIRST_TIMED_STEP = 2


def snowball_flops_per_token(cfg: SnowballConfig, vocab_size: int, seq_len: int) -> float:
    """Forward FLOPs per token, charging full-context attention only on Snowball's long layers."""
    return lm_flops_per_token(
        hidden_dim=cfg.hidden_dim,
        intermediate_dim=cfg.intermediate_dim,
        num_layers=cfg.num_layers,
        num_kv_heads=cfg.num_kv_heads,
        num_heads=cfg.num_heads,
        seq_len=seq_len,
        vocab_size=vocab_size,
        glu=True,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_token,
        num_shared_experts=1,
        shared_intermediate_dim=cfg.shared_expert_intermediate_dim,
        sliding_window=cfg.sliding_window,
        num_full_attention_layers=num_long_attention_layers(cfg.num_layers),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", choices=sorted(PRESETS), required=True)
    parser.add_argument("--layers", type=int, help="Override the preset layer count.")
    parser.add_argument("--seq-len", type=int, help="Override the preset sequence length.")
    parser.add_argument("--batch-size", type=int, help="Global batch in sequences (default: one per device).")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--mp", help="jmp policy override, e.g. 'p=f32,c=bfloat16' (default: preset).")
    parser.add_argument("--expert-axis", type=int, default=1, help="Mesh expert-parallel axis size.")
    parser.add_argument("--context-axis", type=int, default=1, help="Mesh context-parallel axis size.")
    parser.add_argument("--moe-impl", help="MoE backend override, e.g. scatter (default: ring).")
    parser.add_argument("--profile-steps", type=int, default=0, help="Profile this many steps (0 disables).")
    parser.add_argument("--profile-start", type=int, default=10, help="First profiled step.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/snowball-synthetic"))
    parser.add_argument("--run-id", help="Run id under --log-dir (default: size and timestamp).")
    parser.add_argument("--compilation-cache-dir", help="Persistent JAX compilation cache directory.")
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


def build_trainer_config(args: argparse.Namespace, preset: Preset, run_id: str, batch_size: int) -> TrainerConfig:
    compute_mapping = {"batch": ["replica_dcn", "data", "expert"], "vocab": "model"}
    if args.context_axis > 1:
        # Snowball only shards activations over a context axis wider than one device; the loss must agree.
        compute_mapping["position"] = "context"
    mesh = MeshConfig(
        axes={"data": -1, "replica": 1, "model": 1, "context": args.context_axis, "expert": args.expert_axis},
        compute_mapping=compute_mapping,
    )
    return TrainerConfig(
        id=run_id,
        log_dir=args.log_dir,
        mesh=mesh,
        use_explicit_mesh_axes=True,
        # The tiny preset is a CPU check; the default demands an accelerator everywhere except macOS.
        require_accelerator=False,
        mp=jmp.get_policy(args.mp or preset.mp),
        train_batch_size=batch_size,
        num_train_steps=args.steps,
        tracker=JsonLoggerConfig(),
        # Never consulted for saves (the checkpoint hook is not installed); only names the resume-search root.
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


def log_throughput_summary(
    step_durations: list[float], batch_size: int, seq_len: int, flops_per_example: float, num_steps: int
) -> None:
    median = statistics.median(step_durations)
    throughput = compute_instant_throughput(batch_size, median, seq_len, flops_per_example, aggregate_device_flops())
    logger.info(
        "median step over steps %d-%d: %.3f s  %s tokens/s  %s model TFLOP/s  MFU %s%%",
        FIRST_TIMED_STEP,
        num_steps - 1,
        median,
        f"{throughput.tokens_per_second:,.0f}" if throughput.tokens_per_second is not None else "n/a",
        f"{throughput.model_flops_per_second / 1e12:.1f}" if throughput.model_flops_per_second is not None else "n/a",
        f"{throughput.mfu:.2f}" if throughput.mfu is not None else "n/a",
    )


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.size]
    model_cfg = dataclasses.replace(preset.model, attention_implementation="reference", moe_implementation=args.moe_impl)
    if args.layers is not None:
        model_cfg = dataclasses.replace(model_cfg, num_layers=args.layers)
    seq_len = preset.seq_len if args.seq_len is None else args.seq_len
    batch_size = jax.device_count() if args.batch_size is None else args.batch_size
    if batch_size == 1:
        # jnp.roll in the next-token loss slices the (1, seq_len) token grid to (1, 1), and under an explicit mesh JAX
        # drops the sharding of that slice, then rejects concatenating it with the (1, seq_len - 1) remainder.
        raise ValueError("a global batch of 1 fails in the loss under explicit mesh axes; pass --batch-size 2 or more")
    run_id = args.run_id or f"snowball-{args.size}-{time.strftime('%Y%m%d-%H%M%S')}"

    trainer_cfg = build_trainer_config(args, preset, run_id, batch_size)
    profile = args.profile_steps > 0

    def loss_function(model, example, *, key=None):
        return model.compute_next_token_loss(example, key=key)

    levanter.trainer.initialize(trainer_cfg)
    logger.info(
        "size=%s layers=%d hidden=%d experts=%d topk=%d batch=%d seq_len=%d mp=%s expert_axis=%d context_axis=%d",
        args.size,
        model_cfg.num_layers,
        model_cfg.hidden_dim,
        model_cfg.num_experts,
        model_cfg.num_experts_per_token,
        batch_size,
        seq_len,
        trainer_cfg.mp,
        args.expert_axis,
        args.context_axis,
    )
    optimizer = AdamConfig(learning_rate=args.learning_rate, warmup=0.0).build(args.steps)
    step_durations: list[float] = []

    def record_step(info: StepInfo, force: bool = False) -> None:
        # The trainer re-runs every hook with force=True after the last step; that call would duplicate the sample.
        if not force and info.step >= FIRST_TIMED_STEP:
            step_durations.append(info.step_duration)

    with Trainer(trainer_cfg, optimizer, loss_function, add_default_hooks=False) as trainer:
        Pos = model_cfg.max_Pos.resize(seq_len)
        Vocab = Axis("vocab", model_cfg.vocab_size)
        model_key, training_key = jrandom.split(jrandom.PRNGKey(trainer_cfg.seed))
        flops_per_example = 3 * snowball_flops_per_token(model_cfg, Vocab.size, Pos.size) * Pos.size

        trainer.add_hook(callbacks.pbar_logger(total=args.steps), every=1)
        trainer.add_hook(callbacks.log_step_info(args.steps, trainer_cfg.batch_schedule), every=1)
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, trainer_cfg.batch_schedule, flops_per_example), every=1
        )
        trainer.add_hook(record_step, every=1)
        if profile:
            profile_steps = trainer_cfg.profiler.resolve_num_profile_steps(num_train_steps=args.steps)
            profile_dir = str(trainer_cfg.log_dir / trainer.run_id / "profiler")
            trainer.add_hook(
                trainer_cfg.profiler.build(profile_dir, run_id=trainer.run_id, num_steps=profile_steps), every=1
            )

        # Straight to the loader, not through LmDataConfig's mixer: the mixer draws in blocks of 2048 and
        # wraps a shorter dataset back to its start, which would repeat examples and let the model memorize.
        examples = synthetic_examples(args.steps * batch_size, seq_len, model_cfg.vocab_size)
        train_dataset = NamedLmDataset(ListAsyncDataset(examples), Pos)
        start = time.perf_counter()
        state = trainer.initial_state(training_key, model_init=lambda: model_cfg.build(Vocab, key=model_key))
        logger.info("initialized trainer state in %.1f s", time.perf_counter() - start)
        report_memory()

        trainer.train(state, trainer.data_loader(train_dataset))
        trainer.tracker.finish()

    if step_durations:
        log_throughput_summary(step_durations, batch_size, seq_len, flops_per_example, args.steps)
    report_memory()


if __name__ == "__main__":
    main()
