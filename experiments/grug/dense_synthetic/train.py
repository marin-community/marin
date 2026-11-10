# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# GRUG NOVERIFY: a synthetic-token benchmark driven by Levanter's Trainer, not the grug template loop.

"""Train a dense Grug transformer on synthetic tokens through the Levanter trainer and report step throughput.

The dense counterpart of ``experiments/grug/snowball_synthetic``: the same single-process, one-node harness, trainer
setup and summary lines, on the dense model in ``model.py`` (the June Grug block machinery with a SwiGLU MLP and
optional QK-norm). Presets: ``tiny`` (CPU check), ``8b`` (Llama-3 8B) and ``32b`` (Marin 32B, Qwen3 layout). Use
``--layers`` for shallower proxies of the same width. Parameters are FSDP-sharded over every device; ``--model-axis``
adds tensor parallelism. Mixed precision defaults to f32 parameters with bf16 compute.

Every step sees fresh uniform-random tokens and no example repeats, so the loss cannot fall below ln(vocab_size)
(11.76 at the 8b and 32b vocabularies); a lower loss means something leaks the targets. Checkpoints are never
written: only the logging and profiler hooks are installed.

The default batch is one sequence per device. On a single device pass ``--batch-size 2``: a global batch of 1
fails in the next-token loss (see ``main``).

Example, 8 GPUs, 32B at 16 layers, flash attention, profile of steps 10-12::

    python experiments/grug/dense_synthetic/train.py \\
        --size 32b --layers 16 --attention xla_flash --steps 20 --profile-steps 3
"""

import argparse
import dataclasses
import logging
import statistics
import time
from pathlib import Path
from typing import get_args

import jax
import jax.random as jrandom
import jmp
import levanter.trainer
import numpy as np
from haliax import Axis
from levanter import callbacks
from levanter.callbacks._metrics import aggregate_device_flops, compute_instant_throughput
from levanter.callbacks.profiler import ProfileOptionsConfig, ProfilerConfig, XprofUploadConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import NamedLmDataset
from levanter.data.text.examples import GrugLmExample, grug_lm_example_from_named
from levanter.distributed import DistributedConfig
from levanter.grug.attention import GrugAttentionImplementation
from levanter.models.lm_model import LmExample
from levanter.optim.config import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import StepInfo, Trainer, TrainerConfig
from levanter.utils.flop_utils import AttentionFlops, lm_attention_flops_per_token, lm_flops_per_token
from levanter.utils.mesh import MeshConfig

from experiments.grug.dense_synthetic.model import GrugModelConfig, RematMode, Transformer

logger = logging.getLogger("dense_synthetic")

LLAMA3_VOCAB_SIZE = 128256
PRESETS = {
    "tiny": GrugModelConfig(
        vocab_size=128,
        hidden_dim=64,
        intermediate_dim=128,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        max_seq_len=32,
    ),
    # Llama-3 8B, which has no QK-norm.
    "8b": GrugModelConfig(
        vocab_size=LLAMA3_VOCAB_SIZE,
        hidden_dim=4096,
        intermediate_dim=14336,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_seq_len=4096,
        qk_norm=False,
    ),
    # Marin 32B: Qwen3 layout with Llama-3's vocabulary.
    "32b": GrugModelConfig(
        vocab_size=LLAMA3_VOCAB_SIZE,
        hidden_dim=5120,
        intermediate_dim=27648,
        num_layers=64,
        num_heads=40,
        num_kv_heads=8,
        head_dim=128,
        max_seq_len=4096,
    ),
}
DEFAULT_MP = "p=f32,c=bfloat16"
# Steps 0 and 1 are excluded from the summary: step 0 compiles, and the trainer runs per-step hooks from step 2.
FIRST_TIMED_STEP = 2


def dense_flops_per_token(cfg: GrugModelConfig, seq_len: int) -> float:
    """Forward FLOPs per token, as Levanter charges its Llama and Qwen3 models (QK-norm is not counted)."""
    return lm_flops_per_token(
        hidden_dim=cfg.hidden_dim,
        intermediate_dim=cfg.intermediate_dim,
        num_layers=cfg.num_layers,
        num_kv_heads=cfg.num_kv_heads,
        num_heads=cfg.num_heads,
        seq_len=seq_len,
        vocab_size=cfg.vocab_size,
        glu=True,
    )


def dense_attention_flops_per_token(cfg: GrugModelConfig, seq_len: int) -> AttentionFlops:
    """The attention-map part of ``dense_flops_per_token``, non-causal as charged there and causal."""
    return lm_attention_flops_per_token(
        hidden_dim=cfg.hidden_dim, num_heads=cfg.num_heads, num_layers=cfg.num_layers, seq_len=seq_len
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", choices=sorted(PRESETS), required=True)
    parser.add_argument("--layers", type=int, help="Override the preset layer count.")
    parser.add_argument("--seq-len", type=int, help="Override the preset sequence length.")
    parser.add_argument("--batch-size", type=int, help="Global batch in sequences (default: one per device).")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--mp", default=DEFAULT_MP, help=f"jmp policy (default: {DEFAULT_MP}).")
    parser.add_argument("--model-axis", type=int, default=1, help="Tensor-parallel mesh axis size (1 = FSDP only).")
    parser.add_argument("--remat-mode", choices=get_args(RematMode), default="recompute_all")
    parser.add_argument(
        "--attention",
        choices=get_args(GrugAttentionImplementation),
        default="reference",
        help="Attention implementation; reference and xla_flash run on any backend.",
    )
    parser.add_argument("--profile-steps", type=int, default=0, help="Profile this many steps (0 disables).")
    parser.add_argument("--profile-start", type=int, default=10, help="First profiled step.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/dense-synthetic"))
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


def build_trainer_config(args: argparse.Namespace, run_id: str, batch_size: int) -> TrainerConfig:
    # The model shards over the Grug axes: parameters over data (FSDP) and model (TP), the batch over
    # (replica_dcn, data, expert). The expert axis stays at size 1 but must exist for those specs.
    mesh = MeshConfig(
        axes={"data": -1, "replica": 1, "model": args.model_axis, "expert": 1},
        compute_mapping={"batch": ["replica_dcn", "data", "expert"]},
    )
    return TrainerConfig(
        id=run_id,
        log_dir=args.log_dir,
        mesh=mesh,
        use_explicit_mesh_axes=True,
        # The tiny preset is a CPU check; the default demands an accelerator everywhere except macOS.
        require_accelerator=False,
        mp=jmp.get_policy(args.mp),
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
            # The HLO modules let trace tools attribute kernels to HLO ops (spurix's xplane buckets).
            profile_options=ProfileOptionsConfig(enable_hlo_proto=True),
        ),
        jax_compilation_cache_dir=args.compilation_cache_dir,
        log_jaxprs=False,
        log_xla_hlo=False,
    )


def log_model_flops(flops_per_example: float, attention_per_token: AttentionFlops, seq_len: int) -> None:
    """Log the FLOPs per example that MFU charges (3x forward) and their attention-map part, non-causal and causal."""
    if jax.process_index() != 0:
        return
    logger.info(
        "model flops per example: total=%r attention_noncausal=%r attention_causal=%r",
        flops_per_example,
        3 * attention_per_token.noncausal * seq_len,
        3 * attention_per_token.causal * seq_len,
    )


def log_throughput_summary(
    step_durations: list[float], batch_size: int, seq_len: int, flops_per_example: float, num_steps: int
) -> None:
    """Log the median step and its throughput, from rank 0 only: every process times its own steps."""
    if jax.process_index() != 0:
        return
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
    seq_len = PRESETS[args.size].max_seq_len if args.seq_len is None else args.seq_len
    model_cfg = dataclasses.replace(
        PRESETS[args.size],
        max_seq_len=seq_len,
        attention_implementation=args.attention,
        remat_mode=args.remat_mode,
    )
    if args.layers is not None:
        model_cfg = dataclasses.replace(model_cfg, num_layers=args.layers)
    if model_cfg.num_kv_heads % args.model_axis != 0:
        raise ValueError(f"--model-axis {args.model_axis} must divide the {model_cfg.num_kv_heads} KV heads")
    batch_size = jax.device_count() if args.batch_size is None else args.batch_size
    if batch_size == 1:
        # Building the next-token labels slices the (1, seq_len) token grid to (1, 1), and under an explicit mesh JAX
        # drops the sharding of that slice, then rejects concatenating it with the (1, seq_len - 1) remainder.
        raise ValueError("a global batch of 1 fails in the loss under explicit mesh axes; pass --batch-size 2 or more")
    run_id = args.run_id or f"dense-{args.size}-{time.strftime('%Y%m%d-%H%M%S')}"

    trainer_cfg = build_trainer_config(args, run_id, batch_size)
    profile = args.profile_steps > 0

    def loss_function(model: Transformer, example: LmExample, *, key=None):
        batch = grug_lm_example_from_named(example)
        return model.next_token_loss(batch.tokens, batch.loss_weight, mask=batch.attn_mask)

    levanter.trainer.initialize(trainer_cfg)
    logger.info(
        "size=%s layers=%d hidden=%d mlp=%d heads=%d kv=%d batch=%d seq_len=%d mp=%s model_axis=%d remat=%s "
        "attention=%s",
        args.size,
        model_cfg.num_layers,
        model_cfg.hidden_dim,
        model_cfg.intermediate_dim,
        model_cfg.num_heads,
        model_cfg.num_kv_heads,
        batch_size,
        seq_len,
        trainer_cfg.mp,
        args.model_axis,
        args.remat_mode,
        args.attention,
    )
    optimizer = AdamConfig(learning_rate=args.learning_rate, warmup=0.0).build(args.steps)
    step_durations: list[float] = []

    def record_step(info: StepInfo, force: bool = False) -> None:
        # The trainer re-runs every hook with force=True after the last step; that call would duplicate the sample.
        if not force and info.step >= FIRST_TIMED_STEP:
            step_durations.append(info.step_duration)

    with Trainer(trainer_cfg, optimizer, loss_function, add_default_hooks=False) as trainer:
        Pos = Axis("position", seq_len)
        model_key, training_key = jrandom.split(jrandom.PRNGKey(trainer_cfg.seed))
        flops_per_example = 3 * dense_flops_per_token(model_cfg, seq_len) * seq_len
        log_model_flops(flops_per_example, dense_attention_flops_per_token(model_cfg, seq_len), seq_len)

        trainer.add_hook(callbacks.pbar_logger(total=args.steps), every=1)
        trainer.add_hook(callbacks.log_step_info(args.steps, trainer_cfg.batch_schedule), every=1)
        trainer.add_hook(
            callbacks.log_performance_stats(seq_len, trainer_cfg.batch_schedule, flops_per_example), every=1
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
        state = trainer.initial_state(training_key, model_init=lambda: Transformer.init(model_cfg, key=model_key))
        logger.info("initialized trainer state in %.1f s", time.perf_counter() - start)
        report_memory()

        trainer.train(state, trainer.data_loader(train_dataset))
        trainer.tracker.finish()

    if step_durations:
        log_throughput_summary(step_durations, batch_size, seq_len, flops_per_example, args.steps)
    report_memory()


if __name__ == "__main__":
    main()
