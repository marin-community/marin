# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Time the AMD copy of the June 67B-A2B model on synthetic tokens on one or more Slurm nodes, without Fray or Iris.

The AMD counterpart of ``experiments/june_tpu_67b_a2b/moe/synthetic_benchmark.py``: the same presets, synthetic data
and report, on ``experiments/amd/june_67b_a2b``'s model and training loop. Every step sees fresh uniform-random
tokens, so the loss cannot fall below ln(vocab_size). No checkpoint is written. It logs the model FLOPs per example up
front and ends with the median step time from step 2 on, tokens/s, model TFLOP/s and MFU, the median wall step
(the time between consecutive steps' hooks, host work included), then each device's peak memory.

Example, two 8-GPU MI355X nodes, expert parallelism within each node and data parallelism across them::

    srun -N2 --ntasks-per-node=1 python experiments/amd/june_67b_a2b/synthetic_benchmark.py \\
        --size full --steps 30 --gpus-per-node 8 --expert-axis 8 --replica-axis 2 --batch-size 32 \\
        --attention gpu_pallas_triton_flash --watch-interval 0
"""

import argparse
import logging
from pathlib import Path

from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from rigging.log_setup import configure_logging

from experiments.amd.june_67b_a2b.run_config import (
    StepTimes,
    add_run_arguments,
    build_grug_trainer,
    build_model,
    build_trainer,
    global_batch_size,
    initialize_distributed,
    log_run_config,
    shared_run_id,
)
from experiments.amd.june_67b_a2b.train import _run_grug_local
from experiments.june_tpu_67b_a2b.moe.synthetic_benchmark import log_model_flops, report_memory, synthetic_examples
from experiments.june_tpu_67b_a2b.moe.train import GrugRunConfig, _compute_flops


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_run_arguments(parser)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/amd-june-synthetic"))
    parser.add_argument("--run-id", help="Run id under --log-dir (default: size and timestamp).")
    return parser.parse_args(argv)


def build_run_config(args: argparse.Namespace) -> GrugRunConfig:
    """The benchmark's run: the command line's model and mesh on fresh random tokens, flat Adam, no checkpoints."""
    model, default_mp = build_model(args)
    batch_size = global_batch_size(args)
    if args.steps * batch_size >= 2**16:
        raise ValueError(f"{args.steps} batches of {batch_size} sequences exceed the mixture block limit of 65535")
    run_id = shared_run_id(args, "amd-june")
    trainer = build_trainer(
        args,
        batch_size=batch_size,
        num_train_steps=args.steps,
        run_id=run_id,
        default_mp=default_mp,
        tracker=JsonLoggerConfig(),
        checkpointer=CheckpointerConfig(base_path=str(args.log_dir / run_id / "checkpoints")),
    )
    # One fresh example per sequence, and a mixture block exactly as long as the dataset so nothing repeats.
    examples = synthetic_examples(args.steps * batch_size, model.max_seq_len, model.vocab_size)
    data = LmDataConfig(
        components={"synthetic": DirectDatasetComponent(datasets={"train": ListAsyncDataset(examples)})},
        tokenizer="passthrough",
        vocab_size=model.vocab_size,
        mixture_block_size=len(examples),
    )
    return GrugRunConfig(
        model=model,
        data=data,
        resources=ResourceConfig(),
        optimizer=AdamConfig(learning_rate=args.learning_rate, warmup=0.0),
        trainer=build_grug_trainer(args, trainer, save_checkpoints=False),
        eval=None,
    )


def main() -> None:
    args = parse_args()
    # The training loop sets up logging only once it starts; without this, the lines logged before it are dropped.
    configure_logging(logging.INFO)
    initialize_distributed(args)
    config = build_run_config(args)
    model, trainer = config.model, config.trainer.trainer
    batch_size = trainer.train_batch_size
    flops_per_example, flops_summary = _compute_flops(model_config=model)
    log_run_config(args, model, trainer, batch_size, flops_per_example)
    log_model_flops(flops_per_example, flops_summary)
    step_times = StepTimes()
    tracker_log = logging.getLogger(JsonLoggerConfig.logger_name)
    tracker_log.addHandler(step_times)
    try:
        _run_grug_local(config)
    finally:
        tracker_log.removeHandler(step_times)
    step_times.log_summary(batch_size, model.max_seq_len, flops_per_example)
    report_memory()


if __name__ == "__main__":
    main()
