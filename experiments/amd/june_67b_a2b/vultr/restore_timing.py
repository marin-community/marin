# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Time saving and restoring the AMD June trainer's full state, and a restart's time to its first training step.

``--phase save`` trains ``--steps`` steps of the synthetic benchmark's run and writes the final checkpoint to
``--checkpoint-dir``. ``--phase restore`` starts the same run with two more steps, auto-resumes from that checkpoint
and trains the two steps without saving. Both log, on process 0, seconds since the Python process started for: the
checkpoint save or the restore (start and end, the restored arrays ready), and the first training step logged. Every
other option is the synthetic benchmark's::

    srun -N2 --ntasks-per-node=1 python -m experiments.amd.june_67b_a2b.vultr.restore_timing --phase save \\
        --checkpoint-dir $WORK/ckpt-restore/full --steps 2 --size full --seq-len 8192 --batch-size 16 \\
        --gpus-per-node 8 --expert-axis 8 --replica-axis 2 --attention gpu_pallas_triton_flash --watch-interval 0
"""

import time

PROCESS_START = time.time()

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402

import jax  # noqa: E402
from levanter.checkpoint import Checkpointer, CheckpointerConfig  # noqa: E402
from levanter.tracker.json_logger import JsonLoggerConfig  # noqa: E402
from rigging.log_setup import configure_logging  # noqa: E402

from experiments.amd.june_67b_a2b import train  # noqa: E402
from experiments.amd.june_67b_a2b.run_config import initialize_distributed  # noqa: E402
from experiments.amd.june_67b_a2b.synthetic_benchmark import build_run_config  # noqa: E402
from experiments.amd.june_67b_a2b.synthetic_benchmark import parse_args as parse_benchmark_args  # noqa: E402

logger = logging.getLogger("restore_timing")


def _since_start() -> float:
    return time.time() - PROCESS_START


def _mark(event: str) -> None:
    if jax.process_index() == 0:
        logger.info("timing %s at %.1f s since process start", event, _since_start())


class FirstStep(logging.Handler):
    """Marks the first training step the loop logs through the JSON tracker."""

    def __init__(self) -> None:
        super().__init__()
        self.seen = False

    def emit(self, record: logging.LogRecord) -> None:
        event = json.loads(record.getMessage())
        if not self.seen and event.get("event") == "log" and "train/loss" in event.get("metrics", {}):
            self.seen = True
            _mark(f"first step {event['step']} logged")


def _timed_restore(restore):
    def wrapped(*args, **kwargs):
        _mark("restore start")
        state = jax.block_until_ready(restore(*args, **kwargs))
        _mark(f"restore end, step {int(state.step)}")
        return state

    return wrapped


def _timed_final_save(save_final_checkpoint):
    def wrapped(self, *, tree, step):
        _mark(f"final save of step {step} start")
        save_final_checkpoint(self, tree=tree, step=step)
        _mark(f"final save of step {step} end")

    return wrapped


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--phase", choices=["save", "restore"], required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    own, rest = parser.parse_known_args()
    args = parse_benchmark_args(rest)
    configure_logging(logging.INFO)
    initialize_distributed(args)
    _mark("distributed initialized")

    saved_steps = args.steps
    if own.phase == "restore":
        args.steps = saved_steps + 2
    config = build_run_config(args)
    trainer = dataclasses.replace(
        config.trainer.trainer,
        checkpointer=CheckpointerConfig(
            base_path=own.checkpoint_dir, append_run_id_to_base_path=False, save_interval=None, keep=[]
        ),
        load_checkpoint=None if own.phase == "save" else True,
    )
    config = dataclasses.replace(
        config,
        trainer=dataclasses.replace(config.trainer, trainer=trainer, save_checkpoints=own.phase == "save"),
    )

    train.restore_grug_state_from_checkpoint = _timed_restore(train.restore_grug_state_from_checkpoint)
    Checkpointer.save_final_checkpoint = _timed_final_save(Checkpointer.save_final_checkpoint)
    first_step = FirstStep()
    tracker_log = logging.getLogger(JsonLoggerConfig.logger_name)
    tracker_log.addHandler(first_step)
    try:
        train._run_grug_local(config)
    finally:
        tracker_log.removeHandler(first_step)
    _mark("done")


if __name__ == "__main__":
    main()
