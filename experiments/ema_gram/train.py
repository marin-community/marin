# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind EMA-Gram settings to the existing Delphi recipes; dry-run by default."""

import argparse
import json
import logging
import math
import uuid
from dataclasses import replace
from pathlib import Path

import draccus
import jax
from levanter.optim.ema_gram import EmaGramConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from marin.execution.lazy import materialized_config
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.training.training import run_levanter_train_lm
from rigging.filesystem.storage_path import StoragePath

from experiments.ema_gram.plan import HORIZON, PILOT, PILOT_TOKENS, SCALE, Cell, confirmations, read_lock
from experiments.references import train_delphi_seeds as reference


class AttemptLogHandler(logging.FileHandler):
    """Flush every local metric and preserve immutable artifact chunks every 100 records."""

    def __init__(self, local_path: Path, artifact_path: StoragePath):
        super().__init__(local_path)
        self.artifact_path = artifact_path
        self.pending: list[str] = []
        self.chunk = 0
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        super().emit(record)
        self.pending.append(self.format(record))
        if len(self.pending) >= 100:
            self.persist()

    def persist(self):
        if self.pending:
            (self.artifact_path / f"metrics-{self.chunk:06d}.jsonl").write_text("\n".join(self.pending) + "\n")
            self.pending.clear()
            self.chunk += 1

    def close(self):
        self.persist()
        super().close()


def run_training(config):
    """Keep a distinct online W&B run for each process, including preemption retries."""
    if config.output_path is None:
        raise ValueError("Training requires an artifact output path for attempt evidence")
    tracker = config.train_config.trainer.tracker
    attempt = uuid.uuid4().hex[:12]
    tracker = replace(tracker, id=f"{config.train_config.trainer.id}-{attempt}", resume="never", mode="online")
    log_dir = Path("logs") / config.train_config.trainer.id / attempt
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ema_gram_metrics")
    artifact_dir = StoragePath(config.output_path) / "attempts" / attempt
    artifact_dir.mkdirs()
    handler = AttemptLogHandler(log_dir / "metrics.jsonl", artifact_dir)
    logger.addHandler(handler)
    config = replace(
        config,
        train_config=replace(
            config.train_config,
            trainer=replace(config.train_config.trainer, tracker=(tracker, JsonLoggerConfig(logger_name=logger.name))),
        ),
    )
    try:
        return run_levanter_train_lm(config)
    finally:
        memory = [d.memory_stats() for d in jax.local_devices()]
        logger.info(json.dumps({"event": "device_memory_stats", "devices": memory}))
        logger.removeHandler(handler)
        handler.close()
        (log_dir / "memory.json").write_text(json.dumps(memory))
        (artifact_dir / "memory.json").write_text(json.dumps(memory))


def build_training_step(cell: Cell, version: str):
    """Preserve reference architecture, cache order, tokenizer, mesh, precision and schedule."""
    caches = reference.build(((cell.slug, 42),))[0].deps
    base = reference._training_step(cell.slug, cell.seed, caches)
    spec = reference.MODELS[cell.slug]
    steps = (
        math.ceil(PILOT_TOKENS / (spec.batch_size * reference.SEQUENCE_LENGTH))
        if cell.phase == PILOT
        else spec.num_train_steps
    )

    def build_config(ctx):
        pod = base.build_config(ctx)
        cfg = pod.train_config
        opt = EmaGramConfig(
            learning_rate=cell.learning_rate,
            alpha=cell.alpha,
            ridge=cell.ridge,
            gram_beta=1 - 1 / cell.horizon,
            momentum=0.95,
            polar_steps=5,
            adam_lr=spec.adam_lr,
            adam_beta1=0.9,
            adam_beta2=spec.beta2,
            adam_epsilon=spec.epsilon,
            max_grad_norm=0.1,
            weight_decay=0.1,
            lr_schedule="linear",
            warmup=0.1,
            decay=0.2,
            min_lr_ratio=0.0,
            rewarmup=0.0,
        )
        run_id = f"{cell.name}-{version}"
        tracker = replace(
            cfg.trainer.tracker,
            id=run_id,
            name=run_id,
            mode="online",
            resume="never",
            group=f"ema-gram-{version}",
            tags=["ema-gram", cell.phase, cell.slug],
        )
        trainer = replace(
            cfg.trainer,
            id=run_id,
            tracker=tracker,
            num_train_steps=steps,
            steps_per_eval=min(1000, max(1, steps // 10)),
            profiler=replace(cfg.trainer.profiler, enabled=True, start_step=100, num_steps=10),
            checkpointer=replace(cfg.trainer.checkpointer, keep=[{"every": max(1, steps // 10)}]),
        )
        return replace(
            pod, train_config=replace(cfg, optimizer=opt, trainer=trainer, data_seed=cell.seed, hf_save_steps=steps)
        )

    return replace(
        base,
        name=cell.name,
        version=version,
        build_config=build_config,
        run=remote(run_training, resources=reference.ResourceConfig.with_tpu(spec.tpu_type, regions=["us-central2"])),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--version", required=True, help="New identity for an independent campaign")
    parser.add_argument("--output", type=Path, required=True, help="Resolved configs retained before submission")
    parser.add_argument("--prefix", default="gs://marin-us-central2")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--run", action="store_true")
    execution.add_argument("--logging-check", action="store_true", help="Online W&B check with no training")
    parser.add_argument("--lock", type=Path, help="Required pilot-selection lock for confirmation or follow-up")
    parser.add_argument("--reproducible-result", type=Path, help="Reviewed confirmation report required for follow-ups")
    parser.add_argument("--max-concurrent", type=int, default=1)
    args = parser.parse_args()
    cells = [Cell(**c) for c in json.loads(args.plan.read_text())]
    if not cells or len({c.name for c in cells}) != len(cells):
        raise ValueError("Select nonempty unique cells")
    for phase in {c.phase for c in cells} - {PILOT}:
        if args.lock is None:
            raise ValueError("Confirmation requires --lock from completed pilots")
        allowed = confirmations(read_lock(args.lock), phase=phase)
        if any(c not in allowed for c in cells if c.phase == phase):
            raise ValueError("Plan differs from locked settings")
        if phase in (HORIZON, SCALE) and (args.reproducible_result is None or not args.reproducible_result.is_file()):
            raise ValueError("Follow-ups require a reviewed --reproducible-result report")
    steps = [build_training_step(c, args.version) for c in cells]
    steps = [replace(s, override_path=s.path(args.prefix)) for s in steps]
    configs = [draccus.encode(materialized_config(s, args.prefix)) for s in steps]
    args.output.write_text(json.dumps(configs, indent=2) + "\n")
    if args.logging_check:
        cfg = materialized_config(steps[0], args.prefix).train_config
        tracker_cfg = replace(
            cfg.trainer.tracker,
            id=f"ema-gram-logging-check-{uuid.uuid4().hex[:12]}",
            name="ema-gram-logging-check",
            resume="never",
            replicate_path=None,
        )
        tracker = tracker_cfg.init(None)
        try:
            tracker.log_hyperparameters(draccus.encode(cfg))
            tracker.log({"logging_check/success": 1}, step=0)
        finally:
            tracker.finish()
        print(f"https://wandb.ai/{tracker_cfg.entity}/{tracker_cfg.project}/runs/{tracker_cfg.id}")
    if args.run:
        StepRunner().run([s.lower() for s in steps], max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
