"""Policy-only distributed checkpoint conversion."""

from __future__ import annotations

import signal
import sys

import ray
from loguru import logger
from omegaconf import DictConfig
from rigging.training_journal import TrainingJournal

from skyrl_train.checkpoint_exporter import checkpoint_exporter
from skyrl_train.utils.utils import initialize_ray


@ray.remote(num_cpus=1, max_retries=0)
def run_checkpoint_export(cfg: DictConfig):
    """Run conversion away from the Ray head node."""
    root = cfg.trainer.get("native_telemetry_root")
    if root is None:
        return checkpoint_exporter(cfg).run()
    journal = TrainingJournal(
        root + f"/exports/step-{cfg.checkpoint_export.step}", cfg.trainer.run_name, "rl_export", "exporter"
    )
    status = "failed"
    try:
        with journal.events.span("export", step=cfg.checkpoint_export.step):
            result = checkpoint_exporter(cfg).run()
        journal.log({"exported_checkpoint_step": result.step}, step=result.step, commit=True)
        status = "completed"
        return result
    finally:
        journal.finish(status=status)


def run(cfg: DictConfig) -> None:
    initialize_ray(cfg)

    def shutdown_on_sigterm(_signum, _frame):
        ray.shutdown()
        sys.exit(1)

    signal.signal(signal.SIGTERM, shutdown_on_sigterm)
    try:
        result = ray.get(run_checkpoint_export.remote(cfg))
        logger.info(f"Exported global_step_{result.step} to {result.export_path}")
    finally:
        ray.shutdown()
