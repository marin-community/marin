"""Policy-only distributed checkpoint conversion."""

from __future__ import annotations

import signal
import sys

import ray
from loguru import logger
from omegaconf import DictConfig
from ray.util.queue import Queue

from marinskyrl.driver_status import driver_progress_path
from skyrl_train.checkpoint_exporter import checkpoint_exporter
from skyrl_train.entrypoints.workload_progress import report_to_driver, workload_progress_queue, workload_result
from skyrl_train.utils.utils import initialize_ray


@ray.remote(num_cpus=1, max_retries=0)
def run_checkpoint_export(cfg: DictConfig, progress: Queue | None):
    """Run conversion away from the Ray head node."""
    with report_to_driver(progress):
        return checkpoint_exporter(cfg).run()


def run(cfg: DictConfig) -> None:
    initialize_ray(cfg)

    def shutdown_on_sigterm(_signum, _frame):
        ray.shutdown()
        sys.exit(1)

    signal.signal(signal.SIGTERM, shutdown_on_sigterm)
    try:
        with workload_progress_queue(driver_progress_path()) as progress:
            export = run_checkpoint_export.remote(cfg, progress.queue if progress else None)
            result = workload_result(export, progress)
        logger.info(f"Exported global_step_{result.step} to {result.export_path}")
    finally:
        ray.shutdown()
