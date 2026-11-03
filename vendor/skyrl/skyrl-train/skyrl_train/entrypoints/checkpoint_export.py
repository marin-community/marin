"""Hydra entrypoint for policy-only distributed checkpoint conversion."""

from __future__ import annotations

import signal
import sys
from pathlib import Path

import hydra
import ray
from loguru import logger
from omegaconf import DictConfig

from skyrl_train.checkpoint_exporter import checkpoint_exporter
from skyrl_train.utils.utils import initialize_ray

_CONFIG_DIR = str(Path(__file__).parent.parent / "config")


@ray.remote(num_cpus=1, max_retries=0)
def run_checkpoint_export(cfg: DictConfig):
    """Run conversion away from the Ray head node."""
    return checkpoint_exporter(cfg).run()


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


@hydra.main(config_path=_CONFIG_DIR, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
