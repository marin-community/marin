"""Train from unified TaskCompendium task specifications."""

import ray
from omegaconf import DictConfig
from ray.util.queue import Queue

from skyrl_train.entrypoints.main_base import BasePPOExp, run_ray_driver
from skyrl_train.entrypoints.workload_progress import report_to_driver


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig, progress: Queue | None):
    with report_to_driver(progress):
        return BasePPOExp(cfg).run()


def run(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint)
