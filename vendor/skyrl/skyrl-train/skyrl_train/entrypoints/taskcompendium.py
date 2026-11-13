"""Train from unified TaskCompendium task specifications."""

import ray
from omegaconf import DictConfig

from skyrl_train.entrypoints.main_base import BasePPOExp, run_ray_driver


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    BasePPOExp(cfg).run()


def run(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint)
