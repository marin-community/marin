"""Train from unified TaskCompendium task specifications."""

from omegaconf import DictConfig

from skyrl_train.entrypoints.main_base import run_ray_driver, skyrl_entrypoint


def run(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint)
