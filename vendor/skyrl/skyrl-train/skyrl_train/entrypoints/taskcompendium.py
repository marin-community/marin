"""Train from unified TaskCompendium task specifications."""

import hydra
import ray
from omegaconf import DictConfig

from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    BasePPOExp(cfg).run()


def run(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint)


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
