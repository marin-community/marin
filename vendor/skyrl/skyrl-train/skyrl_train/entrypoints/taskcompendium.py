"""Train from unified TaskCompendium tasks with the Shellbox rollout engine."""

import hydra
import ray
from omegaconf import DictConfig

from skyrl_train.dataset.tasks import TaskDataset
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver


class TaskCompendiumExp(BasePPOExp):
    def get_train_dataset(self):
        dataset = TaskDataset(self.cfg.data.train_data, self.tokenizer, self.cfg.trainer.max_prompt_length)
        if len(dataset) < self.cfg.trainer.train_batch_size:
            raise ValueError("The task dataset is smaller than the training batch")
        return dataset

    def get_eval_dataset(self):
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            return TaskDataset(self.cfg.data.val_data, self.tokenizer, self.cfg.trainer.max_prompt_length)
        return None


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    TaskCompendiumExp(cfg).run()


def run(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint)


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
