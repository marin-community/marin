"""Train from unified TaskCompendium task specifications.

With ``data.feed.root``, the tasks come from a feed that a producer extends while training runs
(``skyrl_train.dataset.task_feed``), and ``data.train_data`` is not read.
"""

import hydra
import ray
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from skyrl_train.config.rollout_validation import EntrypointOperation
from skyrl_train.dataset.task_feed import FeedTaskDataset, PolicyEndpoint, write_policy_endpoint
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver


class TaskCompendiumExp(BasePPOExp):
    def get_train_dataset(self):
        if self.cfg.data.feed.root is None:
            return super().get_train_dataset()
        return self._feed_dataset()

    def _feed_dataset(self) -> FeedTaskDataset:
        feed = self.cfg.data.feed
        if self.cfg.data.sampling.kind is not None:
            raise ValueError("data.sampling.kind cannot order data.feed; a task feed trains its rows in feed order")
        if feed.min_rows < self.cfg.trainer.train_batch_size:
            raise ValueError("data.feed.min_rows must be at least trainer.train_batch_size")
        if self.cfg.trainer.algorithm.group_admission.stall_timeout is None:
            raise ValueError(
                "data.feed needs an explicit trainer.algorithm.group_admission.stall_timeout: "
                "the default fails training after 10 minutes without a new task"
            )
        if self.cfg.data.train_data:
            logger.warning("data.feed.root is set, so data.train_data is not read: {}", list(self.cfg.data.train_data))
        return FeedTaskDataset(
            feed.root,
            self.tokenizer,
            self.cfg.trainer.max_prompt_length,
            session=OmegaConf.to_container(feed.session, resolve=True),
            poll_seconds=float(feed.poll_seconds),
            min_rows=int(feed.min_rows),
        )

    def create_inference_engine_client(self, *, operation: EntrypointOperation = EntrypointOperation.TRAIN):
        client = super().create_inference_engine_client(operation=operation)
        root = self.cfg.data.feed.root
        if root is not None and client.enable_http_endpoint:
            # The server listens on 0.0.0.0; the Ray node address reaches it from other pods of the cluster.
            base_url = f"http://{ray.util.get_node_ip_address()}:{client.http_endpoint_port}/v1"
            write_policy_endpoint(root, PolicyEndpoint(base_url=base_url, model=client.model_name))
            logger.info("Wrote the policy endpoint {} for the task feed {}", base_url, root)
        return client


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
