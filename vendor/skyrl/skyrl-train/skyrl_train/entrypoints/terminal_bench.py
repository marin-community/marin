"""
Main entrypoint for training on terminal bench tasks.
"""

import ray
import hydra
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, run_ray_driver
from skyrl_train.dataset.harbor import HarborTaskDataset


class TerminalBenchExp(BasePPOExp):
    def get_trajectory_runner(self, cfg, tokenizer, inference_engine_client):
        del tokenizer
        return self.get_task_runner(cfg, inference_engine_client, harbor_config=cfg.terminal_bench_config)

    def _task_dataset(self, data_files) -> HarborTaskDataset:
        return self.harbor_task_dataset(data_files)

    def get_train_dataset(self):
        prompts_dataset = self._task_dataset(self.cfg.data.train_data)
        # make sure the dataset is large enough to train on
        assert len(prompts_dataset) >= self.cfg.trainer.train_batch_size, (
            f"dataset must be at least as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, "
            f"but has size {len(prompts_dataset)}"
        )
        return prompts_dataset

    def get_eval_dataset(self):
        if self.cfg.trainer.eval_interval <= 0 or not self.cfg.data.val_data:
            return None
        return self._task_dataset(self.cfg.data.val_data)


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    from skyrl_train.utils.fd_monitor import start_fd_monitor  # noqa: PLC0415

    # make sure that the training loop is not run on the head node.
    # Start the file-descriptor monitor on the driver process. This is the
    # process whose logs show "(skyrl_entrypoint pid=...)" and which FD-aborts
    # (uv__epoll_ctl_prep SIGABRT) on long a3 RL chains. Self-contained daemon
    # thread; only runs here (the driver), not in the per-rank Ray workers.
    start_fd_monitor()
    exp = TerminalBenchExp(cfg)
    exp.run()


def run(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint)


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
