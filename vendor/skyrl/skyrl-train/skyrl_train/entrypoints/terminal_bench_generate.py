"""
Main entrypoint for generating rollouts on terminal bench tasks.
"""

import ray
import asyncio
import hydra
from omegaconf import DictConfig

from skyrl_train.entrypoints.main_base import (
    config_dir,
    run_ray_driver,
)
from skyrl_train.entrypoints.terminal_bench import TerminalBenchExp
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.config.rollout_validation import EntrypointOperation
from skyrl_train.trajectory_runners.trajectory_processing import prepare_trajectory_request


class TerminalBenchGenerateExp(TerminalBenchExp):
    async def _generate(self) -> None:
        inference_engine_client = self.create_inference_engine_client(operation=EntrypointOperation.GENERATE)
        trajectory_runner = self.get_trajectory_runner(self.cfg, self.tokenizer, inference_engine_client)

        input_batch, _ = prepare_trajectory_request(
            self.train_dataset.collate_fn(list(self.train_dataset)),
            self.cfg.generator.n_samples_per_prompt,
            get_sampling_params_for_backend(self.cfg.generator.backend, self.cfg.generator.sampling_params),
            "eval",
            0,
        )

        await trajectory_runner.startup()
        try:
            # Generation requests carry the eval phase, which the rollout workers serve only inside an eval session.
            await trajectory_runner.start_eval_session(run_name=self.cfg.trainer.run_name, eval_step=0)
            try:
                await trajectory_runner.run(input_batch)
            finally:
                await trajectory_runner.stop_eval_session()
        finally:
            await trajectory_runner.shutdown()

    def _run(self):
        asyncio.run(self._generate())


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = TerminalBenchGenerateExp(cfg)
    exp.run()


def run(cfg: DictConfig) -> None:
    run_ray_driver(
        cfg,
        skyrl_entrypoint,
        operation=EntrypointOperation.GENERATE,
        failure_message="Generation failed",
    )


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
