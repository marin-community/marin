"""
Main entrypoint for evaluation-only.
"""

import asyncio
from pathlib import Path
from typing import Any, Protocol

import hydra
import ray
from loguru import logger
from omegaconf import DictConfig
from marinskyrl.runtime_options import EvaluationRunner

from skyrl_train.config.pi_training import validate_pi_evaluation_results
from skyrl_train.config.rollout_validation import EntrypointOperation
from skyrl_train.entrypoints.main_base import (
    BasePPOExp,
    config_dir,
)
from skyrl_train.inference_engines.base import NamedWeightsUpdateRequest, lora_disk_load_request
from skyrl_train.utils.utils import validate_generator_cfg, initialize_ray
from skyrl_train.evaluate import evaluate
from skyrl_train.trajectory_runners.trajectory_retention import make_trajectory_sink
from skyrl_train.utils.trainer_utils import build_eval_dataloader


class PolicyAdapterClient(Protocol):
    async def update_named_weights(self, request: NamedWeightsUpdateRequest) -> None: ...


async def load_initial_policy_adapter(inference_engine_client: PolicyAdapterClient, cfg: DictConfig) -> None:
    """Load a configured policy LoRA before an evaluation-only rollout."""
    adapter_path = cfg.trainer.policy.model.lora.adapter_path
    if adapter_path is None:
        return
    if not cfg.generator.run_engines_locally:
        raise ValueError("evaluation-only LoRA loading requires local inference engines")
    if cfg.generator.backend != "vllm":
        raise ValueError("evaluation-only LoRA loading currently requires the vLLM backend")
    path = Path(adapter_path)
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("evaluation-only LoRA adapter_path must be an existing absolute directory")
    await inference_engine_client.update_named_weights(lora_disk_load_request(str(path)))


class EvalOnlyEntrypoint(BasePPOExp):
    def get_train_dataset(self):
        """Override to avoid requiring a train dataset for eval-only runs."""
        return None

    def _run(self) -> dict[str, Any]:
        return asyncio.run(self._evaluate())

    async def _evaluate(self) -> dict[str, Any]:
        assert self.eval_dataset is not None, "The evaluation only entrypoint requires an eval dataset is provided"

        inference_engine_client = self.create_inference_engine_client(operation=EntrypointOperation.GENERATE)
        await load_initial_policy_adapter(inference_engine_client, self.cfg)
        evaluation_runner = EvaluationRunner(self.cfg.trainer.evaluation_runner)
        trajectory_runner = (
            self.get_trajectory_runner(self.cfg, self.tokenizer, inference_engine_client)
            if evaluation_runner is EvaluationRunner.TRAINING
            else self.get_task_runner(
                self.cfg,
                inference_engine_client,
                harbor_config=self.cfg.terminal_bench_config if evaluation_runner is EvaluationRunner.HARBOR else None,
            )
        )

        trajectory_sink = make_trajectory_sink(self.cfg.generator, self.tokenizer)
        trajectory_runner.set_trajectory_sink(trajectory_sink)
        try:
            await trajectory_runner.startup()
            results: dict[str, Any] = await evaluate(
                eval_dataloader=build_eval_dataloader(self.cfg, self.eval_dataset),
                trajectory_runner=trajectory_runner,
                cfg=self.cfg,
                global_step=None,
                tokenizer=self.tokenizer,
                trajectory_sink=trajectory_sink,
            )
            validate_pi_evaluation_results(self.cfg, results)
        finally:
            try:
                await trajectory_runner.shutdown()
            finally:
                trajectory_sink.close()

        with self.get_tracker() as tracker:
            tracker.log(results, step=0, commit=True)

        return results


@ray.remote(num_cpus=1)
def eval_entrypoint(cfg: DictConfig) -> dict:
    exp = EvalOnlyEntrypoint(cfg)
    return exp.run()


def run(cfg: DictConfig) -> None:
    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    metrics = ray.get(eval_entrypoint.remote(cfg))
    logger.info(f"Metrics from eval only run: {metrics}")


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
