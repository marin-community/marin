import logging
import os
from abc import ABC
from typing import ClassVar

import ray

from experiments.evals.resource_configs import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Dependency, Evaluator, ModelConfig
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)


class LevanterTpuEvaluator(Evaluator, ABC):
    """For `Evaluator`s that runs inference with Levanter (primarily Lm Eval Harness) on TPUs."""

    # pip packages to install for running levanter's eval_harness on TPUs
    DEFAULT_PIP_PACKAGES: ClassVar[list[Dependency]] = [
        Dependency(name="levanter>=1.2.dev1163"),
        Dependency(name="lm-eval @ git+https://github.com/nikil-ravi/lm-evaluation-harness.git@bpb-changes"),
    ]

    # Where to store checkpoints, cache inference results, etc.
    CACHE_PATH: str = "/tmp/levanter-lm-eval"

    @staticmethod
    def download_model(model: ModelConfig) -> str:
        """
        Download the model if it's not already downloaded
        """
        downloaded_path: str | None = model.ensure_downloaded(
            local_path=os.path.join(LevanterTpuEvaluator.CACHE_PATH, model.name)
        )
        # Use the model name if a path is not specified (e.g., for Hugging Face models)
        model_name_or_path: str = model.name if downloaded_path is None else downloaded_path
        return model_name_or_path

    @staticmethod
    def cleanup(model: ModelConfig) -> None:
        """
        Clean up resources.
        """
        logger.info("Cleaning up resources.")

        # Delete the checkpoint
        model.destroy()

    _python_version: str = "3.10"
    _pip_packages: ClassVar[list[Dependency]] = DEFAULT_PIP_PACKAGES
    _py_modules: ClassVar[list[Dependency]] = []

    def get_runtime_env(self) -> dict:
        """
        Returns the runtime environment to run the evaluator on the Ray cluster.
        """
        runtime_env: dict = {
            "pip": {
                "packages": [str(package) for package in self._pip_packages],
                "pip_check": False,
                "pip_version": f"==23.0.1;python_version=='{self._python_version}'",
            },
            "env_vars": {
                # Set an env variable needed for lm-eval-harness to trust remote code, required for some of the tasks
                "HF_DATASETS_TRUST_REMOTE_CODE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
        }

        # An empty list of py_modules can cause an error in Ray
        if len(self._py_modules) > 0:
            runtime_env["py_modules"] = [str(module) for module in self._py_modules]

        return runtime_env

    def launch_evaluate_with_ray(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
        resource_config: ResourceConfig | None = None,
    ) -> None:
        """
        Launches the evaluation run with Ray.
        """

        @ray.remote(
            scheduling_strategy=self.scheduling_strategy_fn(resource_config.num_tpu, resource_config.tpu_type),
            runtime_env=self.get_runtime_env(),
        )
        @remove_tpu_lockfile_on_exit
        def launch(
            model: ModelConfig,
            evals: list[EvalTaskConfig],
            output_path: str,
            max_eval_instances: int | None = None,
        ) -> None:
            self.evaluate(model, evals, output_path, max_eval_instances)

        ray.get(launch.remote(model, evals, output_path, max_eval_instances))
