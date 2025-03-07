import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from experiments.evals.resource_configs import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.utils import download_from_gcs, is_remote_path


@dataclass(frozen=True)
class Dependency:
    """Represents a Python dependency e.g., transformers==4.9.2"""

    name: str
    """The name of the dependency e.g., transformers"""

    version: str | None = None
    """The version of the dependency e.g., 4.9.2"""

    def __str__(self):
        return f"{self.name}=={self.version}" if self.version else self.name


@dataclass
class ModelConfig:
    name: str
    """The name of the model e.g., allenai/olmo-7b"""

    path: str | None
    """
    The path to the model checkpoint. Can be a local path or a path on GCS.
    """

    engine_kwargs: dict | None = None
    """
    Additional keyword arguments to pass to the vLLM engine.
    """

    def ensure_downloaded(self, local_path: str | None = None) -> str | None:
        """
        Ensures that the model checkpoint is downloaded to `local_path` if necessary.
        """
        if self.path is None:
            return None
        elif is_remote_path(self.path):
            assert local_path is not None
            download_from_gcs(gcs_path=self.path, destination_path=local_path)
            self.path = local_path
            # Show the contents of self.path
            print(f"Downloaded model checkpoint to {self.path}: {os.listdir(self.path)}")
            return local_path

    def destroy(self) -> None:
        """Deletes the model checkpoint."""
        if self.path and os.path.exists(self.path):
            shutil.rmtree(self.path, ignore_errors=True)
            print(f"Deleted local checkpoint at {self.path}.")


class Evaluator(ABC):

    _python_version: str
    _pip_packages: ClassVar[list[Dependency]]
    _py_modules: ClassVar[list[Dependency]]

    def scheduling_strategy_fn(self, tensor_parallel_size: int, tpu_type: str):
        import ray
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        # One bundle per tensor parallel worker
        pg = ray.util.placement_group(
            [{"TPU": 1, "CPU": 1}] * tensor_parallel_size + [{f"{tpu_type}-head": 1}],
            strategy="STRICT_PACK",  # STRICT_PACK means same node
        )
        return PlacementGroupSchedulingStrategy(pg, placement_group_capture_child_tasks=True)

    @abstractmethod
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

        Args:
            model (ModelConfig): The model configuration of the model we want to evaluate
            evals (List[EvalTaskConfig]): The list of evaluations to run.
            output_path (str): The path to save the evaluation results.
            max_eval_instances (int | None): The maximum number of evaluation instances to run.
            step (ExecutorStep | None): The step to evaluate. Used to get the config for the model and the trainer.
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        model: ModelConfig,
        evals: list[EvalTaskConfig],
        output_path: str,
        max_eval_instances: int | None = None,
    ) -> None:
        """What to run to evaluate."""
        pass
