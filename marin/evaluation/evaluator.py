import os
import shutil
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

from marin.evaluation.utils import download_from_gcs, is_remote_path


@dataclass(frozen=True)
class Dependency:
    """Represents a Python dependency e.g., transformers==4.9.2"""

    name: str
    """The name of the dependency e.g., transformers"""

    version: Optional[str] = None
    """The version of the dependency e.g., 4.9.2"""

    def __str__(self):
        return f"{self.name}=={self.version}" if self.version else self.name


@dataclass
class ModelConfig:
    name: str
    """The name of the model e.g., allenai/olmo-7b"""

    path: Optional[str]
    """
    The path to the model checkpoint. Can be a local path or a path on GCS.
    """

    def ensure_downloaded(self, local_path: Optional[str] = None) -> Optional[str]:
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
        if os.path.exists(self.path):
            shutil.rmtree(self.path, ignore_errors=True)
            print(f"Deleted local checkpoint at {self.path}.")


class Evaluator(ABC):

    _python_version: str
    _pip_packages: List[Dependency]
    _py_modules: List[Dependency]

    @abstractmethod
    def evaluate(self, model: ModelConfig, evals: List[str], output_path: str, max_eval_instances: int | None) -> None:
        """
        Runs the evaluator given the model checkpoint, the list of evaluations to run, and the output path.

        Args:
            model (ModelConfig): The model configuration of the model we want to evaluate
            evals (List[str]): The list of evaluations to run.
            output_path (str): The path to save the evaluation results.
            max_eval_instances (int | None): The maximum number of evaluation instances to run.
        """
        pass
