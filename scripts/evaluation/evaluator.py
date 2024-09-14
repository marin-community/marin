import os
import json
import shutil
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from scripts.evaluation.utils import download_from_gcs, is_remote_path


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
class EvaluatorConfig:
    name: str
    """The name of the evaluator e.g., helm"""

    credentials_path: str
    """
    The path to file containing the credentials e.g., Hugging Face authentication token.
    """

    _credentials: Dict[str, str] = field(default_factory=dict)
    """
    The credentials to use for the evaluator.
    """

    def __post_init__(self) -> None:
        if os.path.exists(self.credentials_path):
            with open(self.credentials_path, "r") as f:
                self._credentials = json.load(f)
                print(f"Loaded credentials from {self.credentials_path}.")
        else:
            print(f"WARNING: No credentials found at {self.credentials_path}")

    def __str__(self) -> str:
        return f"EvaluatorConfig(name={self.name}, credentials_path={self.credentials_path})"

    @property
    def hf_auth_token(self) -> Optional[str]:
        """
        Returns the Hugging Face authentication token if it exists.
        """
        return self._credentials.get("HuggingFaceAuthToken")


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
        """
        Deletes the model checkpoint if it was downloaded from GCS.
        """
        if os.path.exists(self.path):
            shutil.rmtree(self.path, ignore_errors=True)
            print(f"Deleted local checkpoint at {self.path}.")


class Evaluator(ABC):

    _python_version: str
    _pip_packages: List[Dependency]
    _py_modules: List[Dependency]

    def __init__(self, config: EvaluatorConfig):
        self._config: EvaluatorConfig = config

    @abstractmethod
    def evaluate(self, model: ModelConfig, evals: List[str], output_path: str) -> None:
        """
        Runs the evaluator given the model checkpoint, the list of evaluations to run, and the output path.
        """
        pass
