import os
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import subprocess


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

    output_path: str
    """
    The path to save the evaluation results e.g., /path/to/output
    or can be path on GCS gs://bucket/path/to/output.
    """

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
        return (
            f"EvaluatorConfig(name={self.name}, "
            f"output_path={self.output_path}, credentials_path={self.credentials_path})"
        )

    @property
    def hf_auth_token(self) -> Optional[str]:
        """
        Returns the Hugging Face authentication token if it exists.
        """
        return self._credentials.get("HuggingFaceAuthToken")


class Evaluator(ABC):

    _python_version: str
    _pip_packages: List[Dependency]
    _py_modules: List[Dependency]

    def __init__(self, config: EvaluatorConfig):
        self._config: EvaluatorConfig = config

    @staticmethod
    def run_bash_command(command: str, check: bool = True) -> None:
        """Runs a bash command."""
        print(command)
        subprocess.run(command, shell=True, check=check)

    @abstractmethod
    def evaluate(self, model_name_or_path: str, evals: List[str]) -> None:
        """
        Runs the evaluator given the model checkpoint, the list of evaluations to run, and the output path.
        """
        pass

    def authenticate_with_hf(self) -> None:
        """Authenticates with the Hugging Face API using the given token."""
        hf_auth_token: Optional[str] = self._config.hf_auth_token
        if hf_auth_token is None:
            print("WARNING: Skipping logging on with HuggingFace. No token provided.")
        else:
            from huggingface_hub import login

            login(token=self._config.hf_auth_token)
            print("Logged in with Hugging Face.")
