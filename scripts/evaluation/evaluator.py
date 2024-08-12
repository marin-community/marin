from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

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


class Evaluator(ABC):

    _python_version: str
    _pip_packages: List[Dependency]
    _py_modules: List[Dependency]

    @staticmethod
    def run_bash_command(command: str) -> None:
        """Runs a bash command."""
        subprocess.run(command, shell=True, check=True)

    @abstractmethod
    def evaluate(self, model_name_or_path: str, evals: List[str], output_path: str) -> None:
        """
        Runs the evaluator given the model checkpoint, the list of evaluations to run, and the output path.
        """
        pass
