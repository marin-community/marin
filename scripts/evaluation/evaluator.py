from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass


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

    @abstractmethod
    def evaluate(self, model_gcs_path: str, evals: List[str], output_path: str) -> None:
        """
        Run the evaluator given the model checkpoint, the list of evaluations to run, and the output path.
        """
        pass

    def get_runtime_env(self) -> Dict:
        """
        Returns the runtime environment to run the evaluator on the Ray cluster.
        """
        runtime_env = {
            "pip": {
                "packages": [str(package) for package in self._pip_packages],
                "pip_check": False,
                "pip_version": f"==23.0.1;python_version=='{self._python_version}'",
            },
        }

        # An empty list of py_modules can cause an error in Ray
        if len(self._py_modules) > 0:
            runtime_env["py_modules"] = [str(module) for module in self._py_modules]

        return runtime_env
