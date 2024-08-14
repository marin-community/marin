from abc import ABC
from typing import Dict, List

from scripts.evaluation.evaluator import Evaluator, Dependency


class VllmTpuEvaluator(Evaluator, ABC):
    """For `Evaluator`s that runs inference with VLLM on TPUs."""

    # Default pip packages to install for VLLM on TPUs
    # Some versions were fixed in order to resolve dependency conflicts.
    DEFAULT_PIP_PACKAGES: List[Dependency] = [
        Dependency(
            name="https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-nightly+20240726"
            "-cp310-cp310-linux_x86_64.whl",
        ),
        Dependency(
            name="https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly+20240726"
            "-cp310-cp310-linux_x86_64.whl",
        ),
        Dependency(name="aiohttp"),
        Dependency(name="attrs", version="22.2.0"),
        Dependency(name="click", version="8.1.3"),
        Dependency(name="jsonschema", version="4.23.0"),
        Dependency(name="packaging"),
        Dependency(name="starlette", version="0.37.2"),
        Dependency(name="tokenizers", version="0.19.1"),
        Dependency(name="transformers", version="4.43.2"),
    ]

    # VLLM version to install. TODO: Hardcoded for now. Make it configurable.
    # Visit https://github.com/vllm-project/vllm/releases to get the list of available versions.
    VLLM_VERSION: str = "v0.5.4"

    @staticmethod
    def install_vllm_from_source() -> None:
        """
        Runs the necessary commands to install VLLM from source, following the instructions here:
        https://docs.vllm.ai/en/v0.5.0.post1/getting_started/tpu-installation.html
        TPUs require installing VLLM from source.
        """
        # Additional dependencies to install in order for VLLM to work on TPUs
        VllmTpuEvaluator.run_bash_command("sudo apt-get update && sudo apt-get install libopenblas-dev --yes")
        VllmTpuEvaluator.run_bash_command(
            "pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html"
        )
        VllmTpuEvaluator.run_bash_command(
            "pip install torch_xla[pallas] "
            "-f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html "
            "-f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html"
        )
        # Clone the VLLM repository to install it from source. Can fail if the repository already exists.
        VllmTpuEvaluator.run_bash_command("git clone https://github.com/vllm-project/vllm.git", check=False)
        # Runs https://github.com/vllm-project/vllm/blob/main/setup.py with the `tpu` target device
        VllmTpuEvaluator.run_bash_command(
            f"cd vllm && git checkout tags/{VllmTpuEvaluator.VLLM_VERSION} "
            '&& VLLM_TARGET_DEVICE="tpu" pip install -e .'
        )

    _python_version: str = "3.10"
    _pip_packages: List[Dependency] = DEFAULT_PIP_PACKAGES
    _py_modules: List[Dependency] = []

    def get_runtime_env(self) -> Dict:
        """
        Returns the runtime environment to run the evaluator on the Ray cluster.
        """
        runtime_env: Dict = {
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
