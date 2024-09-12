from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import os
import requests
import subprocess
import sys
import time

import ray

from scripts.evaluation.evaluator import Evaluator, Dependency, ModelConfig
from scripts.evaluation.utils import run_bash_command, kill_process_on_port


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
        Dependency(name="transformers", version="4.44.0"),
        # Marin-specific dependencies
        # The olmo library did not set `exists_ok=True` at
        # https://github.com/allenai/OLMo/blob/main/hf_olmo/configuration_olmo.py#L43
        # which causes the following error:
        # ValueError: 'olmo' is already used by a Transformers config, pick another name.
        # Use the following dependency instead where the issue is fixed.
        Dependency(name="git+https://github.com/teetone/OLMo.git@main"),
    ]

    # VLLM version to install. TODO: Hardcoded for now. Make it configurable.
    # Visit https://github.com/vllm-project/vllm/releases to get the list of available versions.
    VLLM_VERSION: str = "v0.5.4"

    # Where to store checkpoints, cache inference results, etc.
    CACHE_PATH: str = "/tmp"

    @staticmethod
    def install_vllm_from_source() -> None:
        """
        Runs the necessary commands to install VLLM from source, following the instructions here:
        https://docs.vllm.ai/en/v0.5.0.post1/getting_started/tpu-installation.html
        TPUs require installing VLLM from source.
        """
        # Additional dependencies to install in order for vLLM to work on TPUs
        start_time: float = time.time()
        run_bash_command("sudo apt-get update && sudo apt-get install libopenblas-dev --yes")
        run_bash_command("pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html")
        run_bash_command(
            "pip install torch_xla[pallas] "
            "-f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html "
            "-f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html"
        )
        # Clone the VLLM repository to install it from source. Can fail if the repository already exists.
        run_bash_command("git clone https://github.com/vllm-project/vllm.git", check=False)
        # Runs https://github.com/vllm-project/vllm/blob/main/setup.py with the `tpu` target device
        run_bash_command(f"cd vllm && git checkout tags/{VllmTpuEvaluator.VLLM_VERSION}")
        run_bash_command('VLLM_TARGET_DEVICE="tpu" pip install -e ./vllm')

        # Get the path to the vllm directory and add the path to sys.path and PYTHONPATH
        current_dir: str = os.path.dirname(os.path.abspath(__file__))
        vllm_path: str = os.path.join(current_dir, "../../vllm")
        sys.path.insert(0, vllm_path)
        os.environ["PYTHONPATH"] = f"{vllm_path}:{os.environ.get('PYTHONPATH', '')}"
        elapsed_time: float = time.time() - start_time
        print(f"Installed vLLM and dependencies. ({elapsed_time}s)")

        # To allow us to specify a really large value for model length for vLLM
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    @staticmethod
    def start_vllm_server_in_background(
        model: ModelConfig, host: str = "127.0.0.1", port: int = 8000, timeout_seconds: int = 3600
    ) -> str:
        """
        Serve the model with a local vLLM server in the background.
        Returns the port the server is running on.
        """
        # Download the model if it's not already downloaded
        downloaded_path: Optional[str] = model.ensure_downloaded(
            local_path=os.path.join(VllmTpuEvaluator.CACHE_PATH, model.name)
        )
        # Use the model name if a path is not specified (e.g., for Hugging Face models)
        model_name_or_path: str = model.name if downloaded_path is None else downloaded_path

        # From https://docs.vllm.ai/en/v0.4.0/models/engine_args.html
        # Set `max_model_len` to a large value to avoid vLLM getting the value from the model config.
        # This value has to be a multiple of 512.
        command: str = (
            f"vllm serve {model_name_or_path} --trust-remote-code --host {host} --port {port} "
            f"--max-model-len {512 * 20}"
        )
        process = subprocess.Popen(command, shell=True)

        # Check that the server has started by sending heartbeat checks
        server_url: str = f"http://{host}:{port}/v1"
        start_time: float = time.time()
        elapsed_time: float = 0
        while True:
            try:
                # Attempt to send a request to the server's health endpoint
                response = requests.get(f"{server_url}/models")
                if response.status_code == 200:
                    raw_response: Dict = response.json()
                    loaded_models: List[str] = [model["id"] for model in raw_response["data"]]

                    # Can be on a machine with a vLLM server up and running, so also check the model is loaded
                    print(f"vLLM server is up and running at {server_url}: {response.text}")
                    if model_name_or_path in loaded_models:
                        print(f"Model {model_name_or_path} is loaded.")
                        break
                    else:
                        print(f"Model {model_name_or_path} is not loaded yet. Loaded models: {loaded_models}")
            except requests.ConnectionError:
                # If the connection is refused, wait and try again
                print(f"vLLM server is not ready yet. Elapsed time in seconds: {elapsed_time}")

            # Check if the timeout has been reached
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                process.kill()
                raise TimeoutError("Failed to start vLLM server within timeout period.")

            time.sleep(5)  # Wait 5 seconds before retrying

        print(f"vLLM server is ready at {server_url} ({elapsed_time}s).")
        return server_url

    @staticmethod
    def cleanup(model: ModelConfig, vllm_port: Optional[int] = None) -> None:
        """
        Clean up the vLLM server and any other resources.
        """
        print("Cleaning up resources.")
        # Kill the vLLM server
        try:
            if vllm_port is not None:
                kill_process_on_port(vllm_port)
        except Exception as e:
            print(f"Failed to kill vLLM server on port {vllm_port}: {e}")

        # Delete the checkpoint
        model.destroy()

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

    @abstractmethod
    def run(self, model: ModelConfig, evals: List[str], output_path: str) -> None:
        """
        Run the evaluator.
        """
        # General setup:
        # Install VLLM from source
        self.install_vllm_from_source()

    def evaluate(self, model: ModelConfig, evals: List[str], output_path: str) -> None:
        """
        Launches the evaluation run with Ray.
        """
        ray.init(runtime_env=self.get_runtime_env())
        result = ray.get(self.run.remote(self, model, evals, output_path))
        print(f"Inference times (in seconds): {result}")
