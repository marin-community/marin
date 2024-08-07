from typing import List

from scripts.evaluation.evaluation_harness import EvaluationHarness, Dependency

import ray


class NoopEvaluationHarness(EvaluationHarness):
    """
    A no-op evaluation harness for testing purposes.
    Doesn't evaluate anything but just ensures that the environment is setup correctly
    by running some simple model inference.
    """

    _python_version: str = "3.9"
    _pip_packages: List[Dependency] = [
        Dependency(name="transformers", version="4.44.0"),
        Dependency(
            name="https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/"
            "torch-nightly+20240601-cp310-cp310-linux_x86_64.whl"
        ),
        Dependency(
            name="https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/"
            "torch_xla-nightly+20240601-cp310-cp310-linux_x86_64.whl"
        ),
    ]
    _py_modules: List[Dependency] = []

    @staticmethod
    @ray.remote(memory=1 * 1024 * 1024 * 1024)  # 1 GB
    def _run_something(model_gcs_path: str, evals: List[str], output_path: str) -> None:
        import torch_xla.core.xla_model as xm
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = xm.xla_device()
        print(f"Device: {device}")

        model_path: str = "gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        prompt: str = "def hello_world():"
        input = tokenizer([prompt], return_tensors="pt").to(device)

        generated_ids = model.generate(**input, max_new_tokens=100)
        print(tokenizer.batch_decode(generated_ids)[0])

    def evaluate(self, model_gcs_path: str, evals: List[str], output_path: str) -> None:
        """
        Run the evaluation harness.
        """
        ray.init(runtime_env=self.get_runtime_env())
        ray.get(self._run_something.remote(model_gcs_path, evals, output_path))
