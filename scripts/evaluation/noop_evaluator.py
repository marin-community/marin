from typing import List

from scripts.evaluation.evaluator import Evaluator, Dependency

import ray


class NoopEvaluator(Evaluator):
    """
    A no-op evaluator for testing purposes.
    Doesn't evaluate anything but just ensures that the environment is setup correctly
    by running some simple model inference.
    """

    _python_version: str = "3.10"
    _pip_packages: List[Dependency] = [
        Dependency(
            name="https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/"
            "torch-nightly+20240601-cp310-cp310-linux_x86_64.whl"
        ),
        Dependency(
            name="https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/"
            "torch_xla-nightly+20240601-cp310-cp310-linux_x86_64.whl"
        ),
        Dependency(
            name="https://storage.googleapis.com/libtpu-nightly-releases/wheels/libtpu-nightly/"
            "libtpu_nightly-0.1.dev20240527%2Bdefault-py3-none-any.whl"
        ),
        Dependency(
            name="https://storage.googleapis.com/jax-releases/nightly/jax/jax-0.4.29.dev20240527-py3-none-any.whl"
        ),
        Dependency(
            name="https://storage.googleapis.com/jax-releases/nightly/nocuda/"
            "jaxlib-0.4.29.dev20240527-cp310-cp310-manylinux2014_x86_64.whl",
        ),
        Dependency(
            name="https://download.pytorch.org/whl/cpu/torchvision-0.19.0%2Bcpu-cp310-cp310-linux_x86_64.whl",
        ),
    ]
    _py_modules: List[Dependency] = []

    @staticmethod
    @ray.remote(memory=1 * 1024 * 1024 * 1024)  # 1 GB
    def _run_something(model_gcs_path: str, evals: List[str], output_path: str) -> None:
        # Following the example code here:
        # https://docs.vllm.ai/en/latest/getting_started/examples/offline_inference_tpu.html
        from vllm import LLM, SamplingParams

        print("Running some simple model inference.")
        prompts = [
            "A robot may not injure a human being",
            "It is only with the heart that one can see rightly;",
            "The greatest glory in living lies not in never falling,",
        ]
        answers = [
            " or, through inaction, allow a human being to come to harm.",
            " what is essential is invisible to the eye.",
            " but in rising every time we fall.",
        ]
        N = 1
        # Currently, top-p sampling is disabled. `top_p` should be 1.0.
        sampling_params = SamplingParams(temperature=0.7, top_p=1.0, n=N, max_tokens=16)

        # Set `enforce_eager=True` to avoid ahead-of-time compilation.
        # In real workloads, `enforace_eager` should be `False`.
        llm = LLM(model="google/gemma-2b", enforce_eager=True)
        print("Model loaded.")

        outputs = llm.generate(prompts, sampling_params)
        print("Model inference done")
        for output, answer in zip(outputs, answers):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            assert generated_text.startswith(answer)

    def evaluate(self, model_path: str, evals: List[str], output_path: str) -> None:
        """
        Run the evaluation harness.
        """
        print(f"Running {evals} on {model_path} and saving results to {output_path}...")
        ray.init(runtime_env=self.get_runtime_env())
        print("Ray initialized with dependencies.")
        ray.get(self._run_something.remote(model_path, evals, output_path))
