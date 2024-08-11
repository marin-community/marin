from typing import List

import ray

from scripts.evaluation.evaluator import Evaluator, Dependency


class SimpleInferenceEvaluator(Evaluator):
    """
    A simple inference evaluator for testing purposes.
    Runs inference and compute some simple metrics to ensure that VLLM on TPUs is working.
    """

    _python_version: str = "3.10"
    _pip_packages: List[Dependency] = [
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
        Dependency(name="jsonschema"),
        Dependency(name="packaging"),
        Dependency(name="starlette", version="0.37.2"),
        Dependency(name="tokenizers", version="0.19.1"),
        Dependency(name="transformers", version="4.43.2"),
    ]
    _py_modules: List[Dependency] = []

    @staticmethod
    @ray.remote(memory=1 * 1024 * 1024 * 1024)  # 1 GB
    def _run_something(model_path: str, evals: List[str], output_path: str) -> None:
        # Install from VLLM source
        # Additional dependencies to install in order for VLLM to work on TPUs
        # From https://docs.vllm.ai/en/v0.5.0.post1/getting_started/tpu-installation.html
        SimpleInferenceEvaluator.run_bash_command("sudo apt-get update && sudo apt-get install libopenblas-dev --yes")
        SimpleInferenceEvaluator.run_bash_command(
            "pip install torch_xla[tpu] " "-f https://storage.googleapis.com/libtpu-releases/index.html"
        )
        SimpleInferenceEvaluator.run_bash_command(
            "pip install torch_xla[pallas] "
            "-f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html "
            "-f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html"
        )
        SimpleInferenceEvaluator.run_bash_command("git clone https://github.com/vllm-project/vllm.git")
        # Runs https://github.com/vllm-project/vllm/blob/main/setup.py
        SimpleInferenceEvaluator.run_bash_command('cd vllm && VLLM_TARGET_DEVICE="tpu" python setup.py develop')

        import vllm

        print("file: " + vllm.__file__)

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
        # Currently, top-p sampling is disabled. `top_p` should be 1.0.
        sampling_params = SamplingParams(temperature=0.7, top_p=1.0, n=1, max_tokens=16)

        # Set `enforce_eager=True` to avoid ahead-of-time compilation.
        # In real workloads, `enforce_eager` should be `False`.
        llm = LLM(model="google/gemma-2b", enforce_eager=True)
        print("Model loaded.")

        outputs = llm.generate(prompts, sampling_params)
        for output, answer in zip(outputs, answers):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            assert generated_text.startswith(answer)

        # TODO: return some notion of stats

    def evaluate(self, model_path: str, evals: List[str], output_path: str) -> None:
        """
        Run the evaluator.
        """
        print(f"Running {evals} on {model_path} and saving results to {output_path}...")
        ray.init(runtime_env=self.get_runtime_env())
        ray.get(self._run_something.remote(model_path, evals, output_path))
