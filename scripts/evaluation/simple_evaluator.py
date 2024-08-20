from dataclasses import dataclass
from typing import Dict, List
import os
import time

import ray

from scripts.evaluation.vllm_tpu_evaluator import VllmTpuEvaluator


@dataclass(frozen=True)
class TestPlan:
    """
    A test plan to run with `SimpleEvaluator`.
    """

    prompts: List[str]
    num_outputs: int
    max_tokens: int
    temperature: float = 0


class SimpleEvaluator(VllmTpuEvaluator):
    """
    A simple evaluator for testing purposes.
    Runs inference with a given model on some prompts and computes the total inference time.
    """

    QUICK_TEST_PLAN: TestPlan = TestPlan(
        prompts=[
            "A robot may not injure a human being",
            "It is only with the heart that one can see rightly;",
            "The greatest glory in living lies not in never falling,",
        ],
        num_outputs=1,
        max_tokens=16,
    )

    LONG_GENERATION_TEST_PLAN: TestPlan = TestPlan(
        prompts=[
            "Write an essay for me on the topic of 'The importance of education in society'.",
            "Write documentation on the VLLM library.",
        ],
        num_outputs=1,
        max_tokens=2000,
        temperature=0.5,
    )

    MANY_PROMPTS_TEST_PLAN: TestPlan = TestPlan(
        prompts=["Hello, my name is"] * 16,
        num_outputs=1,
        max_tokens=100,
        temperature=1.0,
    )

    MANY_OUTPUTS_TEST_PLAN: TestPlan = TestPlan(
        prompts=["I am not afraid of storms, for"],
        num_outputs=16,
        max_tokens=100,
        temperature=1.0,
    )

    NAME_TO_TEST_PLAN: Dict[str, TestPlan] = {
        "quick": QUICK_TEST_PLAN,
        "long": LONG_GENERATION_TEST_PLAN,
        "many_prompts": MANY_PROMPTS_TEST_PLAN,
        "many_outputs": MANY_OUTPUTS_TEST_PLAN,
    }

    @ray.remote(memory=8 * 1024 * 1024 * 1024, resources={"TPU": 4})  # 8 GB of memory, always request 4 TPUs
    def run(self, model_name_or_path: str, evals: List[str]) -> None:
        super().run(model_name_or_path, evals)

        from vllm import LLM, SamplingParams

        # Download the model from GCS if it is a GCS path
        model: str
        if SimpleEvaluator.is_gcs_path(model_name_or_path):
            SimpleEvaluator.download_from_gcs(model_name_or_path)
            model = os.path.split(model_name_or_path)[-1]
        else:
            model = model_name_or_path

        # Set `enforce_eager=True` to avoid ahead-of-time compilation.
        # In real workloads, `enforce_eager` should be `False`.
        llm = LLM(model=model, enforce_eager=False, trust_remote_code=True)

        inference_times: Dict[str, float] = {}
        for eval_name in evals:
            assert eval_name in SimpleEvaluator.NAME_TO_TEST_PLAN, f"Unknown eval: {eval_name}"
            test_plan: TestPlan = SimpleEvaluator.NAME_TO_TEST_PLAN[eval_name]

            # Set sampling parameters based on the test plan
            sampling_params = SamplingParams(
                temperature=test_plan.temperature,
                n=test_plan.num_outputs,
                max_tokens=test_plan.max_tokens,
                # Currently, top-p sampling is disabled. `top_p` should be 1.0.
                top_p=1.0,
            )

            # Run inference and time it
            start_time: float = time.time()
            outputs = llm.generate(test_plan.prompts, sampling_params)
            inference_times[eval_name] = time.time() - start_time

            # Print the outputs for debugging
            for output in outputs:
                prompt: str = output.prompt
                print(f"Prompt: {prompt!r}")
                for i, generation in enumerate(output.outputs):
                    print(f"Generation (#{i + 1} of {test_plan.num_outputs}): {generation.text!r}")
                print("-" * 100)

        print(f"Inference times (in seconds): {inference_times}")
