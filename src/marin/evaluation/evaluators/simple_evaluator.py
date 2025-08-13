import time
import traceback
from dataclasses import dataclass
from typing import ClassVar

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.evaluators.vllm_tpu_evaluator import VllmTpuEvaluator


@dataclass(frozen=True)
class TestPlan:
    """
    A test plan to run with `SimpleEvaluator`.
    """

    prompts: list[str]
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

    LONG_INPUT_TEST_PLAN = TestPlan(
        # Sample 5-shot prompt from MMLU
        prompts=[
            "Answer with only a single letter.\n"
            "The following are multiple choice questions (with answers) about abstract algebra.\n\n"
            "Question: Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. "
            "Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.\n"
            "A. True, True\n"
            "B. False, False\n"
            "C. True, False\n"
            "D. False, True\n"
            "Answer: B\n\n"
            "Question: Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\n"
            "A. 0\n"
            "B. 1\n"
            "C. 2\n"
            "D. 3\n"
            "Answer: B\n\n"
            "Question: Find the characteristic of the ring 2Z.\n"
            "A. 0\n"
            "B. 3\n"
            "C. 12\n"
            "D. 30\n"
            "Answer: A\n\n"
            "Question: Statement 1| Every function from a finite set onto itself must be one to one. "
            "Statement 2 | Every subgroup of an abelian group is abelian.\n"
            "A. True, True\n"
            "B. False, False\n"
            "C. True, False\n"
            "D. False, True\n"
            "Answer: A\n\n"
            "Question: Statement 1 | Every element of a group generates a cyclic subgroup of the group. "
            "Statement 2 | The symmetric group S_10 has 10 elements.\n"
            "A. True, True\n"
            "B. False, False\n"
            "C. True, False\n"
            "D. False, True\n"
            "Answer: C\n\n"
            "Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\n"
            "A. 0\n"
            "B. 4\n"
            "C. 2\n"
            "D. 6\n"
            "Answer:",
        ]
        * 16,
        num_outputs=1,
        max_tokens=1,
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

    NAME_TO_TEST_PLAN: ClassVar[dict[str, TestPlan]] = {
        "quick": QUICK_TEST_PLAN,
        "long": LONG_GENERATION_TEST_PLAN,
        "long_input": LONG_INPUT_TEST_PLAN,
        "many_prompts": MANY_PROMPTS_TEST_PLAN,
        "many_outputs": MANY_OUTPUTS_TEST_PLAN,
    }

    def evaluate(
        self,
        model: ModelConfig,
        evals: list[str],
        output_path: str,
        max_eval_instances: int | None = None,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams

            # Download and load the model with vLLM
            # Use the model name if a path is not specified (e.g., for Hugging Face models)
            model_name_or_path: str = self.download_model(model)
            llm = LLM(model=model_name_or_path, enforce_eager=False, trust_remote_code=True)

            inference_times: dict[str, float] = {}
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
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError("SimpleEvaluator failed. Please check the logs for more information.") from e
        finally:
            self.cleanup(model)
