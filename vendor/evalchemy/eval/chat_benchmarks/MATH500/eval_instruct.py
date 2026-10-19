from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.sources.common import text_specs

import json
import logging
import os
from typing import Any, Dict, List, Optional  # noqa: F401


from eval.task import BaseBenchmark
from eval.generation_stops import END_OF_TURN_SEQUENCES

# Modified version of hendrycks_math with additional instruction to mark the solution with \\boxed
# https://github.com/mlfoundations/evalchemy/blob/e70a45e41cb2ada273d6bb98e75dba303ec31f8b/eval/chat_benchmarks/AMC23/eval_instruct.py#L15
PROMPT = """Problem: {problem}\nMark your solution with \\boxed\nAnswer:"""


class MATH500Benchmark(BaseBenchmark):
    SOURCE = SourceRef("HuggingFaceH4/MATH-500", "test")
    """
    MATH500 Benchmark for evaluating the math reasoning of LLMs.
    Link: https://huggingface.co/datasets/HuggingFaceH4/MATH-500

    Uses shared boxed-answer extraction and mathematical equivalence grading.
    """
    REDUCER = ReducerSpec()

    METRICS = ("accuracy",)
    PRIMARY_METRIC = "accuracy"

    def benchmark_size(self) -> int:
        return len(self.load_questions())

    def __init__(
        self,
        data_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "math500.jsonl"),
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        num_samples: int = 1,
        pass_at_k: Optional[Any] = None,
    ):
        """
        Initialize MATH500 benchmark.

        Args:
            data_file: File containing the MATH500 dataset (id, problem, reference_solution, expected_answer, source)
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
            num_samples: Number of completions per problem. 1 (default) = today's single-sample path.
            pass_at_k: k-list for pass@k aggregation (only used when num_samples > 1).
        """
        super().__init__(
            logger=logger,
            system_instruction=system_instruction,
            num_samples=num_samples,
            pass_at_k=pass_at_k,
        )
        self.data_file = data_file
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.n_repeat = 1

    def iter_specs(self, options):
        return text_specs(
            self.load_questions(), options,
            prompt=lambda example: PROMPT.format(problem=example['problem']),
            target=lambda example: example['answer'], grader="boxed_math",
            generation_kwargs={"do_sample": False, "temperature": 0.7, "max_gen_toks": self.max_new_tokens,
                        "until": list(END_OF_TURN_SEQUENCES)},
            repeats=self.n_repeat,
        )


    def load_questions(self) -> List[Dict[str, str]]:
        """Load MATH500 questions from the data file."""
        with open(self.data_file, "r") as f:
            questions = [json.loads(x) for x in f]

        if self.debug:
            questions = questions[:2]
            self.logger.info(f"Debug mode enabled. Using only {len(questions)} questions.")

        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")
        return questions
