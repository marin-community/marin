from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.sources.common import text_specs

import json
import logging
import os
from typing import Dict, List, Optional

from eval.task import BaseBenchmark

# Modified version of hendrycks_math with additional instruction to mark the solution with \\boxed
# https://github.com/mlfoundations/evalchemy/blob/e70a45e41cb2ada273d6bb98e75dba303ec31f8b/eval/chat_benchmarks/AMC23/eval_instruct.py#L15
PROMPT = """Problem: {problem}\nMark your solution with \\boxed\nAnswer:"""


class AMC23Benchmark(BaseBenchmark):
    SOURCE = SourceRef("zwhe99/amc23", "train")
    METRICS = ("accuracy_avg",)
    PRIMARY_METRIC = "accuracy_avg"
    """
    AMC23 Benchmark for evaluating the math reasoning of LLMs.
    Link: https://huggingface.co/datasets/zwhe99/amc23

    Follows the evaluation logic of hendrycks_math answer extraction.
    Added additional instruction to the prompt to mark the solution with \\boxed.
    """
    REDUCER = ReducerSpec(accuracy="accuracy_avg", stderr="accuracy_std_err")

    def __init__(
        self,
        data_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "amc23.json"),
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize AMC23 benchmark.

        Args:
            data_file: File containing the AMC23 dataset (id, problem, reference_solution, expected_answer, source)
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.n_repeat = 10

    def iter_specs(self, options):
        return text_specs(
            self.load_questions(), options,
            prompt=lambda example: PROMPT.format(problem=example['question']),
            target=lambda example: example['answer'], grader="boxed_math",
            generation_kwargs={"do_sample": False, "temperature": 0.7, "max_gen_toks": self.max_new_tokens},
            repeats=self.n_repeat,
            reference=lambda _row: {"extractor": "hendrycks_math", "comparator": "hendrycks_math"},
        )


    def load_questions(self) -> List[Dict[str, str]]:
        """Load AMC23 questions from the data file."""
        with open(self.data_file, "r") as f:
            questions = [json.loads(x) for x in f]

        if self.debug:
            questions = questions[:2]
            self.logger.info(f"Debug mode enabled. Using only {len(questions)} questions.")
 
        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")
        return questions
