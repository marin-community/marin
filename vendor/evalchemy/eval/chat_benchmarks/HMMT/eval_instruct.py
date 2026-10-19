from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.sources.common import text_specs

import logging
from typing import Dict, List, Optional


from datasets import load_dataset
from eval.task import BaseBenchmark


# Modified version of hendrycks_math with additional instruction to mark the solution with \\boxed
PROMPT = """Problem: {problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nAnswer:"""


class HMMTBenchmark(BaseBenchmark):
    SOURCE = SourceRef("MathArena/hmmt_feb_2025", "train")
    METRICS = ("accuracy_avg",)
    PRIMARY_METRIC = "accuracy_avg"
    """
    HMMT Benchmark for evaluating the math reasoning of LLMs.
    https://huggingface.co/datasets/MathArena/hmmt_feb_2025

    Follows the evaluation logic of hendrycks_math answer extraction.
    """
    REDUCER = ReducerSpec(accuracy="accuracy_avg", stderr="accuracy_std_err")

    def __init__(
        self,
        dataset_name: str = "MathArena/hmmt_feb_2025",
        debug: bool = False,
        max_tokens: int = 32768,
        seed: List[int] = [0, 1234, 1234, 1234],
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize HMMT benchmark.

        Args:
            dataset_name: Dataset containing the HMMT dataset (id, answer)
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.dataset_name = dataset_name
        self.debug = debug
        self.max_new_tokens = max_tokens
        self.seed = seed
        self.n_repeat = 10

    def iter_specs(self, options):
        return text_specs(
            self.load_questions(), options,
            prompt=lambda example: PROMPT.format(problem=example["problem"]),
            target=lambda example: example["answer"], grader="matharena",
            generation_kwargs={"do_sample": False, "max_gen_toks": self.max_new_tokens},
            repeats=self.n_repeat,
        )


    def load_questions(self) -> List[Dict[str, str]]:
        """Load HMMT questions from the configured Hugging Face dataset."""
        questions = [dict(example) for example in load_dataset(self.dataset_name, split="train")]
        if self.debug:
            questions = questions[:2]
            self.logger.info("Debug mode enabled. Using only %s questions.", len(questions))
        return questions
