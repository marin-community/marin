from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.sources.programs import live_code_records
from eval.sources.common import text_specs

import logging
from itertools import islice
import os
from typing import List, Optional

from datasets import load_dataset

from eval.task import BaseBenchmark


HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE")


# Calculate mean and standard error for all metrics


class LiveCodeBenchBenchmark(BaseBenchmark):
    """
    LiveCodeBench Benchmark for evaluating the math reasoning of LLMs.

    Follows the evaluation logic of hendrycks_math answer extraction.
    """
    SOURCE = SourceRef("livecodebench/code_generation_lite", revision="release_v2")
    METRICS = ("accuracy_avg",)
    PRIMARY_METRIC = "accuracy_avg"
    REDUCER = ReducerSpec(accuracy="accuracy_avg", stderr="accuracy_std_err", repeat_ddof=1, group_suffix="_avg")

    REPEATS = 6

    def __init__(
        self,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize LiveCodeBench benchmark.

        Args:
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.debug = debug
        self.max_new_tokens = max_tokens
        self.seed = seed
        self.n_repeat = self.REPEATS

    def iter_specs(self, options):
        def prompt(example):
            instruction = (
                "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."
                if example["is_stdin"] else
                "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."
            )
            return instruction + example["prompt"]

        examples = self.load_questions()
        if self.debug:
            examples = islice(examples, 10)
        return text_specs(
            examples, options, prompt=prompt, target=lambda example: example["test"],
            grader="live_code", generation_kwargs={"do_sample": False, "max_gen_toks": self.max_new_tokens},
            repeats=self.n_repeat, tags=lambda example: {"difficulty": example["difficulty"]},
        )


    def load_questions(self):
        rows = load_dataset(
            "livecodebench/code_generation_lite",
            version_tag="release_v2",
            split="test",
            trust_remote_code=True,
            cache_dir=HF_HUB_CACHE,
        )
        return live_code_records(rows)
