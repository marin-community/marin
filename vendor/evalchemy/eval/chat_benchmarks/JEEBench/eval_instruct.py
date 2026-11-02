from __future__ import annotations
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.sources.common import text_specs
import logging
import os
from typing import List, Optional

from datasets import Dataset, load_dataset

from eval.task import BaseBenchmark

########################################################################

# Adapted from https://github.com/dair-iitd/jeebench/blob/main/inference.py


PROMPT_LIBRARY = {
    "MCQ": "In this problem, only one option will be correct. Give a detailed solution and end the solution with the final answer.",
    "MCQ(multiple)": "In this problem, multiple options can be correct. Give a detailed solution and end the solution with the final answer.",
    "Integer": "In this problem, the final answer will be a non-negative integer. Give a detailed solution and end the solution with the final answer.",
    "Numeric": "In this problem, the final will be a numeric value. Give the numerical answer correct upto the 2nd decimal digit. Give a detailed solution and end the solution with the final answer.",
}

HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE")
if not HF_HUB_CACHE:
    print(
        "WARNING: HF_HUB_CACHE environment variable is not set, using default cache directory ~/.cache/huggingface/hub for JEEBench benchmark"
    )


def format_message(question, prompt_library):
    prefix_prompt = prompt_library[question["type"]]
    suffix_prompt = ""

    stripped_question = question["question"].replace("\n\n", "\n").strip()

    prompt = prefix_prompt + "\n\n" + "Problem: " + stripped_question + suffix_prompt

    content = prompt.strip()
    messages = [{"role": "user", "content": content}]

    return messages


########################################################################


def prompt_for_boxed_answer(prompt_library):
    """
    Mofify prompt_library in-place to elicit final answer in parseable boxed format.
    """

    EXPECTED_KEYS = {"MCQ", "MCQ(multiple)", "Integer", "Numeric"}
    assert (
        set(prompt_library.keys()) == EXPECTED_KEYS
    ), f"Unexpected keys in prompt_library: {set(prompt_library.keys())}"

    prompt_library[
        "MCQ"
    ] += " Mark your solution, which should be exactly one multiple-choice letter, with \\boxed\nAnswer:"
    prompt_library[
        "MCQ(multiple)"
    ] += " Mark your solution, which should be one or more multiple-choice letter(s), with \\boxed\nAnswer:"
    prompt_library["Integer"] += " Mark your solution with \\boxed\nAnswer:"
    prompt_library["Numeric"] += " Mark your solution with \\boxed\nAnswer:"

    return prompt_library


class JEEBenchBenchmark(BaseBenchmark):
    SOURCE = SourceRef("daman1209arora/jeebench", "test")
    METRICS = ("accuracy_avg",)
    PRIMARY_METRIC = "accuracy_avg"
    """
    JEEBench, comprising "515 challenging preengineering mathematics, physics and chemistry problems from the highly competitive IIT JEE-Advanced exam."
    Link: https://huggingface.co/datasets/daman1209arora/jeebench
    """
    REDUCER = ReducerSpec(accuracy="accuracy_avg", stderr="accuracy_std_err", trial_metric=None)

    def __init__(
        self,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize JEEBench benchmark.

        Args:
            debug: If set, only evaluate on 3 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.debug = debug
        self.max_new_tokens = max_tokens
        self.seed = seed
        self.n_repeat = 3

        self.prompt_library = prompt_for_boxed_answer(dict(PROMPT_LIBRARY))



    def load_questions(self) -> Dataset:
        """
        Load JEEBench questions from source.
        """
        self.logger.info("Loading JEEBench questions from source...")
        dataset = load_dataset("daman1209arora/jeebench", split="test", cache_dir=HF_HUB_CACHE)
        self.logger.info(f"{len(dataset)} examples retrieved.")
        return dataset

    def iter_specs(self, options):
        rows = self.load_questions()
        if self.debug:
            rows = list(rows)[:3]
        return text_specs(
            rows, options, prompt=lambda row: format_message(row, self.prompt_library),
            target=lambda row: row["gold"], grader="jee", reference=lambda row: {"type": row["type"]},
            source_id=lambda row: f"{row['description']}/{row['subject']}/{row['index']}",
            generation_kwargs={"do_sample": False, "max_gen_toks": self.max_new_tokens}, repeats=self.n_repeat,
            tags=lambda row: {"subject": row["subject"], "type": row["type"]},
        )
