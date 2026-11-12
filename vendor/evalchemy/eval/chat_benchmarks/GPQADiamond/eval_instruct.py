from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.sources.common import text_specs

import hashlib
import logging
import os
import random
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from eval.task import BaseBenchmark


PROMPT = """Return your final response within \\boxed{{}} and only include the letter choice (A, B, C, or D) as your final response.
Problem: {problem}
Options: {options}
Answer:"""

HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE")
if not HF_HUB_CACHE:
    print(
        "WARNING: HF_HUB_CACHE environment variable is not set, using default cache directory ~/.cache/huggingface/hub for GPQADiamond benchmark"
    )


class GPQADiamondBenchmark(BaseBenchmark):
    SOURCE = SourceRef("Idavidrein/gpqa", "train")
    METRICS = ("accuracy_avg",)
    PRIMARY_METRIC = "accuracy_avg"
    """
    GPQADiamond Benchmark for evaluating multiple choice reasoning of LLMs.
    Link: https://huggingface.co/datasets/Idavidrein/gpqa
    """
    REDUCER = ReducerSpec(accuracy="accuracy_avg", stderr="accuracy_std_err")

    def __init__(
        self,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize GPQADiamond benchmark.

        Args:
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.dataset_name = "Idavidrein/gpqa"
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.n_repeat = 3

    def iter_specs(self, options):
        def records():
            for source in self.load_questions():
                choices, answer = self.generate_multiple_choice_answers(source)
                yield {**source, "multiple_choice_string": choices, "answer": answer}

        return text_specs(
            records(), options,
            prompt=lambda example: PROMPT.format(problem=example["Question"], options=example["multiple_choice_string"]),
            target=lambda example: example["answer"], grader="letter_choice",
            generation_kwargs={"do_sample": True, "temperature": 0.7, "max_gen_toks": self.max_new_tokens},
            repeats=self.n_repeat,
        )


    def load_questions(self) -> List[Dict[str, Any]]:
        """Load GPQADiamond questions from the dataset."""
        dataset = load_dataset(self.dataset_name, "gpqa_diamond", cache_dir=HF_HUB_CACHE)
        questions = [row for row in dataset["train"]]
        if self.debug:
            questions = questions[:2]
        self.logger.info(f"Loaded {len(questions)} questions from {self.dataset_name}")
        return questions

    def generate_multiple_choice_answers(self, data: Dict[str, Any]) -> tuple[str, str]:
        """Generate multiple choice string and correct answer letter."""
        answers = [
            data["Correct Answer"],
            data["Incorrect Answer 1"],
            data["Incorrect Answer 2"],
            data["Incorrect Answer 3"],
        ]
        rnd = random.Random(42 + int(hashlib.md5(data["Question"].encode()).hexdigest(), 16))
        rnd.shuffle(answers)

        options = ["A", "B", "C", "D"]
        options_to_answers = {letter: answer for letter, answer in zip(options, answers)}

        multiple_choice_string = ", ".join(f"{letter}) {options_to_answers[letter]}" for letter in options)
        correct_answer_letter = next(
            letter for letter, answer in options_to_answers.items() if answer == data["Correct Answer"]
        )

        return multiple_choice_string, correct_answer_letter
