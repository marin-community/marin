from eval.sources.common import text_specs
from eval.contracts.eval_spec import ReducerSpec, SourceRef
import logging
import os
from typing import List, Optional

from datasets import Dataset, load_dataset

from eval.task import BaseBenchmark


########################################################################

# Adapted from https://github.com/centerforaisafety/hle/blob/main/hle_eval/run_model_predictions.py

SYSTEM_EXACT_ANSWER = "Your response should be in the following format:\nAnswer: {your chosen multiple choice letter}. Include only the letter, no other text."

SYSTEM_MC = "Your response should be in the following format:\nAnswer: {your chosen multiple choice letter}. Include only the letter, no other text."

HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE")
if not HF_HUB_CACHE:
    print(
        "WARNING: HF_HUB_CACHE environment variable is not set, using default cache directory ~/.cache/huggingface/hub for HLE benchmark"
    )


def format_message(question):
    answer_type = question["answer_type"]
    system_prompt = SYSTEM_EXACT_ANSWER if answer_type == "exact_match" else SYSTEM_MC
    question_text = question["question"]

    text_content = dict(type="text", text=question_text)
    if question["image"]:  # "" if not multi-modal
        image_content = dict(type="image_url", image_url=dict(url=question["image"]))
        content = [text_content, image_content]
    else:
        content = [text_content]

    # system_role = "user" if "o1" in args.model else "system" # o1 no sys prompt
    # messages = [
    #     {"role": system_role, "content": system_prompt},
    #     {"role": "user", "content": content}
    # ]
    content = system_prompt + "\n" + question_text
    messages = [{"role": "user", "content": content}]
    return messages


########################################################################


class HLESubsetBenchmark(BaseBenchmark):
    SOURCE = SourceRef("cais/hle", "test")
    METRICS = ("accuracy_avg",)
    PRIMARY_METRIC = "accuracy_avg"
    """
    Humanity's Last Exam benchmark for evaluating the "frontier of human knowledge".
    Link: https://huggingface.co/datasets/cais/hle
    """

    def __init__(
        self,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize HLE benchmark.

        Args:
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.debug = debug
        self.max_new_tokens = max_tokens
        self.seed = seed
        self.n_repeat = 3




    def load_questions(self) -> Dataset:
        """
        Load HLE questions from source.
        Keep only the multiplechoice and no images.
        """
        self.logger.info("Loading HLE questions from source, filtering for multiplechoice and no images...")
        dataset = load_dataset("cais/hle", split="test", cache_dir=HF_HUB_CACHE)
        dataset = dataset.filter(lambda x: x["answer_type"] == "multipleChoice")
        dataset = dataset.filter(lambda x: x["image"] == "")
        self.logger.info(f"{len(dataset)} examples remaining after filtering for multiplechoice and no images.")
        return dataset

    REDUCER = ReducerSpec(accuracy="accuracy_avg", stderr="accuracy_std_err")

    def iter_specs(self, options):
        rows = self.load_questions()
        if self.debug:
            rows = list(rows)[:2]
        return text_specs(
            rows, options, prompt=format_message, target=lambda row: row["answer"], grader="letter_choice",
            reference=lambda row: {"extractor": "hle"},
            generation_kwargs={"do_sample": False, "max_gen_toks": self.max_new_tokens}, repeats=self.n_repeat,
        )
