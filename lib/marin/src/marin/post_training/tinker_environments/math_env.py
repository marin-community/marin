import logging
import math
import re
from functools import partial
from typing import Literal, Sequence, cast

from datasets import Dataset, concatenate_datasets, get_dataset_config_names, load_dataset
from transformers import PreTrainedTokenizer

from .math_grading import (
    extract_boxed,
    grade_answer,
    grade_answer_math_verify,
    run_with_timeout_signal,
)

logger = logging.getLogger(__name__)


class MathEnv():
    def __init__(
        self,
        problem: str,
        answer: str,
        grader: Literal["sympy", "math_verify"] = "sympy",
        timeout: float = 1.0,
    ):
        self.problem = problem
        self.answer = answer
        self.grader = grader
        self.timeout = timeout

    @classmethod
    def question_suffix(cls) -> str:
        return " Write your answer in \\boxed{} format."

    def get_question(self) -> str:
        return self.problem + self.question_suffix()

    def check_format(self, sample_str: str) -> bool:
        try:
            _ = extract_boxed(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str) -> bool:
        try:
            answer = extract_boxed(sample_str)
        except ValueError:
            return False
        return safe_grade(answer, self.answer, self.grader, self.timeout)

    def get_reference_answer(self) -> str:
        return self.answer

    @staticmethod
    def standard_fewshot_prefix():
        return [
            {
                "role": "user",
                "content": "How many r's are in strawberry?" + MathEnv.question_suffix(),
            },
            {
                "role": "assistant",
                "content": "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}",
            },
        ]


def safe_grade(given_answer: str, ground_truth: str, grader: str = "sympy", timeout: float = 1.0):
    if grader == "sympy":
        grader_func = grade_answer
    elif grader == "math_verify":
        grader_func = grade_answer_math_verify
    else:
        raise ValueError(f"Invalid grader: {grader}")
    out = run_with_timeout_signal(
        grader_func, args=(given_answer, ground_truth), timeout_seconds=int(math.ceil(timeout))
    )
    if out is None:
        logger.warning(f"Timeout grading {given_answer} against {ground_truth}")
        return False
    return out


def _get_hendrycks_math_test() -> Dataset:
    test_dataset = load_dataset("HuggingFaceH4/MATH-500", name="default", split="test")
    return cast(Dataset, test_dataset)


def _get_hendrycks_math_train() -> Dataset:
    # For Hendrycks MATH, the standard is to use both the "train" and "test" splits for
    # training. The "test" split here is NOT the same as the MATH-500 test split above,
    # which is a commonly-held-out subset of 500 of the below 12.5k problems. To construct
    # a clean training set, we filter out problems that exist in the MATH-500 test set,
    # resulting in 12000 train and 500 test problems.

    test_problems: set[str] = {
        problem["problem"]  # pyright: ignore[reportArgumentType, reportCallIssue]
        for problem in _get_hendrycks_math_test()
    }

    dataset_name = "EleutherAI/hendrycks_math"
    configs = get_dataset_config_names(dataset_name)
    pieces = []
    for cfg in configs:
        for split in ("train", "test"):
            ds = load_dataset(dataset_name, name=cfg, split=split)
            ds = ds.filter(lambda example: example["problem"] not in test_problems)
            pieces.append(ds)
    full_dataset = concatenate_datasets(pieces)

    return full_dataset


# Adapted from https://github.com/thinking-machines-lab/tinker-cookbook/blob/20e26a629797188aa8c6f34474b0d4757b20b90d/tinker_cookbook/renderers.py#L165
def parse_response_for_stop_token(
    response: list[int], tokenizer: PreTrainedTokenizer, stop_token: int
) -> tuple[str, bool]:
    """Parse response for a single stop token.

    We expect a properly rendered response to have exactly one stop token; but it may have zero if e.g. the model
    ran out of tokens when sampling, which will incur a format error. If there are > 1, there is likely a bug in the
    sampler and we should error.
    """
    emt_count = response.count(stop_token)
    if emt_count == 0:
        str_response = tokenizer.decode(response)
        logger.debug(f"Response is not a valid assistant response: {str_response}")
        return str_response, False
    elif emt_count == 1:
        str_response = tokenizer.decode(response[: response.index(stop_token)])
        return str_response, True
    else:
        raise ValueError(
            f"When parsing response, expected to split into 1 or 2 pieces using stop tokens, but got {emt_count}. "
            "You probably are using the wrong stop tokens when sampling"
        )
