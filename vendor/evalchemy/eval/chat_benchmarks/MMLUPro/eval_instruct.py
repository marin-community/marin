from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.sources.common import text_specs

import logging
from importlib.resources import files
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from transformers import AutoTokenizer
from eval.contracts.preflight import (
    NetworkDatasetRequirement,
    PackageFileRequirement,
    PythonDependencyRequirement,
)
from eval.task import BaseBenchmark

# --- Prompt construction from Script 1 ---

choices = [chr(ord("A") + i) for i in range(16)]


def select_by_category(df: List[Dict[str, Any]], subject: str) -> List[Dict[str, Any]]:
    return [ex for ex in df if ex["category"] == subject]


def format_cot_example(example: Dict[str, Any], including_answer: bool = True) -> str:
    prompt = "Question:\n" + example["question"] + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(example["options"]):
        prompt += f"{choices[i]}. {opt}\n"
    if including_answer:
        cot = example["cot_content"].replace("A: Let's think step by step.", "Answer: Let's think step by step.")
        prompt += cot + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


PROMPT_PACKAGE = "eval.chat_benchmarks.MMLUPro"
PROMPT_RESOURCE = "initial_prompt.txt"


def generate_cot_prompt(val_df: List[Dict[str, Any]], curr: Dict[str, Any], k: int) -> str:
    # Load base template
    base = files(PROMPT_PACKAGE).joinpath(PROMPT_RESOURCE).read_text()
    subject = curr["category"]
    support = select_by_category(val_df, subject)[:k]
    prompt = base.replace("{$}", subject) + "\n"
    for ex in support:
        prompt += format_cot_example(ex, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


def preprocess(df: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for ex in df:
        opts = [o for o in ex["options"] if o != "N/A"]
        ex["options"] = opts
        out.append(ex)
    return out


# --- MMLUPro Benchmark with CoT prompting ---


class MMLUProBenchmark(BaseBenchmark):
    SOURCE = SourceRef("TIGER-Lab/MMLU-Pro", "test")
    METRICS = ("accuracy_avg",)
    PRIMARY_METRIC = "accuracy_avg"
    """
    MMLU-Pro CoT Benchmark: harness-style but with dynamic few-shot CoT prompts
    and multi-stage regex answer extraction, reporting both overall and per-area accuracy.
    """
    REDUCER = ReducerSpec(accuracy="accuracy_avg", stderr="accuracy_std_err")

    RESOURCE_REQUIREMENTS = (
        PackageFileRequirement(PROMPT_PACKAGE, PROMPT_RESOURCE),
        PythonDependencyRequirement("transformers"),
        NetworkDatasetRequirement("TIGER-Lab/MMLU-Pro"),
    )

    def __init__(
        self,
        num_fewshot: int = 5,
        max_model_length: int = 4096,
        max_tokens: int = 32768,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        seed: List[int] = [0, 1234, 1234, 1234],
    ):
        super().__init__(logger=logger, system_instruction=system_instruction)
        if num_fewshot < 0:
            raise ValueError("num_fewshot must be non-negative")
        self.dataset_name = "TIGER-Lab/MMLU-Pro"
        self.num_fewshot = num_fewshot
        self.max_model_length = max_model_length
        self.max_new_tokens = max_tokens
        self.debug = debug
        self.seed = seed

        self.test_examples = None
        self.val_examples = None
        self.tokenizer: Optional[AutoTokenizer] = None

    def load_questions(self):
        if self.test_examples is None:
            dataset = load_dataset(self.dataset_name)
            self.test_examples = preprocess(dataset["test"])
            self.val_examples = preprocess(dataset["validation"])
        return self.test_examples, self.val_examples

    def validate_prepared_data(self) -> None:
        required_fields = {"answer", "category", "cot_content", "options", "question"}
        test_examples, val_examples = self.load_questions()
        if not test_examples or not val_examples:
            raise ValueError("MMLU-Pro test and validation splits must both be non-empty")
        for split_name, records in (("test", test_examples), ("validation", val_examples)):
            missing = required_fields - set(records[0])
            if missing:
                raise ValueError(f"MMLU-Pro {split_name} records are missing fields: {sorted(missing)}")
        representative = generate_cot_prompt(
            val_examples,
            test_examples[0],
            min(1, self.num_fewshot),
        )
        if not representative.strip():
            raise ValueError("MMLU-Pro representative prompt is empty")

    def iter_specs(self, options):
        test_examples, val_examples = self.load_questions()
        def prompt(example):
            shots = self.num_fewshot
            text = generate_cot_prompt(val_examples, example, shots)
            encode = self.tokenizer.encode if self.tokenizer is not None else options.encode
            if shots and encode is None:
                raise ValueError("MMLU-Pro few-shot prompt sizing requires a tokenizer")
            context = options.context_length or self.max_model_length
            output = options.max_tokens or self.max_new_tokens
            while shots > 0 and len(encode(text)) >= context - output:
                shots -= 1
                text = generate_cot_prompt(val_examples, example, shots)
            return text

        examples = test_examples[:2] if self.debug else test_examples
        return text_specs(
            examples, options, prompt=prompt, target=lambda example: example["answer"],
            grader="mmlu_choice", generation_kwargs={"do_sample": False, "max_gen_toks": self.max_new_tokens},
            tags=lambda example: {"category": example["category"]},
        )
