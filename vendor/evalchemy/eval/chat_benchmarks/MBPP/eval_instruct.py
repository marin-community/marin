import json
import logging
import os
from typing import Dict, Generator, List, Optional

from rigging.filesystem.factory import open_url
import pyarrow.parquet as pq

from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.data_sources import fetch_hf
from eval.sources.common import text_specs
from eval.task import BaseBenchmark


SUITE = 'MBPP'
DATASET_REPO = 'google-research-datasets/mbpp'
DATASET_REVISION = '4bb6404fdc6cacfda99d4ac4205087b89d32030c'
DATA_FILE = 'mbpp.jsonl'



class MBPPBenchmark(BaseBenchmark):
    """
    MBPP (Mostly Basic Python Programming) benchmark for evaluating
    Python code generation capabilities.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        num_examples: int = 3,
        start_idx: int = 10,
        end_idx: int = 510,
        debug: bool = False,
        max_tokens: int = 512,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize MBPP benchmark.

        Args:
            data_dir: Optional directory containing MBPP JSONL fixtures
            max_tokens: Maximum number of tokens for generation
            num_examples: Number of examples to show in few-shot prompt
            start_idx: Start index for evaluation examples
            end_idx: End index for evaluation examples
            debug: If set, only evaluate on 2 examples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_dir = data_dir
        self.max_tokens = max_tokens
        self.num_examples = num_examples
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.debug = debug

    def format_test_example(self, question: str, tests: List[str], code: Optional[str] = None) -> str:
        """Format a single test example."""
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(question.strip(), "\n".join(tests))
        if code:
            code = code.replace("\r", "").replace("\t", "    ")
            prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
        return prompt

    def load_examples(self):
        if self.data_dir is not None:
            with open_url(os.path.join(self.data_dir, DATA_FILE), "rt") as data:
                return [json.loads(line) for line in data if line.strip()]
        rows = []
        for split in ("prompt", "train", "test", "validation"):
            path = fetch_hf(DATASET_REPO, f"full/{split}-00000-of-00001.parquet",
                            revision=DATASET_REVISION, cache_prefix=self.dataset_cache)
            with open_url(path, "rb") as data:
                rows.extend(pq.read_table(data).to_pylist())
        return sorted(rows, key=lambda row: row["task_id"])

    def read_test_examples(self, examples: List[Dict]) -> Generator[Dict[str, str], None, None]:
        examples_str = []
        for i in range(1, self.num_examples + 1):
            ex = examples[i]
            example_prompt = "- Example {}:\n{}".format(
                i, self.format_test_example(ex["text"], ex["test_list"], ex["code"])
            )
            examples_str.append(example_prompt)

        eval_range = range(self.start_idx, min(self.end_idx, len(examples)))
        if self.debug:
            eval_range = list(eval_range)[:2]
            self.logger.info(f"Debug mode: using 2 examples")

        for i in eval_range:
            ex = examples[i]
            prompt = self.format_test_example(ex["text"], ex["test_list"])

            prompt_with_shots = """
Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}
""".strip().format(
                "\n\n".join(examples_str), prompt
            )

            yield {"task_id": ex["task_id"], "prompt": prompt_with_shots}





    SOURCE = SourceRef(DATASET_REPO, revision=DATASET_REVISION)
    METRICS = ("pass@1",)
    PRIMARY_METRIC = "pass@1"
    REDUCER = ReducerSpec(accuracy="pass@1")

    def iter_specs(self, options):
        examples = self.load_examples()
        if self.data_dir is not None:
            with open_url(os.path.join(self.data_dir, "mbpp_test.jsonl"), "rt") as data:
                problems = {row["task_id"]: row for line in data if line.strip() for row in [json.loads(line)]}
        else:
            problems = {row["task_id"]: {"task_id": row["task_id"], "prompt": row["text"],
                                         "test": row["test_list"], "code": row["code"]}
                        for row in examples[10:510]}
        return text_specs(
            self.read_test_examples(examples), options,
            prompt=lambda row: row["prompt"], target=lambda row: problems[row["task_id"]]["test"],
            grader="program_suite",
            reference=lambda row: {"suite": SUITE, "language": "python", "problem": problems[row["task_id"]],
                                   "timeout": 10.0, "mbpp": True},
            generation_kwargs={"do_sample": False, "max_gen_toks": self.max_tokens},
        )
