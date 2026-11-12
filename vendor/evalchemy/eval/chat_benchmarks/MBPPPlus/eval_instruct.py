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


SUITE = 'MBPPPlus'
DATASET_REPO = 'evalplus/mbppplus'
DATASET_REVISION = 'b2d74c91837c3f2a20c1299ae98133cbe7cfa077'
DATA_FILE = 'mbppplus.jsonl'



class MBPPPlusBenchmark(BaseBenchmark):
    """
    MBPPPlus benchmark for evaluating code generation capabilities across different languages.
    """

    SOURCE = SourceRef(DATASET_REPO, revision=DATASET_REVISION)
    METRICS = ("pass@1",)
    PRIMARY_METRIC = "pass@1"

    def benchmark_size(self) -> int:
        return sum(1 for _ in self.read_test_examples(self.load_examples()))

    def __init__(
        self,
        data_dir: str | None = None,
        timeout: float = 3.0,
        debug: bool = False,
        max_tokens: int = 1024,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize MBPPPlus benchmark.

        Args:
            data_dir: Optional directory containing MBPPPlus JSONL fixtures
            max_tokens: Maximum number of tokens for generation
            timeout: Timeout for code execution
            debug: If True, only evaluate first 2 examples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_dir = data_dir
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.debug = debug
        self.num_examples = 3
        self.start_idx = 0
        self.end_idx = 500

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
        path = fetch_hf(DATASET_REPO, "data/test-00000-of-00001-d5781c9c51e02795.parquet",
                        revision=DATASET_REVISION, cache_prefix=self.dataset_cache)
        with open_url(path, "rb") as data:
            return pq.read_table(data).to_pylist()

    def read_test_examples(self, examples: List[Dict]) -> Generator[Dict[str, str], None, None]:
        examples_str = []
        for i in range(1, self.num_examples + 1):
            ex = examples[i]
            example_prompt = "- Example {}:\n{}".format(
                i, self.format_test_example(ex["prompt"], ex["test_list"], ex["code"])
            )
            examples_str.append(example_prompt)

        eval_range = range(self.start_idx, min(self.end_idx, len(examples)))
        if self.debug:
            eval_range = list(eval_range)[:2]
            self.logger.info(f"Debug mode: using 2 examples")

        for i in eval_range:
            ex = examples[i]
            prompt = self.format_test_example(ex["prompt"], ex["test_list"])

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





    REDUCER = ReducerSpec(accuracy="pass@1")

    def iter_specs(self, options):
        examples = self.load_examples()
        problems = {row["task_id"]: row for row in examples}
        return text_specs(
            self.read_test_examples(examples), options,
            prompt=lambda row: row["prompt"], target=lambda row: problems[row["task_id"]]["test"],
            grader="program_suite",
            reference=lambda row: {"suite": SUITE, "language": "python", "problem": problems[row["task_id"]],
                                   "timeout": self.timeout, "mbpp": False},
            generation_kwargs={"do_sample": False, "max_gen_toks": self.max_tokens},
        )
