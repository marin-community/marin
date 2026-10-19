from eval.sources.common import text_specs
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from typing import Dict, List, Generator, Optional
import json
import os
import logging

from eval.task import BaseBenchmark


SUITE = 'MBPP'
DATA_FILE = 'mbpp.jsonl'
TEST_FILE = 'mbpp_test.jsonl'


class MBPPBenchmark(BaseBenchmark):
    """
    MBPP (Mostly Basic Python Programming) benchmark for evaluating
    Python code generation capabilities.
    """

    def __init__(
        self,
        data_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
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
            data_dir: Directory containing MBPP datasets
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

    def read_test_examples(self, data_path: str) -> Generator[Dict[str, str], None, None]:
        """
        Read and format test examples from data file.

        Args:
            data_path: Path to the data file

        Yields:
            Dictionary containing task_id and formatted prompt
        """
        try:
            with open(data_path, "r") as f:
                examples = [json.loads(x) for x in f]
            self.logger.info(f"Loaded {len(examples)} examples from {data_path}")

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

        except Exception as e:
            self.logger.error(f"Error reading examples: {str(e)}")
            raise





    SOURCE = SourceRef("mbpp")
    METRICS = ("pass@1",)
    PRIMARY_METRIC = "pass@1"
    REDUCER = ReducerSpec(accuracy="pass@1")

    def iter_specs(self, options):
        with open(os.path.join(self.data_dir, TEST_FILE)) as data:
            problems = {row["task_id"]: row for line in data if line.strip() for row in [json.loads(line)]}
        return text_specs(
            self.read_test_examples(os.path.join(self.data_dir, DATA_FILE)), options,
            prompt=lambda row: row["prompt"], target=lambda row: problems[row["task_id"]]["test"],
            grader="program_suite",
            reference=lambda row: {"suite": SUITE, "language": "python", "problem": problems[row["task_id"]],
                                   "timeout": 10.0, "mbpp": True},
            generation_kwargs={"do_sample": False, "max_gen_toks": self.max_tokens},
        )
