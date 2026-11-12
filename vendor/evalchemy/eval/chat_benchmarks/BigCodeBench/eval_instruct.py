from dataclasses import replace
import json
import logging
import os
from typing import Optional
import warnings

from rigging.filesystem.factory import open_url
import pyarrow.parquet as pq

from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.data_sources import fetch_hf
from eval.sources.common import text_specs
from eval.task import BaseBenchmark


warnings.filterwarnings("ignore")


BIGCODEBENCH_HF = "bigcode/bigcodebench"
BIGCODEBENCH_VERSION = "v0.1.3"
BIGCODEBENCH_REVISION = "b74c0d0bf70d2c0bc459be537895cca163007f1a"
BIGCODEBENCH_HARD_REVISION = "298d2cc7b96612e15e47313c3603ee124cee0c1f"




class BigCodeBenchBenchmark(BaseBenchmark):
    """
    BigCodeBench benchmark for evaluating code generation capabilities across different languages.
    """

    def __init__(
        self,
        language: str = "python",
        data_dir: str | None = None,
        max_tokens: int = 1280,
        timeout: float = 120,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        instruction_prefix: str = None,
        safe_mode: bool = False,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize BigCodeBench benchmark.

        Args:
            language: Programming language to evaluate
            data_dir: Optional directory containing BigCodeBench JSON fixtures
            max_tokens: Maximum number of tokens for generation
            timeout: Timeout for code execution
            debug: If True, only evaluate first 2 examples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.language = language
        self.data_dir = data_dir
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.debug = debug
        self.prompt_types = ["instruct", "complete", "instruct-hard", "complete-hard"]
        self.safe_mode = safe_mode
        self.instruction_prefix = instruction_prefix or "Please provide a self-contained Python script that solves the following problem in a markdown code block:"




    SOURCE = SourceRef(BIGCODEBENCH_HF, split=BIGCODEBENCH_VERSION, revision=BIGCODEBENCH_REVISION)
    METRICS = ("pass@1",)
    PRIMARY_METRIC = "pass@1"
    REDUCER = ReducerSpec(accuracy="pass@1", groups={"prompt_type": "{group}_{metric}"})

    def iter_specs(self, options):
        for prompt_type in self.prompt_types:
            source = self.SOURCE
            if prompt_type.endswith("-hard"):
                source = SourceRef("bigcode/bigcodebench-hard", split=BIGCODEBENCH_VERSION,
                                   revision=BIGCODEBENCH_HARD_REVISION)
            if self.data_dir is not None:
                with open_url(os.path.join(self.data_dir, f"BigCodeBench-{prompt_type}.json"), "rt") as data:
                    rows = json.load(data)
            else:
                path = fetch_hf(source.dataset, "data/v0.1.3-00000-of-00001.parquet",
                                revision=source.revision, cache_prefix=self.dataset_cache)
                with open_url(path, "rb") as data:
                    raw_rows = pq.read_table(data).to_pylist()
                prompt_field = "complete_prompt" if prompt_type.startswith("complete") else "instruct_prompt"
                rows = [{**{key: row[key] for key in ("task_id", "test", "entry_point", "doc_struct",
                                                     "code_prompt", "canonical_solution")},
                         "prompt": row[prompt_field]} for row in raw_rows]
            if self.debug:
                rows = rows[:2]
            def prompt(row):
                code = row["prompt"].strip()
                if prompt_type.startswith("complete"):
                    return f"                            {self.instruction_prefix}\n                            ```\n                            {code}\n                            ```\n                            "
                return f"                            {self.instruction_prefix}\n                            {code}\n                            "
            yield from text_specs(
                rows, replace(options, source=source), prompt=prompt, namespace=prompt_type,
                target=lambda row: row["test"], grader="program_suite",
                reference=lambda row: {"suite": "BigCodeBench", "language": self.language, "problem": row,
                                       "timeout": self.timeout, "safe_mode": self.safe_mode},
                generation_kwargs={"temperature": 0, "max_gen_toks": self.max_tokens},
                tags=lambda row: {"prompt_type": prompt_type},
            )
