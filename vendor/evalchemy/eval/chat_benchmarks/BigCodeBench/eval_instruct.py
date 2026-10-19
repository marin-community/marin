from eval.sources.common import text_specs
from eval.contracts.eval_spec import ReducerSpec, SourceRef
import json
import logging
import os
from typing import Optional


from eval.task import BaseBenchmark




# suppress warnings
import warnings

warnings.filterwarnings("ignore")


BIGCODEBENCH_PATH = os.environ.get("BIGCODEBENCH_PATH", "eval/chat_benchmarks/BigCodeBench/data")
BIGCODEBENCH_HF = "bigcode/bigcodebench"
BIGCODEBENCH_VERSION = "v0.1.3"




class BigCodeBenchBenchmark(BaseBenchmark):
    """
    BigCodeBench benchmark for evaluating code generation capabilities across different languages.
    """

    def __init__(
        self,
        language: str = "python",
        data_dir: str = BIGCODEBENCH_PATH,
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
            data_dir: Directory containing BigCodeBench datasets
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




    SOURCE = SourceRef(BIGCODEBENCH_HF, revision=BIGCODEBENCH_VERSION)
    METRICS = ("pass@1",)
    PRIMARY_METRIC = "pass@1"
    REDUCER = ReducerSpec(accuracy="pass@1", groups={"prompt_type": "{group}_{metric}"})

    def iter_specs(self, options):
        for prompt_type in self.prompt_types:
            with open(os.path.join(self.data_dir, f"BigCodeBench-{prompt_type}.json")) as data:
                rows = json.load(data)
            if self.debug:
                rows = rows[:2]
            def prompt(row):
                code = row["prompt"].strip()
                if prompt_type.startswith("complete"):
                    return f"                            {self.instruction_prefix}\n                            ```\n                            {code}\n                            ```\n                            "
                return f"                            {self.instruction_prefix}\n                            {code}\n                            "
            yield from text_specs(
                rows, options, prompt=prompt, namespace=prompt_type,
                target=lambda row: row["test"], grader="program_suite",
                reference=lambda row: {"suite": "BigCodeBench", "language": self.language, "problem": row,
                                       "timeout": self.timeout, "safe_mode": self.safe_mode},
                generation_kwargs={"temperature": 0, "max_gen_toks": self.max_tokens},
                tags=lambda row: {"prompt_type": prompt_type},
            )
