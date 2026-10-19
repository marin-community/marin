from eval.sources.common import text_specs
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from typing import List, Optional
import json
import os
import logging

from eval.task import BaseBenchmark


LANUGUAGES = [
    # "adb",
    # "clj",
    "cpp",
    "cs",
    # "dart",
    # "dfy",
    # "dlang",
    # "elixir",
    # "fs",
    # "go",
    # "hs",
    "java",
    "js",
    # "julia",
    # "lean",
    "lua",
    # "luau",
    # "matlab",
    # "ocaml",
    "php",
    # "pl",
    # "python",
    "r",
    # "racket",
    # "ruby",
    "rs",
    # "scala",
    "sh",
    # "swift",
    "ts",
    # "v",
]

LANGUAGE_MAP = {
    "js": "javascript",
    "java": "java",
    "python": "python",
    "cpp": "cpp",
    "cs": "csharp",
    "go": "go",
    "hs": "haskell",
    "php": "php",
    "r": "r",
    "ruby": "ruby",
    "rs": "rust",
    "scala": "scala",
    "sh": "bash",
    "swift": "swift",
    "ts": "typescript",
    "adb": "ada",
    "clj": "clojure",
    "dart": "dart",
    "dfy": "fsharp",
    "dlang": "d",
    "elixir": "elixir",
    "fs": "fsharp",
    "julia": "julia",
    "lean": "lean",
    "lua": "lua",
    "luau": "lua",
    "matlab": "matlab",
    "ocaml": "ocaml",
    "pl": "perl",
    "racket": "racket",
    "v": "vlang",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class MultipleBenchmark(BaseBenchmark):
    """
    Multipl-e benchmark for evaluating code generation capabilities across different languages.
    """

    def __init__(
        self,
        languages: List[str] = LANUGUAGES,
        data_dir: str = DATA_DIR,
        max_tokens: int = 1024,
        timeout: float = 15,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize multipl-e benchmark.

        Args:
            languages: List of programming languages to evaluate
            data_dir: Directory containing multipl-e datasets
            max_tokens: Maximum number of tokens for generation
            num_workers: Number of workers for parallel evaluation
            timeout: Timeout for code execution
            debug: If True, only evaluate first 2 examples
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.languages = languages
        self.data_dir = data_dir
        self.max_tokens = max_tokens or 1024
        self.timeout = timeout
        self.debug = debug
        self.system_prompt = "You are a helpful programming assistant designed to complete code snippets."
        self.task_prompt = """Please generate code to complete the following problem:
        ```{lang}
        {prompt}
        ```
        """




    SOURCE = SourceRef("MultiPL-E")
    METRICS = ("pass@1",)
    PRIMARY_METRIC = "pass@1"
    REDUCER = ReducerSpec(accuracy="pass@1", groups={"language": "{group}_{metric}"})

    def iter_specs(self, options):
        for language in self.languages:
            with open(os.path.join(self.data_dir, f"multipl-e-{language}.json")) as data:
                rows = json.load(data)
            if self.debug:
                rows = rows[:10]
            yield from text_specs(
                rows, options, namespace=language,
                prompt=lambda row: [{"role": "system", "content": self.system_prompt},
                                    {"role": "user", "content": self.task_prompt.format(prompt=row["prompt"], lang=LANGUAGE_MAP[language])}],
                target=lambda row: row["test"], grader="program_suite",
                reference=lambda row: {"suite": "MultiPLE", "language": language, "problem": row, "timeout": self.timeout},
                generation_kwargs={"temperature": 0, "max_gen_toks": self.max_tokens},
                tags=lambda row: {"language": language},
            )
