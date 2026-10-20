import json
import logging
import os
from typing import List, Optional

import fsspec
import pyarrow.parquet as pq

from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.data_sources import fetch_hf
from eval.sources.common import text_specs
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

MULTIPLE_REPO = "nuprl/MultiPL-E"
MULTIPLE_REVISION = "28441b6024e71d4a1c1c0f6bf171c935cd5a43f2"
# Preserve the two historical Lua prompts, including their doctests and whitespace.
LUA_PROMPTS = {
    'HumanEval_69_search': (
        '-- You are given a non-empty table of positive integers. Return the greatest integer that is greater than \n'
        '-- zero, and has a frequency greater than or equal to the value of the integer itself. \n'
        '-- The frequency of an integer is the number of times it appears in the table.\n'
        '-- If no such a value exist, return -1.\n'
        '-- Examples:\n'
        '-- >>> search({4, 1, 2, 2, 3, 1})\n'
        '-- 2\n'
        '-- >>> search({1, 2, 2, 3, 3, 3, 4, 4, 4})\n'
        '-- 3\n'
        '-- >>> search({5, 5, 4, 4, 4})\n'
        '-- -1\n'
        'local function search(lst)\n'
    ),
    'HumanEval_142_sum_squares': (
        '-- "\n'
        '-- This function will take a table of integers. For all entries in the table, the function shall square the integer entry if its index is a \n'
        '-- multiple of 3 and will cube the integer entry if its index is a multiple of 4 and not a multiple of 3. The function will not \n'
        '-- change the entries in the table whose indexes are not a multiple of 3 or 4. The function shall then return the sum of all entries. \n'
        '-- Examples:\n'
        '-- >>> lst\n'
        '-- {1, 2, 3}\n'
        '-- >>> lst\n'
        '-- {}\n'
        '-- >>> lst\n'
        '-- {-1, -5, 2, -1, -5}\n'
        'local function sum_squares(lst)\n'
    ),
}


class MultipleBenchmark(BaseBenchmark):
    """
    Multipl-e benchmark for evaluating code generation capabilities across different languages.
    """

    def __init__(
        self,
        languages: List[str] = LANUGUAGES,
        data_dir: str | None = None,
        max_tokens: int = 1024,
        timeout: float = 15,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize multipl-e benchmark.

        Args:
            languages: List of programming languages to evaluate
            data_dir: Optional directory containing MultiPL-E JSON fixtures
            max_tokens: Maximum number of tokens for generation
            timeout: Timeout for code execution
            debug: If True, only evaluate the first 10 examples per language
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




    SOURCE = SourceRef(MULTIPLE_REPO, revision=MULTIPLE_REVISION)
    METRICS = ("pass@1",)
    PRIMARY_METRIC = "pass@1"
    REDUCER = ReducerSpec(accuracy="pass@1", groups={"language": "{group}_{metric}"})

    def load_examples(self, language):
        if self.data_dir is not None:
            with fsspec.open(os.path.join(self.data_dir, f"multipl-e-{language}.json"), "rt") as data:
                return json.load(data)
        upstream_language = {"julia": "jl", "racket": "rkt"}.get(language, language)
        rows = []
        for suite in (("humaneval",) if language == "dart" else ("humaneval", "mbpp")):
            path = fetch_hf(MULTIPLE_REPO, f"{suite}-{upstream_language}/test-00000-of-00001.parquet",
                            revision=MULTIPLE_REVISION, cache_prefix=self.dataset_cache)
            with fsspec.open(path, "rb") as data:
                examples = pq.read_table(data).to_pylist()
            for example in examples:
                prompt = example["prompt"]
                if language in ("cpp", "cs", "java", "js", "lua", "php", "sh", "ts"):
                    prompt = prompt.replace(f"{language}thon", language)
                if language == "lua" and example["name"] in LUA_PROMPTS:
                    prompt = LUA_PROMPTS[example["name"]]
                rows.append({"task_id": example["name"], "prompt": prompt,
                             "test": example["tests"], "language": example["language"]})
        return rows

    def iter_specs(self, options):
        for language in self.languages:
            rows = self.load_examples(language)
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
