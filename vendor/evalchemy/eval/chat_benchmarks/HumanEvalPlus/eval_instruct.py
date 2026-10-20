import json
import logging
import os
from typing import Any, Dict, List, Optional

import fsspec

from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.data_sources import fetch_hf
from eval.sources.common import text_specs
from eval.task import BaseBenchmark
from .utils.utils import language_settings


SUITE = 'HumanEvalPlus'
HUMANEVALPLUS_REPO = "evalplus/humanevalplus"
HUMANEVALPLUS_REVISION = "d32357cf319e50e9c8d8dab5ea876c72b0fd321b"


class HumanEvalPlusBenchmark(BaseBenchmark):
    """
    HumanEvalPlus benchmark for evaluating code generation capabilities across different languages.
    """

    SOURCE = SourceRef(HUMANEVALPLUS_REPO, revision=HUMANEVALPLUS_REVISION)
    METRICS = ("pass@1",)
    PRIMARY_METRIC = "pass@1"

    def benchmark_size(self) -> int:
        return sum(len(self.load_examples(language)) for language in self.languages)

    def load_examples(self, language: str) -> List[Dict[str, Any]]:
        if self.data_dir is not None:
            path = os.path.join(self.data_dir, f"humanevalplus-{language}.jsonl")
        else:
            if language != "python":
                raise ValueError(f"HumanEvalPlus has no upstream dataset for {language}")
            path = fetch_hf(HUMANEVALPLUS_REPO, "test.jsonl", revision=HUMANEVALPLUS_REVISION,
                            cache_prefix=self.dataset_cache)
        with fsspec.open(path, "rt") as data:
            examples = [json.loads(line) for line in data if line.strip()]
        if self.data_dir is None:
            # The bundled harness explicitly invoked check after defining it.
            for example in examples:
                example["test"] += f"\ncheck({example['entry_point']})"
        return examples[:2] if self.debug else examples

    def __init__(
        self,
        languages: List[str] = ["python"],
        data_dir: str | None = None,
        max_tokens: int = 1024,
        timeout: float = 3.0,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize HumanEvalPlus benchmark.

        Args:
            languages: List of programming languages to evaluate
            data_dir: Optional directory containing HumanEvalPlus JSON fixtures
            max_tokens: Maximum number of tokens for generation
            timeout: Timeout for code execution
            debug: If True, only evaluate first 2 examples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.languages = languages
        self.data_dir = data_dir
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.debug = debug

    def build_deepseekcoder_instruction(self, language: str, question: str) -> str:
        """Build instruction prompt for the model."""
        return """
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```{}
{}
```
""".strip().format(
            language.lower(), question.strip()
        )




    REDUCER = ReducerSpec(accuracy="pass@1", groups={"language": "{group}_{metric}"})

    def iter_specs(self, options):
        for language in self.languages:
            yield from text_specs(
                self.load_examples(language), options,
                prompt=lambda row: self.build_deepseekcoder_instruction(language_settings[language]["full_name"], row["prompt"]),
                target=lambda row: row["test"], grader="program_suite", namespace=language,
                reference=lambda row: {"suite": SUITE, "language": language, "problem": row, "timeout": self.timeout},
                generation_kwargs={"do_sample": False, "max_gen_toks": self.max_tokens},
                tags=lambda row: {"language": language},
            )
