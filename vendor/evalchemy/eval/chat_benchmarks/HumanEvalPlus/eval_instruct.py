from eval.sources.common import text_specs
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from typing import Dict, List, Any, Optional
import json
import os
from pathlib import Path
import logging

from .utils.utils import language_settings
from eval.task import BaseBenchmark


SUITE = 'HumanEvalPlus'


class HumanEvalPlusBenchmark(BaseBenchmark):
    """
    HumanEvalPlus benchmark for evaluating code generation capabilities across different languages.
    """

    SOURCE = SourceRef("HumanEvalPlus")
    METRICS = ("pass@1",)
    PRIMARY_METRIC = "pass@1"

    def benchmark_size(self) -> int:
        return sum(len(self.load_examples(language)) for language in self.languages)

    def load_examples(self, language: str) -> List[Dict[str, Any]]:
        problem_file = Path(self.data_dir) / f"humanevalplus-{language}.jsonl"
        if not problem_file.exists():
            self.logger.warning(f"Dataset file not found: {problem_file}")
            return []
        examples = [json.loads(line) for line in problem_file.read_text().splitlines() if line.strip()]
        return examples[:2] if self.debug else examples

    def __init__(
        self,
        languages: List[str] = ["python"],
        data_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
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
            data_dir: Directory containing HumanEvalPlus datasets
            max_tokens: Maximum number of tokens for generation
            num_workers: Number of workers for parallel evaluation
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
