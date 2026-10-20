from eval.sources.common import text_specs
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from typing import Any, Dict, Generator, Optional
import json
import logging
import os

import fsspec

from eval.data_sources import fetch_hf
from eval.task import BaseBenchmark

HF_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
# The original asks for one placeholder but grades three. Preserve Evalchemy's corrected prompt.
CORRECTED_PLACEHOLDER_PROMPT = (
    "What is inside Shinto shrines? Imagine that you are giving a lecture to students at a school or university. "
    "Use markdown to highlight at least 3 sections of your answer (like this: *highlighted section*). "
    "Your answer must also contain at least 3 placeholders (an example of a placeholder is [address])."
)


class IFEvalBenchmark(BaseBenchmark):
    SOURCE = SourceRef("google/IFEval", "train", HF_REVISION)
    METRICS = ("prompt-level", "instruction-level", "prompt_level_strict", "instruction_level_strict", "prompt_level_loose", "instruction_level_loose")
    PRIMARY_METRIC = "prompt-level"
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
        Initialize Instruction Following Benchmark

        Args:
        data_dir: Directory containing IFEval datasets
        max_tokens: Maximum number of tokens for generation
        num_examples: Number of examples to show in few-shot prompt
        start_idx: Start index for evaluation examples
        end_idx: End index for evaluation examples
        debug: If True, only evaluate the first 2 examples
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

    def load_questions(self, data_path: str | None = None) -> list[dict[str, Any]]:
        """Load instruction inputs and retain the local placeholder correction."""
        if data_path is None and self.data_dir is not None:
            data_path = os.path.join(self.data_dir, "input_data.jsonl")
        canonical = data_path is None
        path = data_path or fetch_hf(
            self.SOURCE.dataset, "ifeval_input_data.jsonl", revision=HF_REVISION, cache_prefix=self.dataset_cache,
        )
        with fsspec.open(path, "rt", encoding="utf-8") as data:
            examples = [json.loads(line) for line in data]
        if canonical:
            for row in examples:
                if row["key"] == 2785:
                    row["prompt"] = CORRECTED_PLACEHOLDER_PROMPT
        self.logger.info("Loaded %d examples from %s", len(examples), path)
        return examples

    def validate_prepared_data(self) -> None:
        examples = self.load_questions()
        if self.data_dir is None and len(examples) != 541:
            raise ValueError(f"IFEval expected 541 questions, found {len(examples)}")
        if not examples or any(
            not row["prompt"].strip() or not row["instruction_id_list"]
            or len(row["instruction_id_list"]) != len(row["kwargs"]) for row in examples
        ):
            raise ValueError("IFEval requires prompts, instruction IDs, and matching grading arguments")

    def read_test_examples(self, data_path: str | None = None) -> Generator[Dict[str, Any], None, None]:
        """
        Read and get prompts from data file.

        Args:
            data_path: Path to the data file

        Yields:
            Dictionary containing key, prompt and other properties required for evaluation
        """
        examples = self.load_questions(data_path)
        eval_range = range(self.start_idx, min(self.end_idx, len(examples)))
        if self.debug:
            eval_range = list(eval_range)[:2]
            self.logger.info("Debug mode: using 2 examples")
        for i in eval_range:
            yield examples[i]




    REDUCER = ReducerSpec(trial_metric="prompt_level_strict", name="instructions")

    def iter_specs(self, options):
        return text_specs(
            self.read_test_examples(), options, prompt=lambda row: row["prompt"], target=lambda row: row["instruction_id_list"],
            grader="instructions", reference=lambda row: {"protocol": 'ifeval'},
            generation_kwargs={"do_sample": False, "temperature": 0, "max_gen_toks": self.max_tokens},
        )
