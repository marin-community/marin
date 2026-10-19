from eval.sources.common import text_specs
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from typing import Dict, Generator, Optional
import json
import logging
import os

from eval.task import BaseBenchmark


class IFEvalBenchmark(BaseBenchmark):
    SOURCE = SourceRef("google/IFEval", "input_data")
    METRICS = ("prompt-level", "instruction-level", "prompt_level_strict", "instruction_level_strict", "prompt_level_loose", "instruction_level_loose")
    PRIMARY_METRIC = "prompt-level"
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
        Initialize Instruction Following Benchmark

        Args:
        data_dir: Directory containing MBPP datasets
        max_tokens: Maximum number of tokens for generation
        num_examples: Number of examples to show in few-shot prompt
        start_idx: Start index for evaluation examples
        end_idx: End index for evaluation examples
        debug_size: If set, only evaluate this many examples
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

    def read_test_examples(self, data_path: str) -> Generator[Dict[str, str], None, None]:
        """
        Read and get prompts from data file.

        Args:
            data_path: Path to the data file

        Yields:
            Dictionary containing key, prompt and other properties required for evaluation
        """
        try:
            with open(data_path, "r") as f:
                examples = [json.loads(x) for x in f]

            self.logger.info(f"Loaded {len(examples)} examples from {data_path}")

            eval_range = range(self.start_idx, min(self.end_idx, len(examples)))
            if self.debug:
                eval_range = list(eval_range)[:2]
                self.logger.info(f"Debug mode: using 2 examples")

            for i in eval_range:
                ex = examples[i]

                yield ex

        except Exception as e:
            self.logger.error(f"Error reading examples: {str(e)}")
            raise




    REDUCER = ReducerSpec(trial_metric="prompt_level_strict", name="instructions")

    def iter_specs(self, options):
        return text_specs(
            self.read_test_examples(os.path.join(self.data_dir, "input_data.jsonl")), options, prompt=lambda row: row["prompt"], target=lambda row: row["instruction_id_list"],
            grader="instructions", reference=lambda row: {"protocol": 'ifeval'},
            generation_kwargs={"do_sample": False, "temperature": 0, "max_gen_toks": self.max_tokens},
        )
