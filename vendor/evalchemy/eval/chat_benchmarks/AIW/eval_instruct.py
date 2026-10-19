from __future__ import annotations
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.sources.common import text_specs
import json
import logging
import os
from typing import Dict, List, Optional


from eval.task import BaseBenchmark


class AIWBenchmark(BaseBenchmark):
    SOURCE = SourceRef("AIW", "local")
    METRICS = ("overall_accuracy", "overall_variance")
    PRIMARY_METRIC = "overall_accuracy"
    REDUCER = ReducerSpec(name="last_trial", accuracy="overall_accuracy", groups={"id": "per_id_stats"})
    """
    AIW Benchmark for evaluating the math reasoning of LLMs.
    """

    TARGET_IDS = {577, 580, 581, 582, 583, 584, 559, 560, 561, 562, 563, 564, 637, 638, 639, 640, 641, 642}

    def __init__(
        self,
        data_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "aiw_data.json"),
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        n_trials: int = 100,  # Run 100 trials
    ):
        """
        Initialize AIW benchmark.

        Args:
            data_file: File containing the AIW dataset
            debug: If set, only evaluate on 2 examples
            seed: Random seed
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file
        self.debug = debug
        self.max_new_tokens = max_tokens
        self.seed = seed
        self.n_repeat = n_trials


    def load_questions(self) -> List[Dict[str, str]]:
        """Load AIW questions from the data file."""
        with open(self.data_file, "r") as f:
            questions = json.load(f)
            
        if self.debug:
            questions = questions[:2]
            self.logger.info(f"Debug mode enabled. Using only {len(questions)} questions.")
 
        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")
        return questions

    def iter_specs(self, options):
        rows = (row for row in self.load_questions() if row["id"] in self.TARGET_IDS)
        return text_specs(
            rows, options, prompt=lambda row: row["prompt"], target=lambda row: str(row["right_answer"]),
            grader="answer_number", reference=lambda _row: {"comparator": "hendrycks_math"}, generation_kwargs={"do_sample": True, "temperature": 1.0,
                                                "max_gen_toks": self.max_new_tokens}, repeats=self.n_repeat,
            tags=lambda row: {"id": str(row["id"])},
        )
