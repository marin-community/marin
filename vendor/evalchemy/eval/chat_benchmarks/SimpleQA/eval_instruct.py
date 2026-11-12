from eval.sources.common import text_specs
from eval.contracts.eval_spec import ReducerSpec
from eval.contracts.eval_spec import SourceRef
"""OpenAI SimpleQA benchmark."""

import ast
import csv
import logging
from rigging.filesystem.factory import open_url
from typing import Any, Dict, List, Optional


from eval.task import BaseBenchmark
from eval.data_sources import fetch_hf

DATASET_SIZE = 4_326
HF_REVISION = "e319282ab125c3dbd0c7fd00be2e4dd54e7e8f94"




class SimpleQABenchmark(BaseBenchmark):
    """Evaluate short-form factual answers with OpenAI's SimpleQA classifier."""

    METRICS = ("accuracy", "accuracy_given_attempted", "f1")
    PRIMARY_METRIC = "accuracy"
    SOURCE = SourceRef("basicv8vc/SimpleQA", "train", HF_REVISION)
    REQUIRES_JUDGE = True

    def __init__(
        self,
        data_file: str | None = None,
        debug: bool = False,
        seed: tuple[int, int, int, int] = (0, 1234, 1234, 1234),
        max_tokens: int = 512,
        annotator_model: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.judge_model = annotator_model

    def benchmark_size(self) -> int:
        return DATASET_SIZE

    def validate_prepared_data(self) -> None:
        questions = self.load_questions()
        expected_size = min(2, self.benchmark_size()) if self.debug else self.benchmark_size()
        if len(questions) != expected_size:
            raise ValueError(f"SimpleQA expected {expected_size} questions, found {len(questions)}")
        if any(not example["question"].strip() or not example["answer"].strip() for example in questions):
            raise ValueError("SimpleQA questions and answers must not be empty")

    def _load_source_questions(self) -> List[Dict[str, Any]]:
        path = self.data_file or fetch_hf(
            self.SOURCE.dataset, "simple_qa_test_set.csv", revision=HF_REVISION, cache_prefix=self.dataset_cache,
        )
        with open_url(path, "rt", newline="", encoding="utf-8") as data:
            questions = []
            for source_index, row in enumerate(csv.DictReader(data)):
                metadata = ast.literal_eval(row["metadata"])
                questions.append(
                    {
                        "source_index": source_index,
                        "question": row["problem"],
                        "answer": row["answer"],
                        "metadata": metadata,
                    }
                )
        return questions

    def load_questions(self) -> List[Dict[str, Any]]:
        """Load the pinned canonical CSV and retain its source row identity."""
        questions = self._load_source_questions()
        if self.debug:
            return questions[:2]
        return questions










    REDUCER = ReducerSpec(name="classification")

    def iter_specs(self, options):
        return text_specs(
            self.load_questions(), options, prompt=lambda row: row["question"], target=lambda row: row["answer"],
            grader="simpleqa", generation_kwargs={"do_sample": False, "temperature": 0,
                                           "max_gen_toks": self.max_new_tokens},
        )
