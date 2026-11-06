# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Google DeepMind's 1,000-question SimpleQA Verified benchmark."""

import csv
import logging
from typing import Any, Dict, List, Optional

from rigging.filesystem.factory import open_url

from eval.chat_benchmarks.SimpleQA.eval_instruct import SimpleQABenchmark
from eval.contracts.eval_spec import SourceRef
from eval.data_sources import fetch_hf

DATASET_SIZE = 1_000
HF_DATASET = "google/simpleqa-verified"
HF_REVISION = "0dc97e0d28d8233463e005cdc4475cc2a13ba2dc"
HF_FILENAME = "simpleqa_verified.csv"
_BOOLEANS = {"True": True, "False": False}


class SimpleQAVerifiedBenchmark(SimpleQABenchmark):
    """Evaluate short-form factuality with the SimpleQA Verified questions and autorater prompt."""

    SOURCE = SourceRef(HF_DATASET, "eval", HF_REVISION)
    GRADER = "simpleqa_verified"

    def __init__(
        self,
        data_file: str | None = None,
        debug: bool = False,
        seed: tuple[int, int, int, int] = (0, 1234, 1234, 1234),
        max_tokens: int = 2048,
        annotator_model: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(
            data_file=data_file,
            debug=debug,
            seed=seed,
            max_tokens=max_tokens,
            annotator_model=annotator_model,
            logger=logger,
            system_instruction=system_instruction,
        )

    def benchmark_size(self) -> int:
        return DATASET_SIZE

    def _load_source_questions(self) -> List[Dict[str, Any]]:
        path = self.data_file or fetch_hf(
            HF_DATASET, HF_FILENAME, revision=HF_REVISION, cache_prefix=self.dataset_cache,
        )
        with open_url(path, "rt", newline="", encoding="utf-8") as data:
            return [
                {
                    "source_index": source_index,
                    "question": row["problem"],
                    "answer": row["answer"],
                    "metadata": {
                        # Row index in OpenAI's original SimpleQA CSV, for pairing with SimpleQA results.
                        "original_index": int(row["original_index"]),
                        "topic": row["topic"],
                        "answer_type": row["answer_type"],
                        "multi_step": _BOOLEANS[row["multi_step"]],
                        "requires_reasoning": _BOOLEANS[row["requires_reasoning"]],
                        "urls": row["urls"],
                    },
                }
                for source_index, row in enumerate(csv.DictReader(data))
            ]
