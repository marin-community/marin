"""Full, pinned text-only OlympiadBench benchmark."""

import logging
from typing import Any, Dict, List, Optional

from eval.chat_benchmarks.OlympiadBench.eval_instruct import OlympiadBenchBenchmark
from eval.contracts.eval_spec import SourceRef

DEFAULT_DATASET = "lmms-lab/OlympiadBench"
DEFAULT_DATASET_REVISION = "c24a1391397fcfe50afaea9210d2c29066494b69"
DEFAULT_SPLIT = "test_en"


class OlympiadBenchFullBenchmark(OlympiadBenchBenchmark):
    """Evaluate every scorable English text-only OlympiadBench problem.

    The source revision is pinned so scores remain reproducible. `OlympiadBench`
    remains the legacy 30-example subset for continuity with historical runs.
    """

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET,
        dataset_revision: str = DEFAULT_DATASET_REVISION,
        dataset_split: str = DEFAULT_SPLIT,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        num_samples: int = 1,
        pass_at_k: Optional[Any] = None,
        annotator_model: Optional[str] = None,
    ):
        super().__init__(
            data_file=None,
            dataset_name=dataset_name,
            dataset_revision=dataset_revision,
            dataset_split=dataset_split,
            debug=debug,
            seed=seed,
            max_tokens=max_tokens,
            logger=logger,
            system_instruction=system_instruction,
            num_samples=num_samples,
            pass_at_k=pass_at_k,
            n_repeat=1,
            annotator_model=annotator_model,
        )

    def load_questions(self) -> List[Dict[str, Any]]:
        """Load the scorable English text-only questions from the pinned source."""
        questions = self._load_from_hf()
        self.logger.info(
            "Loaded %s questions from HF dataset %s@%s[%s]",
            len(questions),
            self.dataset_name,
            self.dataset_revision,
            self.dataset_split,
        )
        if self.debug:
            questions = questions[:2]
            self.logger.info("Debug mode enabled. Using only %s questions.", len(questions))
        return questions
    SOURCE = SourceRef(DEFAULT_DATASET, DEFAULT_SPLIT, DEFAULT_DATASET_REVISION)
