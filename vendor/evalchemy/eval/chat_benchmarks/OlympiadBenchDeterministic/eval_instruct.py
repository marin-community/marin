"""OlympiadBench subset scored only by Minerva/SymPy equivalence."""

import logging
from typing import List, Optional

from eval.chat_benchmarks.OlympiadBench.eval_instruct import OlympiadBenchBenchmark


class OlympiadBenchDeterministicBenchmark(OlympiadBenchBenchmark):
    """Use the historical 30-row subset without the LLM-judge fallback.

    Scores are not directly comparable with judge-backed OlympiadBench runs.
    """

    REQUIRES_JUDGE = False

    def __init__(
        self,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        num_samples: int = 1,
        pass_at_k: Optional[List[int] | str] = None,
        n_repeat: int = 1,
    ):
        super().__init__(
            debug=debug,
            seed=seed,
            max_tokens=max_tokens,
            logger=logger,
            system_instruction=system_instruction,
            num_samples=num_samples,
            pass_at_k=pass_at_k,
            n_repeat=n_repeat,
        )
