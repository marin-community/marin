from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.sources.common import text_specs

import json
import logging
from rigging.filesystem.factory import open_url
from typing import Any, Dict, List, Optional  # noqa: F401


from eval.task import BaseBenchmark
from eval.data_sources import fetch_hf
from eval.generation_stops import END_OF_TURN_SEQUENCES

# Modified version of hendrycks_math with additional instruction to mark the solution with \\boxed
# https://github.com/mlfoundations/evalchemy/blob/e70a45e41cb2ada273d6bb98e75dba303ec31f8b/eval/chat_benchmarks/AMC23/eval_instruct.py#L15
PROMPT = """Problem: {problem}\nMark your solution with \\boxed\nAnswer:"""
HF_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"


class MATH500Benchmark(BaseBenchmark):
    SOURCE = SourceRef("HuggingFaceH4/MATH-500", "test", HF_REVISION)
    """
    MATH500 Benchmark for evaluating the math reasoning of LLMs.
    Link: https://huggingface.co/datasets/HuggingFaceH4/MATH-500

    Uses shared boxed-answer extraction and mathematical equivalence grading.
    """
    REDUCER = ReducerSpec()

    METRICS = ("accuracy",)
    PRIMARY_METRIC = "accuracy"

    def benchmark_size(self) -> int:
        return len(self.load_questions())

    def __init__(
        self,
        data_file: str | None = None,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        num_samples: int = 1,
        pass_at_k: Optional[Any] = None,
    ):
        """
        Initialize MATH500 benchmark.

        Args:
            data_file: File containing the MATH500 dataset (id, problem, reference_solution, expected_answer, source)
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
            num_samples: Number of completions per problem. 1 (default) = today's single-sample path.
            pass_at_k: k-list for pass@k aggregation (only used when num_samples > 1).
        """
        super().__init__(
            logger=logger,
            system_instruction=system_instruction,
            num_samples=num_samples,
            pass_at_k=pass_at_k,
        )
        self.data_file = data_file
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.n_repeat = 1

    def iter_specs(self, options):
        return text_specs(
            self.load_questions(), options,
            prompt=lambda example: PROMPT.format(problem=example['problem']),
            target=lambda example: example['answer'], grader="boxed_math",
            generation_kwargs={"do_sample": False, "temperature": 0.7, "max_gen_toks": self.max_new_tokens,
                        "until": list(END_OF_TURN_SEQUENCES)},
            repeats=self.n_repeat,
        )


    def load_questions(self) -> List[Dict[str, str]]:
        """Load MATH500 questions from the data file."""
        path = self.data_file or fetch_hf(
            self.SOURCE.dataset, "test.jsonl", revision=HF_REVISION, cache_prefix=self.dataset_cache,
        )
        with open_url(path, "rt", encoding="utf-8") as f:
            questions = [json.loads(x) for x in f]

        if self.debug:
            questions = questions[:2]
            self.logger.info(f"Debug mode enabled. Using only {len(questions)} questions.")

        self.logger.info(f"Loaded {len(questions)} questions from {path}")
        return questions

    def validate_prepared_data(self) -> None:
        questions = self.load_questions()
        if not questions or any(not row["problem"].strip() or not row["answer"].strip() for row in questions):
            raise ValueError("MATH500 questions and answers must not be empty")
        if self.data_file is None and len(questions) != (2 if self.debug else 500):
            raise ValueError(f"MATH500 expected {2 if self.debug else 500} questions, found {len(questions)}")
