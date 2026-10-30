from eval.sources.common import text_specs
from eval.contracts.eval_spec import ReducerSpec, SourceRef
"""IFBench: instruction-following eval against the real allenai/IFBench_test set.

IFBench is AI2's harder successor to IFEval -- 300 WildChat-sourced prompts each
carrying one or more *verifiable* constraints drawn from 58 distinct types
(``count:keywords_multiple``, ``format:emoji``, ``ratio:stop_words``,
``words:palindrome``, ...). Constraints are checked programmatically by the
canonical AI2 graders vendored alongside this file (``instructions.py`` etc.),
so the reported accuracy is directly comparable to the IFBench paper's.

This benchmark mirrors the IFEval chat-benchmark structure (load → generate →
evaluate), but pulls its data from HuggingFace rather than a vendored JSONL.
"""

import logging
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset

from eval.task import BaseBenchmark



class IFBenchBenchmark(BaseBenchmark):
    SOURCE = SourceRef("allenai/IFBench_test", "train")
    METRICS = ("strict_prompt_accuracy", "strict_instruction_accuracy", "loose_prompt_accuracy", "loose_instruction_accuracy")
    PRIMARY_METRIC = "strict_prompt_accuracy"
    """Instruction-following benchmark over ``allenai/IFBench_test``."""

    def __init__(
        self,
        num_examples: Optional[int] = None,
        debug: bool = False,
        max_tokens: int = 1024,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """Initialize the IFBench instruction-following benchmark.

        Args:
            num_examples: Cap on the number of examples to evaluate (None = all 300).
            debug: If True, only evaluate 2 examples.
            max_tokens: Maximum number of tokens for generation.
            logger: Optional logger instance.
            system_instruction: Optional system instruction for the model.
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.num_examples = num_examples
        self.debug = debug
        self.max_tokens = max_tokens

    def load_questions(self) -> List[Dict[str, Any]]:
        """Load the IFBench prompts from the ``allenai/IFBench_test`` HF dataset.

        Each row's ``prompt`` already carries the constraint inline as natural
        language (e.g. ``"... Include exactly 5 numbers in the response."``); the
        machine-checkable constraint metadata lives alongside it in
        ``instruction_id_list`` and ``kwargs`` and is consumed by the grader.
        """
        dataset: Dataset = load_dataset("allenai/IFBench_test", split="train")
        self.logger.info("Loaded %d examples from allenai/IFBench_test", len(dataset))

        questions: List[Dict[str, Any]] = []
        for ex in dataset:
            # Coerce numpy/pandas scalar types to native Python so the row is
            # JSON-serializable when we later write the response file.
            questions.append(
                {
                    "key": int(ex["key"]) if ex["key"] is not None else len(questions),
                    "prompt": ex["prompt"],
                    "instruction_id_list": list(ex["instruction_id_list"]),
                    "kwargs": [dict(kw) for kw in ex["kwargs"]],
                }
            )

        if self.debug:
            questions = questions[:2]
            self.logger.info("Debug mode: using 2 examples")
        elif self.num_examples is not None:
            questions = questions[: self.num_examples]

        return questions






    REDUCER = ReducerSpec(trial_metric="strict_prompt_accuracy", name="instructions")

    def iter_specs(self, options):
        return text_specs(
            self.load_questions(), options, prompt=lambda row: row["prompt"], target=lambda row: row["instruction_id_list"],
            grader="instructions", reference=lambda row: {"protocol": 'ifbench'},
            generation_kwargs={"do_sample": False, "temperature": 0, "max_gen_toks": self.max_tokens},
        )
