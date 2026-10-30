from eval.sources.common import text_specs
from eval.contracts.eval_spec import SourceRef
import json
import logging
import os
from typing import Any, Dict, List, Optional  # noqa: F401


from eval.task import BaseBenchmark

# Document-grounded prompt: the question is unanswerable without the supporting
# passage, so the full-page evidence from the source filing is rendered ahead of
# the question. The instruction mirrors the canonical FinanceBench framing -- a
# direct, concise answer grounded in the supplied document excerpt.
PROMPT = """Based on the following financial document excerpt, answer the question.

Document:
{evidence_text}

Question: {question}

Provide a direct, concise answer."""

class FinanceBenchBenchmark(BaseBenchmark):
    """
    FinanceBench Benchmark for evaluating financial document Q&A.

    FinanceBench (PatronusAI/financebench) tests whether an LLM can answer questions that
    require reading and reasoning over financial documents -- 10-K / 10-Q filings and
    earnings-call transcripts -- spanning numerical, boolean, and summary question types.

    Each question is shipped with the supporting passage from the source filing, which is
    injected into the prompt as document context so the answer is grounded in the text
    rather than recalled from parametric memory. Without that context the benchmark is
    unsolvable: the questions are about specific line items in specific filings.

    Grading is LLM-as-judge (SimpleQA-style correct / incorrect / not_attempted) rather
    than exact match, because financial answers frequently differ from the gold only in
    formatting or unit surface form (e.g. "$1,577M" vs "$1577.00"). The shared
    answer-equivalence judge is configured by the shared runtime.

    Link: https://github.com/patronus-ai/financebench
    """

    METRICS = ("accuracy",)
    PRIMARY_METRIC = "accuracy"
    SOURCE = SourceRef("PatronusAI/financebench", "test")
    REQUIRES_JUDGE = True

    def __init__(
        self,
        data_file: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "financebench.jsonl"
        ),
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 4096,
        annotator_model: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize FinanceBench benchmark.

        Args:
            data_file: JSONL file of FinanceBench items
                (id, question, answer, evidence_text, doc_name, company, question_type).
            debug: If set, only evaluate on 2 examples.
            seed: Random seed for reproducibility (deterministic at temperature 0).
            max_tokens: Max generation tokens. 4096 by default -- factual Q&A answers
                are short, but the prompt's document context can be long.
            annotator_model: Override the judge model. The CLI's ``auto`` sentinel is
                treated as unset, then falls back to ``$JUDGE_MODEL`` and finally
                ``gpt-4o-mini`` (the evalchemy-standard cheap judge).
            logger: Optional logger instance.
            system_instruction: Optional system instruction for the model.
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.judge_model = annotator_model





    def load_questions(self) -> List[Dict[str, Any]]:
        """Load FinanceBench questions from the local data file.

        The shipped ``data/financebench.jsonl`` is a 50-item deterministic sample of the
        open-source slice of ``patronus-ai/financebench``
        (``data/financebench_open_source.jsonl``, 150 rows). Each item carries the
        ``evidence_text`` -- the full page of the source filing the question is grounded
        in -- so the benchmark runs offline the same way MATH500 / AIME24 do.
        """
        with open(self.data_file, "r") as f:
            questions = [json.loads(line) for line in f if line.strip()]

        if self.debug:
            questions = questions[:2]
            self.logger.info(
                f"Debug mode enabled. Using only {len(questions)} questions."
            )

        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")
        return questions

    def iter_specs(self, options):
        return text_specs(
            self.load_questions(), options,
            prompt=lambda row: PROMPT.format(question=row["question"], evidence_text=row["evidence_text"]),
            target=lambda row: row["answer"], grader="equivalence_judge",
            reference=lambda row: {"question": row["question"], "answers": [str(row["answer"])]},
            generation_kwargs={"do_sample": False, "temperature": 0, "max_gen_toks": self.max_new_tokens},
        )
