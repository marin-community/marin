from eval.contracts.eval_spec import SourceRef
from eval.sources.common import text_specs
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional


from eval.generation_stops import END_OF_TURN_SEQUENCES
from eval.task import BaseBenchmark

# Same prompt shape as MATH500/AMC23/AIME24 (math reasoning benchmarks in this tree):
# the explicit "Mark your solution with \boxed" instruction makes answer extraction reliable.
PROMPT = """Problem: {problem}\nMark your solution with \\boxed\nAnswer:"""

# This is the hand-selected 30-example stratified export introduced in
# https://github.com/marin-community/evalchemy/pull/24. It came from
# lmms-lab/olympiadbench[test_en], but that export recorded neither an immutable
# dataset revision nor the selection script. Keep it for historical comparison
# only; `OlympiadBenchFull` is the reproducible benchmark.
DEFAULT_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "olympiadbench.jsonl")
DEFAULT_DATASET = "lmms-lab/olympiadbench"
DEFAULT_SPLIT = "test_en"


def _strip_dollars(s: str) -> str:
    """Strip a single pair of surrounding ``$...$`` (or ``$$...$$``) delimiters if present."""
    s = s.strip()
    if len(s) >= 2 and s.startswith("$") and s.endswith("$"):
        inner = s[1:-1].strip()
        if inner.startswith("$") and inner.endswith("$") and len(inner) >= 2:
            inner = inner[1:-1].strip()
        return inner
    return s


def _split_multiple_answers(s: str) -> List[str]:
    """Split a single reference-answer string that may itself carry several answers.

    OlympiadBench packs multiple answers into one ``final_answer`` list entry as
    ``$...$, $...$`` (each wrapped in dollars). Strip the dollars first, then split
    on commas that separate top-level dollar groups. Falls back to the whole string.
    """
    cleaned = _strip_dollars(s)
    if "$" in s:
        # Re-split on the original dollar-delimited groups, then strip each.
        groups = re.findall(r"\$([^$]*)\$", s)
        if groups:
            return [g.strip() for g in groups if g.strip()]
    if cleaned:
        return [cleaned]
    return []


def _flatten_reference_answers(raw: Any) -> List[str]:
    """Normalize the stored ``answer`` field into a flat list of candidate gold strings.

    Accepts a list of strings (the OlympiadBench ``final_answer`` shape, where each
    entry may itself contain several dollar-delimited answers) and returns the union of
    the individual candidate answers. Strings and single-string inputs are also accepted.
    """
    if raw is None:
        return []
    if isinstance(raw, (str, int, float)):
        raw = [str(raw)]
    out: List[str] = []
    for item in raw:
        out.extend(_split_multiple_answers(str(item)))
    return out


class OlympiadBenchBenchmark(BaseBenchmark):
    SOURCE = SourceRef("lmms-lab/olympiadbench", "test_en")
    """
    OlympiadBench Benchmark for evaluating competition math/physics reasoning of LLMs.
    Link: https://huggingface.co/datasets/lmms-lab/olympiadbench

    Uses the text-only English split (``test_en``). Each problem's reference answer may
    be a single value or a list of acceptable values; grading is correct-if-any-match
    through Minerva/SymPy first and an LLM equivalence judge for unresolved answers.
    Follows the MATH500/AMC23 pattern in this tree: the model is asked to box its
    final answer.
    """

    METRICS = ("accuracy",)
    PRIMARY_METRIC = "accuracy"
    REQUIRES_JUDGE = True

    def benchmark_size(self) -> int | None:
        return len(self.load_questions()) if self.data_file is not None else None

    def benchmark_metrics(self):
        if self.num_samples > 1:
            return super().benchmark_metrics()
        return ("accuracy_avg",) if self.n_repeat > 1 else self.METRICS

    def benchmark_primary_metric(self):
        if self.num_samples > 1:
            return super().benchmark_primary_metric()
        return "accuracy_avg" if self.n_repeat > 1 else self.PRIMARY_METRIC

    def __init__(
        self,
        data_file: Optional[str] = DEFAULT_DATA_FILE,
        dataset_name: str = DEFAULT_DATASET,
        dataset_revision: Optional[str] = None,
        dataset_split: str = DEFAULT_SPLIT,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        num_samples: int = 1,
        pass_at_k: Optional[Any] = None,
        n_repeat: int = 10,
        annotator_model: Optional[str] = None,
    ):
        """
        Initialize OlympiadBench benchmark.

        Args:
            data_file: Local JSONL containing the legacy 30-example subset.
            dataset_name: Source dataset identifier retained with the subset provenance.
            dataset_revision: Source revision when loading a reproducible derived task.
            dataset_split: Source dataset split.
            debug: If set, only evaluate on 2 examples.
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            max_tokens: Max generation tokens. These are hard olympiad problems; default 32768.
            logger: Optional logger instance.
            system_instruction: Optional system instruction for the model.
            num_samples: Number of completions per problem. 1 (default) = single-sample path.
            pass_at_k: k-list for pass@k aggregation (only used when num_samples > 1).
            n_repeat: Number of seeded repetitions for the subset's single-sample score.
            annotator_model: Explicit judge model, or ``auto`` to use ``$JUDGE_MODEL``.
        """
        super().__init__(
            logger=logger,
            system_instruction=system_instruction,
            num_samples=num_samples,
            pass_at_k=pass_at_k,
        )
        self.data_file = data_file
        self.dataset_name = dataset_name
        self.dataset_revision = dataset_revision
        self.dataset_split = dataset_split
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.n_repeat = n_repeat
        self.judge_model = annotator_model









    def load_questions(self) -> List[Dict[str, Any]]:
        """Load the bundled legacy OlympiadBench subset."""
        assert self.data_file is not None
        with open(self.data_file, "r") as f:
            questions = [json.loads(x) for x in f]
        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")

        if self.debug:
            questions = questions[:2]
            self.logger.info(f"Debug mode enabled. Using only {len(questions)} questions.")

        return questions

    def _load_from_hf(self) -> List[Dict[str, Any]]:
        """Project the HF dataset into the local JSONL record shape."""
        from datasets import load_dataset

        cache_dir = os.environ.get("HF_HUB_CACHE")
        ds = load_dataset(
            self.dataset_name,
            split=self.dataset_split,
            revision=self.dataset_revision,
            cache_dir=cache_dir,
        )

        out: List[Dict[str, Any]] = []
        for ex in ds:
            source = ex.get("source") or ""
            if "_TO_" not in source:
                continue
            final_answer = ex.get("final_answer")
            if not final_answer:
                continue
            subject = "mathematics" if "maths" in source else ("physics" if "physics" in source else "unknown")
            context = ex.get("context")
            question = ex.get("question") or ""
            out.append(
                {
                    "id": ex.get("question_id"),
                    "problem": (context + "\n\n" if context else "") + question,
                    "question": question,
                    "context": context,
                    "answer": list(final_answer),
                    "subject": subject,
                    "subfield": ex.get("subfield"),
                    "unit": ex.get("unit"),
                    "answer_type": ex.get("answer_type"),
                    "is_multiple_answer": bool(ex.get("is_multiple_answer")),
                    "error": ex.get("error"),
                    "source": source,
                }
            )
        return out



    def iter_specs(self, options):
        return text_specs(
            self.load_questions(), options, prompt=lambda row: PROMPT.format(problem=row["problem"]),
            target=lambda row: row["answer"], grader="math_equivalence",
            reference=lambda row: {"question": row["problem"], "answers": _flatten_reference_answers(row["answer"]),
                                   "judge": self.REQUIRES_JUDGE},
            generation_kwargs={"do_sample": self.n_repeat > 1, "temperature": 0.7, "top_p": 1.0,
                        "max_gen_toks": self.max_new_tokens, "until": list(END_OF_TURN_SEQUENCES)},
            repeats=self.n_repeat,
        )
