from eval.sources.common import conversation_specs
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from dataclasses import dataclass
from typing import Optional
import glob
import json
import os
import logging
from pathlib import Path


from eval.constants import AUTO_ANNOTATOR_MODEL
from eval.task import BaseBenchmark


# MT-Bench source construction needs only these bundled data conventions. Keep
# them here so importing the benchmark does not load FastChat's serving stack.
NEED_REF_CATS = ["math", "reasoning", "coding", "arena-hard-200"]
temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}


def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load the requested contiguous range of bundled MT-Bench questions."""
    with open(question_file) as questions:
        return [json.loads(line) for line in questions if line][begin:end]


def load_model_answers(answer_dir: str):
    """Load bundled reference answers indexed by model name and question id."""
    answers = {}
    for filename in sorted(glob.glob(os.path.join(answer_dir, "*.jsonl"))):
        with open(filename) as answer_file:
            answers[Path(filename).stem] = {
                record["question_id"]: record
                for line in answer_file
                if (record := json.loads(line))
            }
    return answers


def load_judge_prompts(prompt_file: str):
    """Load bundled judge prompts indexed by their stable prompt name."""
    with open(prompt_file) as prompts_file:
        return {
            record["name"]: record
            for line in prompts_file
            if (record := json.loads(line))
        }


@dataclass
class MTBenchConfig:
    """Configuration for MTBench evaluation."""

    question_begin: Optional[int] = None
    question_end: Optional[int] = None
    max_new_token: int = 1024
    judge_file: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fastchat", "llm_judge", "data", "judge_prompts.jsonl"
    )
    judge_model: str = "gpt-4o-mini-2024-07-18"
    mode: str = "single"


class MTBenchBenchmark(BaseBenchmark):
    """
    MTBench benchmark for evaluating multi-turn chat capabilities.
    """

    REQUIRES_OPENAI_ANNOTATOR = False  # Can also be anthropic

    def __init__(
        self,
        base_path: str = os.path.dirname(os.path.abspath(__file__)),
        config: Optional[MTBenchConfig] = None,
        debug: bool = False,
        annotator_model: str = "gpt-4o-mini-2024-07-18",
        max_tokens: int = 1024,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize MTBench benchmark.

        Args:
            base_path: Base directory for bundled MTBench data
            config: MTBench configuration object
            debug: If True, run in debug mode on 2 samples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        base_path = Path(base_path)
        if annotator_model == AUTO_ANNOTATOR_MODEL:
            annotator_model = "gpt-4"
        if config:
            print(f"Warning: Overwriting config.judge_model = {annotator_model} ")
            config.judge_model = annotator_model
        self.config = config or MTBenchConfig(judge_model=annotator_model)
        self.config.max_new_token = max_tokens
        self.debug = debug

        # Setup paths
        self.data_path = base_path / "fastchat/llm_judge/data/mt_bench"
        self.question_file = self.data_path / "question.jsonl"
        self.ref_answer_dir = self.data_path / "reference_answer"
        self.judge_model = self.config.judge_model

    REQUIRES_JUDGE = True
    METRICS = ("Turn 1", "Turn 2", "Average")
    PRIMARY_METRIC = "Average"
    REDUCER = ReducerSpec(trial_metric=None, accuracy="Average")
    SOURCE = SourceRef("lmsys/mt-bench", "test")

    def iter_specs(self, options):
        if self.config.mode != "single":
            raise ValueError("MTBench pairwise mode requires an explicit comparison source")
        questions = load_questions(self.question_file, self.config.question_begin, self.config.question_end)
        if self.debug:
            questions = questions[:2]
        templates = load_judge_prompts(self.config.judge_file)
        references = load_model_answers(self.ref_answer_dir)
        def reference(row):
            reference_based = row["category"] in NEED_REF_CATS
            key = "single-math-v1" if reference_based else "single-v1"
            answers = references["gpt-4"][row["question_id"]]["choices"][0]["turns"] if reference_based else []
            return {"templates": [templates[key], templates[key + "-multi-turn"]], "answers": answers}
        return conversation_specs(
            questions, options, turns=lambda row: row["turns"], target=lambda row: row["turns"],
            grader="mtbench", reference=reference,
            generation_kwargs=lambda row: {"max_gen_toks": self.config.max_new_token,
                                    "do_sample": temperature_config.get(row["category"], 0.7) >= 1e-4,
                                    "temperature": temperature_config.get(row["category"], 0.7)},
            seed_repeats=False,
        )
