from eval.sources.common import text_specs
from eval.contracts.eval_spec import ReducerSpec
from eval.contracts.eval_spec import SourceRef
from typing import Dict, Optional
from dataclasses import dataclass
import os
import logging
from datasets import load_dataset


from eval.constants import AUTO_ANNOTATOR_MODEL
from eval.task import BaseBenchmark

@dataclass
class WildBenchConfig:
    """Configuration for WildBench evaluation."""

    # Dataset configuration
    dataset_version: str = "v2"
    split: str = "test"
    start_idx: int = 0
    end_idx: int = -1

    # Model configuration
    max_words_to_eval: int = 1000
    temperature = 0.0
    top_p = 1.0
    model: str = "gpt-4o-mini-2024-07-18"
    eval_template: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "evaluation", "eval_template.score.v2.md"
    )

    # Task weights
    task_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.task_weights is None:
            self.task_weights = {
                "Creative Tasks": 0.5,
                "Planning & Reasoning": 1.25,
                "Math & Data Analysis": 1.0,
                "Information/Advice seeking": 0.75,
                "Coding & Debugging": 1.25,
            }


class WildBenchBenchmark(BaseBenchmark):
    """
    WildBench benchmark for evaluating diverse real-world tasks.
    """

    REQUIRES_JUDGE = True
    SOURCE = SourceRef("allenai/WildBench", "test", "v2")
    METRICS = ("score", "adjusted_score", "task_macro_score", "adjusted_task_macro_score", "avg_output_length")
    PRIMARY_METRIC = "adjusted_score"

    def __init__(
        self,
        config: Optional[WildBenchConfig] = None,
        annotator_model: str = "gpt-4o-mini-2024-07-18",
        debug: bool = False,
        max_tokens: int = 1024,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize WildBench benchmark.

        Args:
            config: WildBench configuration
            debug: If True, run in debug mode on 2 samples
            logger: Optional logger instance
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        if annotator_model == AUTO_ANNOTATOR_MODEL:
            annotator_model = "gpt-4-1106-preview"
        if config:
            self.logger.warning(f"Overwriting config.judge_model = {annotator_model} ")
            config.model = annotator_model
        self.config = config or WildBenchConfig(model=annotator_model)
        self.judge_model = self.config.model
        self.REDUCER = ReducerSpec(trial_metric=None, name="weighted_macro", accuracy="adjusted_score", group_weights=self.config.task_weights)
        self.debug = debug
        self.max_new_tokens = max_tokens

        # Task category mapping
        self.task_group_mapping = {
            "Information seeking": "Information/Advice seeking",
            "Creative Writing": "Creative Tasks",
            "Coding & Debugging": "Coding & Debugging",
            "Reasoning": "Planning & Reasoning",
            "Editing": "Creative Tasks",
            "Math": "Math & Data Analysis",
            "Planning": "Planning & Reasoning",
            "Brainstorming": "Creative Tasks",
            "Role playing": "Creative Tasks",
            "Advice seeking": "Information/Advice seeking",
            "Data Analysis": "Math & Data Analysis",
            "Others": "Creative Tasks",
        }







    def iter_specs(self, options):
        rows = load_dataset("allenai/WildBench", self.config.dataset_version, split=self.config.split)
        end = self.config.end_idx if self.config.end_idx >= 0 else len(rows)
        rows = list(rows)[self.config.start_idx:end]
        if self.debug:
            rows = rows[:2]
        return text_specs(
            rows, options, prompt=lambda row: row["conversation_input"], target=lambda row: row["checklist"],
            grader="wildbench", reference=lambda row: {"template": self.config.eval_template,
                                                      "max_words": self.config.max_words_to_eval},
            generation_kwargs={"temperature": self.config.temperature, "top_p": self.config.top_p,
                        "max_gen_toks": self.max_new_tokens},
            tags=lambda row: {"category": tuple(sorted(
                {self.task_group_mapping[tag] for tag in [row["primary_tag"], *row["secondary_tags"]]}))},
        )
