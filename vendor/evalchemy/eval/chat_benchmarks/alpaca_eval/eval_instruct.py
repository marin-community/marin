from eval.sources.common import text_specs
from eval.contracts.eval_spec import ReducerSpec
from eval.contracts.eval_spec import SourceRef
from alpaca_eval import constants
from typing import Optional
import logging
import datasets

from alpaca_eval.constants import DEFAULT_ANNOTATOR_CONFIG
from eval.graders.alpaca import annotation_config
from eval.task import BaseBenchmark




class AlpacaBenchmark(BaseBenchmark):
    """
    Alpaca benchmark for evaluating language model responses on instruction following.
    """

    REQUIRES_JUDGE = True
    SOURCE = SourceRef("tatsu-lab/alpaca_eval", "eval", "alpaca_eval")
    METRICS = ("win_rate", "standard_error", "n_total", "avg_length", "discrete_win_rate", "length_controlled_winrate", "lc_standard_error")
    PRIMARY_METRIC = "length_controlled_winrate"

    ANNOTATOR_CONFIG_MAP = {
        "gpt-4o-mini-2024-07-18": "weighted_alpaca_eval_gpt-4o-mini-2024-07-18",
        "gpt-4-1106-preview": "weighted_alpaca_eval_gpt4_turbo",
        "gpt-4": "alpaca_eval_gpt4_0613",
        "auto": DEFAULT_ANNOTATOR_CONFIG,
    }

    def __init__(
        self,
        dataset_name: str = "tatsu-lab/alpaca_eval",
        subset: str = "alpaca_eval",
        split: str = "eval",
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.5,
        do_sample: bool = True,
        debug: bool = False,
        annotator_model: str = "gpt-4o-mini-2024-07-18",
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize Alpaca benchmark.

        Args:
            dataset_name: HuggingFace dataset name
            subset: Dataset subset name
            split: Dataset split to use
            max_tokens: Maximum number of tokens for generation
            temperature: Sampling temperature
            do_sample: Whether to use sampling for generation
            debug: debug: If True, only evaluate first 2 examples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.max_tokens = max_tokens if max_tokens is not None else 1024
        self.temperature = temperature
        self.do_sample = do_sample
        self.debug = debug
        self.annotator_conf = self.ANNOTATOR_CONFIG_MAP[annotator_model]
        self.judge_model = annotation_config(self.annotator_conf)[1]["completions_kwargs"]["model_name"]

    def load_dataset(self) -> datasets.Dataset:
        """Load the evaluation dataset."""
        try:
            dataset = datasets.load_dataset(self.dataset_name, self.subset, trust_remote_code=True)[self.split]

            if self.debug:
                dataset = dataset.select(range(2))
                self.logger.info(f"Debug mode: using 2 examples")

            self.logger.info(f"Loaded {len(dataset)} examples for evaluation")
            return dataset

        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise




    REDUCER = ReducerSpec(trial_metric=None, name="alpaca")

    def iter_specs(self, options):
        references = {row["instruction"]: row for row in constants.ALPACAEVAL_REFERENCE_OUTPUTS()}
        return text_specs(
            self.load_dataset(), options, prompt=lambda row: row["instruction"],
            target=lambda row: references[row["instruction"]]["output"], grader="alpaca",
            reference=lambda row: {"config": self.annotator_conf,
                                   "baseline": references[row["instruction"]]["generator"]},
            generation_kwargs={"temperature": self.temperature, "do_sample": self.do_sample, "max_gen_toks": self.max_tokens},
        )
