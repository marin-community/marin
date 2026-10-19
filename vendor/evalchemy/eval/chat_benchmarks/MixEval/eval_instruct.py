from eval.sources.common import text_specs
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from .mix_eval.prompts.evaluation_prompts import construct_prompt_freeform, construct_prompt_multichoice
from typing import Optional
import logging
import os
import json


from eval.constants import AUTO_ANNOTATOR_MODEL
from eval.task import BaseBenchmark


class MixEvalBenchmark(BaseBenchmark):
    """
    MixEval benchmark for evaluating language model responses on various tasks.
    """

    REQUIRES_JUDGE = True
    SOURCE = SourceRef("MixEval", "test", "2024-06-01")
    METRICS = ("overall",)
    PRIMARY_METRIC = "overall"
    REDUCER = ReducerSpec(name="rounded_split_mean", trial_metric=None)

    def __init__(
        self,
        benchmark: str = "mixeval",
        version: str = "2024-06-01",
        data_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mix_eval", "data"),
        annotator_model: str = "gpt-4o-mini-2024-07-18",
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize MixEval benchmark.

        Args:
            benchmark: Benchmark name
            version: Benchmark version
            data_path: Path to evaluation data
            annotator_model: Model selected for the judge requests
            debug: If set, only evaluate on 2 examples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        if annotator_model == AUTO_ANNOTATOR_MODEL:
            annotator_model = "gpt-3.5-turbo-0125"
        self.judge_model = annotator_model
        self.benchmark = benchmark
        self.version = version
        self.data_path = data_path
        self.debug = debug

    def iter_specs(self, options):
        folder = "mixeval-hard" if self.benchmark == "mixeval_hard" else "mixeval"
        for split, filename, formatter in [("freeform", "free-form.json", construct_prompt_freeform),
                                           ("multichoice", "multiple-choice.json", construct_prompt_multichoice)]:
            path = os.path.join(self.data_path, f"mixeval-{self.version}", folder, filename)
            with open(path) as data:
                rows = [{**row, "id": identity} for identity, row in json.load(data).items()]
            if self.debug:
                rows = rows[:2]
            yield from text_specs(
                rows, options, namespace=split, prompt=formatter, target=lambda row: row["target"],
                grader="mixeval", reference=lambda row: {"kind": split},
                generation_kwargs={"temperature": 1.0, "do_sample": True},
                seed_repeats=False,
                tags=lambda row: {
                    "split": split,
                    "benchmark_split": f"{row['benchmark_name']}|{split}",
                },
            )
