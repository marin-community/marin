from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.sources.common import text_specs
import logging


from datasets import load_dataset
from eval.task import BaseBenchmark
from typing import Optional

PREDS_PATH = "temp_swebench_preds.json"


class SWEBenchBenchmark(BaseBenchmark):
    """
    SWE-bench (Software Engineering) benchmark for evaluating
    Language Models' ability to resolve GitHub Issues.
    """

    def __init__(
        self,
        dataset_name: str = "princeton-nlp/SWE-bench_Lite",
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        max_tokens: int = 32768,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.debug = debug
        self.max_tokens = max_tokens
        self.dataset_name = dataset_name
        """
        Options for <dataset + split(s)>:
        - princeton-nlp/SWE-bench
        - princeton-nlp/SWE-bench_Lite
        - princeton-nlp/SWE-bench_Verified
        """




    SOURCE = SourceRef("princeton-nlp/SWE-bench_Lite")
    METRICS = ("accuracy",)
    PRIMARY_METRIC = "accuracy"
    REDUCER = ReducerSpec(name="outcome_report")

    def iter_specs(self, options):
        problems = {row["instance_id"]: row for row in load_dataset(self.dataset_name, split="test")}
        rows = ({**problems[row["instance_id"]], "text": row["text"]}
                for row in load_dataset("princeton-nlp/SWE-bench_oracle", split="test")
                if row["instance_id"] in problems)
        if self.debug:
            rows = list(rows)[:2]
        # Native reporting counts the full dataset, including unsubmitted rows.
        report_reference = iter(({
            "report_population": tuple(problems),
            "report_metadata": {"total_instances": len(problems), "schema_version": 2},
        },))
        return text_specs(
            rows, options, prompt=lambda row: row["text"], target=lambda row: row["patch"], grader="swe_patch",
            reference=lambda _row: next(report_reference, {}),
            generation_kwargs={"do_sample": False, "temperature": 0.2, "top_p": 0.95, "max_gen_toks": self.max_tokens},
        )
