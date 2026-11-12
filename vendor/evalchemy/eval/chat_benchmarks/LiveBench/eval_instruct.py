from eval.sources.common import conversation_specs
from eval.contracts.eval_spec import ReducerSpec
from eval.contracts.eval_spec import SourceRef
from typing import Optional
import logging
import glob


from eval.chat_benchmarks.LiveBench.livebench.common import get_categories_tasks, load_questions, load_questions_jsonl
from eval.task import BaseBenchmark


class LiveBenchBenchmark(BaseBenchmark):
    """
    LiveBench benchmark for evaluating language model responses.
    """


    def __init__(
        self,
        dataset_name: str = "live_bench",
        question_source: str = "huggingface",
        temperature: float = 0.0,
        debug: bool = False,
        num_choices: int = 1,
        release_date: str = "2024-08-31",
        max_tokens: int = 4096,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize LiveBench benchmark.

        Args:
            dataset_name: Name of the dataset
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.dataset_name = dataset_name
        self.question_source = question_source
        self.debug = debug
        self.release_date = release_date
        if self.debug:
            self.max_tokens = 128
            self.release_date = "2024-06-24"
        else:
            self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_choices = num_choices
        self.all_release_dates = ["2024-07-26", "2024-06-24", "2024-08-31", "2024-11-25"]
        self.data_path = f"eval/chat_benchmarks/LiveBench/data"






    SOURCE = SourceRef("live_bench", "test", "2024-08-31")
    METRICS = ("global_average",)
    PRIMARY_METRIC = "global_average"
    REDUCER = ReducerSpec(
        trial_metric=None,
        selected_trial=0,
        accuracy="global_average",
        groups={"task": "{group}", "category": "{group}", "date": "{group}"},
    )

    def load_questions(self):
        if self.question_source == "huggingface":
            categories, tasks = get_categories_tasks(self.dataset_name)
            for category in sorted(tasks):
                for name in sorted(tasks[category]):
                    rows = load_questions(categories[category], self.all_release_dates, self.release_date, name)
                    yield from rows[:10] if self.debug else rows
            return
        files = glob.glob(f"{self.data_path}/{self.dataset_name}/**/question.jsonl", recursive=True)
        for path in sorted(files):
            rows = load_questions_jsonl(path, self.all_release_dates, self.release_date)
            yield from rows[:10] if self.debug else rows

    def iter_specs(self, options):
        return conversation_specs(
            self.load_questions(), options, turns=lambda row: row["turns"], prompt_renderer="fastchat",
            target=lambda row: row.get("ground_truth", ""),
            grader="livebench", generation_kwargs={"temperature": self.temperature, "do_sample": self.temperature >= 1e-4,
                                            "max_gen_toks": self.max_tokens}, repeats=self.num_choices,
            tags=lambda row: {"task": row["task"], "category": row["category"],
                              "date": "subset_" + (row["livebench_removal_date"] or "no_date")},
        )
