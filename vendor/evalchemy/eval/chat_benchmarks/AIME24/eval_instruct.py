"""AIME 2024 source data."""

import json
from pathlib import Path

from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.generation_stops import END_OF_TURN_SEQUENCES
from eval.sources.common import text_specs
from eval.task import BaseBenchmark

PROMPT = "Problem: {problem}\nMark your solution with \\boxed\nAnswer:"
DATA_FILE = Path(__file__).parent / "data" / "aime24.json"


class AIME24Benchmark(BaseBenchmark):
    SOURCE = SourceRef("zwhe99/aime24", "train")
    METRICS = ("accuracy_avg",)
    PRIMARY_METRIC = "accuracy_avg"
    REDUCER = ReducerSpec(accuracy="accuracy_avg", stderr="accuracy_std_err")

    def __init__(self, data_file: str | Path = DATA_FILE, debug: bool = False, max_tokens: int = 32768, **kwargs):
        super().__init__(**kwargs)
        self.data_file, self.debug, self.max_tokens = Path(data_file), debug, max_tokens
        self.n_repeat = 10

    def load_questions(self):
        rows = [json.loads(line) for line in self.data_file.read_text().splitlines()]
        return rows[:2] if self.debug else rows

    def benchmark_size(self):
        return len(self.load_questions())

    def iter_specs(self, options):
        return text_specs(self.load_questions(), options, prompt=lambda row: PROMPT.format(**row),
                          target=lambda row: row["expected_answer"], grader="boxed_math",
                          generation_kwargs={"do_sample": True, "temperature": 0.7, "max_gen_toks": self.max_tokens,
                                      "until": list(END_OF_TURN_SEQUENCES)},
                          repeats=self.n_repeat, seed_repeats=True,
                          reference=lambda _row: {"comparator": "math_answers_equivalent"})
