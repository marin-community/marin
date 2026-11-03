"""LiveCodeBench variant: dataset selection for the shared protocol."""
from datasets import load_dataset

from eval.chat_benchmarks.LiveCodeBench.eval_instruct import HF_HUB_CACHE, LiveCodeBenchBenchmark
from eval.contracts.eval_spec import SourceRef
from eval.sources.programs import live_code_records

def filter_by_contest_date(example):
    target_months = ["2024-08", "2024-09", "2024-10", "2024-11", "2024-12", "2025-01"]
    return example["contest_date"][:7] in target_months

class LiveCodeBenchV5OfficialBenchmark(LiveCodeBenchBenchmark):
    SOURCE = SourceRef("livecodebench/code_generation_lite", revision="release_v5")
    REPEATS = 3

    def load_questions(self):
        rows = load_dataset(
            "livecodebench/code_generation_lite",
            split="test",
            version_tag="release_v5",
            cache_dir=HF_HUB_CACHE,
        )
        rows = rows.filter(filter_by_contest_date)
        return live_code_records(rows)
