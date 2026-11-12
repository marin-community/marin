"""LiveCodeBench variant: dataset selection for the shared protocol."""
from datasets import load_dataset

from eval.chat_benchmarks.LiveCodeBench.eval_instruct import HF_HUB_CACHE, LiveCodeBenchBenchmark
from eval.contracts.eval_spec import SourceRef
from eval.sources.programs import live_code_records

class LiveCodeBenchV5Benchmark(LiveCodeBenchBenchmark):
    SOURCE = SourceRef("mlfoundations-dev/LCBv5-v2")
    REPEATS = 3

    def load_questions(self):
        rows = load_dataset(
            "mlfoundations-dev/LCBv5-v2", split="test", trust_remote_code=True,
            cache_dir=HF_HUB_CACHE,
        )
        return live_code_records(rows)
