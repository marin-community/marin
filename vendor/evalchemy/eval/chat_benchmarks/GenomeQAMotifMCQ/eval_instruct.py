"""Complete motif-MCQ cell for the bounded GenomeQA comparison."""

from eval.chat_benchmarks.GenomeQA.eval_instruct import GenomeQABenchmark
from eval.contracts.benchmark_metadata import MetricKind


class GenomeQAMotifMCQBenchmark(GenomeQABenchmark):
    METRIC_KIND_OVERRIDES = {"genomeqa_macro": MetricKind.BINARY}
    CELLS = {"6_motif/mcq": 100}
    METRICS = ("genomeqa_macro", "accuracy", "6_motif/mcq/accuracy")
