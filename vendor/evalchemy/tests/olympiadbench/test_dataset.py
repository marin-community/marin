import json

from eval.chat_benchmarks.OlympiadBench.eval_instruct import OlympiadBenchBenchmark
from eval.contracts.eval_spec import RequestKind, SourceRef
from eval.sources.common import SourceOptions
from tests.contracts.source_harness import evaluate_source


class _CorrectBackend:
    async def execute(self, request):
        assert request.kind is RequestKind.GENERATE
        return r"Reasoning. \boxed{\frac{1}{2}}"


def test_subset_source_repeats_before_common_reduction(tmp_path):
    data = tmp_path / "subset.jsonl"
    data.write_text(json.dumps({"id": "half", "problem": "Compute one half.", "answer": ["0.5"]}) + "\n")
    source = OlympiadBenchBenchmark(data_file=data, n_repeat=2)
    metrics, samples = evaluate_source(source, SourceOptions("OlympiadBench", SourceRef("local")), _CorrectBackend())

    assert [sample["sample_repeat"] for sample in samples] == [0, 1]
    assert metrics["accuracy_avg"] == 1.0


def test_subset_source_preserves_all_acceptable_answers(tmp_path):
    data = tmp_path / "subset.jsonl"
    data.write_text(json.dumps({"id": "answers", "problem": "Choose one.", "answer": ["$1$, $2$"]}) + "\n")
    spec = next(OlympiadBenchBenchmark(data_file=data, n_repeat=1).iter_specs(SourceOptions("OlympiadBench", SourceRef("local"))))

    assert spec.grader.reference["answers"] == ["1", "2"]
