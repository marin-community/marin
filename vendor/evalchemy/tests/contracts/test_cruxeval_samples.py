import json

from eval.chat_benchmarks.CruxEval.eval_instruct import CruxEvalBenchmark
from eval.contracts.eval_spec import SourceRef
from eval.sources.common import SourceOptions


def test_cruxeval_source_limits_before_expanding_directions(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [
        {"id": f"sample_{index}", "code": "def f(x): return x + 1", "input": str(index), "output": str(index + 1)}
        for index in range(3)
    ]
    (data_dir / "cruxeval.jsonl").write_text("".join(f"{json.dumps(row)}\n" for row in rows))

    specs = list(CruxEvalBenchmark(data_dir=str(data_dir)).iter_specs(SourceOptions("CruxEval", SourceRef("fixture"), limit=1)))

    assert [(spec.source_id, spec.namespace, spec.target) for spec in specs] == [
        ("sample_0", "input", "0"),
        ("sample_0", "output", "1"),
    ]
