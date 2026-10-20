"""IFEval keeps its historical loose headline while exposing both protocols."""

import json

import pytest

from eval.chat_benchmarks.IFEval.eval_instruct import IFEvalBenchmark
from eval.contracts.eval_spec import SourceRef
from eval.sources.common import SourceOptions
from tests.contracts.source_harness import FixedEndpoint, evaluate_source


def test_ifeval_legacy_metrics_and_headline_use_loose_checks(tmp_path):
    records = [
        {"key": 1, "prompt": "Return JSON.", "instruction_id_list": ["detectable_format:json_format"], "kwargs": [{}]},
        {"key": 2, "prompt": "Say apple and banana.", "instruction_id_list": ["keywords:existence"] * 2,
         "kwargs": [{"keywords": ["apple"]}, {"keywords": ["banana"]}]},
    ]
    (tmp_path / "input_data.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    source = IFEvalBenchmark(data_dir=str(tmp_path), start_idx=0, end_idx=2)

    metrics, _samples = evaluate_source(
        source, SourceOptions("IFEval", SourceRef("fixture")),
        FixedEndpoint('Preface\n{"answer": "apple"}\nEnd'),
    )

    assert metrics["prompt_level_strict"] == 0
    assert metrics["instruction_level_strict"] == pytest.approx(1 / 3)
    assert metrics["prompt_level_loose"] == metrics["prompt-level"] == 0.5
    assert metrics["instruction_level_loose"] == metrics["instruction-level"] == pytest.approx(2 / 3)
    assert source.describe("IFEval").primary_metric == "prompt-level"
