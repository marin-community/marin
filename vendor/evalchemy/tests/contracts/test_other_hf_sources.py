"""Full ordered parity against the datasets bundled before HF migration."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from eval.chat_benchmarks.IFEval.eval_instruct import IFEvalBenchmark
from eval.chat_benchmarks.MATH500.eval_instruct import MATH500Benchmark
from eval.chat_benchmarks.MixEval.eval_instruct import MixEvalBenchmark
from eval.chat_benchmarks.SimpleQA.eval_instruct import SimpleQABenchmark
from eval.chat_benchmarks.SimpleQAMini.eval_instruct import SimpleQAMiniBenchmark
from eval.contracts.eval_spec import SourceRef
from eval.sources.common import SourceOptions


@pytest.mark.data_integration
@pytest.mark.parametrize("name,benchmark,kwargs", [
    ("MATH500", MATH500Benchmark, {}),
    ("SimpleQA", SimpleQABenchmark, {}),
    ("SimpleQAMini", SimpleQAMiniBenchmark, {}),
    ("IFEval", IFEvalBenchmark, {}),
    ("MixEval", MixEvalBenchmark, {}),
    ("MixEvalHard", MixEvalBenchmark, {"benchmark": "mixeval_hard"}),
    ("MixEvalAugust", MixEvalBenchmark, {"version": "2024-08-11"}),
    ("MixEvalHardAugust", MixEvalBenchmark, {"benchmark": "mixeval_hard", "version": "2024-08-11"}),
])
def test_pinned_source_preserves_complete_ordered_specs(name, benchmark, kwargs):
    expected = json.loads((Path(__file__).parent / "fixtures/hf-other/ordered-specs.json").read_text())[name]
    source = benchmark(**kwargs)
    source.validate_prepared_data()
    specs = [asdict(spec) for spec in source.iter_specs(SourceOptions(name, SourceRef("fixture"), seed=7))]
    # Provenance changes to an immutable HF pin; every evaluation input must match.
    for spec in specs:
        spec["source"] = asdict(SourceRef("fixture"))
    digest = hashlib.sha256(json.dumps(specs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    assert {"count": len(specs), "sha256": digest} == expected
