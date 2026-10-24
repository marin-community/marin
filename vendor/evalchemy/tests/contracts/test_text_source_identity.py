"""Repeated dataset rows remain separate scored examples."""

import pytest

from eval.chat_benchmarks.RepoBench import eval_instruct as repobench
from eval.sources.common import SourceOptions
from tests.contracts.source_harness import FixedEndpoint, evaluate_source


@pytest.fixture
def duplicate_repo_rows():
    row = {
        "repo_name": "example", "file_path": "src/example.py", "context": [],
        "import_statement": "", "cropped_code": "value = 1", "next_line": "print(value)",
    }
    return [dict(row), dict(row)]


def test_identical_repo_rows_are_scored_across_namespaces_and_trials(monkeypatch, duplicate_repo_rows):
    monkeypatch.setattr(repobench, "load_dataset", lambda *_args, **_kwargs: {
        "cross_file_first": duplicate_repo_rows, "in_file": duplicate_repo_rows,
    })
    source = repobench.RepoBenchmark(languages=["python"], subsets=["cross_file_first", "in_file"])
    options = SourceOptions("RepoBench", source.SOURCE, num_samples=2)
    backend = FixedEndpoint(lambda request: "print(value)" if request.metadata["ordinal"] == 0 else "wrong")

    metrics, samples = evaluate_source(source, options, backend)

    assert metrics["EM"] == 50.0
    assert len(samples) == 8
    assert len({sample["sample_id"] for sample in samples}) == 4
    assert {(sample["sample_namespace"], sample["source_id"], sample["sample_repeat"]) for sample in samples} == {
        (namespace, ordinal, trial)
        for namespace in ("python:cross_file_first", "python:in_file")
        for ordinal in (0, 1)
        for trial in (0, 1)
    }
    assert {sample["EM"] for sample in samples if sample["source_id"] == 0} == {100.0}
    assert {sample["EM"] for sample in samples if sample["source_id"] == 1} == {0.0}
    assert all(sample["doc"] == duplicate_repo_rows[0] for sample in samples)
