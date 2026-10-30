"""Source selection preserves the full benchmark's generation contract."""

import json
import sys

import pytest
from argparse import Namespace
from dataclasses import replace

from eval.contracts.benchmark_metadata import infer_benchmark_metadata
from eval.eval import _attach_benchmark_metadata, setup_custom_parser
from eval.execution.run_spec import RunSpec
from eval.lm_eval_compat import parse_eval_args

from eval.contracts.eval_spec import EvalSpec, GraderSpec, ModelRequest, RequestKind, SourceRef
from eval.contracts.sample_manifest import SampleCoverageError
from eval.source_selection import selected_specs


def test_noncontiguous_selection_preserves_prompts_seeds_and_every_repeat(monkeypatch, tmp_path):
    def source():
        for ordinal in range(6):
            for repeat in range(3):
                yield EvalSpec(
                    task="task",
                    source=SourceRef("dataset"),
                    source_id=ordinal,
                    ordinal=ordinal,
                    namespace="output",
                    trial=repeat,
                    requests=(
                        ModelRequest(
                            RequestKind.GENERATE,
                            f"fewshot-context-{ordinal}",
                            {"seed": 1234 + repeat, "max_gen_toks": 8192},
                            context_length=32768,
                        ),
                    ),
                    grader=GraderSpec("exact"),
                )

    full = tuple(source())
    keys = ['{"namespace":"output","source_id":1}', '{"namespace":"output","source_id":4}']
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evalchemy",
            "--tasks",
            "MATH500",
            "--finestore_output_path",
            str(tmp_path / "results"),
            "--selected_source_keys",
            json.dumps(keys),
        ],
    )
    args = parse_eval_args(setup_custom_parser())
    run = RunSpec.from_args(args)
    chosen = selected_specs((source(),), run.selected_source_keys)
    assert chosen == tuple(item for item in full if item.source_id in {1, 4})
    assert len(chosen) == 6
    assert [item.ordinal for item in chosen] == [1, 1, 1, 4, 4, 4]


def test_selection_does_not_end_parent_source_early_or_mix_input_and_output():
    visited = []

    def source():
        for ordinal, namespace in enumerate(["input", "output", "input", "output"]):
            visited.append(ordinal)
            yield EvalSpec("task", SourceRef("dataset"), "sample", ordinal, (), GraderSpec("exact"), namespace=namespace)

    chosen = selected_specs((source(),), ('{"namespace":"output","source_id":"sample"}',))
    assert visited == [0, 1, 2, 3]
    assert [item.namespace for item in chosen] == ["output", "output"]


def test_unknown_keys_fail_before_returning_any_source_items():
    source = (EvalSpec("task", SourceRef("dataset"), 0, 0, (), GraderSpec("exact")),)
    with pytest.raises(SampleCoverageError, match="absent from parent dataset"):
        selected_specs((source,), ('{"namespace":"default","source_id":99}',))


def test_subset_metadata_keeps_parent_size_and_reports_selected_size():
    parent = replace(infer_benchmark_metadata("task", {"accuracy": 0.5}, n_attempted=100), n_benchmark=100)
    keys = '["{\\"namespace\\":\\"default\\",\\"source_id\\":1}","{\\"namespace\\":\\"default\\",\\"source_id\\":4}"]'
    args = Namespace(resolved_benchmark_metadata=(parent,), selected_source_keys=keys)
    result = {"results": {"task": {"accuracy": 0.5}}, "task_outcomes": {"task": {"expected_count": 2}}}
    _attach_benchmark_metadata(result, args)
    assert result["benchmark_metadata"]["task"]["n_benchmark"] == 2
    assert result["benchmark_metadata"]["task"]["n_attempted"] == 2
    assert result["task_selection"]["parent_benchmark_metadata"]["n_benchmark"] == 100
    assert result["canonical_results"]["task"]["accuracy"] == 0.5
