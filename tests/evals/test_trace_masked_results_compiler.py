# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from marin.evaluation.trace_masked_results import compile_trace_masked_results_fn


def _write_results(
    path: Path,
    *,
    assistant_text: float,
    final_assistant: float,
    tool: float,
    obs: float,
    patch: float,
    patch_00: float,
    patch_50: float,
) -> None:
    results = {
        "datasets": {
            "local_a": {
                "metadata": {
                    "contrastive_outcome": False,
                    "row_prefix_fraction": None,
                },
                "metrics": {
                    "trace_masked_eval/local_a/assistant_text/bpb": assistant_text,
                    "trace_masked_eval/local_a/final_assistant/bpb": final_assistant,
                    "trace_masked_eval/local_a/tool_call/bpb": tool,
                    "trace_masked_eval/local_a/observation/bpb": obs,
                },
            },
            "outcome_a": {
                "metadata": {
                    "contrastive_outcome": True,
                    "row_prefix_fraction": None,
                },
                "metrics": {
                    "trace_masked_eval/outcome_a/patch/bpb": patch,
                    "trace_masked_eval/outcome_a/outcome_contrastive/normalized_auroc": 0.6,
                    "trace_masked_eval/outcome_a/outcome_contrastive/prefix_25/normalized_auroc": 0.5,
                    "trace_masked_eval/outcome_a/outcome_contrastive/prefix_50/normalized_auroc": 0.55,
                    "trace_masked_eval/outcome_a/outcome_contrastive/prefix_75/normalized_auroc": 0.58,
                    "trace_masked_eval/outcome_a/outcome_contrastive/prefix_100/normalized_auroc": 0.6,
                },
            },
            "outcome_a_patch_prefix_00": {
                "metadata": {
                    "contrastive_outcome": False,
                    "row_prefix_fraction": 0.0,
                },
                "metrics": {
                    "trace_masked_eval/outcome_a_patch_prefix_00/patch/bpb": patch_00,
                },
            },
            "outcome_a_patch_prefix_50": {
                "metadata": {
                    "contrastive_outcome": False,
                    "row_prefix_fraction": 0.5,
                },
                "metrics": {
                    "trace_masked_eval/outcome_a_patch_prefix_50/patch/bpb": patch_50,
                },
            },
        }
    }
    path.write_text(json.dumps(results))


def test_compile_trace_masked_results_fn_writes_compact_and_expanded_tables(tmp_path):
    model_a_path = tmp_path / "model_a_results.json"
    model_b_path = tmp_path / "model_b_results.json"
    _write_results(
        model_a_path,
        assistant_text=0.5,
        final_assistant=0.6,
        tool=0.7,
        obs=0.8,
        patch=0.3,
        patch_00=0.45,
        patch_50=0.35,
    )
    _write_results(
        model_b_path,
        assistant_text=0.4,
        final_assistant=0.5,
        tool=0.6,
        obs=0.7,
        patch=0.2,
        patch_00=0.31,
        patch_50=0.24,
    )

    output_path = tmp_path / "compiled"
    compile_trace_masked_results_fn(
        {
            "inputs": [
                {"model_name": "Model A", "results_path": str(model_a_path)},
                {"model_name": "Model B", "results_path": str(model_b_path)},
            ],
            "output_path": str(output_path),
        }
    )

    compiled_results = json.loads((output_path / "compiled_results.json").read_text())
    assert [row["model"] for row in compiled_results["rows"]] == ["Model A", "Model B"]
    assert compiled_results["rows"][0]["assistant_text"] == 0.5
    assert compiled_results["rows"][0]["final_assistant"] == 0.6
    assert compiled_results["rows"][0]["tool"] == 0.7
    assert compiled_results["rows"][0]["obs"] == 0.8
    assert compiled_results["rows"][0]["patch"] == 0.3
    assert compiled_results["rows"][0]["patch_00"] == 0.45
    assert compiled_results["rows"][0]["patch_50"] == 0.35
    assert compiled_results["rows"][0]["patch_gain"] == pytest.approx(0.15)
    assert compiled_results["rows"][0]["auc"] == 0.6
    assert compiled_results["rows"][0]["auc25"] == 0.5

    compact_md = (output_path / "compiled_results_compact.md").read_text()
    expanded_txt = (output_path / "compiled_results_expanded.txt").read_text()
    compact_csv = (output_path / "compiled_results_compact.csv").read_text()
    assert "| model | assistant_text | final_assistant | patch | tool | obs | patch_gain |" in compact_md
    assert "Model A" in expanded_txt
    assert "patch_gain" in compact_csv.splitlines()[0]
