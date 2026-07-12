# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed eval-output artifacts: read metrics back through the artifact, not the directory.

Each toy step *writes* the on-disk shape its real producer writes (the levanter evaluator's
top-level ``results.json``; the evalchemy compiler's ``compiled_results.json`` /
``averaged_results.json``), so resolving the step and reading through the typed accessor
exercises the real round-trip without running an eval.
"""

import json

import fsspec
import pytest
from marin.evaluation.eval_result import LevanterEvalResult
from marin.execution.lazy import ArtifactStep, resolve

from experiments.evals.evalchemy_results_compiler import CompiledEvalResult


def _writer(files: dict[str, object]):
    """A step ``run`` that writes ``{filename: json}`` under the step's output dir."""

    def run(config: dict) -> None:
        fs, _, _ = fsspec.get_fs_token_paths(config["out"])
        fs.makedirs(config["out"], exist_ok=True)
        for filename, payload in files.items():
            with fs.open(f"{config['out']}/{filename}", "w") as f:
                f.write(json.dumps(payload))

    return run


def _step(name: str, kind: type, files: dict[str, object]) -> ArtifactStep:
    return ArtifactStep(
        name=name,
        version="2026.06.28",
        artifact_type=kind,
        run=_writer(files),
        build_config=lambda ctx: {"out": ctx.output_path},
    )


def test_levanter_eval_result_reads_metrics_and_averages(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    results = {
        "results": {"hellaswag": {"acc,none": 0.5, "acc_stderr,none": 0.01, "alias": "hellaswag"}},
        "averages": {"macro_avg_acc": 0.5, "micro_avg_acc": 0.42},
    }
    result = resolve(_step("evaluation/toy", LevanterEvalResult, {"results.json": results}))

    # String aliases are dropped; only numeric metrics survive.
    assert result.task_metrics() == {"hellaswag": {"acc,none": 0.5, "acc_stderr,none": 0.01}}
    assert result.averages() == {"macro_avg_acc": 0.5, "micro_avg_acc": 0.42}


def test_compiled_eval_result_reads_compiled_and_averaged(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    compiled = [{"id": "q1", "correct": 1, "dataset_name": "aime", "model_name": "m_seed1"}]
    averaged = [{"base_model_name": "m", "dataset_name": "aime", "num_seeds": 2, "correct_mean": 0.75}]
    result = resolve(
        _step(
            "evaluation/toy-compile",
            CompiledEvalResult,
            {"compiled_results.json": compiled, "averaged_results.json": averaged},
        )
    )

    assert result.compiled() == compiled
    assert result.averaged() == averaged


def test_compiled_eval_result_averaged_missing_raises(tmp_path, monkeypatch):
    """When the compile produced no averages, ``averaged()`` fails loudly rather than returning empty."""
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    compiled = [{"id": "q1", "correct": 0, "dataset_name": "aime", "model_name": "m"}]
    result = resolve(_step("evaluation/toy-noavg", CompiledEvalResult, {"compiled_results.json": compiled}))

    assert result.compiled() == compiled
    with pytest.raises(FileNotFoundError):
        result.averaged()
