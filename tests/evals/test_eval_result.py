# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed eval-output artifact + report aggregation: read metrics back through the artifact.

Each toy fixture writes the on-disk shape evalchemy's real producer writes — lm-eval's native
``{task}_{n}shot/<model>/results_<ts>.json`` tree — so reading through the typed accessor and
compiling a report exercises the real round-trip without running an eval.
"""

import json

import fsspec
import pytest
from marin.evaluation.eval_result import (
    EvalchemyResult,
    ReportEntry,
    compile_eval_report,
)
from marin.execution.artifact import result_type_name
from marin.execution.lazy import ArtifactStep, resolve


def _writer(files: dict[str, object]):
    """A step ``run`` that writes ``{relpath: json}`` under the step's output dir (relpath may nest)."""

    def run(config: dict) -> None:
        fs, _, _ = fsspec.get_fs_token_paths(config["out"])
        for relpath, payload in files.items():
            full = f"{config['out']}/{relpath}"
            fs.makedirs(full.rsplit("/", 1)[0], exist_ok=True)
            with fs.open(full, "w") as f:
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


# evalchemy's aggregated output: one results_<ts>.json per task, nested under <task>_<n>shot/<model>/.
_GSM8K = {"results": {"gsm8k": {"exact_match,none": 0.3, "exact_match_stderr,none": 0.02, "alias": "gsm8k"}}}
_ARC = {"results": {"arc_easy": {"acc,none": 0.5, "acc_norm,none": 0.66, "alias": "arc_easy"}}}


def test_evalchemy_result_merges_nested_task_files(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    files = {
        "gsm8k_8shot/vllm/results_2026-07-16T00-00-00.json": _GSM8K,
        "arc_easy_0shot/vllm/results_2026-07-16T00-01-00.json": _ARC,
    }
    result = resolve(_step("evaluation/toy-evalchemy", EvalchemyResult, files))

    # String aliases are dropped; only numeric metrics survive; each task's file is merged.
    assert result.task_metrics() == {
        "gsm8k": {"exact_match,none": 0.3, "exact_match_stderr,none": 0.02},
        "arc_easy": {"acc,none": 0.5, "acc_norm,none": 0.66},
    }
    # evalchemy records no cross-task average; the report computes suite rollups instead.
    assert result.averages() == {}


def test_evalchemy_result_missing_results_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    # A step that writes nothing under its output dir: the accessor must fail loudly, not return {}.
    result = resolve(_step("evaluation/toy-empty", EvalchemyResult, {}))
    with pytest.raises(FileNotFoundError, match="no evalchemy results"):
        result.task_metrics()


def test_compile_eval_report_merges_across_groups(tmp_path):
    gen_dir = tmp_path / "gen"
    mcq_dir = tmp_path / "mcq"
    report_dir = tmp_path / "report"
    (gen_dir / "gsm8k_8shot" / "vllm").mkdir(parents=True)
    (mcq_dir / "arc_easy_0shot" / "vllm").mkdir(parents=True)
    (gen_dir / "gsm8k_8shot" / "vllm" / "results_2026-07-16T00-00-00.json").write_text(json.dumps(_GSM8K))
    (mcq_dir / "arc_easy_0shot" / "vllm" / "results_2026-07-16T00-01-00.json").write_text(json.dumps(_ARC))

    entries = [
        ReportEntry(str(gen_dir), result_type_name(EvalchemyResult), "gen"),
        ReportEntry(str(mcq_dir), result_type_name(EvalchemyResult), "mcq"),
    ]
    report = compile_eval_report(entries, str(report_dir))

    assert report.task_metrics == {
        "gsm8k": {"exact_match,none": 0.3, "exact_match_stderr,none": 0.02},
        "arc_easy": {"acc,none": 0.5, "acc_norm,none": 0.66},
    }
    # evalchemy contributes no per-result averages.
    assert report.averages == {}
    # The human-readable report.json is written alongside.
    written = json.loads((report_dir / "report.json").read_text())
    assert written["task_metrics"]["gsm8k"] == {"exact_match,none": 0.3, "exact_match_stderr,none": 0.02}


def test_compile_eval_report_rejects_duplicate_task(tmp_path):
    """Two results carrying the same task key must fail loudly, not silently drop one."""
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "gsm8k_8shot" / "vllm").mkdir(parents=True)
    (b / "gsm8k_5shot" / "vllm").mkdir(parents=True)
    (a / "gsm8k_8shot" / "vllm" / "results_2026-07-16T00-00-00.json").write_text(json.dumps(_GSM8K))
    (b / "gsm8k_5shot" / "vllm" / "results_2026-07-16T00-01-00.json").write_text(json.dumps(_GSM8K))  # same "gsm8k"
    entries = [
        ReportEntry(str(a), result_type_name(EvalchemyResult), "group_a"),
        ReportEntry(str(b), result_type_name(EvalchemyResult), "group_b"),
    ]
    with pytest.raises(ValueError, match="duplicate task 'gsm8k'"):
        compile_eval_report(entries, str(tmp_path / "report"))
