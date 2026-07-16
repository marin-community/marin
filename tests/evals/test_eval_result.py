# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed eval-output artifacts + the report aggregation: read metrics back through the artifact.

Each toy fixture writes the on-disk shape its real producer writes — the Levanter evaluator's flat
top-level ``results.json`` and lm-eval's native ``{task}_{n}shot/<model>/results_<ts>.json`` tree — so
reading through the typed accessor and compiling a report exercises the real round-trip without running
an eval.
"""

import json

import fsspec
import pytest
from marin.evaluation.eval_result import (
    LevanterEvalResult,
    LmEvalHarnessResult,
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


_LEVANTER_RESULTS = {
    "results": {"hellaswag": {"acc,none": 0.5, "acc_stderr,none": 0.01, "alias": "hellaswag"}},
    "averages": {"macro_avg_acc": 0.5, "micro_avg_acc": 0.42},
}

# lm-eval's aggregated output: one results_<ts>.json per task, nested under <task>_<n>shot/<model>/.
_LM_EVAL_GSM8K = {"results": {"gsm8k": {"exact_match,none": 0.3, "exact_match_stderr,none": 0.02, "alias": "gsm8k"}}}
_LM_EVAL_IFEVAL = {"results": {"ifeval": {"prompt_level_strict_acc,none": 0.6, "alias": "ifeval"}}}


def test_levanter_eval_result_reads_metrics_and_averages(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    result = resolve(_step("evaluation/toy", LevanterEvalResult, {"results.json": _LEVANTER_RESULTS}))

    # String aliases are dropped; only numeric metrics survive.
    assert result.task_metrics() == {"hellaswag": {"acc,none": 0.5, "acc_stderr,none": 0.01}}
    assert result.averages() == {"macro_avg_acc": 0.5, "micro_avg_acc": 0.42}


def test_lm_eval_harness_result_merges_nested_task_files(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    files = {
        "gsm8k_8shot/vllm/results_2026-07-16T00-00-00.json": _LM_EVAL_GSM8K,
        "ifeval_0shot/vllm/results_2026-07-16T00-01-00.json": _LM_EVAL_IFEVAL,
    }
    result = resolve(_step("evaluation/toy-lmeval", LmEvalHarnessResult, files))

    assert result.task_metrics() == {
        "gsm8k": {"exact_match,none": 0.3, "exact_match_stderr,none": 0.02},
        "ifeval": {"prompt_level_strict_acc,none": 0.6},
    }
    # lm-eval records no cross-task average; the report computes suite rollups instead.
    assert result.averages() == {}


def test_compile_eval_report_merges_across_backends(tmp_path):
    levanter_dir = tmp_path / "levanter"
    lmeval_dir = tmp_path / "lmeval"
    report_dir = tmp_path / "report"
    (levanter_dir).mkdir()
    (lmeval_dir / "gsm8k_8shot" / "vllm").mkdir(parents=True)
    (levanter_dir / "results.json").write_text(json.dumps(_LEVANTER_RESULTS))
    (lmeval_dir / "gsm8k_8shot" / "vllm" / "results_2026-07-16T00-00-00.json").write_text(json.dumps(_LM_EVAL_GSM8K))

    entries = [
        ReportEntry(str(levanter_dir), result_type_name(LevanterEvalResult), "mcq"),
        ReportEntry(str(lmeval_dir), result_type_name(LmEvalHarnessResult), "gen"),
    ]
    report = compile_eval_report(entries, str(report_dir))

    assert report.task_metrics == {
        "hellaswag": {"acc,none": 0.5, "acc_stderr,none": 0.01},
        "gsm8k": {"exact_match,none": 0.3, "exact_match_stderr,none": 0.02},
    }
    # Levanter's averages are namespaced by the result's label; lm-eval contributes none.
    assert report.averages == {"mcq/macro_avg_acc": 0.5, "mcq/micro_avg_acc": 0.42}
    # The human-readable report.json is written alongside.
    written = json.loads((report_dir / "report.json").read_text())
    assert written["task_metrics"]["gsm8k"] == {"exact_match,none": 0.3, "exact_match_stderr,none": 0.02}


def test_compile_eval_report_rejects_duplicate_task(tmp_path):
    """Two results carrying the same task key must fail loudly, not silently drop one."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "results.json").write_text(json.dumps(_LEVANTER_RESULTS))
    (b / "results.json").write_text(json.dumps(_LEVANTER_RESULTS))  # same "hellaswag" key
    entries = [
        ReportEntry(str(a), result_type_name(LevanterEvalResult), "group_a"),
        ReportEntry(str(b), result_type_name(LevanterEvalResult), "group_b"),
    ]
    with pytest.raises(ValueError, match="duplicate task 'hellaswag'"):
        compile_eval_report(entries, str(tmp_path / "report"))
