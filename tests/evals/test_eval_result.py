# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed eval-output artifact + report aggregation: read metrics back through the artifact.

Each toy fixture writes the on-disk shape evalchemy's real producer writes — lm-eval's native
``<task_dir>/<model>/results_<ts>.json`` tree — so reading through the typed accessor and compiling a
report exercises the real round-trip without running an eval. The fixtures pin the behaviour that
matters: metrics are keyed by the upload dir (unique per task-config), not by the bare task name
lm-eval writes inside the JSON, so shot variants of one task do not overwrite each other.
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


# evalchemy's aggregated output: one results_<ts>.json per task-config, nested under <task_dir>/<model>/.
# lm-eval keys its `results` block by the bare task name, so both hellaswag shot variants say "hellaswag".
_GSM8K = {"results": {"gsm8k": {"exact_match,none": 0.3, "exact_match_stderr,none": 0.02, "alias": "gsm8k"}}}
_ARC = {"results": {"arc_easy": {"acc,none": 0.5, "acc_norm,none": 0.66, "alias": "arc_easy"}}}
_HELLASWAG_0 = {"results": {"hellaswag": {"acc,none": 0.50, "alias": "hellaswag"}}}
_HELLASWAG_10 = {"results": {"hellaswag": {"acc,none": 0.62, "alias": "hellaswag"}}}
# A group task writes several entries (the group aggregate plus subgroups) into one file.
_MMLU = {
    "results": {
        "mmlu": {"acc,none": 0.41, "alias": "mmlu"},
        "mmlu_stem": {"acc,none": 0.38, "alias": " - stem"},
    }
}


def test_evalchemy_result_keys_by_task_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    files = {
        "gsm8k_8shot/vllm/results_2026-07-16T00-00-00.json": _GSM8K,
        "arc_easy_0shot/vllm/results_2026-07-16T00-01-00.json": _ARC,
    }
    result = resolve(_step("evaluation/toy-evalchemy", EvalchemyResult, files))

    # Keyed by the upload dir; string aliases dropped, only numeric metrics survive.
    assert result.task_metrics() == {
        "gsm8k_8shot": {"exact_match,none": 0.3, "exact_match_stderr,none": 0.02},
        "arc_easy_0shot": {"acc,none": 0.5, "acc_norm,none": 0.66},
    }
    # evalchemy records no cross-task average; the report computes suite rollups instead.
    assert result.averages() == {}


def test_evalchemy_result_keeps_shot_variants_of_one_task_distinct(tmp_path, monkeypatch):
    """Two shot cuts of hellaswag share the inner task name "hellaswag" but upload to different dirs;
    keying by the dir keeps both instead of the later file overwriting the earlier."""
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    files = {
        "hellaswag_0shot/vllm/results_2026-07-16T00-00-00.json": _HELLASWAG_0,
        "hellaswag_10shot/vllm/results_2026-07-16T00-01-00.json": _HELLASWAG_10,
    }
    result = resolve(_step("evaluation/toy-hellaswag", EvalchemyResult, files))

    assert result.task_metrics() == {
        "hellaswag_0shot": {"acc,none": 0.50},
        "hellaswag_10shot": {"acc,none": 0.62},
    }


def test_evalchemy_result_namespaces_group_subtasks(tmp_path, monkeypatch):
    """A group task's file carries several entries; each is namespaced under the task dir so the group
    aggregate and its subtasks stay separable and never collide with another group's dir."""
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    files = {"mmlu_5shot/vllm/results_2026-07-16T00-00-00.json": _MMLU}
    result = resolve(_step("evaluation/toy-mmlu", EvalchemyResult, files))

    assert result.task_metrics() == {
        "mmlu_5shot/mmlu": {"acc,none": 0.41},
        "mmlu_5shot/mmlu_stem": {"acc,none": 0.38},
    }


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
        "gsm8k_8shot": {"exact_match,none": 0.3, "exact_match_stderr,none": 0.02},
        "arc_easy_0shot": {"acc,none": 0.5, "acc_norm,none": 0.66},
    }
    # evalchemy contributes no per-result averages.
    assert report.averages == {}
    # The human-readable report.json is written alongside.
    written = json.loads((report_dir / "report.json").read_text())
    assert written["task_metrics"]["gsm8k_8shot"] == {"exact_match,none": 0.3, "exact_match_stderr,none": 0.02}


def test_compile_eval_report_rejects_duplicate_task_dir(tmp_path):
    """Two results that both carry the same task dir must fail loudly, not silently drop one."""
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "gsm8k_8shot" / "vllm").mkdir(parents=True)
    (b / "gsm8k_8shot" / "vllm").mkdir(parents=True)
    (a / "gsm8k_8shot" / "vllm" / "results_2026-07-16T00-00-00.json").write_text(json.dumps(_GSM8K))
    (b / "gsm8k_8shot" / "vllm" / "results_2026-07-16T00-01-00.json").write_text(json.dumps(_GSM8K))
    entries = [
        ReportEntry(str(a), result_type_name(EvalchemyResult), "group_a"),
        ReportEntry(str(b), result_type_name(EvalchemyResult), "group_b"),
    ]
    with pytest.raises(ValueError, match="duplicate task 'gsm8k_8shot'"):
        compile_eval_report(entries, str(tmp_path / "report"))
