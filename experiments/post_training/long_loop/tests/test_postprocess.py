# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from experiments.post_training.long_loop import postprocess
from experiments.post_training.long_loop.graph import execute_local_graph


def test_tail_graph_adopts_producers_and_runs_only_postprocessing(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "outputs"))
    roots = {}
    for key in ("inputs", "sft", "sft_adoption", "export", "rl", "finalization", "eval_inputs", "eval"):
        directory = tmp_path / "original" / key
        directory.mkdir(parents=True)
        (directory / "producer.txt").write_text(f"original-{key}")
        roots[key] = str(directory)
    calls = []

    def merge(*, output, original, selection, **kwargs):
        assert original == roots["eval"]
        assert selection == roots["eval_inputs"]
        calls.append("merge")
        Path(output, "view.txt").write_text("recovered view")

    def report(*, output, stages, rl_terminal, **kwargs):
        assert rl_terminal == roots["rl"]
        assert Path(stages[-1], "view.txt").read_text() == "recovered view"
        assert stages[:-1] == [roots[key] for key in ("inputs", "sft_adoption", "export", "rl", "eval_inputs", "eval")]
        calls.append("report")
        Path(output, "report.txt").write_text("qualified report")

    def accounting(*, output, trained, settings):
        assert trained == roots["sft"]
        assert calls == ["merge", "report"]
        calls.append("accounting")
        Path(output, "accounting.txt").write_text("original SFT consumption")

    def archive(*, output, artifacts, trained, evaluated, completed_report, **kwargs):
        assert trained == roots["rl"]
        assert artifacts["native_eval"] == roots["eval"]
        assert artifacts["eval"] == evaluated
        assert Path(completed_report, "report.txt").read_text() == "qualified report"
        assert Path(artifacts["sft_data_accounting"], "accounting.txt").read_text() == "original SFT consumption"
        calls.append("archive")
        Path(output, "archive.txt").write_text("evidence archive")

    monkeypatch.setattr(postprocess, "merge_verifier_recovery", merge)
    monkeypatch.setattr(postprocess, "write_report", report)
    monkeypatch.setattr(postprocess, "write_sft_accounting", accounting)
    monkeypatch.setattr(postprocess, "archive_recovered_run", archive)
    monkeypatch.setattr(postprocess, "remote", lambda fn, **kwargs: fn)
    resolved = {
        "name": "test/loop",
        "evaluation": {"expected_rollouts": 2},
        "adopted_sft": {},
        "archive": {"additional_evidence": [], "sft_data_accounting": {}},
        "execution": {
            "archive": {"device": {"kind": "cpu"}, "cpu": 4, "ram": "8GB", "disk": "8GB"},
            "task_timeouts": {"accounting": 1800, "archive": 5400},
            "coordinator": {
                "cluster": "fake",
                "cluster_config": "fake.yaml",
                "cpu": 4,
                "memory": "16GB",
                "disk": "64GB",
                "priority": "production",
                "max_retries": 0,
                "target_cluster": None,
                "parent_cluster_config": None,
                "required_env_vars": [],
                "coordinator_timeout_hours": 109,
            },
        },
    }
    plan = {
        "version": "2026.11.15.17",
        "artifacts": roots,
        "additional_evidence": [],
        "accepted_unscored_attempts": [],
        "sft_geometry": "Adopted native SFT step203",
        "recovery": {"uri": "original-answer-recovery", "manifest_sha256": "a" * 64},
    }
    terminal = postprocess.build_postprocessing_graph(plan, resolved)
    execute_local_graph(terminal)
    assert calls == ["merge", "report", "accounting", "archive"]
    assert Path(terminal.path(), "archive.txt").read_text() == "evidence archive"
    for key, root in roots.items():
        assert Path(root, "producer.txt").read_text() == f"original-{key}"


def test_recovered_archive_includes_original_native_files_without_mutating_settings(tmp_path, monkeypatch):
    evaluated = tmp_path / "view"
    evaluated.mkdir()
    names = ["bixbench-pi.jsonl", "aggregated_results.jsonl"]
    (evaluated / "merge-provenance.json").write_text(json.dumps({"source_files": [{"name": name} for name in names]}))
    settings = {"stage_files": {"eval": [{"path": "terminal.json", "category": "config"}]}}
    observed = []
    monkeypatch.setattr(postprocess, "archive_graph_run", lambda **kwargs: observed.append(kwargs))
    postprocess.archive_recovered_run(
        output="archive",
        settings=settings,
        resolved_spec={},
        artifacts={},
        trained="original-rl",
        evaluated=str(evaluated),
        completed_report="report",
    )
    assert settings == {"stage_files": {"eval": [{"path": "terminal.json", "category": "config"}]}}
    assert observed[0]["settings"]["stage_files"]["eval"] == [
        {"path": "terminal.json", "category": "config"},
        {"path": "merge-provenance.json", "category": "eval_trace"},
        *[{"path": f"original-native/{name}", "category": "eval_trace"} for name in names],
    ]
