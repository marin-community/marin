# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from experiments.post_training.long_loop import postprocess
from experiments.post_training.long_loop.graph import execute_local_graph


@pytest.mark.parametrize("adopt_report", [False, True])
@pytest.mark.parametrize("adopt_accounting", [False, True])
def test_tail_graph_adopts_producers_and_runs_only_postprocessing(tmp_path, monkeypatch, adopt_report, adopt_accounting):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "outputs"))
    roots = {}
    for key in ("inputs", "sft", "sft_adoption", "export", "rl", "finalization", "eval_inputs", "eval"):
        directory = tmp_path / "original" / key
        directory.mkdir(parents=True)
        (directory / "producer.txt").write_text(f"original-{key}")
        roots[key] = str(directory)
    if adopt_report:
        for key, filename, content in (
            ("recovered_eval", "view.txt", "recovered view"),
            ("report", "report.txt", "qualified report"),
        ):
            directory = tmp_path / "original" / key
            directory.mkdir()
            (directory / filename).write_text(content)
            (directory / "producer.txt").write_text(f"original-{key}")
            roots[key] = str(directory)
    calls = []
    if adopt_accounting:
        directory = tmp_path / "original" / "sft_data_accounting"
        directory.mkdir()
        (directory / "producer.txt").write_text("original-sft_data_accounting")
        (directory / "accounting.txt").write_text("original SFT consumption")
        roots["sft_data_accounting"] = str(directory)
    remote_timeouts = {}

    def bind_remote(fn, **kwargs):
        remote_timeouts[fn.__name__] = kwargs["timeout"]
        return fn

    def merge(*, output, original, selection, **kwargs):
        assert not adopt_report
        assert original == roots["eval"]
        assert selection == roots["eval_inputs"]
        calls.append("merge")
        Path(output, "view.txt").write_text("recovered view")

    def report(*, output, stages, rl_terminal, **kwargs):
        assert not adopt_report
        assert rl_terminal == roots["rl"]
        assert Path(stages[-1], "view.txt").read_text() == "recovered view"
        assert stages[:-1] == [roots[key] for key in ("inputs", "sft_adoption", "export", "rl", "eval_inputs", "eval")]
        calls.append("report")
        Path(output, "report.txt").write_text("qualified report")

    def accounting(*, output, trained, settings):
        assert trained == roots["sft"]
        assert calls == ([] if adopt_report else ["merge", "report"])
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
    monkeypatch.setattr(postprocess, "remote", bind_remote)
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
        "task_timeouts": {"accounting": 21600},
    }
    terminal = postprocess.build_postprocessing_graph(plan, resolved)
    if not adopt_accounting:
        assert remote_timeouts["accounting"] == postprocess.Duration.from_seconds(21600)
    assert remote_timeouts["archive"] == postprocess.Duration.from_seconds(5400)
    execute_local_graph(terminal)
    assert calls == [
        *([] if adopt_report else ["merge", "report"]),
        *([] if adopt_accounting else ["accounting"]),
        "archive",
    ]
    assert Path(terminal.path(), "archive.txt").read_text() == "evidence archive"
    for key, root in roots.items():
        assert Path(root, "producer.txt").read_text() == f"original-{key}"
