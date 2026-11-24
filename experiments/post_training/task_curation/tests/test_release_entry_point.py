# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the atlas-release entry point (DESIGN.md §1, M2c)."""

import json
from pathlib import Path

import pytest
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.release_stages import __main__ as entry_point
from experiments.post_training.task_curation.release_stages import graph as release_graph_module
from experiments.post_training.task_curation.release_stages import preflight as preflight_module
from experiments.post_training.task_curation.release_stages import preflight_slices
from experiments.post_training.task_curation.release_stages.graph import write_plan
from experiments.post_training.task_curation.tests.release_graph_fixtures import fixture_plan

EXPECTED_NODE_COUNT = 28


@pytest.fixture
def plan_document(tmp_path: Path) -> Path:
    """A fixture plan document sealed under a throwaway local release root."""
    plan = fixture_plan(release_root=str(tmp_path / "release"), slice_output_root=str(tmp_path / "slice"))
    write_plan(plan, plan.release_root)
    return Path(write_plan(plan, plan.release_root)["uri"])


def _argv(plan_document: Path, mode: str, *extra: str) -> list[str]:
    return ["--version", "testrelease", "--plan", str(plan_document), mode, *extra]


def test_plan_mode_lists_the_whole_graph_io_free(tmp_path, plan_document, capsys):
    release_root = Path(plan_document).parent
    before = sorted(p.name for p in release_root.iterdir())
    exit_code = entry_point.main(_argv(plan_document, "plan"))
    output = capsys.readouterr().out
    assert exit_code == 0
    assert output.count("NODE ") == EXPECTED_NODE_COUNT
    assert "atlas-release-testrelease-exposure_probe" in output
    assert "PLAN testrelease" in output and "nothing executed" in output
    after = sorted(p.name for p in release_root.iterdir())
    assert after == before


def test_plan_mode_refuses_a_version_mismatch(plan_document, capsys):
    exit_code = entry_point.main(["--version", "other", "--plan", str(plan_document), "plan"])
    assert exit_code == 2
    assert "binds release" in capsys.readouterr().err


def test_preflight_mode_reports_blocked_stages_without_sealing(plan_document, capsys):
    exit_code = entry_point.main(_argv(plan_document, "preflight", "--stage", "catalog_load"))
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "BLOCKED catalog_load:" in output
    assert "blocked until their upstream stages run" in output
    assert not (Path(plan_document).parent / "preflight").exists()


def test_preflight_mode_runs_a_slice_and_seals_a_fresh_receipt(tmp_path, plan_document, monkeypatch, capsys):
    checked: list[str] = []
    monkeypatch.setattr(preflight_module, "type_check_module", lambda module, repo: checked.append(module) or "ok")

    def fake_runner(node, plan, selection):
        return lambda: {"fake": True, "node": node}

    monkeypatch.setattr(preflight_slices, "slice_runner", fake_runner)
    monkeypatch.setattr(preflight_slices, "slice_inputs", lambda node, plan, selection: {"node": node})
    exit_code = entry_point.main(_argv(plan_document, "preflight", "--stage", "census_inputs"))
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "OK census_inputs:" in output
    assert checked == ["experiments/post_training/task_curation/release_stages/graph.py"]
    receipt = json.loads(
        StoragePath(f"{tmp_path / 'release'}/preflight/census_inputs/preflight-receipt.json").read_text()
    )
    assert receipt["ok"] is True and receipt["slice_evidence"]["node"] == "census_inputs"
    # A second preflight serves the fresh receipt without re-running the slice.
    exit_code = entry_point.main(_argv(plan_document, "preflight", "--stage", "census_inputs"))
    assert exit_code == 0
    assert capsys.readouterr().out.count("OK census_inputs:") == 1


def test_preflight_mode_reports_a_failed_slice_and_exits_nonzero(tmp_path, plan_document, monkeypatch, capsys):
    monkeypatch.setattr(preflight_module, "type_check_module", lambda module, repo: "ok")
    monkeypatch.setattr(
        preflight_slices,
        "slice_runner",
        lambda node, plan, selection: (_ for _ in ()).throw(ValueError("slice exploded")),
    )
    exit_code = entry_point.main(_argv(plan_document, "preflight", "--stage", "census_inputs"))
    output = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL census_inputs:" in output and "slice exploded" in output


def test_preflight_mode_refuses_unknown_nodes(plan_document, capsys):
    exit_code = entry_point.main(_argv(plan_document, "preflight", "--stage", "not_a_node"))
    assert exit_code == 2
    assert "unknown graph nodes" in capsys.readouterr().err


def test_run_mode_builds_the_terminal_for_an_unrestricted_run(tmp_path, plan_document, monkeypatch, capsys):
    """A plain run builds the terminal stage, not just the plan document."""
    captured = {}

    def fake_run(*handles, **kwargs):
        captured["handles"] = list(handles)
        return []

    monkeypatch.setattr(entry_point, "assert_submission_allowed", lambda *a, **k: None)
    monkeypatch.setattr(entry_point, "run_handles", fake_run)
    exit_code = entry_point.main(_argv(plan_document, "run"))
    assert exit_code == 0
    handles = captured["handles"]
    assert handles[0].name.endswith("-plan")
    # The terminal handle is the exposure probe: the graph's last stage. The
    # entry point constructs its own graph, so compare identity by node key.
    built = release_graph_module.release_graph(entry_point.load_plan(plan_document))
    assert handles[-1].name == built.terminal.name
    assert handles[-1].version == built.terminal.version


def test_run_mode_gates_on_the_submission_guard(tmp_path, plan_document, monkeypatch, capsys):
    def refuse(*args, **kwargs):
        raise ValueError("checkout HEAD is not landed on atlas-env-audit")

    monkeypatch.setattr(entry_point, "assert_submission_allowed", refuse)
    exit_code = entry_point.main(_argv(plan_document, "run"))
    assert exit_code == 2
    assert "not landed" in capsys.readouterr().err


def test_resume_mode_forces_failed_stages_alone(tmp_path, plan_document, monkeypatch, capsys):
    """Resume forces failed stages alone through the same terminal run.

    Regression: the dispatch passed ``force_run_failed`` to ``mode_resume``, which
    does not accept it, so every resume died at the handler boundary.
    """
    captured = {}

    def fake_run(*handles, **kwargs):
        captured["force_run_failed"] = kwargs["force_run_failed"]
        return []

    monkeypatch.setattr(entry_point, "assert_submission_allowed", lambda *a, **k: None)
    monkeypatch.setattr(entry_point, "run_handles", fake_run)
    exit_code = entry_point.main(_argv(plan_document, "resume"))
    assert exit_code == 0
    assert captured["force_run_failed"] is True
    assert "RESUME testrelease" in capsys.readouterr().out


def test_status_mode_reports_missing_state(tmp_path, plan_document, capsys):
    exit_code = entry_point.main(_argv(plan_document, "status", "--json"))
    output = capsys.readouterr().out
    assert exit_code == 0
    document = json.loads(output)
    assert document["release_id"] == "testrelease"
    nodes = {row["node"]: row for row in document["nodes"]}
    assert len(nodes) == 27
    assert nodes["census"]["preflight"] == "missing"
    assert nodes["census"]["artifact"] in ("absent", "unknown-prefix")


def test_ensure_preflight_refuses_a_blocked_node(plan_document, capsys):
    plan = entry_point._plan_document(type("Args", (), {"version": "testrelease", "plan": str(plan_document)})())
    with pytest.raises(ValueError, match="blocked"):
        preflight_slices.ensure_preflight("catalog_load", plan)


def test_ensure_preflight_serves_a_fresh_receipt_without_rerunning(tmp_path, plan_document, monkeypatch):
    plan = fixture_plan(release_root=str(tmp_path / "release"), slice_output_root=str(tmp_path / "slice"))
    selection = preflight_slices.SliceSelection()
    digests = preflight_module.input_digests(preflight_slices.slice_inputs("census", plan, selection))
    receipt = preflight_module.PreflightReceipt(
        stage="census",
        mode="preflight",
        repo_sha=entry_point.repo_head(entry_point.REPO_ROOT),
        input_digests=digests,
        ok=True,
        checked_at="2026-11-24T00:00:00+00:00",
    )
    preflight_module.write_receipt(str(tmp_path / "release"), receipt)

    def _no_slice(*args, **kwargs):
        raise AssertionError("a fresh receipt must be served without re-running the slice")

    monkeypatch.setattr(preflight_module, "run_preflight", _no_slice)
    out = preflight_slices.ensure_preflight("census", plan)
    assert out.ok and out.input_digests == digests


def test_slice_inputs_are_digest_stable(plan_document):
    plan = entry_point._plan_document(type("Args", (), {"version": "testrelease", "plan": str(plan_document)})())
    selection = preflight_slices.SliceSelection()
    first = preflight_slices.slice_inputs("census", plan, selection)
    second = preflight_slices.slice_inputs("census", plan, selection)
    assert first == second
    assert preflight_module.input_digests(first) == preflight_module.input_digests(second)
    assert first["only"] == "Marin:fixture"


def test_module_path_resolves_the_stage_module():
    assert preflight_slices.module_path("census").endswith("release_stages/census.py")
    assert preflight_slices.module_path("release_check_postpublish").endswith("release_stages/release_check.py")
