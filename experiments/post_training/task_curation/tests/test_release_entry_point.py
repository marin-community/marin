# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the atlas-release entry point (DESIGN.md §1, M2c)."""

import json
from pathlib import Path

import pytest
from marin.execution import lazy
from rigging.filesystem.step_status import read_step_status
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.release_stages import __main__ as entry_point
from experiments.post_training.task_curation.release_stages import graph as release_graph_module
from experiments.post_training.task_curation.release_stages import preflight as preflight_module
from experiments.post_training.task_curation.release_stages import preflight_slices
from experiments.post_training.task_curation.release_stages.graph import write_plan
from experiments.post_training.task_curation.release_stages.plan_derivation import DerivedRelease
from experiments.post_training.task_curation.tests.release_graph_fixtures import fixture_plan

EXPECTED_NODE_COUNT = 31


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


def test_resume_with_cluster_submits_the_resume_job(tmp_path, plan_document, monkeypatch):
    """`resume --cluster` submits an in-region resume job instead of running locally."""
    captured = {}

    def fake_submit(args, *, verb):
        captured["verb"] = verb
        captured["cluster"] = args.cluster
        return 0

    monkeypatch.setattr(entry_point, "_submit_root", fake_submit)
    exit_code = entry_point.main(_argv(plan_document, "resume", "--cluster", "cw-us-east-02a"))
    assert exit_code == 0
    assert captured == {"verb": "resume", "cluster": "cw-us-east-02a"}


def test_submit_mode_submits_the_run_job(tmp_path, plan_document, monkeypatch):
    captured = {}

    def fake_submit(args, *, verb):
        captured["verb"] = verb
        return 0

    monkeypatch.setattr(entry_point, "_submit_root", fake_submit)
    exit_code = entry_point.main(_argv(plan_document, "submit", "--cluster", "cw-us-east-02a"))
    assert exit_code == 0
    assert captured == {"verb": "run"}


def test_interrupted_graph_resumes_and_skips_sealed_nodes(tmp_path, monkeypatch):
    """The auto-resume semantics: a retry runs resume at the same version.

    The iris-level retry reruns the root entrypoint with the resume verb; the
    sealed steps skip by fingerprint and only the failed or unstarted nodes re-run.
    A deterministic failure stays bounded: without force_run_failed the run
    refuses instead of silently re-running.
    """
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))

    def _mark(name: str) -> None:
        with (tmp_path / f"{name}.calls").open("a", encoding="utf-8") as handle:
            handle.write("x\n")

    def _count(name: str) -> int:
        path = tmp_path / f"{name}.calls"
        return len(path.read_text().splitlines()) if path.exists() else 0

    def sealed() -> dict:
        _mark("sealed")
        return {"n": 1}

    root = lazy.apply("atlas-resume-fixture-sealed", sealed, version="2026.11.25.1")

    def interrupted(sealed) -> dict:
        _mark("node")
        raise ValueError("the node died mid-run")

    broken = lazy.apply("atlas-resume-fixture-node", interrupted, version="2026.11.25.1", sealed=root)
    with pytest.raises(RuntimeError, match="step\\(s\\) failed"):
        lazy.run(broken, force_run_failed=False)
    assert (_count("sealed"), _count("node")) == (1, 1)
    assert read_step_status(broken.path(str(tmp_path))) == "FAILED"

    def fixed(sealed) -> dict:
        _mark("node")
        return {"n": 2}

    resumed = lazy.apply("atlas-resume-fixture-node", fixed, version="2026.11.25.1", sealed=root)
    lazy.run(resumed, force_run_failed=True)
    assert (_count("sealed"), _count("node")) == (1, 2)


def test_status_mode_reports_missing_state(tmp_path, plan_document, capsys):
    exit_code = entry_point.main(_argv(plan_document, "status", "--json"))
    output = capsys.readouterr().out
    assert exit_code == 0
    document = json.loads(output)
    assert document["release_id"] == "testrelease"
    nodes = {row["node"]: row for row in document["nodes"]}
    assert len(nodes) == 30
    assert nodes["main_source_sync"]["preflight"] == "missing"
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


# ---------------------------------------------------------------------------
# The release mode: derive from the reviewed config, dry-run or seal+submit.


def _release_config(tmp_path: Path, release_id: str = "testrelease") -> Path:
    """A fixture release config; the derivation's ports are injected in the tests."""
    config = {
        "kind": "atlas-release-config",
        "release_id": release_id,
        "cluster": "cw-us-east-02a",
        "freeze_source_revision": "a" * 40,
        "prior_release": {
            "release_id": "priorrelease",
            "declarations": {"uri": "s3://bucket/prior/declarations.json", "sha256": "c" * 64},
            "identity": {"uri": "s3://bucket/prior/identity.json", "sha256": "c" * 64},
            "backend": {"uri": "s3://bucket/prior/backend.json", "sha256": "c" * 64},
            "composed_release": {"uri": "s3://bucket/prior/release.json", "sha256": "c" * 64},
            "composed_coverage": {"uri": "s3://bucket/prior/coverage.json", "sha256": "c" * 64},
            "oracle_inventory": {"uri": "s3://bucket/prior/oracle-inventory.json", "sha256": "c" * 64},
            "source_counts": {"uri": "s3://bucket/prior/source-counts.json", "sha256": "c" * 64},
        },
        "production_version": 41,
        "preview_base_version": 30,
    }
    path = tmp_path / "release-config.json"
    path.write_text(json.dumps(config))
    return path


def _derived_fixture(tmp_path: Path):
    """A fixture DerivedRelease over the shared fixture plan."""
    plan = fixture_plan(release_root=str(tmp_path / "release"), slice_output_root=str(tmp_path / "slice"))
    receipt = {
        "release_id": plan.release_id,
        "artifact_version": plan.artifact_version,
        "measured": {
            "declared_source_count": plan.expected_source_count,
            "prior_source_count": plan.expected_prior_sources,
            "targets": list(plan.targets),
            "expected_population_tasks": plan.expected_population_tasks,
            "expected_panels": plan.expected_panels,
            "expected_contract_tasks": plan.expected_contract_tasks,
            "population_task_counts": {"Marin:fixture": plan.expected_population_tasks},
            "population_evidence_kinds": {"Marin:fixture": "population"},
        },
    }
    return DerivedRelease(plan_document=plan.model_dump(mode="json"), receipt=receipt)


def _release_argv(config: Path, *extra: str) -> list[str]:
    return ["release", "--release", "testrelease", "--config", str(config), *extra]


def test_release_mode_dry_run_derives_and_prints_the_graph(tmp_path, monkeypatch, capsys):
    """The dry run derives from state, prints the graph, writes nothing."""
    derived = _derived_fixture(tmp_path)
    monkeypatch.setattr(entry_point, "derive_release_plan", lambda *args, **kwargs: derived)
    preflight_calls = {}
    monkeypatch.setattr(
        entry_point,
        "_run_preflight",
        lambda plan, **kwargs: preflight_calls.update({"release_id": plan.release_id}) or 0,
    )
    exit_code = entry_point.main(_release_argv(_release_config(tmp_path), "--dry-run"))
    output = capsys.readouterr().out
    assert exit_code == 0
    assert f"DERIVED testrelease at freeze {'a' * 12}" in output
    assert output.count("NODE ") == EXPECTED_NODE_COUNT
    assert "RELEASE testrelease: plan derived from state" in output
    assert preflight_calls == {"release_id": "testrelease"}
    assert not (tmp_path / "release" / "release-plan.json").exists()


def test_release_mode_seals_and_submits_the_derived_plan(tmp_path, monkeypatch, capsys):
    """Without --dry-run the derivation seals write-once and composes the submitter."""
    derived = _derived_fixture(tmp_path)
    monkeypatch.setattr(entry_point, "derive_release_plan", lambda *args, **kwargs: derived)
    sealed_uri = str(tmp_path / "release" / "release-plan.json")
    monkeypatch.setattr(entry_point, "seal_derivation", lambda release_root, d: {"uri": sealed_uri, "sha256": "b" * 64})
    captured = {}

    def fake_submit(args, *, verb):
        captured["args"] = args
        captured["verb"] = verb
        return 0

    monkeypatch.setattr(entry_point, "_submit_root", fake_submit)
    exit_code = entry_point.main(_release_argv(_release_config(tmp_path)))
    output = capsys.readouterr().out
    assert exit_code == 0
    assert f"SEALED {sealed_uri}" in output
    assert captured["verb"] == "run"
    assert captured["args"].version == "testrelease"
    assert captured["args"].plan == sealed_uri
    assert captured["args"].cluster == "cw-us-east-02a"


def test_release_mode_refuses_a_release_mismatch(tmp_path, capsys):
    exit_code = entry_point.main(
        ["release", "--release", "other", "--config", str(_release_config(tmp_path)), "--dry-run"]
    )
    assert exit_code == 2
    assert "binds release" in capsys.readouterr().err


def test_release_mode_refuses_a_cluster_mismatch(tmp_path, monkeypatch, capsys):
    derived = _derived_fixture(tmp_path)
    monkeypatch.setattr(entry_point, "derive_release_plan", lambda *args, **kwargs: derived)
    exit_code = entry_point.main(_release_argv(_release_config(tmp_path), "--cluster", "cw-us-west-01b", "--dry-run"))
    assert exit_code == 2
    assert "pins cluster" in capsys.readouterr().err
