import hashlib
import json
import shutil
from pathlib import Path

import pytest

from capability_pipeline import grading_diagnostics, regrade


def _write(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def _toolchain(tmp_path: Path) -> Path:
    source = tmp_path / "taskcompendium"
    source.mkdir(exist_ok=True)
    (source / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0'\n")
    (source / "uv.lock").write_text("version = 1\n")
    return source


def _fixture_source() -> Path:
    return Path(__file__).resolve().parents[1] / "data/c32-evaluation-diagnostic-004"


def _runner(
    *,
    plan_bundle,
    plan_sha256,
    output,
    taskcompendium_source,
    timeout_seconds,
    mode="ready",
):
    assert plan_bundle.is_dir()
    assert taskcompendium_source.is_dir()
    assert timeout_seconds > 0
    plan = json.loads((plan_bundle / "plan.json").read_text())
    controls = json.loads((plan_bundle / "input/bundle/controls.json").read_text())
    cases = {case["id"]: case for case in controls["cases"]}
    rows = []
    for ordinal, cell in enumerate(plan["cells"], 1):
        case = cases[cell["case_id"]]
        reward = case["expect"]["reward_min"]
        if mode == "semantic_failed" and ordinal == 1:
            reward = 0.0
        fingerprint = {
            "schema_version": "capability-grading-input-fingerprint-v1",
            "submission_sha256": "a" * 64,
            "grading_input_sha256": "b" * 64,
            "specification_sha256": "c" * 64,
            "protocol_sha256": "d" * 64,
            "transcript_sha256": "e" * 64,
            "payload_sha256": "f" * 64,
            "step_index": 0,
            "workspace_file_count": 1,
        }
        if mode == "different_input" and ordinal == 2:
            fingerprint["grading_input_sha256"] = "0" * 64
        trial = output / "runtime-trials" / cell["trial_name"]
        candidate = {"sandbox_id": f"candidate-{ordinal}"}
        verifier = {"sandbox_id": f"verifier-{ordinal}"}
        grade = {
            "status": "graded",
            "reward": reward,
            "detail": {
                "grading_input_fingerprint": fingerprint,
                "verifier_sandbox_id": verifier["sandbox_id"],
            },
        }
        rows.append(
            {
                **cell,
                "status": "graded",
                "reward": reward,
                "outcome_class": "graded",
                "private_verifier": verifier,
                "candidate_environment": candidate,
                "trial_sha256": _write(
                    trial / "result.json",
                    json.dumps(
                        {
                            "exception_info": None,
                            "verifier_result": {"rewards": {"reward": reward}},
                        }
                    ).encode(),
                ),
                "grading_sha256": _write(
                    trial / "verifier/taskcompendium-result.json",
                    json.dumps(grade).encode(),
                ),
                "grading_input_fingerprint": fingerprint,
            }
        )
        _write(trial / "daytona-environment.json", json.dumps(candidate).encode())
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_bytes((plan_bundle / "plan.json").read_bytes())
    # This intentionally lies: the wrapper must recompute from raw cells.
    (output / "regrade.json").write_text(
        json.dumps(
            {
                "plan_sha256": plan_sha256,
                "identities_stable": True,
                "summary": {"state": "passed"},
                "cells": rows,
            }
        )
    )
    return 0


@pytest.fixture(autouse=True)
def _mock_isolation(monkeypatch):
    # These older raw Harbor artifact fixtures exercise the legacy validator.
    # Real Docker plans now select capture_once; separate captured fixtures
    # cover that route and the production plan builder stays unchanged.
    original_build_plan = regrade.build_plan

    def legacy_harbor_plan(*args, **kwargs):
        plan = original_build_plan(*args, **kwargs)
        if plan["binding_kind"] == "docker":
            plan["grading_strategy"] = "harbor_replay"
        return plan

    monkeypatch.setattr(regrade, "build_plan", legacy_harbor_plan)

    def provider(path, _root, seen):
        value = json.loads(path.read_text())
        if value["sandbox_id"] in seen:
            raise RuntimeError("duplicate candidate")
        seen.add(value["sandbox_id"])
        return value

    def verifier(value, seen, candidates, **_kwargs):
        sandbox = value["detail"].get("verifier_sandbox_id")
        # The test fixture records this separately in the row, so bind it from
        # the raw grade JSON rather than trusting report fields.
        if sandbox is None:
            raise RuntimeError("missing verifier")
        if sandbox in seen or sandbox in candidates:
            raise RuntimeError("duplicate verifier")
        seen.add(sandbox)
        return {"sandbox_id": sandbox}

    monkeypatch.setattr(grading_diagnostics, "provider_isolation_record", provider)
    monkeypatch.setattr(grading_diagnostics, "verifier_isolation_record", verifier)


def test_grading_diagnostics_ready_reuses_verified_raw_artifacts(tmp_path):
    source = _fixture_source()
    result = grading_diagnostics.run_grading_diagnostics(
        source,
        _toolchain(tmp_path),
        tmp_path / "diagnostics",
        parallelism=8,
        runner=_runner,
    )
    assert result["state"] == "ready"
    assert result["reviewable"] is True
    assert result["summary"]["state"] == "passed"
    assert len(result["extra_files"]) == 5

    # Editable summary/result booleans are ignored on reuse.
    receipt = tmp_path / "diagnostics/grading-diagnostics.json"
    receipt.write_text(json.dumps({"state": "ready", "reviewable": True}))
    reused = grading_diagnostics.run_grading_diagnostics(
        source,
        _toolchain(tmp_path),
        tmp_path / "diagnostics",
        parallelism=8,
        runner=lambda **_: (_ for _ in ()).throw(AssertionError("must reuse")),
    )
    assert reused["state"] == "ready"
    drifted = grading_diagnostics.run_grading_diagnostics(
        source,
        _toolchain(tmp_path),
        tmp_path / "diagnostics",
        parallelism=4,
    )
    assert drifted["state"] == "pending"
    assert "identity drifted" in drifted["issues"][0]

    report = json.loads((tmp_path / "diagnostics/regrade/regrade.json").read_text())
    report["cells"][0]["trial_sha256"] = "0" * 64
    (tmp_path / "diagnostics/regrade/regrade.json").write_text(json.dumps(report))
    tampered = grading_diagnostics.run_grading_diagnostics(
        source,
        _toolchain(tmp_path),
        tmp_path / "diagnostics",
        parallelism=8,
    )
    assert tampered["state"] == "pending"
    assert tampered["reviewable"] is False


def test_grading_diagnostics_classifies_semantic_and_input_drift_as_distinct(tmp_path):
    source = _fixture_source()
    semantic = grading_diagnostics.run_grading_diagnostics(
        source,
        _toolchain(tmp_path),
        tmp_path / "semantic",
        runner=lambda **kwargs: _runner(**kwargs, mode="semantic_failed"),
    )
    assert semantic["state"] == "semantic_failed"
    assert semantic["reviewable"] is True

    different = grading_diagnostics.run_grading_diagnostics(
        source,
        _toolchain(tmp_path),
        tmp_path / "different",
        runner=lambda **kwargs: _runner(**kwargs, mode="different_input"),
    )
    assert different["state"] == "pending"
    assert different["reviewable"] is False
    assert "grading input bytes differ" in different["issues"][0]


def test_grading_diagnostics_returns_unassessed_for_unsupported_verifier(tmp_path):
    source = tmp_path / "unsupported"
    shutil.copytree(_fixture_source(), source)
    specification = source / "bundle/specification.json"
    value = json.loads(specification.read_text())
    value["steps"][0]["verifier"]["mode"] = "other"
    specification.write_text(json.dumps(value))
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["files"]["bundle/specification.json"] = hashlib.sha256(
        specification.read_bytes()
    ).hexdigest()
    (source / "manifest.json").write_text(json.dumps(manifest))
    result = grading_diagnostics.run_grading_diagnostics(
        source, _toolchain(tmp_path), tmp_path / "out"
    )
    assert result["state"] == "unsupported"
    assert result["unassessed"] is True
    assert result["reviewable"] is False


def _report(path: Path) -> dict:
    return json.loads((path / "regrade/regrade.json").read_text())


def _save_report(path: Path, report: dict) -> None:
    (path / "regrade/regrade.json").write_text(json.dumps(report))


def test_raw_grade_and_cross_sandbox_receipts_cannot_be_forged(tmp_path):
    root = tmp_path / "diagnostics"
    grading_diagnostics.run_grading_diagnostics(
        _fixture_source(), _toolchain(tmp_path), root, runner=_runner
    )
    report = _report(root)
    row = report["cells"][0]
    grade_path = (
        root
        / "regrade/runtime-trials"
        / row["trial_name"]
        / "verifier/taskcompendium-result.json"
    )
    grade = json.loads(grade_path.read_text())
    grade["reward"] = 0.5
    row["grading_sha256"] = _write(grade_path, json.dumps(grade).encode())
    _save_report(root, report)
    assert (
        grading_diagnostics.run_grading_diagnostics(
            _fixture_source(), _toolchain(tmp_path), root
        )["state"]
        == "pending"
    )

    root = tmp_path / "cross"
    grading_diagnostics.run_grading_diagnostics(
        _fixture_source(), _toolchain(tmp_path), root, runner=_runner
    )
    report = _report(root)
    row = report["cells"][0]
    row["private_verifier"] = {"sandbox_id": "candidate-1"}
    grade_path = (
        root
        / "regrade/runtime-trials"
        / row["trial_name"]
        / "verifier/taskcompendium-result.json"
    )
    grade = json.loads(grade_path.read_text())
    grade["detail"]["verifier_sandbox_id"] = "candidate-1"
    row["grading_sha256"] = _write(grade_path, json.dumps(grade).encode())
    _save_report(root, report)
    assert (
        grading_diagnostics.run_grading_diagnostics(
            _fixture_source(), _toolchain(tmp_path), root
        )["state"]
        == "pending"
    )


def test_no_tool_private_receipt_drift_and_symlinks_remain_pending(tmp_path):
    source = Path(__file__).resolve().parents[1] / "data/c17-repeated-evaluation-003"
    if not source.is_dir():
        pytest.skip("frozen no-tool fixture unavailable")
    result = grading_diagnostics.run_grading_diagnostics(
        source, _toolchain(tmp_path), tmp_path / "none", runner=_runner
    )
    assert result["state"] == "ready"
    report = _report(tmp_path / "none")
    row = report["cells"][0]
    grade_path = (
        tmp_path
        / "none/regrade/runtime-trials"
        / row["trial_name"]
        / "verifier/taskcompendium-result.json"
    )
    grade = json.loads(grade_path.read_text())
    grade["detail"].pop("verifier_sandbox_id")
    row["grading_sha256"] = _write(grade_path, json.dumps(grade).encode())
    _save_report(tmp_path / "none", report)
    assert (
        grading_diagnostics.run_grading_diagnostics(
            source, _toolchain(tmp_path), tmp_path / "none"
        )["state"]
        == "pending"
    )

    copied = tmp_path / "source"
    shutil.copytree(_fixture_source(), copied)
    root = tmp_path / "drift"
    grading_diagnostics.run_grading_diagnostics(
        copied, _toolchain(tmp_path), root, runner=_runner
    )
    (copied / "bundle/controls.json").write_text("{}")
    assert (
        grading_diagnostics.run_grading_diagnostics(copied, _toolchain(tmp_path), root)[
            "state"
        ]
        == "pending"
    )

    root = tmp_path / "symlink"
    grading_diagnostics.run_grading_diagnostics(
        _fixture_source(), _toolchain(tmp_path), root, runner=_runner
    )
    (root / "plan-bundle/input/bundle/unsafe").symlink_to(root / "binding.json")
    assert (
        grading_diagnostics.run_grading_diagnostics(
            _fixture_source(), _toolchain(tmp_path), root
        )["state"]
        == "pending"
    )

    root = tmp_path / "output-symlink"
    grading_diagnostics.run_grading_diagnostics(
        _fixture_source(), _toolchain(tmp_path), root, runner=_runner
    )
    (root / "regrade/unsafe").symlink_to(root / "binding.json")
    assert (
        grading_diagnostics.run_grading_diagnostics(
            _fixture_source(), _toolchain(tmp_path), root
        )["state"]
        == "pending"
    )


def test_invalid_outcome_partial_output_and_updated_report_hash_stay_pending(tmp_path):
    root = tmp_path / "outcome"
    grading_diagnostics.run_grading_diagnostics(
        _fixture_source(), _toolchain(tmp_path), root, runner=_runner
    )
    report = _report(root)
    report["cells"][0]["outcome_class"] = "runner_exception"
    report["cells"][0]["exception"] = {"type": "RuntimeError"}
    _save_report(root, report)
    assert (
        grading_diagnostics.run_grading_diagnostics(
            _fixture_source(), _toolchain(tmp_path), root
        )["state"]
        == "pending"
    )

    root = tmp_path / "partial"
    grading_diagnostics.run_grading_diagnostics(
        _fixture_source(), _toolchain(tmp_path), root, runner=_runner
    )
    (root / "regrade/regrade.json").unlink()
    called = []
    result = grading_diagnostics.run_grading_diagnostics(
        _fixture_source(),
        _toolchain(tmp_path),
        root,
        runner=lambda **_: called.append(True),
    )
    assert result["state"] == "pending"
    assert not called

    root = tmp_path / "inventory"
    grading_diagnostics.run_grading_diagnostics(
        _fixture_source(), _toolchain(tmp_path), root, runner=_runner
    )
    report = _report(root)
    row = report["cells"][0]
    trial = root / "regrade/runtime-trials" / row["trial_name"] / "result.json"
    row["trial_sha256"] = _write(trial, b'{"changed":true}')
    _save_report(root, report)
    assert (
        grading_diagnostics.run_grading_diagnostics(
            _fixture_source(), _toolchain(tmp_path), root
        )["state"]
        == "pending"
    )
