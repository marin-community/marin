"""Fixed composed-check replay is complete, stable, and never resampled."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from capability_pipeline import composite_grading_diagnostics as mod


def _cell() -> dict:
    return {
        "case_id": "positive-1", "step_index": 0, "check_index": 0,
        "check_id": "gate", "capture_path": "capture",
        "capture_manifest_sha256": "a" * 64,
        "fingerprint": {"workspace": "frozen"}, "original_reward": 0,
        "image": "image@sha256:" + "b" * 64,
        "supervisor_python": "python3", "original_isolation": {"sandbox_id": "original"},
    }


def _write_grades(root: Path, cell: dict, scores: list[float]) -> None:
    raw = root / "raw"
    raw.mkdir(exist_ok=True)
    for repeat, score in enumerate(scores, 1):
        mod._write(raw / f"cell-000-repeat-{repeat:02d}.json", {
            "schema_version": mod.SCHEMA, "cell_index": 0, "repeat": repeat,
            "cell": cell, "status": "graded", "reward": score,
            "detail": {
                "fixed_grading_capture_manifest_sha256": cell["capture_manifest_sha256"],
                "grading_input_fingerprint": cell["fingerprint"],
                "verifier_sandbox_id": f"private-{repeat}",
            },
        })


def _proof(root: Path, cell: dict) -> None:
    binding = root / "binding.json"
    if not binding.exists():
        mod._write(binding, {})
    mod._write(root / "derivation-proof.json", {
        "schema_version": mod.SCHEMA,
        "binding_sha256": mod.sha256(binding),
        "cells_sha256": mod._digest([cell]),
        "derived_checks": 1,
    })


def test_fixed_zero_score_is_ready_with_ten_independent_grades(tmp_path, monkeypatch):
    cell = _cell()
    _write_grades(tmp_path, cell, [0] * 10)
    _proof(tmp_path, cell)
    monkeypatch.setattr(mod, "_validate_input", lambda *_: ([cell], {"candidate_ids": {"candidate"}, "verifier_ids": {"original"}}))

    def isolation(row, seen, candidates, **_):
        sandbox = row["detail"]["verifier_sandbox_id"]
        assert sandbox not in seen | candidates
        seen.add(sandbox)
        return {"sandbox_id": sandbox}

    monkeypatch.setattr(mod, "verifier_isolation_record", isolation)
    state, issues, summary = mod._classify(tmp_path, {})
    assert (state, issues, summary["graded"]) == ("ready", [], 10)


def test_score_change_is_semantic_and_missing_cell_is_pending(tmp_path, monkeypatch):
    cell = _cell()
    _write_grades(tmp_path, cell, [0] * 9 + [1])
    _proof(tmp_path, cell)
    monkeypatch.setattr(mod, "_validate_input", lambda *_: ([cell], {"candidate_ids": set(), "verifier_ids": set()}))
    monkeypatch.setattr(mod, "verifier_isolation_record", lambda *_args, **_kwargs: {})
    state, issues, summary = mod._classify(tmp_path, {})
    assert state == "semantic_failed" and len(issues) == 1 and summary["graded"] == 10
    (tmp_path / "raw/cell-000-repeat-10.json").unlink()
    state, issues, summary = mod._classify(tmp_path, {})
    assert state == "pending" and summary == {"expected": 10, "observed": 9}


def test_started_attempt_is_never_replayed(tmp_path, monkeypatch):
    item = tmp_path / "item"
    item.mkdir()
    cell = _cell()
    source = {"authoring": "fixed"}
    monkeypatch.setattr(mod, "_source", lambda *_args: (source, tmp_path, tmp_path))
    monkeypatch.setattr(mod, "_validate_input", lambda *_args: ([cell], {"candidate_ids": set(), "verifier_ids": set()}))
    monkeypatch.setattr(mod, "verifier_isolation_record", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(mod, "_controller", lambda: {"controller": "fixed"})

    def freeze(attempt, _source, _first, _task, binding, _helper):
        attempt.mkdir(parents=True)
        mod._write(attempt / "binding.json", binding)

    calls = []

    def runner(*, toolchain, attempt, timeout):
        calls.append((toolchain, timeout))
        _proof(attempt, cell)
        _write_grades(attempt, cell, [0] * 10)
        return 0

    monkeypatch.setattr(mod, "_freeze", freeze)
    first = mod.run_composite_grading_diagnostics(item, {}, object(), 30, runner=runner)
    assert first["state"] == "ready" and first["expected_private_grades"] == 10
    second = mod.run_composite_grading_diagnostics(item, {}, object(), 30, runner=runner)
    assert second["state"] == "ready" and len(calls) == 1
    assert json.loads(Path(first["report_artifact"]).read_text())["state"] == "ready"


def test_incomplete_started_attempt_remains_pending(tmp_path, monkeypatch):
    item = tmp_path / "item"
    item.mkdir()
    cell = _cell()
    monkeypatch.setattr(mod, "_source", lambda *_args: ({"authoring": "fixed"}, tmp_path, tmp_path))
    monkeypatch.setattr(mod, "_validate_input", lambda *_args: ([cell], {"candidate_ids": set(), "verifier_ids": set()}))
    monkeypatch.setattr(mod, "_controller", lambda: {"controller": "fixed"})

    def freeze(attempt, _source, _first, _task, binding, _helper):
        attempt.mkdir(parents=True)
        mod._write(attempt / "binding.json", binding)

    calls = []

    def runner(*, toolchain, attempt, timeout):
        calls.append(1)
        _proof(attempt, cell)
        _write_grades(attempt, cell, [0] * 3)
        return 1

    monkeypatch.setattr(mod, "_freeze", freeze)
    first = mod.run_composite_grading_diagnostics(item, {}, object(), 30, runner=runner)
    second = mod.run_composite_grading_diagnostics(item, {}, object(), 30, runner=runner)
    assert first["state"] == second["state"] == "pending"
    assert len(calls) == 1


def test_freeze_selects_authored_positive_oracle_and_negative_control(tmp_path):
    first = tmp_path / "attempt/evaluation/attempts/001"
    first.mkdir(parents=True)
    task = tmp_path / "task"
    task.mkdir()
    for name in ("composite-specification.json", "renderings.json", "composite-verifier.json"):
        (task / name).write_text("{}")
    bundle = tmp_path / "attempt/inputs/bundle"
    bundle.mkdir(parents=True)
    controls = {"cases": [{"id": "p", "class": "positive"}, {"id": "n", "class": "negative"}]}
    mod._write(bundle / "controls.json", controls)
    mod._write(bundle / "binding.json", {"environment": {"kind": "docker"}})
    positive = first / "runtime-trials/oracle-p"
    negative = first / "runtime-trials/control-n"
    for trial in (positive, negative):
        mod._write(trial / "result.json", {"exception_info": None})
        mod._write(trial / "verifier/taskcompendium-result.json", {"status": "graded", "reward": 0})
    oracle_grade = positive / "verifier/taskcompendium-result.json"
    negative_grade = negative / "verifier/taskcompendium-result.json"
    mod._write(first / "authored-oracle.json", {"cases": [{
        "case_id": "p", "step_index": 0,
        "grading_artifact": "runtime-trials/oracle-p/verifier/taskcompendium-result.json",
        "grading_sha256": mod.sha256(oracle_grade),
        "trial_artifact": "runtime-trials/oracle-p/result.json",
        "trial_sha256": mod.sha256(positive / "result.json"),
        "result": mod._read(oracle_grade),
    }]})
    mod._write(first / "runtime-evidence.json", {"cases": [{
        "id": "n", "step_index": 0, "control_type": "authored_adversarial_control",
        "artifact": "runtime-trials/control-n/verifier/taskcompendium-result.json",
        "artifact_sha256": mod.sha256(negative_grade),
        "result": mod._read(negative_grade),
    }]})
    mod._write(first / "receipt.json", {})
    mod._write(first / "artifacts.manifest.json", {})
    frozen = tmp_path / "frozen"
    mod._freeze(frozen, {}, first, task, {"source": "fixed"})
    selected = mod._read(frozen / "input/selected-trials.json")["trials"]
    assert {(row["case_id"], row["source"]) for row in selected} == {
        ("p", "authored_positive"), ("n", "authored_negative")
    }
    assert (frozen / "input/runtime-trials/oracle-p/result.json").is_file()
    assert (frozen / "input/runtime-trials/control-n/result.json").is_file()


def test_validate_input_binds_original_oracle_capture_and_isolation(tmp_path, monkeypatch):
    from capability_pipeline import composite_policy

    inputs = tmp_path / "input"
    trial = inputs / "runtime-trials/oracle-p"
    verifier = trial / "verifier"
    capture_root = verifier / "composite-machine-captures/check-000"
    capture_root.mkdir(parents=True)
    (inputs / "composite-specification.json").write_bytes(b"specification")
    (inputs / "renderings.json").write_bytes(b"renderings")
    mod._write(inputs / "controls.json", {"cases": [{"id": "p", "class": "positive"}]})
    mod._write(inputs / "task-binding.json", {"environment": {"kind": "docker"}})
    mod._write(inputs / "runtime-evidence.json", {"cases": []})
    mod._write(inputs / "receipt.json", {})
    mod._write(inputs / "artifacts.manifest.json", {})
    check = {
        "id": "gate", "role": "gate", "script_path": "checks/gate.sh",
        "args": [], "image": "python@sha256:" + "a" * 64, "timeout": 60,
    }
    config = {
        "schema_version": composite_policy.SCHEMA_VERSION,
        "specification_sha256": mod.sha256(inputs / "composite-specification.json"),
        "implementation": {
            "taskcompendium_revision": "dc6b501c8604bcd2e3c20c1e9947679845fdfef8",
            "adapter_sha256": mod.sha256(Path(mod.__file__).with_name("composite_verifier.py")),
            "policy_sha256": mod.sha256(Path(mod.__file__).with_name("composite_policy.py")),
            "native_judge_protocol_sha256": mod.sha256(Path(mod.__file__).with_name("native_judge_protocol.py")),
        },
        "steps": [{"step_index": 0, "machine_checks": [check], "judge": {
            "criterion_weights": [1.0], "critical_indices": [0],
            "critical_min": 1.0, "conditional_caps": [],
        }}],
    }
    mod._write(inputs / "composite-verifier.json", config)
    fingerprint = {"workspace": "same"}
    synthetic_spec, synthetic_protocol = b"synthetic-spec", b"synthetic-protocol"
    manifest_sha = "b" * 64
    machine = {
        "id": "gate", "status": "graded", "reward": 0,
        "detail": {"grading_input_fingerprint": fingerprint, "verifier_sandbox_id": "private-original"},
    }
    grade = {"status": "graded", "reward": 0, "detail": {
        "machine_results": [machine],
        "composite_machine_captures": [{
            "check_id": "gate", "check_index": 0, "step_index": 0,
            "path": "composite-machine-captures/check-000", "manifest_sha256": manifest_sha,
            "source_specification_sha256": mod.sha256(inputs / "composite-specification.json"),
            "source_renderings_sha256": mod.sha256(inputs / "renderings.json"),
            "config_sha256": mod.sha256(inputs / "composite-verifier.json"),
        }],
    }}
    mod._write(verifier / "taskcompendium-result.json", grade)
    mod._write(trial / "result.json", {"exception_info": None, "verifier_result": {"rewards": {"reward": 0}}})
    mod._write(trial / "daytona-environment.json", {
        "adapter": "taskcompendium-daytona", "daytona_sdk_version": "0.200.2",
        "image": "candidate-image", "sandbox_id": "candidate-original",
        "snapshot": "candidate-snapshot", "network_block_all": True,
    })
    mod._write(inputs / "authored-oracle.json", {"cases": [{
        "case_id": "p", "step_index": 0,
        "grading_artifact": "runtime-trials/oracle-p/verifier/taskcompendium-result.json",
        "grading_sha256": mod.sha256(verifier / "taskcompendium-result.json"),
        "trial_artifact": "runtime-trials/oracle-p/result.json",
        "trial_sha256": mod.sha256(trial / "result.json"), "result": grade,
    }]})
    mod._write(inputs / "selected-trials.json", {"trials": [{
        "case_id": "p", "trial_path": "runtime-trials/oracle-p",
        "grading_path": "runtime-trials/oracle-p/verifier/taskcompendium-result.json",
        "source": "authored_positive",
    }]})
    source = {
        "first_runtime_evidence_sha256": mod.sha256(inputs / "runtime-evidence.json"),
        "first_authored_oracle_sha256": mod.sha256(inputs / "authored-oracle.json"),
        "first_receipt_sha256": mod.sha256(inputs / "receipt.json"),
        "first_artifact_manifest_sha256": mod.sha256(inputs / "artifacts.manifest.json"),
        "controls_sha256": mod.sha256(inputs / "controls.json"),
        "task_binding_sha256": mod.sha256(inputs / "task-binding.json"),
        "specification_sha256": mod.sha256(inputs / "composite-specification.json"),
        "renderings_sha256": mod.sha256(inputs / "renderings.json"),
        "config_sha256": mod.sha256(inputs / "composite-verifier.json"),
    }
    binding = {"source": source, "probe_specific_source": True}
    mod._write(tmp_path / "binding.json", binding)
    mod._write(inputs / "files.json", {"files": mod._files(inputs, exclude={"files.json"})})

    def capture(path, *, expected_manifest_sha256):
        assert path == capture_root and expected_manifest_sha256 == manifest_sha
        return {
            "manifest_sha256": manifest_sha,
            "manifest": {
                "fingerprint": fingerprint,
                "source_specification_sha256": __import__("hashlib").sha256(synthetic_spec).hexdigest(),
                "source_renderings_sha256": __import__("hashlib").sha256(synthetic_protocol).hexdigest(),
            },
            "specification": synthetic_spec, "protocol": synthetic_protocol,
        }

    monkeypatch.setattr(mod, "load_capture", capture)
    monkeypatch.setattr(mod, "_expected_machine_delivery", lambda *_args: (synthetic_spec, synthetic_protocol))

    def isolation(row, seen, candidates, **_kwargs):
        sandbox = row["detail"]["verifier_sandbox_id"]
        assert sandbox not in seen | candidates
        seen.add(sandbox)
        return {"sandbox_id": sandbox}

    monkeypatch.setattr(mod, "verifier_isolation_record", isolation)
    cells, ids = mod._validate_input(tmp_path, binding, derive=True)
    assert len(cells) == 1 and cells[0]["original_reward"] == 0
    assert ids == {"candidate_ids": {"candidate-original"}, "verifier_ids": {"private-original"}}
    # The production branch also binds every copied trial byte to the first
    # evaluator's independently sealed artifact inventory.
    original_files = {
        path.relative_to(inputs).as_posix(): {
            "sha256": mod.sha256(path), "bytes": path.stat().st_size,
        }
        for path in trial.rglob("*") if path.is_file()
    }
    (inputs / "artifacts.manifest.json").write_text(json.dumps({"files": original_files}))
    source["first_artifact_manifest_sha256"] = mod.sha256(inputs / "artifacts.manifest.json")
    binding.pop("probe_specific_source")
    (tmp_path / "binding.json").write_text(json.dumps(binding))
    (inputs / "files.json").write_text(json.dumps({"files": mod._files(inputs, exclude={"files.json"})}))
    assert len(mod._validate_input(tmp_path, binding, derive=True)[0]) == 1
    provider = mod._read(trial / "daytona-environment.json")
    provider["sandbox_id"] = "tampered-candidate"
    (trial / "daytona-environment.json").write_text(json.dumps(provider))
    (inputs / "files.json").write_text(json.dumps({"files": mod._files(inputs, exclude={"files.json"})}))
    with pytest.raises(ValueError, match="first evaluator closure"):
        mod._validate_input(tmp_path, binding, derive=True)


def test_remote_inner_does_not_write_bytecode_into_frozen_helper(tmp_path, monkeypatch):
    import importlib.util

    from capability_pipeline.synthesis import SOURCE_LOCK

    tools = tmp_path / "input/tools"
    tools.mkdir(parents=True)
    (tools / "dt.py").write_text("VALUE = 1\n")
    cell = _cell()
    cell["requested_resource_profile"] = None
    monkeypatch.setenv("CAPABILITY_REMOTE_COMPOSITE_GRADE", "1")
    monkeypatch.setenv("DAYTONA_API_KEY", "test-only")
    monkeypatch.setattr(mod, "_controller", lambda: {"pinned": "same"})
    monkeypatch.setattr(mod, "_validate_input", lambda *_args, **_kwargs: ([cell], {}))
    binding = {
        "controller": {"pinned": "same"},
        "taskcompendium_source_lock_sha256": mod.sha256(SOURCE_LOCK),
    }
    mod._write(tmp_path / "binding.json", binding)

    def grade(*_args, **_kwargs):
        module = importlib.util.module_from_spec(
            importlib.util.spec_from_file_location("frozen_dt_probe", tools / "dt.py")
        )
        module.__spec__.loader.exec_module(module)
        assert module.VALUE == 1
        return SimpleNamespace(status=SimpleNamespace(value="graded"), reward=0, detail={})

    monkeypatch.setitem(sys.modules, "capability_pipeline.daytona_verifier", SimpleNamespace(grade_captured_in_daytona=grade))
    prior = sys.dont_write_bytecode
    try:
        assert mod._inner(tmp_path, mod.sha256(tmp_path / "binding.json")) == 0
    finally:
        sys.dont_write_bytecode = prior
    assert not (tools / "__pycache__").exists()
    assert len(list((tmp_path / "raw").glob("*.json"))) == 10
