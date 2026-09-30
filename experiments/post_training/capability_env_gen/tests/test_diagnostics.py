import hashlib
import json
from pathlib import Path

import pytest

from capability_pipeline import diagnostics


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def item(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "item"
    (root / "contract").mkdir(parents=True)
    (root / "harbor").mkdir(parents=True)
    (root / "workspace" / "task").mkdir(parents=True)
    (root / "contract" / "accepted.json").write_text('{"proposal_hash":"a"}')
    (root / "harbor" / "manifest.json").write_text("{}")
    bundle = root / "workspace" / "task"
    (bundle / "specification.json").write_text("{}")
    (bundle / "binding.json").write_text('{"environment":{"kind":"none"}}')
    (bundle / "controls.json").write_text(
        '{"cases":[{"id":"positive","class":"positive"}]}'
    )
    source = tmp_path / "toolchain"
    source.mkdir()
    return root, source


def test_repeat_diagnostics_require_resolved_primary_attack_review(
    tmp_path, monkeypatch,
):
    from capability_pipeline import attack_adjudication, synthesis

    root, _ = item(tmp_path)
    (root / "harbor" / "specification.json").write_text('{"lowered":true}')
    (root / "runtime-evidence.json").write_text('{"attestation":{}}')

    def attestation_issues(_evidence, _evidence_path, _bundle, _harbor,
                           _controls, specification_sha256):
        # The runtime attests the exported Harbor specification, which can
        # differ from the authored input after TaskCompendium lowering.
        assert specification_sha256 == sha(root / "harbor" / "specification.json")
        assert specification_sha256 != sha(root / "workspace/task/specification.json")
        return ["independent adversary report needs adjudication or retry"]

    monkeypatch.setattr(
        synthesis, "_attestation_issues",
        attestation_issues,
    )
    with pytest.raises(ValueError, match="lack resolved"):
        diagnostics._validated_primary_adversary(root)
    review = root.parent.parent / "attack-adjudication" / root.name / "attempt-1"
    review.mkdir(parents=True)
    (review / "result.json").write_text('{"state":"resolved"}')
    monkeypatch.setattr(
        attack_adjudication, "validate_resolution",
        lambda item_root, review_root: {"state": "resolved"},
    )
    gate = diagnostics._validated_primary_adversary(root)
    assert gate["state"] == "resolved"
    assert gate["runtime_evidence_sha256"] == sha(root / "runtime-evidence.json")
    assert gate["adjudication_result_sha256"] == sha(review / "result.json")


def install_fake_evaluation(monkeypatch, *, passed=True):
    def make_plan(args):
        args.out.write_text(
            json.dumps(
                {
                    "attempts": 3,
                    "inputs": {
                        "package": {"path": "package", "sha256": diagnostics.evaluation.input_hash(args.package)},
                        "bundle": {"path": "bundle", "sha256": diagnostics.evaluation.input_hash(args.bundle)},
                        "controls": {"path": "bundle/controls.json", "sha256": sha(args.controls)},
                    },
                    "controller_files": {"capability_pipeline/evaluation.py": "c" * 64},
                    "unassessed_recipe_rows": diagnostics.evaluation.UNASSESSED,
                    "critical_control_ids": [],
                }
            )
        )
        return 0

    def validate(path, expected):
        plan = json.loads(path.read_text())
        assert sha(path) == expected
        return plan, {}

    def run(args):
        args.out.mkdir()
        plan, _ = validate(args.plan, args.plan_sha256)
        receipts = []
        for number in range(1, 4):
            root = args.out / "attempts" / f"{number:03d}"
            root.mkdir(parents=True)
            (root / "runtime-evidence.json").write_text("{}")
            receipt = {
                "attempt": number,
                "state": "valid",
                "oracle_passed": True,
                "solver_passed": passed,
                "authored_controls_passed": True,
                "independent_attacks_passed": True,
                "sandbox_ids": [],
            }
            artifact = diagnostics.evaluation.artifact_manifest(root)
            (root / "artifacts.manifest.json").write_text(json.dumps(artifact))
            receipt["artifact_manifest_sha256"] = sha(root / "artifacts.manifest.json")
            (root / "receipt.json").write_text(json.dumps(receipt))
            receipts.append(receipt)
        matrix = diagnostics.evaluation.aggregate(plan, receipts)
        matrix_path = args.out / "matrix.json"
        matrix_path.write_text(json.dumps(matrix))
        (args.out / "matrix.attestation.json").write_text(
            json.dumps(
                {
                    "plan_sha256": args.plan_sha256,
                    "matrix_sha256": sha(matrix_path),
                    "input_and_controller_hashes_unchanged_at_finish": True,
                    "initial_inputs": plan["inputs"],
                    "initial_controller": plan["controller_files"],
                    "final_controller": plan["controller_files"],
                    "final_input_sha256": {key: value["sha256"] for key, value in plan["inputs"].items()},
                    "attempt_receipts": {
                        f"{number:03d}": sha(
                            args.out / "attempts" / f"{number:03d}" / "receipt.json"
                        )
                        for number in range(1, 4)
                    },
                }
            )
        )
        return 0 if passed else 2

    monkeypatch.setattr(diagnostics.evaluation, "make_plan", make_plan)
    monkeypatch.setattr(diagnostics.evaluation, "run_evaluation", run)
    monkeypatch.setattr(diagnostics.evaluation, "validate_plan", validate)
    monkeypatch.setattr(
        diagnostics,
        "_toolchain_identity",
        lambda _: {"source_lock_sha256": "b" * 64, "files": {}},
    )
    monkeypatch.setattr(
        diagnostics.evaluation,
        "controller_hashes",
        lambda: {"capability_pipeline/evaluation.py": "c" * 64},
    )


def test_stages_immutable_inputs_and_returns_full_quality_packet(tmp_path, monkeypatch):
    root, source = item(tmp_path)
    install_fake_evaluation(monkeypatch)
    result = diagnostics.run_repeated_diagnostics(root, source, 30)
    assert result["state"] == "ready"
    assert result["reviewable"] is True
    assert isinstance(result["extra_files"]["controller/diagnostics/matrix.json"], Path)
    assert isinstance(
        result["extra_files"][
            "controller/diagnostics/evaluation/attempts/001/receipt.json"
        ],
        Path,
    )
    inputs = root / "diagnostics" / "attempt-1" / "inputs"
    manifest = json.loads((inputs / "manifest.json").read_text())
    assert manifest["files"]["package/manifest.json"] == sha(
        inputs / "package" / "manifest.json"
    )
    assert manifest["plan_sha256"] == sha(inputs / "plan.json")
    assert result["evaluation_payload"]["attempts/001/receipt.json"] == sha(
        root
        / "diagnostics"
        / "attempt-1"
        / "evaluation"
        / "attempts"
        / "001"
        / "receipt.json"
    )


def test_no_status_required_and_status_updates_do_not_break_reuse(
    tmp_path, monkeypatch
):
    root, source = item(tmp_path)
    install_fake_evaluation(monkeypatch)
    first = diagnostics.run_repeated_diagnostics(root, source, 30)
    assert first["reused"] is False
    (root / "status.json").write_text('{"state":"pending_quality_review"}')
    second = diagnostics.run_repeated_diagnostics(root, source, 30)
    assert second["reused"] is True
    assert second["state"] == "ready"


def test_config_drift_creates_new_attempt(tmp_path, monkeypatch):
    root, source = item(tmp_path)
    install_fake_evaluation(monkeypatch)
    resources = tmp_path / "resources.json"
    resources.write_text('{"cpu":2,"memory_gb":2,"disk_gb":10}')
    diagnostics.run_repeated_diagnostics(
        root, source, 30, candidate_resources=resources
    )
    resources.write_text('{"cpu":3,"memory_gb":2,"disk_gb":10}')
    result = diagnostics.run_repeated_diagnostics(
        root, source, 30, candidate_resources=resources
    )
    assert result["reused"] is False
    assert result["attempt"].endswith("attempt-2")
    helper = tmp_path / "dt.py"
    helper.write_text("first")
    diagnostics.run_repeated_diagnostics(root, source, 30, daytona_helper=helper)
    helper.write_text("second")
    helper_result = diagnostics.run_repeated_diagnostics(
        root, source, 30, daytona_helper=helper
    )
    assert helper_result["reused"] is False
    assert helper_result["attempt"].endswith("attempt-4")


def test_tampered_wrapper_result_is_not_trusted(tmp_path, monkeypatch):
    root, source = item(tmp_path)
    install_fake_evaluation(monkeypatch)
    diagnostics.run_repeated_diagnostics(root, source, 30)
    cached = root / "diagnostics" / "attempt-1" / "result.json"
    cached.write_text('{"state":"pending","reviewable":false,"extra_files":{}}')
    result = diagnostics.run_repeated_diagnostics(root, source, 30)
    assert result["reused"] is True
    assert result["state"] == "ready"
    assert result["reviewable"] is True


def test_helper_drift_during_execution_is_pending(tmp_path, monkeypatch):
    root, source = item(tmp_path)
    helper = tmp_path / "dt.py"
    helper.write_text("before")
    install_fake_evaluation(monkeypatch)
    original = diagnostics.evaluation.run_evaluation

    def mutate_helper(args):
        result = original(args)
        helper.write_text("after")
        return result

    monkeypatch.setattr(diagnostics.evaluation, "run_evaluation", mutate_helper)
    result = diagnostics.run_repeated_diagnostics(
        root, source, 30, daytona_helper=helper
    )
    assert result["state"] == "pending"
    assert result["reviewable"] is False


def test_tampered_payload_or_partial_matrix_is_pending_not_reused(
    tmp_path, monkeypatch
):
    root, source = item(tmp_path)
    install_fake_evaluation(monkeypatch)
    diagnostics.run_repeated_diagnostics(root, source, 30)
    evidence = (
        root
        / "diagnostics"
        / "attempt-1"
        / "evaluation"
        / "attempts"
        / "001"
        / "runtime-evidence.json"
    )
    evidence.write_text('{"tampered":true}')
    result = diagnostics.run_repeated_diagnostics(root, source, 30)
    assert result["state"] == "pending"
    assert result["reviewable"] is False
    assert result["attempt"].endswith("attempt-1")

    root2, source2 = item(tmp_path / "partial")
    diagnostics.run_repeated_diagnostics(root2, source2, 30)
    matrix = root2 / "diagnostics" / "attempt-1" / "evaluation" / "matrix.json"
    document = json.loads(matrix.read_text())
    document["cells"] = document["cells"][:2]
    matrix.write_text(json.dumps(document))
    partial = diagnostics.run_repeated_diagnostics(root2, source2, 30)
    assert partial["state"] == "pending"
    assert partial["reviewable"] is False


def test_complete_semantic_gate_failure_is_pending_but_reviewable(
    tmp_path, monkeypatch
):
    root, source = item(tmp_path)
    install_fake_evaluation(monkeypatch, passed=False)
    result = diagnostics.run_repeated_diagnostics(root, source, 30)
    assert result["state"] == "pending"
    assert result["reviewable"] is True
    assert result["issues"] == ["repeated runtime diagnostics are incomplete"]
    assert "resource_envelope" in result["unassessed_recipe_rows"]
