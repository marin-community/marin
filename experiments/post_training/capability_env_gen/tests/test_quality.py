import copy
import json
from types import SimpleNamespace

import pytest

from capability_pipeline.quality import (
    AXES,
    COMMON_CONDITIONS,
    has_executable_verifier,
    prepare_packet,
    run_review,
    validate_receipt,
)


@pytest.mark.parametrize(
    "verifier",
    [
        {"kind": "code_answer", "verifier": {}},
        {"kind": "tasktrove", "mode": "script"},
        {"kind": "tasktrove", "mode": "exact", "runtime": {"kind": "container"}},
    ],
)
def test_executable_verifier_detection_is_independent_of_proposal_label(verifier):
    assert has_executable_verifier({"steps": [{"verifier": verifier}]})


def build(root):
    for path, contents in {
        "contract/accepted.json": "{}",
        "workspace/task/specification.json": '{"fixture":"not a real TaskSpec"}',
        "workspace/task/node_modules/required.txt": "final bundle resource must survive",
        "workspace/node_modules/ignored.txt": "dependency tree",
        "runtime-evidence.json": '{"fixture":"not real execution"}',
    }.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)


def test_prepare_packet_includes_controller_sidecar_without_adding_item_files(
    tmp_path,
):
    root, output = tmp_path / "item", tmp_path / "review"
    build(root)
    sidecar = tmp_path / "adjudication.json"
    sidecar.write_text('{"state":"resolved"}')
    manifest = prepare_packet(
        root,
        output,
        {"controller/attack-adjudication/result.json": sidecar},
    )
    assert "controller/attack-adjudication/result.json" in manifest["files"]
    assert "controller/attack-adjudication/result.json" not in manifest["item_files"]


def receipt(manifest):
    path = "workspace/task/specification.json"
    return {
        "schema_version": "capability-quality-review-v1",
        "snapshot_hash": manifest["snapshot_hash"],
        "decision": "accept",
        "scores": dict.fromkeys(AXES, 4),
        "findings": [
            {
                "axis": axis,
                "severity": "pass",
                "claim": "fixture claim",
                "citations": [
                    {
                        "path": path,
                        "sha256": manifest["files"][path],
                        "supports": "fixture only",
                    }
                ],
            }
            for axis in sorted(AXES)
        ],
        "build_conditions": [
            {
                "id": identifier,
                "state": "passed",
                "axis": "reproducibility",
                "severity": "pass",
                "claim": "Unit fixture only, not empirical proof",
                "citations": [
                    {
                        "path": path,
                        "sha256": manifest["files"][path],
                        "supports": "Unit fixture only",
                    }
                ],
            }
            for identifier in manifest["required_build_conditions"]
        ],
        "required_changes": [],
        "limitations": ["Unit fixture only; no task validity established"],
    }


def test_packet_keeps_complete_final_bundle_and_rejects_reuse(tmp_path):
    root, output = tmp_path / "build", tmp_path / "review"
    build(root)
    manifest = prepare_packet(root, output)
    assert "workspace/task/node_modules/required.txt" in manifest["files"]
    assert "workspace/node_modules/ignored.txt" not in manifest["files"]
    with pytest.raises(ValueError, match="fresh review"):
        prepare_packet(root, output)
    with pytest.raises(ValueError, match="outside"):
        prepare_packet(root, root / "review")


def test_changed_snapshot_or_false_green_cannot_accept(tmp_path):
    root, output = tmp_path / "build", tmp_path / "review"
    build(root)
    manifest = prepare_packet(root, output)
    value = receipt(manifest)
    validate_receipt(value, manifest, output / "input")
    changed = copy.deepcopy(value)
    changed["required_changes"] = ["Fix incorrect answer"]
    with pytest.raises(ValueError, match="contradicts"):
        validate_receipt(changed, manifest, output / "input")
    changed = copy.deepcopy(value)
    changed["snapshot_hash"] = "other"
    with pytest.raises(ValueError, match="another snapshot"):
        validate_receipt(changed, manifest, output / "input")
    path = output / "input/workspace/task/specification.json"
    path.chmod(0o644)
    path.write_text("changed")
    with pytest.raises(ValueError, match="snapshot was modified"):
        validate_receipt(value, manifest, output / "input")


def test_build_conditions_cannot_be_skipped_or_declared_missing_on_accept(tmp_path):
    root, output = tmp_path / "build", tmp_path / "review"
    build(root)
    manifest = prepare_packet(root, output)
    manifest["required_build_conditions"] = ["engine-probe"]
    value = receipt(manifest)
    value["build_conditions"] = []
    with pytest.raises(ValueError, match="every required"):
        validate_receipt(value, manifest, output / "input")
    value["build_conditions"] = [
        {**value["findings"][0], "id": "engine-probe", "state": "missing"}
    ]
    with pytest.raises(ValueError, match="condition unresolved"):
        validate_receipt(value, manifest, output / "input")


def test_source_mutation_during_review_invalidates_receipt(tmp_path):
    root, output = tmp_path / "build", tmp_path / "review"
    build(root)

    def invoke(workspace, *args):
        manifest = json.loads((workspace / "input-manifest.json").read_text())
        (workspace / "review.json").write_text(json.dumps(receipt(manifest)))
        (root / "runtime-evidence.json").write_text("changed after snapshot")
        return {"returncode": 0, "timed_out": False, "stdout": "", "stderr": ""}

    result = run_review(root, output, SimpleNamespace(invoke=invoke, model="fixture"))
    assert result["state"] == "pending"
    assert "changed during" in result["issues"][0]
    assert result["publication_certified"] is False


def test_blueprint_gates_remain_required_without_a_pilot_checklist(tmp_path):
    root, output = tmp_path / "build", tmp_path / "review"
    build(root)
    (root / "contract/accepted.json").write_text(
        json.dumps(
            {
                "proposal": {
                    "builder_plan": [
                        {"acceptance_checks": ["Measure reproducible reset"]}
                    ],
                    "validation_plan": ["Test held-out boundary cases"],
                },
                "review": {
                    "issues": ["Confirm the engine supports the required operation"]
                },
            }
        )
    )
    manifest = prepare_packet(root, output)
    assert set(COMMON_CONDITIONS) <= set(manifest["required_build_conditions"])
    assert len(manifest["required_build_conditions"]) == len(COMMON_CONDITIONS) + 3
    assert (
        manifest["planned_build_conditions"]["plan-validation-1"]
        == "Test held-out boundary cases"
    )
    incomplete = receipt(manifest)
    incomplete["build_conditions"] = [
        c for c in incomplete["build_conditions"] if c["id"] != "quality-oracle"
    ]
    with pytest.raises(ValueError, match="every required"):
        validate_receipt(incomplete, manifest, output / "input")
