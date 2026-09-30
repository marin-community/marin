import json
import sys
from types import SimpleNamespace

import pytest

from capability_pipeline.synthesis import _repeat_and_grading_diagnostics


@pytest.mark.parametrize(
    "repeat_state,grade_state,expected_state,reviewable",
    [
        ("ready", "ready", "ready", True),
        ("ready", "semantic_failed", "pending", True),
        ("ready", "pending", "pending", False),
        ("pending", "ready", "pending", True),
    ],
)
def test_composed_machine_replay_is_separate_from_judge_calibration(
    tmp_path, monkeypatch, repeat_state, grade_state, expected_state, reviewable
):
    (tmp_path / "contract").mkdir()
    (tmp_path / "contract/accepted.json").write_text(
        json.dumps({"proposal": {"verification": "judge"}})
    )
    (tmp_path / "workspace/task").mkdir(parents=True)
    (tmp_path / "workspace/task/composite-verifier.json").write_text("{}")
    repeated = {
        "state": repeat_state, "reviewable": True,
        "unassessed_recipe_rows": ["reward_determinism_10_regrades"],
        "extra_files": {"repeated": "original.json"},
    }
    monkeypatch.setattr(
        "capability_pipeline.diagnostics.run_repeated_diagnostics",
        lambda *a, **k: repeated,
    )
    calls = []

    def grade(*args):
        calls.append(args)
        return {
            "state": grade_state, "reviewable": grade_state != "pending",
            "extra_files": {"composed": "replays.json"},
        }

    monkeypatch.setitem(
        sys.modules, "capability_pipeline.composite_grading_diagnostics",
        SimpleNamespace(run_composite_grading_diagnostics=grade),
    )
    toolchain = SimpleNamespace(package_root=tmp_path, source_package_root=None)
    result = _repeat_and_grading_diagnostics(tmp_path, toolchain, 60, None)
    assert calls == [(tmp_path, repeated, toolchain, 60, None)]
    assert result["state"] == expected_state
    assert result["reviewable"] is reviewable
    assert result["fixed_grading"]["scope"] == "composed_executable_checks_only"
    assert result["fixed_grading"]["judge_evidence"] == "preceding repeated blind calibration"
    assert result["unassessed_recipe_rows"] == ["reward_determinism_10_regrades"]
    assert set(result["extra_files"]) == {"repeated", "composed"}
