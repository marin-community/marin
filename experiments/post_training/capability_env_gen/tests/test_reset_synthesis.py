import json
import sys
from types import SimpleNamespace

import pytest

from capability_pipeline import synthesis


@pytest.mark.parametrize(
    "state,reviewable,expected_state,expected_reviewable",
    [
        ("ready", True, "ready", True),
        ("semantic_failed", True, "pending", True),
        ("pending", False, "pending", False),
    ],
)
def test_docker_reset_failure_routes_to_review_only_with_complete_evidence(
    tmp_path, monkeypatch, state, reviewable, expected_state, expected_reviewable
):
    contract = tmp_path / "contract"
    contract.mkdir()
    (contract / "accepted.json").write_text(
        json.dumps({"proposal": {"environment": "container", "verification": "judge"}})
    )
    monkeypatch.setattr(
        synthesis,
        "_repeat_and_grading_diagnostics",
        lambda *_: {
            "state": "ready",
            "reviewable": True,
            "extra_files": {"runtime": "runtime.json"},
            "unassessed_recipe_rows": ["reset_conformance"],
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "capability_pipeline.reset_runner",
        SimpleNamespace(
            run_frozen_reset=lambda *_: {
                "state": state,
                "reviewable": reviewable,
                "extra_files": {"reset": "reset.json"},
            }
        ),
    )
    result = synthesis._repeated_quality_diagnostics(tmp_path, None, 30, None)
    assert result["state"] == expected_state
    assert result["reviewable"] is expected_reviewable
    assert set(result["extra_files"]) == {"runtime", "reset"}
    assert result["unassessed_recipe_rows"] == ["reset_conformance"]


def test_incomplete_runtime_does_not_start_new_reset_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        synthesis,
        "_repeat_and_grading_diagnostics",
        lambda *_: {"state": "pending", "reviewable": False},
    )
    monkeypatch.setitem(
        sys.modules,
        "capability_pipeline.reset_runner",
        SimpleNamespace(
            run_frozen_reset=lambda *_: pytest.fail(
                "must retain incomplete runtime without new reset"
            )
        ),
    )
    assert synthesis._repeated_quality_diagnostics(tmp_path, None, 30, None) == {
        "state": "pending",
        "reviewable": False,
    }


@pytest.mark.parametrize("environment", ["reasoning", "shellsim"])
def test_non_docker_reset_evidence_is_required(tmp_path, monkeypatch, environment):
    (tmp_path / "contract").mkdir()
    (tmp_path / "contract/accepted.json").write_text(
        json.dumps({"proposal": {"environment": environment}})
    )
    monkeypatch.setattr(
        synthesis, "_repeat_and_grading_diagnostics",
        lambda *_: {"state": "ready", "reviewable": True},
    )
    calls = []

    def reset(*args, **kwargs):
        calls.append((args, kwargs))
        return {"state": "pending", "reviewable": False, "extra_files": {}}

    monkeypatch.setitem(
        sys.modules, "capability_pipeline.non_docker_reset",
        SimpleNamespace(run_frozen_non_docker_reset=reset),
    )
    result = synthesis._repeated_quality_diagnostics(tmp_path, None, 30, None)
    assert len(calls) == 1
    assert result["state"] == "pending"
    assert result["reviewable"] is False
