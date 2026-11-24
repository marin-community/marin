# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The reconcile's judge repair qualifies pinned routes or applies a versioned re-declaration."""

import json
import sys
from contextlib import contextmanager

from experiments.post_training.task_curation.difficulty import judge_repair


@contextmanager
def _fake_judge_serving(tier, priority, shared_model=None):
    yield "https://together.fixture/v1", shared_model or "fixture-model", "fixture-key", ["job-1"]


def _run(monkeypatch, model: str, result_path) -> dict:
    monkeypatch.setattr(judge_repair, "judge_serving", _fake_judge_serving)
    monkeypatch.setattr(
        judge_repair, "probe_declared_judge_endpoint", lambda *args: {"tools_thinking_probe": {"ok": True}}
    )
    monkeypatch.setattr(sys, "argv", ["judge_repair", "--model", model, "--result", str(result_path)])
    judge_repair.main()
    return json.loads(result_path.read_text())


def test_reconcile_applies_the_standing_ruling_judge_route_repair(tmp_path, monkeypatch):
    """An unserved judge with a versioned repair resolves to the served substitute route."""
    result = _run(monkeypatch, "openai/gpt-4o-mini", tmp_path / "result.json")
    assert result["status"] == "ready"
    assert result["model"] == "openai/gpt-4o-mini"
    assert result["served_model"] == "zai-org/GLM-5.3-Flash"
    assert result["repair_version"] == "2026.11.24.1"


def test_reconcile_still_blocks_a_judge_without_a_repair(tmp_path, monkeypatch):
    """The fail-closed posture holds for unserved judges the ruling does not cover."""
    result = _run(monkeypatch, "openai/gpt-5", tmp_path / "result.json")
    assert result == {
        "status": "blocked",
        "reason": "JUDGE_ROUTE_UNQUALIFIED",
        "detail": "Original judge has no registered immutable serving route",
        "model": "openai/gpt-5",
    }
