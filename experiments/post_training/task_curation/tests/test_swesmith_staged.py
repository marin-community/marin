# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controls protect stage admission from false completion and altered tests."""

import json
import subprocess

import pytest

from experiments.post_training.task_curation.swesmith_handoff import repair_grade
from experiments.post_training.task_curation.swesmith_staged import (
    SourceTask,
    candidate_integrity_violations,
    partial_hunks,
    pilot_task,
)


@pytest.mark.parametrize("failure", ["timeout", "docker_error"])
def test_pilot_infrastructure_failure_is_not_a_task_exclusion(tmp_path, monkeypatch, failure):
    task = SourceTask(
        "source/task",
        "train",
        "https://example.org/repo",
        "branch",
        "",
        "",
        "task_hash",
        "solution_hash",
        {"tests/verifier.toml": b"paths = []\nmust_pass = []\n"},
        {},
    )

    def run(command, **_kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 900)
        return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"provider unavailable")

    monkeypatch.setattr(subprocess, "run", run)
    result = pilot_task(task, tmp_path, "image", "module", "")
    assert result["status"] == "infrastructure_failed"
    assert result["controls"] == {}
    assert result["exclusions"] == []
    persisted = json.loads((tmp_path / "pilots/task/attempt-0001/result.json").read_text())
    assert persisted == result


def test_handoff_integrity_accepts_source_repair_without_added_tests_and_rejects_weakened_tests():
    baseline = {"library.py": b"return wrong\n", "tests/test_public.py": b"assert actual == expected\n"}
    repaired = {**baseline, "library.py": b"return correct\n"}
    assert candidate_integrity_violations(repaired, baseline, {"library.py"}) == []
    # Hidden tests were never part of this actor checkout; no new test is required.
    deleted = {"library.py": repaired["library.py"]}
    assert candidate_integrity_violations(deleted, baseline, {"library.py"}) == ["tests/test_public.py"]
    weakened = {**repaired, "tests/test_public.py": b"assert True\n"}
    assert candidate_integrity_violations(weakened, baseline, {"library.py"}) == ["tests/test_public.py"]


def test_partial_recipe_retains_a_real_unrepaired_behavior(tmp_path):
    (tmp_path / "first.py").write_text("answer = 0\n")
    (tmp_path / "second.py").write_text("answer = 0\n")
    patch = """diff --git a/first.py b/first.py
--- a/first.py
+++ b/first.py
@@ -1 +1 @@
-answer = 0
+answer = 1
diff --git a/second.py b/second.py
--- a/second.py
+++ b/second.py
@@ -1 +1 @@
-answer = 0
+answer = 2
"""
    recipes = partial_hunks(patch)
    assert len(recipes) == 1
    subprocess.run(["git", "apply", "-"], input=recipes[0].encode(), cwd=tmp_path, check=True)
    assert (tmp_path / "first.py").read_text() == "answer = 1\n"
    assert (tmp_path / "second.py").read_text() == "answer = 0\n"
    subprocess.run(["git", "apply", "--reverse", "-"], input=recipes[0].encode(), cwd=tmp_path, check=True)
    subprocess.run(["git", "apply", "-"], input=patch.encode(), cwd=tmp_path, check=True)
    assert (tmp_path / "second.py").read_text() == "answer = 2\n"


def test_grader_rejects_nested_and_top_level_hidden_tests_misclassified_as_source(tmp_path):
    # Nested and top-level test files must never become actor repair targets.
    for path in ("glom/test/test_mutation.py", "test_addict.py"):
        contract = {
            "tracked_sha256": {"library.py": "unused"},
            "changed_source_paths": ["library.py", path],
            "verifier": {"paths": [path]},
        }
        result = repair_grade(contract, {"/testbed/library.py": b"candidate"}, tmp_path)
        assert result == {"passed": False, "error": "invalid_task_repair_scope_contains_tests_or_untracked_paths"}
