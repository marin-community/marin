# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controls protect stage admission from false completion and altered tests."""

import subprocess

from experiments.post_training.task_curation.swesmith_handoff import repair_grade
from experiments.post_training.task_curation.swesmith_staged import candidate_integrity, partial_hunks


def test_handoff_integrity_accepts_source_repair_without_added_tests_and_rejects_weakened_tests():
    baseline = {"library.py": b"return wrong\n", "tests/test_public.py": b"assert actual == expected\n"}
    repaired = {**baseline, "library.py": b"return correct\n"}
    assert candidate_integrity(repaired, baseline, {"library.py"}) == []
    # Hidden tests were never part of this actor checkout; no new test is required.
    deleted = {"library.py": repaired["library.py"]}
    assert candidate_integrity(deleted, baseline, {"library.py"}) == ["tests/test_public.py"]
    weakened = {**repaired, "tests/test_public.py": b"assert True\n"}
    assert candidate_integrity(weakened, baseline, {"library.py"}) == ["tests/test_public.py"]


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
    # The first importer version mislabeled these real source layouts and copied
    # their hidden tests into the actor, then crashed when collecting missing files.
    for path in ("glom/test/test_mutation.py", "test_addict.py"):
        contract = {
            "tracked_sha256": {"library.py": "unused"},
            "changed_source_paths": ["library.py", path],
            "verifier": {"paths": [path]},
        }
        result = repair_grade(contract, {"/testbed/library.py": b"candidate"}, tmp_path)
        assert result == {"passed": False, "error": "invalid_task_repair_scope_contains_tests_or_untracked_paths"}
