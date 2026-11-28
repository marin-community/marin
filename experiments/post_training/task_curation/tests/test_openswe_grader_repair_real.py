# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The grader-repair regression against a real upstream row (coordinator
ruling 2026-11-28): gate v5 found repaired rows whose no-op control still
scored 1 — the eval_script wiped the whole working tree with a suffixed
``git checkout HEAD -- . 2>/dev/null || true`` the reset pattern did not
match, so the agent's diff was discarded before the tests ran. The fixture is
kornia-2476's own bytes from the keep-only release; the repair must scope
every unscoped reset to the test-patch files, excise the golden heredocs, pin
the orphaned status captures, and leave a grader that grades the agent's
workspace."""

import json

from experiments.post_training.task_curation.datasets.openswe import grader_repair
from experiments.post_training.task_curation.datasets.openswe.sources import OpenSweRow

REAL_ROW = json.loads(
    '{\n "base_commit": "9b37ee6cc5b0744de75796e89478bd999e7fd622",\n "eval_script": "#!/bin/bash\\n# Activate the testbed environment if needed\\nif [ -f \\"/opt/conda/etc/profile.d/conda.sh\\" ]; then\\n    . /opt/conda/etc/profile.d/conda.sh\\n    conda activate testbed 2>/dev/null || true\\nfi\\ncd /testbed\\n\\n# Clear any existing patches to ensure clean state\\ngit checkout HEAD -- . 2>/dev/null || true\\n\\necho \\"=== PHASE 1: Run tests with ONLY test patch (no golden patch) ===\\"\\n\\n# Apply only the test patch (without golden patch)\\necho \\"Applying test patch...\\"\\ngit apply -v - <<\'EOF_114329324912\'\\ndiff --git a/test/feature/test_matching.py b/test/feature/test_matching.py\\nindex 209d57f..25c15c7 100644\\n--- a/test/feature/test_matching.py\\n+++ b/test/feature/test_matching.py\\n@@ -468,8 +468,11 @@ class TestLightGlueDISK:\\n         assert idxs.shape[0] == dists.shape[0]\\n         assert dists.shape[0] <= data_dev[\'descs1\'].shape[0]\\n         assert dists.shape[0] <= data_dev[\'descs2\'].shape[0]\\n-        expected_idxs = data_dev[\'lightglue_disk_idxs\'].long()\\n-        assert_close(idxs, expected_idxs, rtol=1e-4, atol=1e-4)\\n+        # key doesn\'t exist in data_dev\\n+        # expected_idxs = data_dev[\'lightglue_disk_idxs\'].long()\\n+        data_dev[\'expected_idxs\'].long()\\n+        # TODO: fix this\\n+        # assert_close(idxs, expected_idxs, rtol=1e-4, atol=1e-4)\\n \\n     @pytest.mark.parametrize(\\"data\\", [\\"lightglue_idxs\\"], indirect=True)\\n     def test_single_nocrash(self, device, dtype, data):\\nEOF_114329324912\\n\\n# Run tests with only test patch (should fail because assertion is commented out)\\necho \\">>>>> Start Test Output (Test Patch Only)\\"\\npytest test/feature/test_matching.py::TestLightGlueDISK::test_real \\\\\\n  -v \\\\\\n  --device cpu \\\\\\n  --dtype float32 \\\\\\n  --no-header \\\\\\n  -rA \\\\\\n  --tb=short \\\\\\n  -k \\"lightglue_idxs\\"\\nrc1=$?\\necho \\">>>>> End Test Output (Test Patch Only)\\"\\necho \\"TEST_PATCH_ONLY_EXIT_CODE=$rc1\\"\\n\\n# Clean up test patch\\ngit checkout HEAD -- test/feature/test_matching.py\\n\\necho \\"=== PHASE 2: Run tests with BOTH test patch AND golden patch ===\\"\\n\\n# Apply golden patch first\\necho \\"Applying golden patch...\\"\\ngit apply -v - <<\'EOF_114329324912\'\\ndiff --git a/kornia/__init__.py b/kornia/__init__.py\\nindex b9f6844..3499c3d 100644\\n--- a/kornia/__init__.py\\n+++ b/kornia/__init__.py\\n@@ -13,4 +13,4 @@ from kornia.testing import xla_is_available\\n from kornia.utils import eye_like, vec_like, create_meshgrid, image_to_tensor, tensor_to_image\\n \\n # Version variable\\n-__version__ = \\"0.6.13-dev\\"\\n+__version__ = \\"0.7.0\\"\\nEOF_114329324912\\n\\n# Apply test patch on top of golden patch\\necho \\"Applying test patch on top of golden patch...\\"\\ngit apply -v - <<\'EOF_114329324912\'\\ndiff --git a/test/feature/test_matching.py b/test/feature/test_matching.py\\nindex 209d57f..25c15c7 100644\\n--- a/test/feature/test_matching.py\\n+++ b/test/feature/test_matching.py\\n@@ -468,8 +468,11 @@ class TestLightGlueDISK:\\n         assert idxs.shape[0] == dists.shape[0]\\n         assert dists.shape[0] <= data_dev[\'descs1\'].shape[0]\\n         assert dists.shape[0] <= data_dev[\'descs2\'].shape[0]\\n-        expected_idxs = data_dev[\'lightglue_disk_idxs\'].long()\\n-        assert_close(idxs, expected_idxs, rtol=1e-4, atol=1e-4)\\n+        # key doesn\'t exist in data_dev\\n+        # expected_idxs = data_dev[\'lightglue_disk_idxs\'].long()\\n+        data_dev[\'expected_idxs\'].long()\\n+        # TODO: fix this\\n+        # assert_close(idxs, expected_idxs, rtol=1e-4, atol=1e-4)\\n \\n     @pytest.mark.parametrize(\\"data\\", [\\"lightglue_idxs\\"], indirect=True)\\n     def test_single_nocrash(self, device, dtype, data):\\nEOF_114329324912\\n\\n# Run tests with both patches (should still fail because test patch comments out assertion)\\necho \\">>>>> Start Test Output (Both Patches)\\"\\npytest test/feature/test_matching.py::TestLightGlueDISK::test_real \\\\\\n  -v \\\\\\n  --device cpu \\\\\\n  --dtype float32 \\\\\\n  --no-header \\\\\\n  -rA \\\\\\n  --tb=short \\\\\\n  -k \\"lightglue_idxs\\"\\nrc2=$?\\necho \\">>>>> End Test Output (Both Patches)\\"\\necho \\"BOTH_PATCHES_EXIT_CODE=$rc2\\"\\n\\n# Clean up all patches\\ngit checkout HEAD -- . 2>/dev/null || true\\n\\necho \\"=== PHASE 3: Run tests with ONLY golden patch (original tests) ===\\"\\n\\n# Apply only golden patch (original tests should pass)\\necho \\"Applying golden patch only...\\"\\ngit apply -v - <<\'EOF_114329324912\'\\ndiff --git a/kornia/__init__.py b/kornia/__init__.py\\nindex b9f6844..3499c3d 100644\\n--- a/kornia/__init__.py\\n+++ b/kornia/__init__.py\\n@@ -13,4 +13,4 @@ from kornia.testing import xla_is_available\\n from kornia.utils import eye_like, vec_like, create_meshgrid, image_to_tensor, tensor_to_image\\n \\n # Version variable\\n-__version__ = \\"0.6.13-dev\\"\\n+__version__ = \\"0.7.0\\"\\nEOF_114329324912\\n\\n# Run tests with only golden patch (should pass)\\necho \\">>>>> Start Test Output (Golden Patch Only)\\"\\npytest test/feature/test_matching.py::TestLightGlueDISK::test_real \\\\\\n  -v \\\\\\n  --device cpu \\\\\\n  --dtype float32 \\\\\\n  --no-header \\\\\\n  -rA \\\\\\n  --tb=short \\\\\\n  -k \\"lightglue_idxs\\"\\nrc3=$?\\necho \\">>>>> End Test Output (Golden Patch Only)\\"\\necho \\"GOLDEN_PATCH_ONLY_EXIT_CODE=$rc3\\"\\n\\n# Final cleanup\\ngit checkout HEAD -- . 2>/dev/null || true\\n\\n# Determine final exit code based on expected behavior:\\n# - Test patch only should make test fail (rc1 != 0)\\n# - Both patches should still make test fail (rc2 != 0) \\n# - Golden patch only should make test pass (rc3 == 0)\\necho \\"=== SUMMARY ===\\"\\necho \\"Test Patch Only Exit Code: $rc1 (expected non-zero)\\"\\necho \\"Both Patches Exit Code: $rc2 (expected non-zero)\\"\\necho \\"Golden Patch Only Exit Code: $rc3 (expected zero)\\"\\n\\n# The primary test we care about is that the golden patch fixes the issue\\n# So we exit with the exit code from golden patch only run\\necho \\"OPENSWE_EXIT_CODE=$rc3\\"",\n "instance_id": "kornia__kornia-2476",\n "patch": "diff --git a/kornia/__init__.py b/kornia/__init__.py\\nindex b9f6844..3499c3d 100644\\n--- a/kornia/__init__.py\\n+++ b/kornia/__init__.py\\n@@ -13,4 +13,4 @@ from kornia.testing import xla_is_available\\n from kornia.utils import eye_like, vec_like, create_meshgrid, image_to_tensor, tensor_to_image\\n \\n # Version variable\\n-__version__ = \\"0.6.13-dev\\"\\n+__version__ = \\"0.7.0\\"\\n",\n "repo": "kornia/kornia",\n "test_patch": "diff --git a/test/feature/test_matching.py b/test/feature/test_matching.py\\nindex 209d57f..25c15c7 100644\\n--- a/test/feature/test_matching.py\\n+++ b/test/feature/test_matching.py\\n@@ -468,8 +468,11 @@ class TestLightGlueDISK:\\n         assert idxs.shape[0] == dists.shape[0]\\n         assert dists.shape[0] <= data_dev[\'descs1\'].shape[0]\\n         assert dists.shape[0] <= data_dev[\'descs2\'].shape[0]\\n-        expected_idxs = data_dev[\'lightglue_disk_idxs\'].long()\\n-        assert_close(idxs, expected_idxs, rtol=1e-4, atol=1e-4)\\n+        # key doesn\'t exist in data_dev\\n+        # expected_idxs = data_dev[\'lightglue_disk_idxs\'].long()\\n+        data_dev[\'expected_idxs\'].long()\\n+        # TODO: fix this\\n+        # assert_close(idxs, expected_idxs, rtol=1e-4, atol=1e-4)\\n \\n     @pytest.mark.parametrize(\\"data\\", [\\"lightglue_idxs\\"], indirect=True)\\n     def test_single_nocrash(self, device, dtype, data):\\n"\n}'  # noqa: E501
)


def real_row() -> OpenSweRow:
    # The classifier reads the grader bytes; the fields it does not read carry
    # inert placeholders so the fixture stays grader-only.
    return OpenSweRow(
        **REAL_ROW,
        problem_statement="grader-only fixture",
        dockerfile="FROM openswe-python-3.11\nCOPY repo /testbed\n",
        image_name="openswe--kornia__kornia-2476",
        license_partition={},
    )


def test_real_kornia_row_classifies_repairable_with_scoped_resets():
    classification = grader_repair.classify_grader(real_row())
    assert classification.verdict == "repairable"
    assert len(classification.golden_sites) == 2
    # The suffixed working-tree wipes are found and scoped, not missed.
    assert len(classification.reset_sites) == 3
    assert all("2>/dev/null || true" in site.suffix for site in classification.reset_sites)


def test_real_kornia_row_repair_grades_the_agent_workspace():
    row = real_row()
    repaired = grader_repair.repaired_script(row, grader_repair.classify_grader(row))
    # No unscoped working-tree reset survives: the agent's diff is what runs.
    assert "git checkout HEAD -- ." not in repaired
    # Three suffixed wipes scoped plus the script's own pre-existing scoped
    # cleanup between its phases.
    assert repaired.count("git checkout HEAD -- test/feature/test_matching.py") == 4
    # Both golden heredocs are gone; the recorded golden patch bytes never appear.
    assert row.patch not in repaired
    assert repaired.count("git apply") == 2  # the two test-patch applications only
    # The script keeps its phases, its exit marker, and its test patch.
    assert "OPENSWE_EXIT_CODE=$rc3" in repaired
    assert row.test_patch in repaired


def test_real_kornia_row_repaired_grader_reclassifies_clean():
    row = real_row()
    repaired = grader_repair.repaired_script(row, grader_repair.classify_grader(row))
    classification = grader_repair.classify_grader(
        OpenSweRow(
            **{
                **REAL_ROW,
                "eval_script": repaired,
                "problem_statement": "grader-only fixture",
                "dockerfile": "FROM openswe-python-3.11\nCOPY repo /testbed\n",
                "image_name": "openswe--kornia__kornia-2476",
                "license_partition": {},
            }
        )
    )
    assert classification.verdict == "clean"
    assert not classification.golden_sites
    assert not classification.reset_sites
