# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The no-op grader scan and repair (coordinator ruling 2026-11-27): a grader
that applies the golden patch before testing passes an empty-diff agent; the
classifier finds it from the row's own bytes, the repair drops the golden
application so the agent's diff is what gets tested, and unrepairable rows
exclude with the ruling's typed reason."""

import pytest

from experiments.post_training.task_curation.datasets.openswe import convert, grader_repair
from experiments.post_training.task_curation.datasets.openswe.sources import OpenSweRow
from experiments.post_training.task_curation.tests.test_openswe_conversion import (
    CONDENSED,
    LICENSE_PARTITION,
)
from experiments.post_training.task_curation.tests.test_openswe_conversion import (
    row as make_row,
)

GOLDEN = "diff --git a/w.py b/w.py\n--- a/w.py\n+++ b/w.py\n@@ -1 +1,2 @@\n+fixed\n"
TEST_PATCH = "diff --git a/test_w.py b/test_w.py\n--- a/test_w.py\n+++ b/test_w.py\n@@ -1 +1,2 @@\n+assert fixed\n"


def grader_row(**overrides) -> OpenSweRow:
    payload = {"patch": GOLDEN, "test_patch": TEST_PATCH}
    payload.update(overrides)
    return make_row(**payload)


def direct_golden_script(golden: str = GOLDEN, test: str = TEST_PATCH) -> str:
    return (
        "#!/bin/bash\ncd /testbed\n"
        '# Apply the golden patch to update source code\necho "Applying golden patch..."\n'
        f"git apply -v --allow-empty - <<'EOF_114329324912'\n{golden}EOF_114329324912\n"
        f"git apply -v --allow-empty - <<'EOF_114329324912'\n{test}EOF_114329324912\n"
        'pytest -q\necho "OPENSWE_EXIT_CODE=$rc"\n'
    )


def test_direct_golden_heredoc_is_repairable_and_excised():
    script = direct_golden_script()
    classification = grader_repair.classify_grader(grader_row(eval_script=script))
    assert classification.verdict == "repairable"
    assert len(classification.golden_sites) == 1
    assert classification.golden_sites[0].match == "exact"
    repaired = grader_repair.repaired_script(grader_row(eval_script=script), classification)
    assert GOLDEN not in repaired
    assert "Applying golden patch" not in repaired
    assert TEST_PATCH in repaired
    assert "pytest -q" in repaired
    assert "OPENSWE_EXIT_CODE" in repaired
    # The repaired grader re-classifies clean against the same row.
    assert grader_repair.classify_grader(grader_row(eval_script=repaired)).verdict == "clean"


def test_clean_script_stays_verbatim():
    script = '#!/bin/bash\ncd /testbed\npytest -q\necho "OPENSWE_EXIT_CODE=$rc"\n'
    classification = grader_repair.classify_grader(grader_row(eval_script=script))
    assert classification.verdict == "clean"
    assert classification.repaired_script is None


def test_unknown_heredoc_is_not_golden():
    setup = "diff --git a/setup.cfg b/setup.cfg\n--- a/setup.cfg\n+++ b/setup.cfg\n@@ -1 +1 @@\n-x\n+y\n"
    script = (
        "#!/bin/bash\ncd /testbed\n"
        f"git apply -v --allow-empty - <<'EOF_SETUP'\n{setup}EOF_SETUP\n"
        f"git apply -v --allow-empty - <<'EOF_114329324912'\n{TEST_PATCH}EOF_114329324912\n"
        'pytest -q\necho "OPENSWE_EXIT_CODE=$rc"\n'
    )
    classification = grader_repair.classify_grader(grader_row(eval_script=script))
    assert classification.verdict == "clean"
    assert len(classification.unknown_sites) == 1
    assert len(classification.test_sites) == 1


def test_truncated_golden_heredoc_is_still_golden():
    truncated = GOLDEN[: len(GOLDEN) - 6] + "\n"
    script = direct_golden_script(golden=truncated)
    classification = grader_repair.classify_grader(grader_row(eval_script=script))
    assert classification.verdict == "repairable"
    assert classification.golden_sites[0].match == "prefix"


def test_unmatched_golden_heredoc_fails_closed():
    """An embedded diff that matches neither recorded patch but touches the
    golden patch's own files is a modified golden application: fail closed."""
    modified = "diff --git a/w.py b/w.py\n--- a/w.py\n+++ b/w.py\n@@ -1 +1 @@\n-different\n"
    script = direct_golden_script(golden=modified)
    classification = grader_repair.classify_grader(grader_row(eval_script=script))
    assert classification.verdict == "unrepairable"
    assert classification.reason == grader_repair.EXCLUSION_REASON


def test_file_based_golden_application_is_noted_not_verdict_driving():
    """A ``git apply <file>`` carries no bytes to verify, so the classifier
    records it and stays clean; the fidelity gate's no-op control adjudicates
    file-based applications behaviorally on its sample."""
    script = (
        "#!/bin/bash\ncd /testbed\n"
        "git apply /tmp/golden.patch\n"
        f"git apply -v --allow-empty - <<'EOF_114329324912'\n{TEST_PATCH}EOF_114329324912\n"
        'pytest -q\necho "OPENSWE_EXIT_CODE=$rc"\n'
    )
    classification = grader_repair.classify_grader(grader_row(eval_script=script))
    assert classification.verdict == "clean"
    assert "git apply /tmp/golden.patch" in classification.detail


def test_assignment_golden_body_empties_and_wipe_scopes():
    script = (
        "#!/bin/bash\ncd /testbed\n"
        "apply_testWithFix_patch() {\n"
        '    echo "Resetting to clean state for testWithFix scenario..."\n'
        "    git checkout -- .\n"
        "    local golden_patch_content=$(cat <<'EOF'\n"
        f"{GOLDEN}EOF\n"
        "    )\n"
        '    echo "$golden_patch_content" | git apply -v --allow-empty -\n'
        "}\n"
        f"git apply -v --allow-empty - <<'EOF_114329324912'\n{TEST_PATCH}EOF_114329324912\n"
        'pytest -q\necho "OPENSWE_EXIT_CODE=$rc"\n'
    )
    classification = grader_repair.classify_grader(grader_row(eval_script=script))
    assert classification.verdict == "repairable"
    assert len(classification.golden_sites) == 1
    assert classification.golden_sites[0].consuming == "assignment"
    assert len(classification.reset_sites) == 1
    repaired = grader_repair.repaired_script(grader_row(eval_script=script), classification)
    assert GOLDEN not in repaired
    # The assignment wrapper stays valid; the golden variable becomes empty.
    assert "local golden_patch_content=$(cat <<'EOF'" in repaired
    assert "git checkout -- ." not in repaired
    assert "git checkout HEAD -- test_w.py" in repaired
    assert TEST_PATCH in repaired


def test_reset_scopes_to_the_test_patch_files_only_when_present():
    script = (
        "#!/bin/bash\ncd /testbed\n"
        f"git apply -v --allow-empty - <<'EOF_114329324912'\n{GOLDEN}EOF_114329324912\n"
        "git checkout -- .\n"
        f"git apply -v --allow-empty - <<'EOF_114329324912'\n{TEST_PATCH}EOF_114329324912\n"
        'pytest -q\necho "OPENSWE_EXIT_CODE=$rc"\n'
    )
    classification = grader_repair.classify_grader(grader_row(eval_script=script))
    assert classification.verdict == "repairable"
    repaired = grader_repair.repaired_script(grader_row(eval_script=script), classification)
    assert "git checkout -- ." not in repaired
    assert "git checkout HEAD -- test_w.py" in repaired


def test_reset_drops_when_no_test_files_are_named():
    script = (
        "#!/bin/bash\ncd /testbed\n"
        f"git apply -v --allow-empty - <<'EOF_114329324912'\n{GOLDEN}EOF_114329324912\n"
        'echo "Resetting to clean state..."\n'
        "git checkout -- .\n"
        'pytest -q\necho "OPENSWE_EXIT_CODE=$rc"\n'
    )
    classification = grader_repair.classify_grader(grader_row(eval_script=script, test_patch="\n"))
    assert classification.verdict == "repairable"
    repaired = grader_repair.repaired_script(grader_row(eval_script=script, test_patch="\n"), classification)
    assert "git checkout" not in repaired
    assert "Resetting to clean state" not in repaired


def test_status_capture_after_golden_excision_pins_to_zero():
    """The capture that read the excised application's exit code would record
    some unrelated command's status; the repair pins it to 0 so the script's
    own gating keeps its meaning (gate v5's mrpro-234 and smart_open-845
    evidence)."""
    script = (
        "#!/bin/bash\ncd /testbed\n"
        'echo "Applying golden patch..."\n'
        f"git apply -v --allow-empty - <<'EOF_114329324912'\n{GOLDEN}EOF_114329324912\n"
        "\n"
        "patch_rc=$?\n"
        "if [ $patch_rc -ne 0 ]; then\n"
        '  echo "manual correction"\n'
        "fi\n"
        f"git apply -v --allow-empty - <<'EOF_114329324912'\n{TEST_PATCH}EOF_114329324912\n"
        'pytest -q\necho "OPENSWE_EXIT_CODE=$rc"\n'
    )
    classification = grader_repair.classify_grader(grader_row(eval_script=script))
    repaired = grader_repair.repaired_script(grader_row(eval_script=script), classification)
    assert "patch_rc=0" in repaired
    assert "patch_rc=$?" not in repaired
    # The test patch's own status capture is untouched.
    assert "git apply" in repaired


def test_suffixed_unscoped_reset_is_found_and_scoped():
    """Gate v5's missed pattern: `git checkout HEAD -- . 2>/dev/null || true`
    discards the agent's diff and the reset pattern required the line to end at
    the command; the pattern now carries trailing constructs into the scoped
    replacement."""
    script = (
        "#!/bin/bash\ncd /testbed\n"
        f"git apply -v --allow-empty - <<'EOF_114329324912'\n{GOLDEN}EOF_114329324912\n"
        "git checkout HEAD -- . 2>/dev/null || true\n"
        f"git apply -v --allow-empty - <<'EOF_114329324912'\n{TEST_PATCH}EOF_114329324912\n"
        'pytest -q\necho "OPENSWE_EXIT_CODE=$rc"\n'
    )
    classification = grader_repair.classify_grader(grader_row(eval_script=script))
    assert len(classification.reset_sites) == 1
    assert classification.reset_sites[0].suffix.strip() == "2>/dev/null || true"
    repaired = grader_repair.repaired_script(grader_row(eval_script=script), classification)
    assert "git checkout HEAD -- . 2>/dev/null || true" not in repaired
    assert "git checkout HEAD -- test_w.py 2>/dev/null || true" in repaired


def test_script_that_does_not_parse_fails_closed():
    """A grader bash refuses to parse scores 0 for every agent state; it can
    never grade, so it cannot be repaired into one that does (gate v5's
    Coqtail/mqttasgi/production-stack evidence)."""
    script = (
        "#!/bin/bash\ncd /testbed\n" "if [ -f x ]; then\n" "    }\n" "fi\n" 'pytest -q\necho "OPENSWE_EXIT_CODE=$rc"\n'
    )
    classification = grader_repair.classify_grader(grader_row(eval_script=script))
    assert classification.verdict == "unrepairable"
    assert classification.reason == "eval_script_syntax_error"
    assert "does not parse" in classification.detail


def test_script_that_echoes_the_verdict_multiple_times_fails_closed():
    """The verifier's ambiguity rule scores any run with several
    OPENSWE_EXIT_CODE lines 0; a phased script that echoes the verdict per
    phase can never award 1 (gate v5's pySBOL2/eiffel evidence)."""
    script = (
        "#!/bin/bash\ncd /testbed\n"
        f"git apply -v --allow-empty - <<'EOF_114329324912'\n{TEST_PATCH}EOF_114329324912\n"
        'pytest -q tests/a\necho "OPENSWE_EXIT_CODE=$rc1"\n'
        'pytest -q tests/b\necho "OPENSWE_EXIT_CODE=$rc2"\n'
    )
    classification = grader_repair.classify_grader(grader_row(eval_script=script))
    assert classification.verdict == "unrepairable"
    assert classification.reason == "ambiguous_exit_code"
    assert "2 times" in classification.detail


def test_single_exit_marker_stays_repairable():
    script = (
        "#!/bin/bash\ncd /testbed\n"
        f"git apply -v --allow-empty - <<'EOF_114329324912'\n{GOLDEN}EOF_114329324912\n"
        'pytest -q\necho "OPENSWE_EXIT_CODE=$rc"\n'
    )
    classification = grader_repair.classify_grader(grader_row(eval_script=script))
    assert classification.verdict == "repairable"


def test_scan_counts_classes():
    rows = [
        grader_row(
            instance_id="acme__clean-1",
            eval_script='#!/bin/bash\ncd /testbed\npytest -q\necho "OPENSWE_EXIT_CODE=$rc"\n',
        ),
        grader_row(instance_id="acme__golden-2", eval_script=direct_golden_script()),
        grader_row(
            instance_id="acme__stuck-3",
            eval_script=direct_golden_script(
                golden="diff --git a/w.py b/w.py\n--- a/w.py\n+++ b/w.py\n@@ -1 +1 @@\n-other\n"
            ),
        ),
    ]
    scan = grader_repair.scan_rows(rows)
    assert scan["counts"] == {"scanned": 3, "clean": 1, "repairable": 1, "unrepairable": 1}
    assert scan["unrepairable_by_reason"] == {grader_repair.EXCLUSION_REASON: 1}


def test_conversion_repairs_the_grader_and_excludes_the_unrepairable():
    repaired_row = grader_row(instance_id="acme__golden-1", eval_script=direct_golden_script())
    files = convert.package_files(repaired_row, CONDENSED)
    emitted = files["tests/eval_script.sh"].decode()
    assert GOLDEN not in emitted
    assert TEST_PATCH in emitted
    assert b"grader_repair" in files["task.toml"]
    stuck_row = grader_row(
        instance_id="acme__stuck-2",
        eval_script=direct_golden_script(
            golden="diff --git a/w.py b/w.py\n--- a/w.py\n+++ b/w.py\n@@ -1 +1 @@\n-altered\n"
        ),
    )
    with pytest.raises(convert.OpenSweRowExcluded) as raised:
        convert.package_files(stuck_row, CONDENSED)
    assert raised.value.reason == grader_repair.EXCLUSION_REASON


def test_license_partition_rides_the_repaired_row():
    repaired_row = grader_row(instance_id="acme__golden-1", eval_script=direct_golden_script())
    files = convert.package_files(repaired_row, CONDENSED)
    assert LICENSE_PARTITION["repo_pinned_commit"] in files["task.toml"].decode()
