# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for PII exposure in the Marin codebase.

Runs the PII scanner against all git-tracked files and asserts that no
new unallowed PII patterns have been introduced. Existing acceptable
findings (contributor emails, test fixtures from public data) are
suppressed via the allowlist.
"""

from pathlib import Path

from scripts.security.scan_pii import Finding, is_allowed, load_allowlist, scan_repo

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST_PATH = REPO_ROOT / "scripts" / "security" / "pii_allowlist.yaml"


def test_no_new_pii_findings():
    """Fail if any PII finding is not covered by the allowlist."""
    unallowed, _allowed = scan_repo(REPO_ROOT, ALLOWLIST_PATH)
    if unallowed:
        details = "\n".join(f"  {f.file}:{f.line_number} [{f.pattern_name}] {f.matched_text}" for f in unallowed)
        raise AssertionError(
            f"{len(unallowed)} PII finding(s) not in allowlist:\n{details}\n\n"
            "If these are acceptable, add them to scripts/security/pii_allowlist.yaml"
        )


def test_allowlist_suppresses_known_findings():
    """Verify the allowlist is actually suppressing findings (not vacuously passing)."""
    _, allowed = scan_repo(REPO_ROOT, ALLOWLIST_PATH)
    assert len(allowed) > 0, "Expected allowlist to suppress at least some findings"


def test_allowlist_matching():
    """Verify allowlist glob matching works for representative patterns."""
    allowlist = load_allowlist(ALLOWLIST_PATH)
    assert len(allowlist) > 0

    contributor_finding = Finding(
        file="lib/levanter/CONTRIBUTORS.md",
        line_number=5,
        pattern_name="email",
        matched_text="test@example.com",
    )
    assert is_allowed(contributor_finding, allowlist)

    snapshot_finding = Finding(
        file="tests/snapshots/ar5iv/inputs/arxiv_1.html",
        line_number=100,
        pattern_name="email",
        matched_text="author@university.edu",
    )
    assert is_allowed(snapshot_finding, allowlist)

    # An email in application source code should NOT be allowed
    src_finding = Finding(
        file="lib/marin/src/marin/processing/some_module.py",
        line_number=10,
        pattern_name="email",
        matched_text="leaked@personal.com",
    )
    assert not is_allowed(src_finding, allowlist)

    # An API key anywhere should NOT be allowed (no api_key entries in allowlist)
    api_key_finding = Finding(
        file="lib/levanter/CONTRIBUTORS.md",
        line_number=1,
        pattern_name="api_key",
        matched_text="sk-abcdefghijklmnopqrstuvwxyz",
    )
    assert not is_allowed(api_key_finding, allowlist)
