# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from iac.stack_drift import MARKER, StackDrift, fingerprint, load_stack_drifts, render_markdown


def _write_artifact(root: Path, stack: str, severity: str, diff: str, **counts: int) -> None:
    directory = root / f"iac-preview-{stack}"
    directory.mkdir(parents=True)
    meta = {"stack": stack, "ok": severity != "error", "severity": severity, **counts}
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (directory / "diff.txt").write_text(diff, encoding="utf-8")


def _drift(stack: str, severity: str, diff: str = "", **counts: int) -> StackDrift:
    filled = {key: counts.get(key, 0) for key in ("create", "update", "delete", "replace", "import")}
    return StackDrift(stack=stack, severity=severity, counts=filled, diff=diff)


def test_load_stack_drifts_reads_each_artifact_directory(tmp_path):
    _write_artifact(tmp_path, "marin", "change", "~ role", update=1)
    _write_artifact(tmp_path, "cw-rno2a", "none", "")

    drifts = load_stack_drifts(tmp_path)

    assert [(d.stack, d.severity, d.clean) for d in drifts] == [
        ("cw-rno2a", "none", True),
        ("marin", "change", False),
    ]
    assert drifts[1].counts["update"] == 1
    assert drifts[1].diff == "~ role"


def test_fingerprint_ignores_diff_text_churn():
    # A refreshed diff carries provider-side etag churn; the same drift shape must not re-notify.
    a = _drift("marin", "change", "~ role [etag=aaa]", update=1)
    b = _drift("marin", "change", "~ role [etag=bbb]", update=1)

    assert fingerprint([a]) == fingerprint([b])


def test_fingerprint_changes_when_the_drift_shape_changes():
    one = _drift("marin", "change", "~ role", update=1)
    two = _drift("marin", "change", "~ role", update=2)
    deleting = _drift("marin", "delete", "- role", delete=1)

    assert fingerprint([one]) != fingerprint([two])
    assert fingerprint([one]) != fingerprint([deleting])


def test_fingerprint_is_stack_order_independent():
    a = _drift("marin", "change", update=1)
    b = _drift("cw-rno2a", "none")

    assert fingerprint([a, b]) == fingerprint([b, a])


def test_render_markdown_embeds_marker_fingerprint_and_only_dirty_diffs():
    dirty = _drift("marin", "change", "~ gcp:projects/iAMCustomRole", update=1)
    clean = _drift("cw-rno2a", "none", "")

    body = render_markdown([dirty, clean], run_url="https://run")

    assert MARKER in body
    assert f"fingerprint:{fingerprint([dirty, clean])}" in body
    assert "1 to update" in body
    # Every stack appears in the summary list...
    assert "`cw-rno2a`" in body
    # ...but only the drifted one gets a diff section.
    assert "## `marin`" in body
    assert "## `cw-rno2a`" not in body
    assert "https://run" in body


def test_render_markdown_truncates_an_oversized_diff():
    huge = _drift("marin", "change", "~ x" * 20_000, update=1)

    body = render_markdown([huge], run_url="https://run")

    assert "characters truncated" in body
    assert len(body) < 65_536
