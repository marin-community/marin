# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Policy overlays replace selected prompts and reject invalid field mappings.

Invariants: none (pins which overlay fields move the stage digest).
"""

import json

import pytest

from taskforge.defaults import DEFAULT_POLICY, DEFAULTS_DIR
from taskforge.policy import POLICY_FILE, load_policy


def test_an_overlay_replaces_only_the_fields_it_names_and_changes_the_digest(tmp_path):
    (tmp_path / "nudge.md").write_text("Call a tool; submit `{program}` when it builds.\n")
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps({"author.nudge": "nudge.md"}))

    policy = load_policy(DEFAULTS_DIR, overlay)

    assert policy.author.nudge.render(program="program.py") == "Call a tool; submit `program.py` when it builds."
    assert policy.author.system == DEFAULT_POLICY.author.system
    assert policy.proposer == DEFAULT_POLICY.proposer
    assert policy.stage_digest != DEFAULT_POLICY.stage_digest
    assert load_policy(DEFAULTS_DIR).stage_digest == DEFAULT_POLICY.stage_digest


def test_a_proposer_overlay_leaves_the_stage_digest_unchanged(tmp_path):
    (tmp_path / "plan.md").write_text("Plan {n} slots.\n")
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps({"proposer.plan": "plan.md"}))

    policy = load_policy(DEFAULTS_DIR, overlay)

    assert policy.proposer.plan.render(n=2) == "Plan 2 slots."
    assert policy.stage_digest == DEFAULT_POLICY.stage_digest


def test_a_policy_file_naming_an_unknown_field_or_missing_one_is_refused(tmp_path):
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps({"author.turns": "turns.md"}))
    with pytest.raises(ValueError, match=r"names no such policy field: author\.turns"):
        load_policy(DEFAULTS_DIR, overlay)

    defaults = json.loads((DEFAULTS_DIR / POLICY_FILE).read_text())
    del defaults["author.nudge"]
    (tmp_path / POLICY_FILE).write_text(json.dumps(defaults))
    with pytest.raises(ValueError, match=r"has no file for author\.nudge"):
        load_policy(tmp_path)
