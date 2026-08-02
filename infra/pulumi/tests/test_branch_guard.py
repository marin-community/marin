# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iac.branch_guard import guard_pulumi_up

GUARDED = frozenset({"marin", "cw-rno2a"})


class Recorder:
    """Records prompts/warnings and returns a fixed confirm decision."""

    def __init__(self, decision):
        self.decision = decision
        self.prompts = []
        self.warnings = []

    def confirm(self, prompt):
        self.prompts.append(prompt)
        return self.decision

    def warn(self, message):
        self.warnings.append(message)


def _guard(rec, *, stack="marin", branch="feature", is_preview=False):
    guard_pulumi_up(
        stack=stack,
        branch=branch,
        is_preview=is_preview,
        main_only_stacks=GUARDED,
        confirm=rec.confirm,
        warn=rec.warn,
    )


def test_declining_aborts_the_update():
    rec = Recorder(decision=False)
    with pytest.raises(SystemExit):
        _guard(rec, stack="marin", branch="feature")
    assert len(rec.prompts) == 1


def test_confirming_proceeds():
    rec = Recorder(decision=True)
    _guard(rec, stack="marin", branch="feature")
    assert len(rec.prompts) == 1
    assert rec.warnings == []


def test_no_terminal_proceeds_with_warning():
    rec = Recorder(decision=None)
    _guard(rec, stack="marin", branch="feature")
    assert len(rec.warnings) == 1


def test_preview_never_prompts():
    rec = Recorder(decision=False)
    _guard(rec, stack="marin", branch="feature", is_preview=True)
    assert rec.prompts == []


def test_main_branch_never_prompts():
    rec = Recorder(decision=False)
    _guard(rec, stack="marin", branch="main")
    assert rec.prompts == []


def test_unguarded_stack_never_prompts():
    rec = Recorder(decision=False)
    _guard(rec, stack="marin-grafana", branch="feature")
    assert rec.prompts == []


def test_detached_head_is_guarded():
    rec = Recorder(decision=False)
    with pytest.raises(SystemExit):
        _guard(rec, stack="cw-rno2a", branch=None)
    assert "detached HEAD" in rec.prompts[0]
