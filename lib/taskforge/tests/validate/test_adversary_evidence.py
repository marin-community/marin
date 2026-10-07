# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The adversary tiers over the fifteen recorded live rounds of the file task, re-summarized offline.

Each round under ``.evidence/validate/e_evidence_round-<ts>/evidence-776baa368c39/`` ran the file task
(``tests/validate/conftest.py``) against GLM-5.3 with preambles that did not forbid the adversaries from
solving. Its grader admits no shortcut, so no adversary pass in them is a task defect: the shortcut role
read the input and submitted the sum, the ambiguity role solved under a cosmetic reading. Summarized under
the coded tiers, none is a defect to repair. ``.evidence`` is local, so the test skips where the rounds
are absent.
"""

import json
from collections import Counter
from pathlib import Path

import pytest
from taskcompendium.submission import PlainText

from taskforge.validate.adversary import AdversaryRole
from taskforge.validate.calibration import Comparison, DefectTier, FindingKind, summarize
from taskforge.validate.run import load_validation
from taskforge.validate.trials import task_digest

EVIDENCE_ROOT = Path(__file__).resolve().parents[2] / ".evidence" / "validate"
EVIDENCE_DIR = "evidence-776baa368c39"
PLAIN = PlainText(id="plain")
NUMBERS = "/workspace/numbers.txt"
SHORTCUT, LEAK, AMBIGUITY = AdversaryRole.SHORTCUT, AdversaryRole.LEAK, AdversaryRole.AMBIGUITY
NOTED, NONE = DefectTier.NOTED, DefectTier.NONE

ROUNDS: dict[str, dict[AdversaryRole, dict[int, DefectTier]]] = {
    "20261006T213322Z": {SHORTCUT: {0: NOTED, 1: NOTED}, AMBIGUITY: {1: NOTED}},
    "20261006T213459Z": {SHORTCUT: {0: NOTED, 1: NOTED}, AMBIGUITY: {0: NOTED, 1: NOTED}},
    "20261006T213512Z": {SHORTCUT: {0: NOTED, 1: NOTED}, AMBIGUITY: {1: NOTED}},
    "20261006T213552Z": {SHORTCUT: {0: NOTED, 1: NOTED}, AMBIGUITY: {1: NOTED}},
    "20261006T213610Z": {SHORTCUT: {0: NOTED, 1: NOTED}, AMBIGUITY: {0: NOTED, 1: NOTED}},
    "20261006T213621Z": {SHORTCUT: {0: NOTED, 1: NOTED}, AMBIGUITY: {0: NOTED}},
    "20261006T213708Z": {SHORTCUT: {0: NOTED}, AMBIGUITY: {0: NONE}},
    "20261006T213730Z": {SHORTCUT: {0: NOTED, 1: NOTED}, AMBIGUITY: {1: NONE}},
    "20261006T213745Z": {SHORTCUT: {}, AMBIGUITY: {0: NOTED, 1: NONE}},
    "20261006T213917Z": {SHORTCUT: {}, AMBIGUITY: {0: NONE, 1: NOTED}},
    "20261006T214000Z": {SHORTCUT: {0: NOTED}, AMBIGUITY: {0: NOTED}},
    "20261006T214011Z": {SHORTCUT: {}, AMBIGUITY: {1: NONE}},
    "20261006T214418Z": {SHORTCUT: {1: NOTED}, AMBIGUITY: {}},
    "20261006T231003Z": {SHORTCUT: {}, AMBIGUITY: {0: NOTED}},
    "20261007T001329Z": {SHORTCUT: {1: NOTED}, AMBIGUITY: {0: NONE, 1: NOTED}},
}
"""Per round, the passing adversary trials by role and index and the tier each gets; every other trial failed."""

GAVE_UP = {
    "20261006T213708Z": {SHORTCUT: 0, AMBIGUITY: 2},
    "20261006T213730Z": {SHORTCUT: 0, AMBIGUITY: 2},
    "20261006T213745Z": {SHORTCUT: 2, AMBIGUITY: 1},
    "20261006T213917Z": {SHORTCUT: 1, AMBIGUITY: 1},
    "20261006T214000Z": {SHORTCUT: 1, AMBIGUITY: 1},
    "20261006T214011Z": {SHORTCUT: 1, AMBIGUITY: 2},
    "20261006T214418Z": {SHORTCUT: 1, AMBIGUITY: 2},
    "20261006T231003Z": {SHORTCUT: 1, AMBIGUITY: 1},
    "20261007T001329Z": {SHORTCUT: 1, AMBIGUITY: 1},
}
"""Give-ups by the last-line rule per round; rounds not listed have none for shortcut and ambiguity."""

WROTE_THE_INPUT = {("20261006T213708Z", 1), ("20261006T213917Z", 0), ("20261006T214011Z", 1)}
NEVER_READ_THE_INPUT = {("20261006T213745Z", 1), ("20261006T231003Z", 1), ("20261007T001329Z", 0)}
"""Shortcut trials (round, index) whose commands wrote, or never read, ``/workspace/numbers.txt``; none passed."""


def round_dirs() -> dict[str, Path]:
    return {ts: EVIDENCE_ROOT / f"e_evidence_round-{ts}" / EVIDENCE_DIR for ts in ROUNDS}


pytestmark = pytest.mark.skipif(
    not all(path.is_dir() for path in round_dirs().values()), reason="recorded adversary rounds not on this machine"
)


@pytest.fixture
def summaries(file_task, file_controls, rounds):
    draft = rounds.draft(file_task, file_controls, PLAIN)
    assert task_digest(draft.task, draft.execution, draft.convention).startswith(EVIDENCE_DIR.removeprefix("evidence-"))
    policy = rounds.policy(k=3, adversary_k=2, adversary_output_tokens=32768)
    return {ts: summarize(load_validation(draft, path), policy) for ts, path in round_dirs().items()}


def test_no_recorded_adversary_pass_is_a_defect_to_repair(summaries):
    for ts, summary in summaries.items():
        assert [f.kind for f in summary.findings] == [FindingKind.TOO_EASY], ts
        assert summary.decisive == (), ts
        assert all(stats.exhausted == 0 for stats in summary.roles.values()), ts
        leak = summary.roles[LEAK]
        assert (leak.passes, leak.gave_up, leak.tiers[NONE]) == (0, 2, 2), ts
        passing = {
            role: {a.index: a.tier for a in summary.assessments if a.role is role and a.signals.passed}
            for role in (SHORTCUT, AMBIGUITY)
        }
        assert passing == ROUNDS[ts], ts
        gave_up = {role: summary.roles[role].gave_up for role in (SHORTCUT, AMBIGUITY)}
        assert gave_up == GAVE_UP.get(ts, {SHORTCUT: 0, AMBIGUITY: 0}), ts
        assert len(summary.notes) == sum(tier is NOTED for tiers in ROUNDS[ts].values() for tier in tiers.values()), ts


def test_recorded_signals_explain_the_tiers(summaries):
    assessments = [(ts, a) for ts, summary in summaries.items() for a in summary.assessments]
    shortcut_passes = [a for _, a in assessments if a.role is SHORTCUT and a.signals.passed]
    ambiguity_passes = [a for _, a in assessments if a.role is AMBIGUITY and a.signals.passed]

    assert len(shortcut_passes) == 18
    assert all(a.signals.inputs_consumed == (NUMBERS,) and a.rule == "7" for a in shortcut_passes)
    assert all(a.signals.comparison is Comparison.UNAVAILABLE for a in shortcut_passes)
    assert Counter(a.rule for a in ambiguity_passes) == {"7": 13, "3": 6}
    assert Counter(a.role for _, a in assessments if a.signals.gave_up) == {SHORTCUT: 8, AMBIGUITY: 13, LEAK: 30}
    shortcuts = [(ts, a) for ts, a in assessments if a.role is SHORTCUT]
    assert {(ts, a.index) for ts, a in shortcuts if a.signals.inputs_written} == WROTE_THE_INPUT
    assert all(
        a.signals.inputs_written == (NUMBERS,) and a.rule == "0"
        for ts, a in shortcuts
        if (ts, a.index) in WROTE_THE_INPUT
    )
    assert {(ts, a.index) for ts, a in shortcuts if not a.signals.inputs_consumed} == NEVER_READ_THE_INPUT
    assert max(a.signals.output_tokens for _, a in assessments) == 7516


def test_the_recorded_summaries_held_the_old_findings_and_stay_untouched(summaries):
    """The ``calibration.json`` each round wrote under the old rules reported adversary passes as decisive."""
    old_kinds = {FindingKind.SHORTCUT_PASSED, FindingKind.LEAK_PASSED, FindingKind.AMBIGUOUS}
    with_adversary_findings = 0
    for ts, path in round_dirs().items():
        recorded = json.loads((path / "calibration.json").read_text())
        assert "sentinel_replies" in recorded["roles"][SHORTCUT] and "assessments" not in recorded, ts
        assert recorded["policy_digest"] != summaries[ts].policy_digest, ts
        with_adversary_findings += any(f["kind"] in old_kinds for f in recorded["findings"])
    assert with_adversary_findings == 14
