# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The recorded live adversary rounds, read offline.

``<evidence_root>/validate/e_evidence_round-<ts>/evidence-776baa368c39/`` holds eighteen rounds of the file task
(``tests/validate/conftest.py``) against GLM-5.3 under the retired role protocol (shortcut, leak and ambiguity
roles, one final reply graded once). Their attempt files carry no submission record, so only the one transcript
regex the submission protocol keeps, the input-read rule (``consumed_inputs``), is pinned on them. Their
``calibration.json`` files hold the retired summary shape and are never rewritten.

``g_adversary_round-<ts>/`` and ``h_built_adversary_round-<ts>/`` hold rounds under the submission protocol
(``test_evidence_live``); their invariants are checked here. ``evidence_root`` is the fixture in ``tests/conftest.py``
and evidence stays on the machine that recorded it, so each test skips where its rounds are absent.
"""

import json
from pathlib import Path

import pydantic
import pytest
from taskcompendium.submission import PlainText

from taskforge.validate.adversary import AdversaryRole, adversary_brief
from taskforge.validate.attempts import load_adversary_attempt, load_outcome, trial_files
from taskforge.validate.calibration import (
    DefectTier,
    consumed_inputs,
    load_summary,
    shell_commands,
    solved,
    summarize,
)
from taskforge.validate.outcome import Graded, TrialKind
from taskforge.validate.run import load_validation
from taskforge.validate.submissions import passing
from taskforge.validate.trials import task_digest

ROUND_EVIDENCE = "evidence-776baa368c39"
PLAIN = PlainText(id="plain")
NUMBERS = "/workspace/numbers.txt"
LIVE_SUBMISSIONS = 10

OLD_ROUNDS = (
    "20261006T213322Z",
    "20261006T213459Z",
    "20261006T213512Z",
    "20261006T213552Z",
    "20261006T213610Z",
    "20261006T213621Z",
    "20261006T213708Z",
    "20261006T213730Z",
    "20261006T213745Z",
    "20261006T213917Z",
    "20261006T214000Z",
    "20261006T214011Z",
    "20261006T214418Z",
    "20261006T231003Z",
    "20261007T001329Z",
    "20261007T183819Z",
    "20261007T184028Z",
    "20261007T184144Z",
)
"""Fifteen rounds whose preambles let the roles solve, then three whose preambles forbade it."""

NEVER_READ_THE_INPUT = {
    ("20261006T213745Z", 1),
    ("20261006T231003Z", 1),
    ("20261007T001329Z", 0),
    ("20261007T183819Z", 0),
    ("20261007T183819Z", 1),
    ("20261007T184028Z", 0),
    ("20261007T184144Z", 0),
    ("20261007T184144Z", 1),
}
"""Shortcut trials (round, index) whose shell commands never read ``numbers.txt`` for content; none passed."""


@pytest.fixture
def evidence_dir(evidence_root: Path) -> Path:
    return evidence_root / "validate"


@pytest.fixture
def old_round_dirs(evidence_dir: Path) -> dict[str, Path]:
    dirs = {ts: evidence_dir / f"e_evidence_round-{ts}" / ROUND_EVIDENCE for ts in OLD_ROUNDS}
    if not all(path.is_dir() for path in dirs.values()):
        pytest.skip("recorded adversary rounds not on this machine")
    return dirs


@pytest.fixture
def submission_round_dirs(evidence_dir: Path) -> list[Path]:
    dirs = sorted(evidence_dir.glob("g_adversary_round-*/evidence-*"))
    if not dirs:
        pytest.skip("no adversary round under the submission protocol on this machine")
    return dirs


def test_consumed_inputs_on_recorded_rounds(old_round_dirs):
    never_read = set()
    for ts, directory in old_round_dirs.items():
        for name, files in trial_files(directory, TrialKind.ADVERSARY).items():
            role, index = name.split("/")
            if role != AdversaryRole.SHORTCUT:
                continue
            assert files.last_path is not None
            outcome = load_outcome(files.last_path)
            assert outcome.rollout is not None, (ts, name)
            if not consumed_inputs(shell_commands(outcome.rollout), (NUMBERS,)):
                never_read.add((ts, int(index)))
                assert isinstance(outcome, Graded) and not solved(outcome), (ts, name)

    assert never_read == NEVER_READ_THE_INPUT


def test_the_recorded_summaries_held_the_old_findings_and_stay_untouched(old_round_dirs):
    """The ``calibration.json`` each round wrote names the retired roles and parses as no summary today; its attempt
    files carry no submission record."""
    for ts, path in old_round_dirs.items():
        recorded = json.loads((path / "calibration.json").read_text())
        assert {"leak", "ambiguity"} <= set(recorded["roles"]), ts
        with pytest.raises(pydantic.ValidationError):
            load_summary(path / "calibration.json")
        last = trial_files(path, TrialKind.ADVERSARY)["shortcut/0"].last_path
        assert last is not None
        with pytest.raises(ValueError, match="not an adversary attempt"):
            load_adversary_attempt(last)


def test_recorded_submission_rounds_keep_their_invariants(submission_round_dirs, file_task, file_controls, rounds):
    draft = rounds.draft(file_task, file_controls, PLAIN)
    policy = rounds.policy(k=3, adversary_k=2, adversary_submissions=LIVE_SUBMISSIONS, adversary_repair_submissions=3)
    for directory in submission_round_dirs:
        assert directory.name == f"evidence-{task_digest(draft.task, draft.execution, draft.convention)[:12]}"
        evidence = load_validation(draft, directory)
        for trial in evidence.adversaries[AdversaryRole.SHORTCUT]:
            assert isinstance(trial.outcome, Graded), directory
            assert trial.system == adversary_brief(LIVE_SUBMISSIONS, "")
            assert len(trial.submissions) <= LIVE_SUBMISSIONS
            assert all(s.passed == passing(s.grade) for s in trial.submissions)
        summary = summarize(evidence, policy)
        # The file task's grader compares against a constant, so no accepted submission is a grader defect.
        assert all(a.tier is not DefectTier.REPAIR for a in summary.assessments), directory
