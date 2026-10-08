# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attempt files are the evidence: read them back, and tell which trials are settled.

``trials.run_trial`` writes one ``<evidence_dir>/<kind>/<trial>/attempt-<n>.json`` per attempt
(``trials.outcome_json``). ``load_outcome`` is its inverse, so validation can resume per trial and
review can run offline on the files alone. A trial is settled when its last attempt is graded, or
ungraded for a cause a fresh attempt cannot change; a re-entered validation re-runs only the
unsettled trials, numbering their attempts after the ones on disk.

An adversary attempt file is the same record with one more key, ``ADVERSARY_KEY``: the system turn the
attempt ran under, its parsed verdict and every verifier submission (``adversary_attempt_json``).
``load_outcome`` ignores that key, so ``trial_files`` reads adversary directories unchanged;
``load_adversary_attempt`` reads all of it.

An adversary attempt file keeps the engine's rollout envelope: the adversary brief is ``messages[0]`` and the
full agent loop, every tool call and result, is in ``steps[-1].messages``.
"""

import dataclasses
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter
from rolloutengine.contracts import RolloutData

from taskforge.validate.outcome import RETRYABLE, Cause, Graded, Outcome, Ungraded
from taskforge.validate.submissions import SUBMISSIONS, AdversaryTrial, Submission, trial_claim
from taskforge.validate.trials import outcome_json

RERUNNABLE: frozenset[Cause] = RETRYABLE | {Cause.TOKEN_CONTRACT}
"""Causes a re-entered validation runs again: infrastructure flakiness and the transport's sampling noise."""

ATTEMPT_FILE = re.compile(r"attempt-(\d+)\.json")
ROLLOUT = TypeAdapter(RolloutData)
ADVERSARY_KEY = "adversary"


def load_outcome(path: Path) -> Outcome:
    """The inverse of ``trials.outcome_json``: ``Graded(rollout)`` or ``Ungraded(cause, detail, rollout)``."""
    record = json.loads(path.read_bytes())
    rollout = None if record["rollout"] is None else ROLLOUT.validate_python(record["rollout"])
    if record["outcome"] == "ungraded":
        return Ungraded(Cause(record["cause"]), record["detail"], rollout)
    if rollout is None:
        raise ValueError(f"{path} records a graded attempt without its rollout")
    return Graded(rollout)


@dataclass(frozen=True)
class TrialFiles:
    """The attempt files of one trial: how many there are (the next attempt number), the last outcome and its file."""

    trial: str
    attempts: int
    last: Outcome | None
    last_path: Path | None

    @property
    def settled(self) -> bool:
        """Graded, or Ungraded for a cause outside RERUNNABLE: re-running cannot change it."""
        return isinstance(self.last, Graded) or (self.last is not None and self.last.cause not in RERUNNABLE)


def trial_files(evidence_dir: Path, kind: str) -> dict[str, TrialFiles]:
    """Every trial directory under ``<evidence_dir>/<kind>/`` and its last attempt, keyed by trial name.

    A trial name may hold a slash (``shortcut/1``); its directory is then nested.
    """
    root = evidence_dir / kind
    numbers: dict[Path, list[int]] = {}
    for path in root.rglob("attempt-*.json"):
        match = ATTEMPT_FILE.fullmatch(path.name)
        if match is not None:
            numbers.setdefault(path.parent, []).append(int(match.group(1)))
    trials = {}
    for directory, attempts in numbers.items():
        last = max(attempts)
        trial = directory.relative_to(root).as_posix()
        path = directory / f"attempt-{last}.json"
        trials[trial] = TrialFiles(trial, last + 1, load_outcome(path), path)
    return dict(sorted(trials.items()))


def adversary_attempt_json(outcome: Outcome, submissions: Sequence[Submission], system: str) -> bytes:
    """``outcome_json(outcome)`` with ``ADVERSARY_KEY``: ``system`` (the exact system text of every request of the
    attempt), ``claim`` (``trial_claim(outcome)``) and ``submissions`` in order, each grade in full."""
    record = json.loads(outcome_json(outcome))
    record[ADVERSARY_KEY] = {
        "system": system,
        "claim": dataclasses.asdict(trial_claim(outcome)),
        "submissions": SUBMISSIONS.dump_python(tuple(submissions), mode="json"),
    }
    return json.dumps(record, indent=1).encode()


def load_adversary_attempt(path: Path) -> AdversaryTrial:
    """The inverse of ``adversary_attempt_json``.

    Raises:
        ValueError: the record has no ``ADVERSARY_KEY``: a solver or control file read as an adversary's.
    """
    record = json.loads(path.read_bytes())
    adversary = record.get(ADVERSARY_KEY)
    if adversary is None:
        raise ValueError(f"{path} is not an adversary attempt: it has no {ADVERSARY_KEY!r} record")
    return AdversaryTrial(
        outcome=load_outcome(path),
        system=adversary["system"],
        submissions=SUBMISSIONS.validate_python(adversary["submissions"]),
    )
