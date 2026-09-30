"""A relaunch under a new run name must keep quality acceptances (2026-09-29).

Construction jobs are relaunched under a new run name; the worker restores the
prior durable snapshot into ``/tmp/capability-pipeline/<new-run>/results`` and
resumes.  Before this fix ``synthesize_one`` treated ``quality_accepted`` as a
non-terminal state and re-ran every gate plus semantic review, so accepted
items were demoted by flaky runtime controls or a second review.
"""

import json
import shutil
from pathlib import Path

import pytest

from capability_pipeline import acceptance, quality, synthesis
from capability_pipeline.inference import atomic_json


def _item():
    return {
        "proposal_hash": "b" * 64,
        "proposal": {"capability_id": "d18.nursing.bedside.infection", "slot": 5},
    }


KEY = "d18.nursing.bedside.infection:5"
NAME = f"d18.nursing.bedside.infection-5-{'b' * 12}"


def _results(tmp_path: Path, run_name: str) -> Path:
    return tmp_path / "capability-pipeline" / run_name / "results"


def _accept(root: Path) -> dict:
    """Build an item the way _synthesize_attempt leaves it after acceptance."""
    item = _item()
    item_root = root / "items" / NAME
    atomic_json(item_root / "contract/accepted.json", item)
    (item_root / "contract/task_contract.md").write_text("contract\n")
    task = item_root / "workspace/task"
    task.mkdir(parents=True)
    (task / "specification.json").write_text('{"steps": [{"prompt": "reconcile"}]}\n')
    (task / "controls.json").write_text('{"schema_version": "1", "cases": []}\n')
    (item_root / "workspace/notes.md").write_text("builder scratch\n")
    (item_root / "harbor/tests").mkdir(parents=True)
    (item_root / "harbor/task.toml").write_text("name = 'x'\n")
    (item_root / "harbor/tests/test.sh").write_text("exit 0\n")
    (item_root / "runtime-evidence.json").write_text('{"passed": true}\n')
    review_root = root / "quality" / NAME / "attempt-1"
    manifest = quality.prepare_packet(item_root, review_root)
    atomic_json(
        review_root / "result.json",
        {"state": "accept", "snapshot_hash": manifest["snapshot_hash"], "issues": []},
    )
    export = root / "validated" / NAME
    shutil.copytree(item_root / "harbor", export)
    result = {
        "key": KEY,
        "proposal_hash": item["proposal_hash"],
        "sessions": [{"session": "build", "status": "complete"}],
        "state": "quality_accepted",
        "issues": [],
        "item_root": str(item_root),
        "runtime_evidence": str(item_root / "runtime-evidence.json"),
        "runtime_validated": True,
        "taskcompendium": {"id": "task-x"},
        "quality_review": {
            "state": "accept",
            "artifact": str(review_root / "result.json"),
            "artifact_sha256": quality.sha256(review_root / "result.json"),
            "snapshot_hash": manifest["snapshot_hash"],
        },
        "export": str(export),
    }
    acceptance.record_acceptance(item, KEY, root, item_root, result)
    atomic_json(item_root / "status.json", result)
    return result


def _relaunch(old: Path, new: Path) -> None:
    """What sync_results restore does: same relative files, a new work root."""
    shutil.copytree(old, new)
    shutil.rmtree(new / "validated")  # synthesize() wipes validated/ at start


def _forbid_rebuild(monkeypatch) -> list:
    calls = []

    def attempt(*_args, **_kwargs):
        calls.append("attempt")
        return {"state": "failed", "issues": ["runtime controls failed: timeout"]}

    def review(*_args, **_kwargs):
        calls.append("review")
        return {"state": "reject"}

    monkeypatch.setattr(synthesis, "_synthesize_attempt", attempt)
    monkeypatch.setattr(quality, "run_review", review)
    monkeypatch.setenv("CAPABILITY_MAX_REPAIR_ROUNDS", "0")
    return calls


def _run(root: Path, **kwargs) -> dict:
    return synthesis.synthesize_one(_item(), root, object(), None, None, 60, **kwargs)


def test_fresh_acceptance_records_content_identity(tmp_path):
    root = _results(tmp_path, "cap-construct-003-hc1-d2")
    result = _accept(root)
    record = result["acceptance"]
    assert record["review_dir"] == f"quality/{NAME}/attempt-1"
    assert record["task_artifact_count"] == 5  # accepted.json, 2 task, 2 harbor
    assert "cap-construct-003" not in json.dumps(record)
    ledger = json.loads(acceptance.ledger_path(root, root / "items" / NAME).read_text())
    assert ledger["record"] == record


@pytest.mark.parametrize("legacy", [False, True])
def test_relaunch_under_new_run_name_keeps_acceptance_without_rereview(
    tmp_path, monkeypatch, legacy
):
    old = _results(tmp_path, "cap-construct-003-hc1-d2")
    accepted = _accept(old)
    if legacy:
        # Accepted before this fix: no acceptance record, no ledger.
        accepted.pop("acceptance")
        atomic_json(old / "items" / NAME / "status.json", accepted)
        shutil.rmtree(old / "acceptances")
    new = _results(tmp_path, "cap-construct-003-hc1-e1")
    _relaunch(old, new)
    calls = _forbid_rebuild(monkeypatch)

    result = _run(new)

    assert calls == []
    assert result["state"] == "quality_accepted"
    item_root = new / "items" / NAME
    assert result["item_root"] == str(item_root)
    assert result["runtime_evidence"] == str(item_root / "runtime-evidence.json")
    assert result["quality_review"]["artifact"] == str(
        new / "quality" / NAME / "attempt-1" / "result.json"
    )
    assert not (new / "quality" / NAME / "attempt-2").exists()
    assert Path(result["export"]) == new / "validated" / NAME
    assert (new / "validated" / NAME / "tests/test.sh").read_text() == "exit 0\n"
    assert result["acceptance_resumed"]["prior_root"] == str(old)
    assert json.loads((item_root / "status.json").read_text()) == result
    assert "cap-construct-003-hc1-d2" not in json.dumps(
        {k: v for k, v in result.items() if k != "acceptance_resumed"}
    )
    # A second relaunch is equally stable.
    newer = _results(tmp_path, "cap-construct-003-hc1-f1")
    _relaunch(new, newer)
    assert _run(newer)["state"] == "quality_accepted"
    assert calls == []


@pytest.mark.parametrize(
    "path",
    ["workspace/task/specification.json", "harbor/tests/test.sh", "harbor/new.txt"],
)
def test_changed_task_content_is_rereviewed(tmp_path, monkeypatch, path):
    old = _results(tmp_path, "cap-construct-003-hc1-d2")
    _accept(old)
    new = _results(tmp_path, "cap-construct-003-hc1-e1")
    _relaunch(old, new)
    (new / "items" / NAME / path).write_text("edited\n")
    calls = _forbid_rebuild(monkeypatch)

    result = _run(new)

    assert calls == ["attempt"]
    assert result["state"] == "failed"


def test_non_task_evidence_changes_do_not_reopen_acceptance(tmp_path, monkeypatch):
    old = _results(tmp_path, "cap-construct-003-hc1-d2")
    _accept(old)
    new = _results(tmp_path, "cap-construct-003-hc1-e1")
    _relaunch(old, new)
    (new / "items" / NAME / "workspace/notes.md").write_text("more scratch\n")
    (new / "items" / NAME / "contract/task_contract.md").write_text("newer docs\n")
    calls = _forbid_rebuild(monkeypatch)
    assert _run(new)["state"] == "quality_accepted"
    assert calls == []


@pytest.mark.parametrize("tamper", ["verdict", "admitted", "proposal_hash"])
def test_unverifiable_acceptance_fails_closed(tmp_path, monkeypatch, tamper):
    old = _results(tmp_path, "cap-construct-003-hc1-d2")
    accepted = _accept(old)
    new = _results(tmp_path, "cap-construct-003-hc1-e1")
    _relaunch(old, new)
    shutil.rmtree(new / "acceptances")
    item_root = new / "items" / NAME
    if tamper == "verdict":
        result_path = new / "quality" / NAME / "attempt-1" / "result.json"
        verdict = json.loads(result_path.read_text())
        atomic_json(result_path, {**verdict, "state": "reject"})
    elif tamper == "admitted":
        atomic_json(item_root / "contract/accepted.json", {**_item(), "extra": 1})
    else:
        atomic_json(item_root / "status.json", {**accepted, "proposal_hash": "c" * 64})
    calls = _forbid_rebuild(monkeypatch)
    assert _run(new)["state"] == "failed"
    assert calls == ["attempt"]


def test_ledger_restores_acceptance_after_status_was_overwritten(tmp_path, monkeypatch):
    old = _results(tmp_path, "cap-construct-003-hc1-d2")
    _accept(old)
    # A pre-fix relaunch re-ran the gates and overwrote status.json with a flake.
    atomic_json(
        old / "items" / NAME / "status.json",
        {"key": KEY, "proposal_hash": "b" * 64, "state": "failed",
         "issues": ["runtime controls failed: timed out after 14400 seconds"]},
    )
    new = _results(tmp_path, "cap-construct-003-hc1-e1")
    _relaunch(old, new)
    calls = _forbid_rebuild(monkeypatch)
    result = _run(new)
    assert calls == []
    assert result["state"] == "quality_accepted"
    assert result["acceptance_resumed"]["source"] == "ledger"


def test_operator_seed_recovers_demoted_item_only_when_content_matches(
    tmp_path, monkeypatch
):
    old = _results(tmp_path, "cap-construct-003-hc1-d2")
    pulled = _accept(old)
    pulled.pop("acceptance")  # a pull made before this fix
    shutil.rmtree(old / "acceptances")
    atomic_json(
        old / "items" / NAME / "status.json",
        {"key": KEY, "proposal_hash": "b" * 64, "state": "failed", "issues": ["flake"]},
    )
    seed_dir = tmp_path / "done" / NAME / "items" / NAME
    atomic_json(seed_dir / "status.json", pulled)
    seeds = acceptance.load_seeds(tmp_path / "done")
    assert list(seeds) == [KEY]

    new = _results(tmp_path, "cap-construct-003-hc1-e1")
    _relaunch(old, new)
    calls = _forbid_rebuild(monkeypatch)
    result = _run(new, acceptance_seeds=seeds[KEY])
    assert calls == []
    assert result["state"] == "quality_accepted"
    assert result["acceptance_resumed"]["source"] == "seed"

    changed = _results(tmp_path, "cap-construct-003-hc1-e2")
    _relaunch(old, changed)
    (changed / "items" / NAME / "harbor/task.toml").write_text("name = 'y'\n")
    # The retained (demoted) status stands; the seed has no effect.
    rejected = _run(changed, acceptance_seeds=seeds[KEY])
    assert rejected["state"] == "failed"
    assert "acceptance_resumed" not in rejected
    assert not (changed / "acceptances").exists()


def _gate(root: Path, attempt: str = "attempt-1", state: str = "ready") -> None:
    atomic_json(root / "quality" / NAME / attempt / acceptance.GATE_RECEIPT,
                {"repeated_diagnostics_state": state})


def _demote(root: Path, issues: list[str], gate: bool = True) -> None:
    """A pre-ledger job re-ran the gates and overwrote the accepted status."""
    if gate:
        for attempt in (root / "quality" / NAME).glob("attempt-*"):
            _gate(root, attempt.name)
    shutil.rmtree(root / "acceptances")
    atomic_json(
        root / "items" / NAME / "status.json",
        {"key": KEY, "proposal_hash": "b" * 64, "state": "failed", "issues": issues,
         "taskcompendium": {"id": "task-x"}, "item_root": str(root / "items" / NAME)},
    )


def test_accepting_review_record_recovers_demoted_item_without_ledger_or_seed(
    tmp_path, monkeypatch
):
    old = _results(tmp_path, "cap-construct-003-shard-071-c6")
    _accept(old)
    _demote(old, ["KC2: reward is below reward_min"])
    new = _results(tmp_path, "cap-construct-003-shard-071-k3")
    _relaunch(old, new)
    calls = _forbid_rebuild(monkeypatch)

    result = _run(new)

    assert calls == []
    assert result["state"] == "quality_accepted"
    assert result["issues"] == []
    assert result["acceptance_resumed"]["source"] == "quality_review"
    assert result["acceptance_resumed"]["gate_proof"] == acceptance.GATE_PROOF_RECEIPT
    assert result["gate_proof"] == acceptance.GATE_PROOF_RECEIPT
    assert result["acceptance"]["review_dir"] == f"quality/{NAME}/attempt-1"
    assert result["superseded_status"] == {
        "state": "failed", "issues": ["KC2: reward is below reward_min"]}
    assert result["taskcompendium"] == {"id": "task-x"}
    assert Path(result["export"]) == new / "validated" / NAME
    # Recorded in the ledger, so the next relaunch resumes from it directly.
    newer = _results(tmp_path, "cap-construct-003-shard-071-k4")
    _relaunch(new, newer)
    again = _run(newer)
    assert again["state"] == "quality_accepted"
    assert again["acceptance_resumed"]["source"] == "status"
    assert calls == []


def test_newest_accepting_review_is_preferred(tmp_path, monkeypatch):
    old = _results(tmp_path, "cap-construct-003-shard-071-c6")
    _accept(old)
    first = old / "quality" / NAME / "attempt-1"
    shutil.copytree(first, old / "quality" / NAME / "attempt-2")
    _demote(old, ["flake"])
    new = _results(tmp_path, "cap-construct-003-shard-071-k3")
    _relaunch(old, new)
    _forbid_rebuild(monkeypatch)
    result = _run(new)
    assert result["acceptance"]["review_dir"] == f"quality/{NAME}/attempt-2"


@pytest.mark.parametrize("verdict", ["repair", "reject"])
def test_non_accepting_review_record_recovers_nothing(tmp_path, monkeypatch, verdict):
    old = _results(tmp_path, "cap-construct-003-shard-071-c6")
    _accept(old)
    result_path = old / "quality" / NAME / "attempt-1" / "result.json"
    atomic_json(result_path, {**json.loads(result_path.read_text()), "state": verdict})
    _demote(old, ["flake"])
    new = _results(tmp_path, "cap-construct-003-shard-071-k3")
    _relaunch(old, new)
    _forbid_rebuild(monkeypatch)
    # The retained (demoted) status stands.
    result = _run(new)
    assert result["state"] == "failed"
    assert "acceptance_resumed" not in result


def test_review_record_does_not_cover_task_edited_after_review(tmp_path, monkeypatch):
    old = _results(tmp_path, "cap-construct-003-shard-071-c6")
    _accept(old)
    _demote(old, ["flake"])
    (old / "items" / NAME / "workspace/task/specification.json").write_text("{}\n")
    new = _results(tmp_path, "cap-construct-003-shard-071-k3")
    _relaunch(old, new)
    _forbid_rebuild(monkeypatch)
    result = _run(new)
    assert result["state"] == "failed"
    assert "acceptance_resumed" not in result
    assert not (new / "acceptances").exists()


@pytest.mark.parametrize("gate", [None, "pending"])
def test_accepting_review_without_passed_gate_is_not_an_acceptance(
    tmp_path, monkeypatch, gate
):
    """'semantic review accepted despite failed repeated runtime gates' never accepted.

    With no receipt the legacy reconstruction decides; this fixture retains no
    repeated diagnostics, so the gate state is unknown and nothing is recovered
    (tests/test_legacy_acceptance_gate.py covers reconstructed verdicts).
    """
    old = _results(tmp_path, "cap-construct-003-shard-039-c6")
    _accept(old)
    _demote(old, ["semantic review accepted despite failed repeated runtime gates"], gate=False)
    if gate is not None:
        _gate(old, state=gate)
    new = _results(tmp_path, "cap-construct-003-shard-039-k3")
    _relaunch(old, new)
    _forbid_rebuild(monkeypatch)
    result = _run(new)
    assert result["state"] == "failed"
    assert "acceptance_resumed" not in result
