"""Legacy post-review gate reconstruction (2026-09-29 acceptance recovery).

Pre-receipt controllers accepted an item only if ``repeated diagnostics state ==
"ready"`` when the semantic review ran; a review can run on merely
``reviewable`` diagnostics ("semantic review accepted despite failed repeated
runtime gates", e.g. shard-039 counting-processes-1).  These fixtures lay out
an item exactly as the old controller did (file names and JSON shapes copied
from run-003 snapshots) and compute every identity with the old modules' own
functions, then freeze the review snapshot with ``quality.prepare_packet``.
"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from capability_pipeline import (
    acceptance,
    diagnostics,
    evaluation,
    grading_diagnostics,
    non_docker_reset,
    quality,
    reset_runner,
    synthesis,
)
from capability_pipeline.inference import atomic_json
from capability_pipeline.legacy_acceptance_gate import (
    NOT_READY,
    READY,
    UNKNOWN,
    reconstruct,
)

TIMEOUT = 14400
PROPOSAL_HASH = "c" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _item(environment: str, verification: str) -> dict:
    return {
        "proposal_hash": PROPOSAL_HASH,
        "proposal": {
            "capability_id": "d01.uncertainty.probability.counting-processes",
            "slot": 1,
            "verification": verification,
            "environment": environment,
        },
    }


def _name() -> str:
    return f"d01.uncertainty.probability.counting-processes-1-{PROPOSAL_HASH[:12]}"


@pytest.fixture
def stub_toolchain(monkeypatch):
    """The TaskCompendium lock identity is environment; stub it, keep its lock digest."""
    lock = _sha(synthesis.SOURCE_LOCK)

    def identity(_source):
        return {"source_lock_sha256": lock, "revision": "dc6b501c", "package_root": "src", "files": {}}

    monkeypatch.setattr(diagnostics, "_toolchain_identity", identity)
    return lock


class Item:
    """An item root plus the old controller's diagnostics evidence."""

    def __init__(self, root: Path, *, kind: str = "none", verification: str = "code",
                 composed: bool = False, supported: bool = True):
        self.root = root
        self.kind = kind
        self.item = _item("container" if kind == "docker" else "reasoning", verification)
        self.path = root / "items" / _name()
        p = self.path
        atomic_json(p / "contract/accepted.json", self.item)
        atomic_json(p / "harbor/binding.json", {"environment": {"kind": kind}, "tools": []})
        atomic_json(p / "harbor/manifest.json", {"step_names": ["step-1"]})
        (p / "harbor/task.toml").write_text("name = 'counting'\n")
        atomic_json(p / "harbor/specification.json", {"steps": [{"resources": []}]})
        atomic_json(p / "harbor/renderings.json", [{"prompt": "count"}])
        (p / "harbor/instruction.md").write_text("Count the arrivals.\n")
        if kind == "docker":
            (p / "harbor/environment").mkdir(parents=True)
            (p / "harbor/environment/Dockerfile").write_text("FROM python:3.12\n")
        verifier = {"kind": "tasktrove", "mode": "script",
                    "runtime": {"kind": "container", "image": "img@sha256:" + "d" * 64,
                                "supervisor_python": "/usr/bin/python3"}}
        steps = [{"verifier": verifier}] if supported else [{"verifier": verifier}, {"verifier": verifier}]
        atomic_json(p / "workspace/task/specification.json", {"steps": steps})
        atomic_json(p / "workspace/task/binding.json",
                    {"environment": {"kind": kind}, "tools": []})
        atomic_json(p / "workspace/task/controls.json", {"schema_version": "1", "cases": [
            {"id": "gold", "class": "positive", "response": "{\"p\": 0.25}",
             "expect": {"status": "graded", "reward_min": 1, "reward_max": 1}},
            {"id": "empty", "class": "malformed", "response": "", "expect": {"status": "extraction_error"}},
        ]})
        if composed:
            atomic_json(p / "workspace/task/composite-verifier.json", {"checks": []})
        self.helper = None
        if kind == "docker" or composed:
            self.helper = p / "workspace/tools/daytona/dt.py"
            self.helper.parent.mkdir(parents=True)
            self.helper.write_text("# pinned Daytona helper\n")
        if kind == "docker":
            atomic_json(p / "workspace/task/candidate-resources.json",
                        {"cpu": 2, "memory_gb": 4, "disk_gb": 10})
            atomic_json(p / "workspace/task/reset-policy.json", {"public_root": "/workspace"})
        self.bridge = None
        if kind == "shellsim":
            self.bridge = root / "shellsim-bridge.json"
            self.bridge.write_text('{"bridge": 1}\n')
        (p / "runtime-evidence.json").write_text('{"passed": true}\n')

    # -- the old controller's evidence -------------------------------------

    def controller(self, variant: str | None = None) -> dict:
        controller = json.loads(json.dumps(diagnostics._controller_identity()))
        if variant is not None:
            controller["evaluation_controller_files"]["capability_pipeline/synthesis.py"] = (
                hashlib.sha256(variant.encode()).hexdigest()
            )
        return controller

    def identity(self, *, timeout: int = TIMEOUT) -> dict:
        resources = self.path / "workspace/task/candidate-resources.json"
        invocation = diagnostics._invocation_identity(
            timeout, self.helper, self.bridge, resources if self.kind == "docker" else None,
        )
        invocation["primary_adversarial_gate"] = {
            "schema_version": "capability-primary-adversarial-gate-v1",
            "state": "passed",
            "runtime_evidence_sha256": _sha(self.path / "runtime-evidence.json"),
            "adjudication_result_sha256": None,
            "new_attacks_in_repeated_measurement": False,
        }
        return diagnostics._item_identity(self.path.resolve(), self.root, invocation)

    def repeated(self, attempt: str, *, passed: bool = True, valid: bool = True,
                 controller: dict | None = None, identity: dict | None = None,
                 result: dict | None = None, manifest: bool = True) -> Path:
        base = self.path / "diagnostics" / attempt
        (base / "evaluation").mkdir(parents=True)
        if manifest:
            atomic_json(base / "inputs/manifest.json", {
                "schema_version": "capability-runtime-evaluation-bundle-v1",
                "plan_sha256": hashlib.sha256(attempt.encode()).hexdigest(),
                "item_identity": identity if identity is not None else self.identity(),
                "controller": controller if controller is not None else self.controller(),
                "files": {}, "directories": ["bundle", "package"],
            })
        cells = [{"attempt": n, "state": "valid" if valid or n < 3 else "timeout",
                  "oracle_passed": True, "solver_passed": passed or n == 1,
                  "authored_controls_passed": True, "primary_adversarial_review_bound": True,
                  "sandbox_ids": [f"{attempt}-sb-{n}"]} for n in (1, 2, 3)]
        matrix = evaluation.aggregate({"attempts": 3, "purpose": "primary_bound_repeatability",
                                       "unassessed_recipe_rows": evaluation.UNASSESSED,
                                       "critical_control_ids": []}, cells)
        atomic_json(base / "evaluation/matrix.json", matrix)
        reviewable = matrix["complete_attempt_inventory"] and all(c["state"] == "valid" for c in cells)
        ready = reviewable and matrix["state"] == "repeated_runtime_passed"
        atomic_json(base / "result.json", result if result is not None else {
            "schema_version": "capability-repeated-diagnostics-result-v1",
            "state": "ready" if ready else "pending", "reused": False,
            "attempt": str(base), "evaluation_state": matrix["state"], "reviewable": reviewable,
            "unassessed_recipe_rows": evaluation.UNASSESSED,
            **({} if ready else {"issues": ["repeated runtime diagnostics are incomplete"]}),
        })
        return base

    def grading(self, attempt: str, state: str = "ready", *, controller: dict | None = None) -> None:
        base = self.path / "diagnostics" / attempt / "fixed-grading"
        files = (controller or self.controller())["evaluation_controller_files"]
        atomic_json(base / "binding.json", {
            "schema_version": grading_diagnostics.DIAGNOSTIC_SCHEMA,
            "parallelism": 8, "timeout_seconds": TIMEOUT,
            "controller": {**files, "capability_pipeline/grading_diagnostics.py":
                           files["capability_pipeline/grading_diagnostics.py"]},
        })
        atomic_json(base / "grading-diagnostics.json", {
            "state": state, "reviewable": state in {"ready", "semantic_failed"},
            "unassessed": False, "issues": [] if state == "ready" else [f"grading {state}"],
        })

    def composite(self, attempt: str, state: str = "ready") -> None:
        from capability_pipeline import composite_grading_diagnostics as composite

        repeated = self.path / "diagnostics" / attempt
        source = {
            "repeated_plan_sha256": json.loads((repeated / "inputs/manifest.json").read_text())["plan_sha256"],
            "repeated_matrix_sha256": _sha(repeated / "evaluation/matrix.json"),
            "current_controls_sha256": _sha(self.path / "workspace/task/controls.json"),
            "controls_sha256": _sha(self.path / "workspace/task/controls.json"),
            "current_harbor_manifest_sha256": _sha(self.path / "harbor/manifest.json"),
            "daytona_helper_sha256": _sha(self.helper),
        }
        base = self.path / "diagnostics/composite-grading" / f"attempt-{composite._digest(source)[:20]}"
        atomic_json(base / "binding.json", {
            "schema_version": composite.SCHEMA, "source": source, "controller": composite._controller(),
            "taskcompendium_source_lock_sha256": _sha(synthesis.SOURCE_LOCK),
            "timeout_seconds": TIMEOUT, "repeats": composite.REPEATS,
        })
        atomic_json(base / "summary.json", {"schema_version": composite.SCHEMA, "state": state, "issues": []})

    def reset(self, state: str | None = "ready", *, controller_variant: str | None = None) -> Path:
        if self.kind == "docker":
            source, details = reset_runner._source(self.path, None)
            binding = reset_runner._binding(source, details["resource_receipt"], TIMEOUT, None)
            if controller_variant is not None:
                binding["controller"]["capability_pipeline/synthesis.py"] = (
                    hashlib.sha256(controller_variant.encode()).hexdigest()
                )
            base = self.path / "diagnostics/reset" / f"attempt-{reset_runner._attempt_key(source)}"
            schema = reset_runner.SCHEMA
        else:
            identity, _ = non_docker_reset._source(self.path, self.bridge)
            key = hashlib.sha256(
                json.dumps(non_docker_reset._portable(identity), sort_keys=True).encode()
            ).hexdigest()[:20]
            binding = {"schema_version": non_docker_reset.SCHEMA, "source": identity,
                       "timeout_seconds": TIMEOUT}
            base = self.path / "diagnostics/non-docker-reset" / f"attempt-{key}"
            schema = non_docker_reset.SCHEMA
        atomic_json(base / "binding.json", binding)
        (base / "raw").mkdir()
        atomic_json(base / "raw/report.json", {"schema_version": "capability-reset-diagnostics-v1"})
        if state is not None:
            atomic_json(base / "summary.json", {
                "schema_version": schema, "state": state,
                "reviewable": state == "semantic_failed", "issues": [],
                "full_quality_reset_gate": "unassessed",
            })
        return base

    def review(self, attempt: str = "attempt-1", verdict: str = "accept") -> Path:
        review_root = self.root / "quality" / self.path.name / attempt
        manifest = quality.prepare_packet(self.path, review_root)
        atomic_json(review_root / "result.json", {
            "schema_version": "capability-quality-result-v1",
            "snapshot_hash": manifest["snapshot_hash"], "state": verdict, "issues": [],
        })
        return review_root


def _ready_item(tmp_path: Path, **kwargs) -> Item:
    item = Item(tmp_path / "results", **kwargs)
    item.repeated("attempt-1")
    if kwargs.get("supported", True) and kwargs.get("verification", "code") != "judge":
        item.grading("attempt-1")
    item.reset()
    return item


# ------------------------------------------------------------ decisions


@pytest.mark.parametrize("kind", ["none", "docker"])
def test_ready_diagnostics_reconstruct_ready(tmp_path, stub_toolchain, kind):
    item = _ready_item(tmp_path, kind=kind)
    result = reconstruct(item.review())
    assert result["verdict"] == READY, result
    (candidate,) = result["candidates"]
    assert candidate["attempt"] == "attempt-1"
    assert candidate["grading"]["state"] == "ready"
    assert candidate["reset"]["kind"] == ("docker_reset" if kind == "docker" else "non_docker_reset")


def test_shard039_shape_reviewable_but_not_ready_is_not_ready(tmp_path, stub_toolchain):
    """Evaluation needs_review with three valid cells: reviewed, but never accepted."""
    item = Item(tmp_path / "results")
    item.repeated("attempt-1", passed=False)
    item.grading("attempt-1")
    item.reset()
    result = reconstruct(item.review())
    assert result["verdict"] == NOT_READY, result
    assert "needs_review" in result["reason"]


@pytest.mark.parametrize("stage", ["grading", "reset"])
def test_semantic_failure_after_ready_repeat_is_not_ready(tmp_path, stub_toolchain, stage):
    item = Item(tmp_path / "results")
    item.repeated("attempt-1")
    item.grading("attempt-1", "semantic_failed" if stage == "grading" else "ready")
    item.reset("semantic_failed" if stage == "reset" else "ready")
    assert reconstruct(item.review())["verdict"] == NOT_READY


def test_docker_reset_without_policy_is_semantic_failed_not_ready(tmp_path, stub_toolchain):
    item = Item(tmp_path / "results", kind="docker")
    (item.path / "workspace/task/reset-policy.json").unlink()
    item.repeated("attempt-1")
    item.grading("attempt-1")
    result = reconstruct(item.review())
    assert result["verdict"] == NOT_READY
    assert result["candidates"][0]["reset"]["reason"] == "reset-policy.json is required"


def test_judge_without_composite_skips_fixed_grading(tmp_path, stub_toolchain):
    item = _ready_item(tmp_path, verification="judge")
    result = reconstruct(item.review())
    assert result["verdict"] == READY
    assert result["candidates"][0]["grading"]["kind"] == "judge_not_applicable"


@pytest.mark.parametrize("state,verdict", [("ready", READY), ("semantic_failed", NOT_READY)])
def test_composed_judge_uses_composite_replay(tmp_path, stub_toolchain, state, verdict):
    item = Item(tmp_path / "results", verification="judge", composed=True)
    item.repeated("attempt-1")
    item.composite("attempt-1", state)
    item.reset()
    result = reconstruct(item.review())
    assert result["verdict"] == verdict, result
    assert result["candidates"][0]["grading"]["kind"] == "composite_grading"


def test_unsupported_regrade_keeps_repeated_state(tmp_path, stub_toolchain):
    item = _ready_item(tmp_path, supported=False)  # multistep: outside regrade
    result = reconstruct(item.review())
    assert result["verdict"] == READY
    assert result["candidates"][0]["grading"]["kind"] == "unsupported"
    assert "multistep" in result["candidates"][0]["grading"]["reason"]


def test_supported_bundle_without_grading_evidence_is_unknown(tmp_path, stub_toolchain):
    item = Item(tmp_path / "results")
    item.repeated("attempt-1")
    item.reset()  # no fixed-grading: grading would have been pending -> no review
    assert reconstruct(item.review())["verdict"] == UNKNOWN


def test_reset_without_summary_is_never_ready(tmp_path, stub_toolchain):
    item = Item(tmp_path / "results", kind="docker")
    item.repeated("attempt-1")
    item.grading("attempt-1")
    item.reset(None)
    assert reconstruct(item.review())["verdict"] == UNKNOWN


# ------------------------------------------------ attempt selection


def test_stale_attempt_for_older_task_bytes_is_ignored(tmp_path, stub_toolchain):
    item = Item(tmp_path / "results")
    item.repeated("attempt-1")  # ready, but for the task before a repair
    item.grading("attempt-1")
    (item.path / "harbor/instruction.md").write_text("Count the arrivals precisely.\n")
    item.repeated("attempt-2", passed=False)
    item.grading("attempt-2")
    item.reset()
    result = reconstruct(item.review())
    assert result["verdict"] == NOT_READY
    assert [c["attempt"] for c in result["candidates"]] == ["attempt-2"]


def test_selection_follows_old_lexicographic_attempt_order(tmp_path, stub_toolchain):
    """sorted(glob) visits attempt-10 before attempt-2; the first match wins."""
    item = Item(tmp_path / "results")
    item.repeated("attempt-10", passed=False)
    item.grading("attempt-10")
    item.repeated("attempt-2")
    item.grading("attempt-2")
    item.reset()
    result = reconstruct(item.review())
    assert [c["attempt"] for c in result["candidates"]] == ["attempt-10"]
    assert result["verdict"] == NOT_READY


def test_earlier_attempt_with_unreadable_metadata_blocks_review(tmp_path, stub_toolchain):
    item = Item(tmp_path / "results")
    item.repeated("attempt-1", manifest=False)  # result.json but no inputs/manifest.json
    item.repeated("attempt-2")
    item.grading("attempt-2")
    item.reset()
    result = reconstruct(item.review())
    assert result["verdict"] == UNKNOWN
    assert "attempt-1 metadata is unreadable" in result["reason"]


def test_candidate_environments_must_agree(tmp_path, stub_toolchain):
    """Two controller versions measured the same bytes; the review env is unknown."""
    item = Item(tmp_path / "results")
    item.repeated("attempt-1", passed=False, controller=item.controller("v1"))
    item.grading("attempt-1", controller=item.controller("v1"))
    item.repeated("attempt-2")
    item.grading("attempt-2")
    item.reset()
    result = reconstruct(item.review())
    assert result["verdict"] == UNKNOWN
    assert {c["outcome"] for c in result["candidates"]} == {READY, NOT_READY}


def test_environment_that_could_not_have_reviewed_is_eliminated(tmp_path, stub_toolchain):
    """A docker reset frozen by the current controller rules out the v1 environment."""
    item = Item(tmp_path / "results", kind="docker")
    item.repeated("attempt-1", passed=False, controller=item.controller("v1"))
    item.grading("attempt-1", controller=item.controller("v1"))
    item.repeated("attempt-2")
    item.grading("attempt-2")
    item.reset()
    result = reconstruct(item.review())
    assert result["verdict"] == READY, result
    assert [c["attempt"] for c in result["candidates"]] == ["attempt-2"]
    assert "frozen by another controller" in result["eliminated"][0]["why"]


def test_all_environments_eliminated_is_unknown(tmp_path, stub_toolchain):
    item = Item(tmp_path / "results", kind="docker")
    item.repeated("attempt-1")
    item.grading("attempt-1")
    item.reset(controller_variant="another-deployment")
    result = reconstruct(item.review())
    assert result["verdict"] == UNKNOWN
    assert "no candidate environment" in result["reason"]


# --------------------------------------------------- evidence integrity


def test_listed_but_missing_input_is_unknown(tmp_path, stub_toolchain):
    item = _ready_item(tmp_path)
    review = item.review()
    target = review / "input/diagnostics/attempt-1/evaluation/matrix.json"
    target.chmod(0o644)
    target.unlink()
    result = reconstruct(review)
    assert result["verdict"] == UNKNOWN
    assert "matrix.json is listed but unreadable" in result["reason"]


def test_altered_input_bytes_are_unknown(tmp_path, stub_toolchain):
    item = _ready_item(tmp_path)
    review = item.review()
    target = review / "input/diagnostics/attempt-1/evaluation/matrix.json"
    target.chmod(0o644)
    target.write_text(json.dumps({**json.loads(target.read_text()), "state": "needs_review"}))
    result = reconstruct(review)
    assert result["verdict"] == UNKNOWN
    assert "bytes differ from the review manifest" in result["reason"]


def test_edited_manifest_is_unknown(tmp_path, stub_toolchain):
    item = _ready_item(tmp_path)
    review = item.review()
    manifest = json.loads((review / "input-manifest.json").read_text())
    manifest["item_files"] = manifest["item_files"][:-1]
    atomic_json(review / "input-manifest.json", manifest)
    result = reconstruct(review)
    assert result["verdict"] == UNKNOWN
    assert "snapshot_hash" in result["reason"]


def test_result_disagreeing_with_matrix_is_unknown(tmp_path, stub_toolchain):
    item = Item(tmp_path / "results")
    item.repeated("attempt-1", result={"state": "pending", "reviewable": False,
                                       "issues": ["diagnostic source or controller changed"]})
    item.grading("attempt-1")
    item.reset()
    assert reconstruct(item.review())["verdict"] == UNKNOWN


def test_reconstruction_executes_nothing(tmp_path, stub_toolchain, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("reconstruction must not execute anything")

    item = _ready_item(tmp_path, kind="docker")
    review = item.review()
    import subprocess

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(diagnostics, "run_repeated_diagnostics", forbidden)
    monkeypatch.setattr(reset_runner, "run_frozen_reset", forbidden)
    monkeypatch.setattr(non_docker_reset, "run_frozen_non_docker_reset", forbidden)
    assert reconstruct(review)["verdict"] == READY


# ------------------------------------------------ acceptance recovery


def _forbid_rebuild(monkeypatch) -> list:
    calls = []

    def attempt(*_args, **_kwargs):
        calls.append("attempt")
        return {"state": "failed", "issues": ["KC2: reward is below reward_min"]}

    monkeypatch.setattr(synthesis, "_synthesize_attempt", attempt)
    monkeypatch.setattr(quality, "run_review", lambda *_a, **_k: calls.append("review"))
    monkeypatch.setenv("CAPABILITY_MAX_REPAIR_ROUNDS", "0")
    return calls


def _demote_and_relaunch(item: Item, tmp_path: Path, issue: str) -> Path:
    atomic_json(item.path / "status.json", {
        "key": "d01.uncertainty.probability.counting-processes:1",
        "proposal_hash": PROPOSAL_HASH, "state": "failed", "issues": [issue],
        "item_root": str(item.path),
    })
    new = tmp_path / "relaunch" / "results"
    shutil.copytree(item.root, new)
    return new


def _run(item: Item, root: Path) -> dict:
    return synthesis.synthesize_one(item.item, root, object(), None, None, TIMEOUT)


def test_demoted_legacy_acceptance_is_recovered_with_reconstruction_proof(
    tmp_path, stub_toolchain, monkeypatch
):
    item = _ready_item(tmp_path)
    item.review("attempt-1")
    new = _demote_and_relaunch(item, tmp_path, "KC2: reward is below reward_min")
    calls = _forbid_rebuild(monkeypatch)

    result = _run(item, new)

    assert calls == []
    assert result["state"] == "quality_accepted"
    assert result["gate_proof"] == acceptance.GATE_PROOF_LEGACY
    assert result["acceptance_resumed"]["source"] == "quality_review"
    assert result["acceptance_resumed"]["gate_proof"] == acceptance.GATE_PROOF_LEGACY
    assert result["gate_reconstruction"]["verdict"] == READY
    assert result["gate_reconstruction"]["diagnostics"] == [
        {"attempt": "attempt-1", "repeated": "ready", "grading": "ready", "reset": "ready"}
    ]
    assert result["superseded_status"]["issues"] == ["KC2: reward is below reward_min"]
    ledger = json.loads(acceptance.ledger_path(new, new / "items" / item.path.name).read_text())
    assert ledger["status"]["gate_proof"] == acceptance.GATE_PROOF_LEGACY


def test_accepted_despite_failed_gates_is_not_recovered(tmp_path, stub_toolchain, monkeypatch):
    item = Item(tmp_path / "results")
    item.repeated("attempt-1", passed=False)
    item.grading("attempt-1")
    item.reset()
    item.review("attempt-1")
    new = _demote_and_relaunch(
        item, tmp_path, "semantic review accepted despite failed repeated runtime gates")
    _forbid_rebuild(monkeypatch)
    result = _run(item, new)
    # The retained (demoted) status stands; nothing is recorded as accepted.
    assert result["state"] == "failed"
    assert "acceptance_resumed" not in result
    assert not (new / "acceptances").exists()


@pytest.mark.parametrize("receipt,expected", [("ready", "quality_accepted"), ("pending", "failed")])
def test_receipt_is_authoritative_over_reconstruction(
    tmp_path, stub_toolchain, monkeypatch, receipt, expected
):
    item = _ready_item(tmp_path)  # reconstruction alone would say ready
    review = item.review("attempt-1")
    atomic_json(review / acceptance.GATE_RECEIPT, {"repeated_diagnostics_state": receipt})
    new = _demote_and_relaunch(item, tmp_path, "flake")
    _forbid_rebuild(monkeypatch)
    result = _run(item, new)
    assert result["state"] == expected
    if receipt == "ready":
        assert result["gate_proof"] == acceptance.GATE_PROOF_RECEIPT
        assert "gate_reconstruction" not in result


def test_unreadable_receipt_does_not_fall_back(tmp_path, stub_toolchain):
    item = _ready_item(tmp_path)
    review = item.review("attempt-1")
    (review / acceptance.GATE_RECEIPT).write_text("{not json")
    assert acceptance.post_review_gate_proof(review) == (None, None)
    (review / acceptance.GATE_RECEIPT).unlink()
    proof, record = acceptance.post_review_gate_proof(review)
    assert proof == acceptance.GATE_PROOF_LEGACY and record["verdict"] == READY


def test_legacy_review_of_a_since_repaired_task_is_not_recovered(
    tmp_path, stub_toolchain, monkeypatch
):
    item = _ready_item(tmp_path)
    item.review("attempt-1")
    (item.path / "workspace/task/controls.json").write_text('{"cases": []}\n')
    new = _demote_and_relaunch(item, tmp_path, "flake")
    _forbid_rebuild(monkeypatch)
    result = _run(item, new)
    assert result["state"] == "failed"
    assert "acceptance_resumed" not in result
    assert not (new / "acceptances").exists()
