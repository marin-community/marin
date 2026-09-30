"""Runtime-gate infrastructure failures retry automatically; task defects do not."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from test_conveyor import Script, _config, _run
from test_synthesis import FakeAgent, accepted

from capability_pipeline import (
    conveyor,
    infrastructure_health,
    synthesis,
    verifier_budget,
)
from capability_pipeline.conveyor import TERMINAL, WAITING, classify_result
from capability_pipeline.synthesis import (
    OfficialToolchain,
    SynthesisError,
    _infrastructure_hold,
    _runtime_infrastructure_failures,
    _trial_infrastructure_cause,
    _validate_infrastructure_health_receipt,
    synthesize_one,
)

SANDBOXED = "capability_pipeline.daytona_verifier:DaytonaSemanticVerifier"
NATIVE = "taskcompendium.harbor.verifier:SemanticVerifier"


def trial(exception_type, message, verifier=SANDBOXED):
    return {
        "verifier_result": None,
        "config": {"verifier": {"import_path": verifier}},
        "exception_info": {"exception_type": exception_type, "exception_message": message},
    }


def grading(error, **extra):
    return trial("GradingInfrastructureError", json.dumps({"error": error, **extra}))


def write_trial(item_root, name, value):
    path = item_root / "runtime-trials" / name / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def runtime_issue(name, exception_type="VerifierTimeoutError"):
    return (
        "runtime controls failed: Traceback (most recent call last):\n  ...\n"
        f"RuntimeError: authored reference gold failed in Harbor ({exception_type}); "
        f"inspect runtime-trials/{name}/result.json"
    )


def item_paths(tmp_path):
    item = accepted()
    name = f"cap.test-1-{item['proposal_hash'][:12]}"
    return item, tmp_path / "items" / name


# -- classification -----------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "cause"),
    [
        (trial("VerifierTimeoutError", "Verifier execution timed out after 600.0 seconds"), "VerifierTimeoutError"),
        (grading("SiloNotFoundError: sandbox 'slb6fc4a' not found"), "SiloNotFoundError"),
        (grading("SiloError: sandbox 'slb1' is destroying"), "SiloSandboxLost"),
        (grading("TransportError: cannot reach http://10.0.0.1:1/sandboxes: Remote end closed"), "SiloTransportError"),
        (grading("Malformed judge verdict", reply="yes", model="glm-5.3"), "GLMJudgeMalformedVerdict"),
        (grading("Judge request failed: timed out"), "GLMJudgeTransport"),
        (
            grading("FileNotFoundError: [Errno 2] No such file or directory: "
                    "'/app/capability-pipeline-staging/submissions/run-1/daytona-tools/dt.py'"),
            "StagingCodeMissing",
        ),
        (
            trial("RuntimeError", "GLM transport unavailable for the whole infrastructure hold (1800 s, 2 attempts): {}"),
            "GLMTransportUnavailable",
        ),
        (trial("GradingInfrastructureError", "ProviderRateLimitExhausted"), "ProviderRateLimitExhausted"),
        (trial("SiloNotFoundError", "sandbox 'x' not found"), "SiloNotFoundError"),
    ],
)
def test_ungraded_provider_and_transport_failures_are_infrastructure(value, cause):
    assert _trial_infrastructure_cause(value) == cause


@pytest.mark.parametrize(
    "value",
    [
        # A native verifier timing out is not provisioning time.
        trial("VerifierTimeoutError", "Verifier execution timed out after 600.0 seconds", verifier=NATIVE),
        # The authored grader crashed or wrote to a read-only mount: a task defect.
        grading("RuntimeError: script '__taskcompendium_script_wrapper.py' exited 1 without reporting a reward; "
                "stderr tail: 'PermissionError: [Errno 13] Permission denied: '/app/x.json''"),
        grading("math-verify cannot parse expected 'superseded'"),
        # The independent solver ran out of turns: a solver outcome.
        trial("TurnCapExhaustedError", "Shell-tool agent exhausted its turn budget"),
        trial("ExtractionError", json.dumps({"error": "Submission is empty"})),
    ],
)
def test_task_and_solver_outcomes_are_not_infrastructure(value):
    assert _trial_infrastructure_cause(value) is None


def test_named_trial_binds_classification_not_stale_trial_directories(tmp_path):
    item_root = tmp_path / "item"
    stale = write_trial(item_root, "oracle-old", trial("VerifierTimeoutError", "timed out"))
    write_trial(item_root, "control-crash", grading("RuntimeError: script 'x' exited 1 without reporting a reward"))
    failed = {"state": "failed", "issues": [runtime_issue("control-crash", "GradingInfrastructureError")]}
    assert _runtime_infrastructure_failures(item_root, failed) == []
    failed["issues"] = [runtime_issue("oracle-old")]
    failures = _runtime_infrastructure_failures(item_root, failed)
    assert failures == [
        {
            "trial": "oracle-old",
            "artifact": str(stale),
            "artifact_sha256": hashlib.sha256(stale.read_bytes()).hexdigest(),
            "exception_type": "VerifierTimeoutError",
            "provider_cause": "VerifierTimeoutError",
        }
    ]


def test_toolchain_integrity_hold_for_judge_calibration(tmp_path):
    for head in (
        "native judge calibration is incomplete: TaskCompendium source hash mismatch: uv.lock",
        (
            "native judge calibration is incomplete: TaskCompendium archive file set mismatch; "
            "extra=['src/taskcompendium/composed_verifier/__init__.py'], missing=[]"
        ),
    ):
        hold = _infrastructure_hold(tmp_path, {"state": "pending_judge_calibration", "issues": [head]})
        assert hold["gate"] == "judge_calibration"
        assert hold["families"] == ["toolchain"]
    measured = {"state": "pending_judge_calibration", "issues": ["native judge calibration did not pass"]}
    assert _infrastructure_hold(tmp_path, measured) is None


def _diagnostics(item_root, cells, trials):
    attempt = item_root / "diagnostics" / "attempt-1"
    evaluation = attempt / "evaluation"
    evaluation.mkdir(parents=True)
    (evaluation / "matrix.json").write_text(json.dumps({"cells": cells}))
    for number, value in trials.items():
        path = evaluation / "attempts" / f"{number:03d}" / "runtime-trials" / "oracle" / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(value))
    # The status carries the writing job's absolute work root, not this one.
    return {
        "state": "pending_repeated_diagnostics",
        "issues": ["repeated runtime diagnostics lack complete validated evidence"],
        "repeated_diagnostics": {
            "state": "pending",
            "attempt": "/tmp/capability-pipeline/other-run/results/items/x/diagnostics/attempt-1",
        },
    }


def test_repeated_diagnostics_runtime_errors_on_infrastructure_are_held(tmp_path):
    result = _diagnostics(
        tmp_path,
        [{"attempt": 1, "state": "valid"}, {"attempt": 2, "state": "runtime_error"}, {"attempt": 3, "state": "runtime_error"}],
        {2: trial("VerifierTimeoutError", "timed out"), 3: grading("Malformed judge verdict")},
    )
    hold = _infrastructure_hold(tmp_path, result)
    assert hold["gate"] == "repeated_diagnostics"
    assert hold["causes"] == ["GLMJudgeMalformedVerdict", "VerifierTimeoutError"]
    assert hold["families"] == ["glm", "sandbox"]


def test_repeated_diagnostics_evidence_disagreement_is_not_infrastructure(tmp_path):
    result = _diagnostics(
        tmp_path,
        [{"attempt": 1, "state": "invalid_evidence"}, {"attempt": 2, "state": "runtime_error"}],
        {2: trial("VerifierTimeoutError", "timed out")},
    )
    assert _infrastructure_hold(tmp_path, result) is None


# -- synthesize_one: automatic retry -------------------------------------------


def _passing_probe(monkeypatch, calls):
    def probe(item_root, output, *, daytona_tools, **_kwargs):
        calls.append(item_root)
        receipt = {
            "schema_version": "capability-daytona-health-v1",
            "state": "passed",
            "snapshot": "cap-verifier-00000000000000000000",
            "snapshot_id": "snap-1",
            "sandbox_id": "health-sandbox",
            "network_block_all": True,
            "network_block_all_requested": True,
            "network_block_all_observed": True,
            "provisioning_attempts": [{"attempt": 1, "state": "created"}],
            "deleted": True,
            "lookup_after_delete": "not_found",
            "deletion_lookups": [{"elapsed_seconds": 0, "state": "not_found"}],
            "completed_at": datetime.now(UTC).isoformat(),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt))
        return receipt

    monkeypatch.setattr(infrastructure_health, "probe_item_provider", probe)


def _held_failure(item_root, name="oracle-gold"):
    artifact = write_trial(item_root, name, trial("VerifierTimeoutError", "Verifier execution timed out after 600.0 seconds"))
    (item_root / "status.json").write_text(
        json.dumps({"state": "failed", "item_root": str(item_root), "issues": [runtime_issue(name)]})
    )
    return artifact


def test_held_runtime_timeout_is_retried_automatically_after_a_passing_probe(tmp_path, monkeypatch):
    synthesis._PROBE_CACHE.clear()
    item, item_root = item_paths(tmp_path)
    artifact = _held_failure(item_root)
    probes = []
    _passing_probe(monkeypatch, probes)
    reruns = []

    def rerun(*args, **_kwargs):
        reruns.append(args)
        return {"state": "pending_quality_review", "item_root": str(item_root), "issues": ["review"]}

    monkeypatch.setattr(synthesis, "_synthesize_attempt", rerun)
    result = synthesize_one(item, tmp_path, FakeAgent(), None, None, 30)

    assert len(probes) == 1 and len(reruns) == 1
    record = result["infrastructure_revalidation"]
    assert record["automatic"] is True and record["attempt"] == 1
    assert record["gate"] == "runtime_controls" and record["causes"] == ["VerifierTimeoutError"]
    assert record["health_sandbox_id"] == "health-sandbox"
    _validate_infrastructure_health_receipt(Path(record["health_receipt"]))
    history = tmp_path / "infrastructure-history" / item_root.name / "revalidation-1"
    assert (history / "runtime-trials/oracle-gold/result.json").is_file()
    assert record["prior_failures"][0]["original_artifact"] == str(artifact)
    assert not (item_root / "runtime-trials").exists()
    assert result["state"] == "pending_quality_review"
    status = json.loads((item_root / "status.json").read_text())
    # The retained failure stays in the transition history of the retried item.
    assert [entry["state"] for entry in status["transitions"]][:1] == ["failed"]


def test_failed_probe_keeps_waiting_without_spending_a_gate_rerun(tmp_path, monkeypatch):
    synthesis._PROBE_CACHE.clear()
    item, item_root = item_paths(tmp_path)
    _held_failure(item_root)

    def probe(item_root, output, *, daytona_tools, **_kwargs):
        receipt = {"schema_version": "capability-daytona-health-v1", "state": "failed", "error_type": "ProviderRateLimitExhausted"}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt))
        return receipt

    monkeypatch.setattr(infrastructure_health, "probe_item_provider", probe)
    monkeypatch.setattr(synthesis, "_synthesize_attempt", lambda *a, **k: pytest.fail("must not rerun"))
    result = synthesize_one(item, tmp_path, FakeAgent(), None, None, 30)

    assert result["state"] == "pending_runtime_infrastructure"
    assert result["issues"][0].startswith("runtime infrastructure hold (runtime_controls): VerifierTimeoutError")
    assert result["issues"][1].startswith("runtime controls failed:")
    record = result["runtime_infrastructure"]
    assert record["revalidations_used"] == 0 and record["max_revalidations"] == 3
    assert record["last_probe"]["checks"]["sandbox"]["error_type"] == "ProviderRateLimitExhausted"
    assert (item_root / "runtime-trials/oracle-gold/result.json").is_file()
    classification = classify_result(result, repair_actionable=lambda value: False)
    assert (classification.klass, classification.stage) == (WAITING, "runtime_infrastructure")


def test_fresh_infrastructure_failure_waits_instead_of_ending_terminal(tmp_path, monkeypatch):
    item, item_root = item_paths(tmp_path)

    def attempt(*_args, **_kwargs):
        write_trial(item_root, "control-p1", grading("SiloNotFoundError: sandbox 'slb1' not found"))
        return {"state": "failed", "item_root": str(item_root), "issues": [runtime_issue("control-p1", "GradingInfrastructureError")]}

    monkeypatch.setattr(synthesis, "_synthesize_attempt", attempt)
    result = synthesize_one(item, tmp_path, FakeAgent(), None, None, 30)
    assert result["state"] == "pending_runtime_infrastructure"
    assert result["runtime_infrastructure"]["hold"]["causes"] == ["SiloNotFoundError"]
    assert result["runtime_infrastructure"]["gate_state"] == "failed"


def test_durable_retry_cap_ends_terminal_with_a_visible_reason(tmp_path, monkeypatch):
    item, item_root = item_paths(tmp_path)
    _held_failure(item_root)
    for number in (1, 2, 3):
        (tmp_path / "infrastructure-history" / item_root.name / f"revalidation-{number}").mkdir(parents=True)
    monkeypatch.setattr(synthesis, "_probe_infrastructure", lambda *a, **k: pytest.fail("must not probe"))
    result = synthesize_one(item, tmp_path, FakeAgent(), None, None, 30)
    assert result["state"] == "failed"
    assert result["failure_stage"] == "runtime_infrastructure"
    assert result["issues"][0] == (
        "runtime infrastructure retries exhausted after 3 gate re-run(s) (runtime_controls): VerifierTimeoutError"
    )
    assert result["runtime_infrastructure"]["exhausted"] is True
    classification = classify_result(result, repair_actionable=lambda value: False)
    assert (classification.klass, classification.stage) == (TERMINAL, "runtime_infrastructure")


def test_conveyor_wait_exhaustion_reenters_on_relaunch(tmp_path, monkeypatch):
    synthesis._PROBE_CACHE.clear()
    item, item_root = item_paths(tmp_path)
    _held_failure(item_root)
    held = synthesis._hold_for_infrastructure(tmp_path, item_root, json.loads((item_root / "status.json").read_text()))
    exhausted = {
        **held,
        "state": "failed",
        "issues": ["wait_budget_exhausted:pending_runtime_infrastructure", *held["issues"]],
        "wait_exhausted": {"kind": "runtime_infrastructure", "cause": "deadline"},
    }
    (item_root / "status.json").write_text(json.dumps(exhausted))
    probes = []
    _passing_probe(monkeypatch, probes)
    monkeypatch.setattr(
        synthesis, "_synthesize_attempt",
        lambda *a, **k: {"state": "quality_accepted", "item_root": str(item_root), "issues": []},
    )
    result = synthesize_one(item, tmp_path, FakeAgent(), None, None, 30)
    assert probes and result["state"] == "quality_accepted"
    assert result["infrastructure_revalidation"]["attempt"] == 1


def test_judge_toolchain_hold_heals_before_rerunning(tmp_path, monkeypatch):
    item, item_root = item_paths(tmp_path)
    item_root.mkdir(parents=True)
    (item_root / "status.json").write_text(json.dumps({
        "state": "pending_judge_calibration",
        "item_root": str(item_root),
        "issues": ["native judge calibration is incomplete: TaskCompendium source hash mismatch: uv.lock"],
    }))
    healed = []
    toolchain = SimpleNamespace(package_root=tmp_path / "overlay", heal=lambda: healed.append(1) or {"drift": "uv.lock"})
    monkeypatch.setattr(
        synthesis, "_synthesize_attempt",
        lambda *a, **k: {"state": "validated", "item_root": str(item_root), "issues": []},
    )
    result = synthesize_one(item, tmp_path, FakeAgent(), toolchain, None, 30)
    assert healed == [1]
    record = result["infrastructure_revalidation"]
    assert record["gate"] == "judge_calibration"
    assert record["probe"]["checks"]["toolchain"] == {"state": "passed", "healed": {"drift": "uv.lock"}}


def test_scheduler_retries_runtime_infrastructure_then_exhausts_to_terminal(tmp_path):
    hold = {
        "state": "pending_runtime_infrastructure",
        "issues": ["runtime infrastructure hold (runtime_controls): VerifierTimeoutError"],
    }
    recovers = Script(tmp_path, "recovers", [hold, hold, {"state": "quality_accepted"}])
    stuck = Script(tmp_path, "stuck", [hold])
    config = _config(CAPABILITY_WAIT_RUNTIME_INFRASTRUCTURE_ATTEMPTS=4)
    outcome = _run(tmp_path, [recovers, stuck], concurrency=2, config=config)
    first, second = outcome.results
    assert first["state"] == "quality_accepted" and len(recovers.calls) == 3
    assert second["state"] == "failed" and len(stuck.calls) == 4
    assert second["failure_stage"] == "runtime_infrastructure"
    assert second["issues"][0] == "wait_budget_exhausted:pending_runtime_infrastructure"


def test_default_runtime_infrastructure_budget():
    budget = conveyor.ConveyorConfig.from_env({}).budgets["runtime_infrastructure"]
    assert (budget.window_seconds, budget.max_attempts, budget.backoff_seconds) == (43_200, 16, 900)


# -- in-job health probe -------------------------------------------------------


class NotFound(Exception):
    status_code = 404


class RateLimited(Exception):
    status_code = 429
    headers = MappingProxyType({"Retry-After": "1"})


class FakeClient:
    def __init__(self, *, snapshots, create_errors=0):
        self.snapshots = snapshots
        self.create_errors = create_errors
        self.deleted = set()
        self.snapshot = SimpleNamespace(get=self._snapshot)

    def _snapshot(self, name):
        if name not in self.snapshots:
            raise NotFound(f"snapshot {name!r} not found")
        return SimpleNamespace(name=name, id="snap-" + name[-4:], state=self.snapshots[name])

    def create(self, params, timeout):
        assert params.network_block_all is True
        if self.create_errors:
            self.create_errors -= 1
            raise RateLimited("capacity exhausted")
        return SimpleNamespace(id="sbx-1", delete=lambda: self.deleted.add("sbx-1"))

    def get(self, sandbox_id):
        if sandbox_id in self.deleted:
            raise NotFound(f"sandbox {sandbox_id!r} not found")
        return SimpleNamespace(id=sandbox_id, network_block_all=True)


def test_probe_receipt_passes_the_operator_receipt_check(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPABILITY_SANDBOX_PROVIDER", "silo")
    monkeypatch.setenv("SILO_API_TOKEN", "t")
    monkeypatch.setenv("SILO_BROKER_URL", "http://broker")
    client = FakeClient(snapshots={"cap-verifier-missing": "active", "cap-verifier-aaaa": "active"})
    client.snapshots.pop("cap-verifier-missing")
    receipt = infrastructure_health.run_probe(
        ["cap-verifier-missing", "cap-verifier-aaaa"], client_factory=lambda: client, sleeper=lambda _s: None
    )
    assert receipt["state"] == "passed" and receipt["snapshot"] == "cap-verifier-aaaa"
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    assert _validate_infrastructure_health_receipt(path)["sandbox_id"] == "sbx-1"


def test_probe_fails_on_exhausted_capacity_and_is_unavailable_without_snapshot(monkeypatch):
    monkeypatch.setenv("CAPABILITY_SANDBOX_PROVIDER", "silo")
    monkeypatch.setenv("SILO_API_TOKEN", "t")
    monkeypatch.setenv("SILO_BROKER_URL", "http://broker")
    busy = FakeClient(snapshots={"cap-verifier-aaaa": "active"}, create_errors=10)
    receipt = infrastructure_health.run_probe(["cap-verifier-aaaa"], client_factory=lambda: busy, sleeper=lambda _s: None)
    assert receipt["state"] == "failed" and receipt["error_type"] == "ProviderRateLimitExhausted"
    empty = FakeClient(snapshots={})
    receipt = infrastructure_health.run_probe(["cap-verifier-aaaa"], client_factory=lambda: empty, sleeper=lambda _s: None)
    assert receipt["state"] == "unavailable" and receipt["error_type"] == "NoActiveProbeSnapshot"


def test_probe_snapshots_come_from_graded_trials_then_the_spec(tmp_path):
    item_root = tmp_path / "item"
    graded = item_root / "runtime-trials/oracle/verifier/taskcompendium-result.json"
    graded.parent.mkdir(parents=True)
    graded.write_text(json.dumps({"detail": {"verifier_snapshot": "cap-verifier-0123456789abcdef0123"}}))
    image = "python:3.12-slim@sha256:" + "a" * 64
    spec = item_root / "harbor/specification.json"
    spec.parent.mkdir(parents=True)
    spec.write_text(json.dumps({"steps": [{"verifier": {"verifier": {"runtime": {
        "kind": "container", "image": image, "supervisor_python": "python3", "timeout": 600.0}}}}]}))
    names = infrastructure_health.probe_snapshots(item_root)
    from capability_pipeline.daytona_policy import verifier_snapshot_recipe
    from capability_pipeline.daytona_resources import VERIFIER_DEFAULT, snapshot_name

    assert names == [
        "cap-verifier-0123456789abcdef0123",
        snapshot_name("cap-verifier", verifier_snapshot_recipe(image, "python3"), VERIFIER_DEFAULT),
    ]


# -- verifier budget ------------------------------------------------------------


def test_sandboxed_verifier_budget_adds_provisioning_to_the_grader_timeout(tmp_path):
    package = tmp_path / "harbor"
    package.mkdir()
    (package / "task.toml").write_text('version = "1.0"\n')
    assert verifier_budget.verifier_override_seconds(package, {}, None) is None
    assert verifier_budget.verifier_override_seconds(package, {0: 600.0}, None, environ={}) == 600 + 900 + 120
    assert verifier_budget.verifier_override_seconds(
        package, {0: 60.0}, None, environ={"CAPABILITY_VERIFIER_PROVISIONING_SECONDS": "0"}
    ) == 600.0  # never below the declared (default) budget
    (package / "task.toml").write_text('[verifier]\ntimeout_sec = 3000.0\n')
    assert verifier_budget.verifier_override_seconds(package, {0: 600.0}, None, environ={}) == 3000.0


def test_composite_verifier_budget_covers_each_machine_check(tmp_path):
    package = tmp_path / "harbor"
    package.mkdir()
    config = {"steps": [{
        "step_index": 0,
        "machine_checks": [{"timeout": 60}, {"timeout": 60}],
        "judge": {"criterion_weights": [1.0]},
    }]}
    from capability_pipeline.composite_timeout import composite_verifier_timeout

    budget = verifier_budget.verifier_override_seconds(package, {}, config, frozenset({0}), environ={})
    assert budget == composite_verifier_timeout(config, 0) + 2 * (900 + 120 - 300)


# -- toolchain isolation and healing --------------------------------------------


def _toolchain_fixture(tmp_path, monkeypatch):
    source_lock = tmp_path / "source.lock.json"
    project = tmp_path / "project"
    source = tmp_path / "source"
    overlay = tmp_path / "overlay"
    names = {
        "pyproject.toml": b"[project]\nname = 'fixture'\n",
        "uv.lock": b"version = 1\n",
        "src/untouched.py": b"unchanged\n",
        "src/taskcompendium/harbor/verifier.py": b"base verifier\n",
    }
    patched = {"src/taskcompendium/harbor/verifier.py": b"patched verifier\n"}
    for root, files in ((source, names), (overlay, {**names, **patched})):
        for name, raw in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    expected = {
        name: {"base_sha256": hashlib.sha256(names[name]).hexdigest(), "patched_sha256": hashlib.sha256(raw).hexdigest()}
        for name, raw in patched.items()
    }
    source_lock.write_text(json.dumps({"files": {n: hashlib.sha256(r).hexdigest() for n, r in names.items()}}))
    extension = project / "vendor/task_spec"
    extension.mkdir(parents=True)
    patch = project / "overlay.patch"
    patch.write_text("approved patch\n")
    (extension / "composite_extension.lock.json").write_text(json.dumps(
        {"files": expected, "patch": "overlay.patch", "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest()}
    ))
    monkeypatch.setattr(synthesis, "SOURCE_LOCK", source_lock)
    monkeypatch.setattr(synthesis, "PROJECT_ROOT", project)
    monkeypatch.setattr(OfficialToolchain, "_expected_overlay_files", staticmethod(lambda: expected))
    from capability_pipeline import composite_extension

    base_sha = expected["src/taskcompendium/harbor/verifier.py"]["base_sha256"]
    patched_sha = expected["src/taskcompendium/harbor/verifier.py"]["patched_sha256"]

    def guard(root, *, expected_base):
        path = root / "src/taskcompendium/harbor/verifier.py"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_base
        path.write_bytes(patched["src/taskcompendium/harbor/verifier.py"])
        return patched_sha

    monkeypatch.setattr(composite_extension, "BASE_VERIFIER_SHA256", base_sha)
    monkeypatch.setattr(composite_extension, "PATCHED_VERIFIER_SHA256", patched_sha)
    monkeypatch.setattr(composite_extension, "apply_taskcompendium_guard", guard)
    return OfficialToolchain(overlay, "uv", source_package_root=source)


def test_agents_get_a_separate_copy_and_a_drifted_overlay_heals(tmp_path, monkeypatch):
    toolchain = _toolchain_fixture(tmp_path, monkeypatch)
    toolchain.validate_runtime_overlay()
    assert toolchain.heal() is None
    shared = toolchain.builder_copy()
    assert shared.builder_root != toolchain.package_root
    assert synthesis._agent_package_root(shared) == shared.builder_root
    # An agent rewriting uv.lock or adding modules in its copy cannot touch the controller's.
    (shared.builder_root / "uv.lock").write_text("version = 2\n")
    shared.validate_runtime_overlay()
    # Drift in the controller overlay itself (the measured failure) heals from the pinned source.
    (toolchain.package_root / "uv.lock").write_text("version = 2\n")
    (toolchain.package_root / "src/taskcompendium/composite_policy.py").write_text("copied by an agent\n")
    with pytest.raises(SynthesisError, match="source hash mismatch: uv.lock"):
        toolchain.validate_runtime_overlay()
    record = toolchain.heal()
    assert "uv.lock" in record["drift"]
    toolchain.validate_runtime_overlay()
    assert not (toolchain.package_root / "src/taskcompendium/composite_policy.py").exists()


def test_heal_refuses_a_corrupted_source(tmp_path, monkeypatch):
    toolchain = _toolchain_fixture(tmp_path, monkeypatch)
    (toolchain.source_package_root / "uv.lock").write_text("version = 3\n")
    with pytest.raises(SynthesisError, match="source hash mismatch"):
        toolchain.heal()


def test_lost_candidate_sandbox_is_infrastructure_even_after_a_grade():
    lost = trial("SiloNotFoundError", "sandbox 'slb1' not found")
    lost["verifier_result"] = {"rewards": {"reward": 1.0}}
    assert _trial_infrastructure_cause(lost) == "SiloNotFoundError"
    graded_timeout = trial("VerifierTimeoutError", "timed out")
    graded_timeout["verifier_result"] = {"rewards": {"reward": 1.0}}
    assert _trial_infrastructure_cause(graded_timeout) is None


# -- repeated-evaluation evidence check ------------------------------------------


def _evaluation_attempt(tmp_path, monkeypatch, gold_reward, state="passed"):
    from test_evaluation import _inputs

    from capability_pipeline import adversary, evaluation
    from capability_pipeline.runtime import sha256

    package, bundle, controls = _inputs(tmp_path)
    root = tmp_path / "attempt"
    root.mkdir()
    for name in ("oracle.json", "solver.json"):
        (root / name).write_text(json.dumps({"cases": []}))
    (root / "runtime-evidence.json").write_text(json.dumps({
        "attestation": {
            "solver": {"retry_limit": 1, "state": state},
            "adversarial": {"diagnostic_new_attacks": "not_run_primary_bound"},
            "oracle_artifact": "oracle.json",
            "solver_artifact": "solver.json",
            "adversary_artifact": None,
        },
        "cases": [
            {"id": "gold", "source_author": "author", "category": "solver",
             "control_type": "independent_solver", "result": {"status": "graded", "reward": gold_reward}},
            {"id": "negative", "source_author": "author", "category": "control",
             "control_type": "authored_adversarial_control", "result": {"status": "graded", "reward": 0}},
        ],
    }))
    monkeypatch.setattr(synthesis, "_attestation_issues", lambda *args, **kwargs: [])
    monkeypatch.setattr(adversary, "assess_attack_results", lambda *args, **kwargs: [])
    plan = {
        "purpose": "primary_bound_repeatability",
        "specification_sha256": sha256(bundle / "specification.json"),
        "lowered_specification_sha256": sha256(bundle / "specification.json"),
        "critical_control_ids": [],
    }
    return evaluation.inspect_attempt(root, {"package": package, "bundle": bundle, "controls": controls}, plan)


def test_solver_below_authored_reward_min_is_a_graded_result_not_invalid_evidence(tmp_path, monkeypatch):
    # The runtime attests "passed" at reward >= 0.8; the author demanded 1.0.
    cell = _evaluation_attempt(tmp_path, monkeypatch, 0.92)
    assert cell["state"] == "valid"
    assert cell["solver_passed"] is False
    assert "gold: reward is below reward_min" in cell["control_results"]["solver"]["issues"]


def test_attested_solver_state_contradicting_the_grades_is_still_invalid(tmp_path, monkeypatch):
    cell = _evaluation_attempt(tmp_path, monkeypatch, 0.5, state="passed")
    assert cell["state"] == "invalid_evidence"
    assert "solver state disagrees with recorded grades" in cell["issues"]


def test_solver_state_checker_defect_cells_are_retried_as_controller_holds(tmp_path):
    result = _diagnostics(
        tmp_path,
        [
            {"attempt": 1, "state": "invalid_evidence", "issues": ["solver state disagrees with recorded grades"]},
            {"attempt": 2, "state": "valid"},
            {"attempt": 3, "state": "runtime_error"},
        ],
        {3: grading("Judge request failed: timed out")},
    )
    hold = _infrastructure_hold(tmp_path, result)
    assert hold["causes"] == ["GLMJudgeTransport", "SolverStateCheck"]
    assert hold["families"] == ["controller", "glm"]


def test_one_sandbox_probe_serves_every_held_item_within_the_ttl(tmp_path, monkeypatch):
    synthesis._PROBE_CACHE.clear()
    probes = []
    _passing_probe(monkeypatch, probes)
    hold = {"gate": "runtime_controls", "families": ["sandbox"], "causes": ["VerifierTimeoutError"], "failures": []}
    first = synthesis._probe_infrastructure(tmp_path, tmp_path / "items/a", hold, None, tmp_path / "tools")
    second = synthesis._probe_infrastructure(tmp_path, tmp_path / "items/b", hold, None, tmp_path / "tools")
    assert first["state"] == second["state"] == "passed"
    assert len(probes) == 1 and second["checks"]["sandbox"]["shared"] is True
    monkeypatch.setenv("CAPABILITY_INFRA_PROBE_TTL_SECONDS", "0")
    synthesis._probe_infrastructure(tmp_path, tmp_path / "items/c", hold, None, tmp_path / "tools")
    assert len(probes) == 2
    synthesis._PROBE_CACHE.clear()


def test_graded_turn_capped_solver_attempt_is_a_controller_hold_not_a_task_defect(tmp_path):
    capped = trial("TurnCapExhaustedError", "Shell-tool agent exhausted its turn budget")
    capped["verifier_result"] = {"rewards": {"reward": 1.0}}
    assert _trial_infrastructure_cause(capped, "control-gold-attempt-1") == "SolverTurnCapGateAbort"
    assert _trial_infrastructure_cause(capped, "oracle-gold") is None
    item_root = tmp_path / "item"
    write_trial(item_root, "control-gold-attempt-1", capped)
    hold = _infrastructure_hold(
        item_root,
        {"state": "failed", "issues": [runtime_issue("control-gold-attempt-1", "TurnCapExhaustedError")]},
    )
    assert hold["families"] == ["controller"] and hold["causes"] == ["SolverTurnCapGateAbort"]


def test_runtime_counts_a_graded_turn_capped_solver_attempt(tmp_path):
    from capability_pipeline.runtime import solver_limit_graded

    artifact = tmp_path / "taskcompendium-result.json"
    artifact.write_text("{}")
    capped = SimpleNamespace(
        exception_info=SimpleNamespace(exception_type="TurnCapExhaustedError"),
        verifier_result={"rewards": {"reward": 0.0}},
    )
    assert solver_limit_graded(True, capped, artifact) is True
    assert solver_limit_graded(False, capped, artifact) is False  # authored controls never
    ungraded = SimpleNamespace(exception_info=capped.exception_info, verifier_result=None)
    assert solver_limit_graded(True, ungraded, artifact) is False
    timeout = SimpleNamespace(
        exception_info=SimpleNamespace(exception_type="VerifierTimeoutError"), verifier_result={"x": 1}
    )
    assert solver_limit_graded(True, timeout, artifact) is False
    assert solver_limit_graded(True, capped, tmp_path / "missing.json") is False


def test_verifier_budget_never_raises(tmp_path):
    package = tmp_path / "harbor"
    package.mkdir()
    broken = {"steps": [{"step_index": 5}]}
    assert verifier_budget.verifier_override_seconds(package, {}, broken, frozenset({0}), environ={}) == 600 + 900 + 120
    assert verifier_budget.provisioning_seconds({"CAPABILITY_VERIFIER_PROVISIONING_SECONDS": "lots"}) == 900
