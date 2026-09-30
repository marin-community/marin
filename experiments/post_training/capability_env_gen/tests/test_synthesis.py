import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from capability_pipeline.inference import digest
from capability_pipeline.synthesis import (
    INCOMPLETE_ADVERSARY_ISSUE,
    OfficialToolchain,
    SynthesisError,
    _attack_adjudication,
    _bounded_incomplete_adversary,
    _build_acceptance_issues,
    _build_checklist_path,
    _compatible_private_verifier_record,
    _container_supervisor_runtimes,
    _continuation_prompt,
    _controls_pass,
    _fresh_construction_repair_allowed,
    _handoff,
    _nondefault_container_supervisors,
    _omp_transcript_totals,
    _private_verifier_record,
    _record_image_compatibility_failure,
    _repair_feedback,
    _run,
    _runtime_image_compatibility_failures,
    _runtime_infrastructure_failures,
    _safe_name,
    _synthesize_attempt,
    _validate_controls,
    _validate_infrastructure_health_receipt,
    _validated_pending_adversary_retry,
    load_accepted,
    synthesize_one,
)


def test_synthesis_requires_supervisor_receipt_for_nondefault_container(tmp_path):
    image = "registry.example/c32/verifier@sha256:" + "a" * 64
    supervisor = "/opt/py312/bin/python3"
    (tmp_path / "specification.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "verifier": {
                            "runtime": {
                                "kind": "container",
                                "image": image,
                                "supervisor_python": supervisor,
                            }
                        }
                    }
                ]
            }
        )
    )
    assert _nondefault_container_supervisors(tmp_path) == {0: (image, supervisor)}
    assert _container_supervisor_runtimes(tmp_path) == {0: (image, supervisor)}
    record, issue = _private_verifier_record(
        {"detail": {}}, "adapter", runtime_image=image, supervisor_python=supervisor
    )
    assert record is None
    assert issue == "lacks bound Daytona private-verifier evidence"


def test_default_container_accepts_exact_legacy_or_strong_receipt():
    from capability_pipeline.daytona_policy import (
        verifier_bootstrap_sha256,
        verifier_snapshot_recipe,
    )
    from capability_pipeline.runtime import verifier_isolation_record

    adapter = hashlib.sha256(
        (Path(__file__).resolve().parents[1] / "capability_pipeline/daytona_verifier.py").read_bytes()
    ).hexdigest()
    image = "python:3.12-slim@sha256:" + "b" * 64
    bootstrap = verifier_bootstrap_sha256()
    recipe = hashlib.sha256(verifier_snapshot_recipe(image, "python3").encode()).hexdigest()
    base_detail = {
        "verifier_isolation": "daytona-network-block-all",
        "verifier_adapter_sha256": adapter,
        "verifier_bootstrap_sha256": bootstrap,
        "verifier_sandbox_id": "sandbox-1",
        "verifier_snapshot": "legacy-snapshot",
    }
    legacy, issue = _compatible_private_verifier_record(
        {"detail": base_detail}, adapter, (image, "python3")
    )
    assert issue is None
    assert set(legacy) == {"sandbox_id", "snapshot", "adapter_sha256", "bootstrap_sha256"}

    from capability_pipeline.daytona_resources import VERIFIER_DEFAULT, snapshot_name

    strong_detail = {
        **base_detail,
        "verifier_snapshot": snapshot_name(
            "cap-verifier", verifier_snapshot_recipe(image, "python3"), VERIFIER_DEFAULT
        ),
        "verifier_supervisor_python": "python3",
        "verifier_bootstrap_command_sha256": verifier_bootstrap_sha256("python3"),
        "verifier_snapshot_recipe_sha256": recipe,
        "verifier_requested_resource_profile": VERIFIER_DEFAULT.receipt(),
    }
    # Live Harbor grades retain deletion evidence in the raw result and in
    # runtime's summarized private-verifier record. Synthesis must compare the
    # same complete shape, including cleanup, rather than fabricating a mismatch.
    cleanup = {
        "state": "deleted",
        "attempts": [{"attempt": 1, "state": "delete_requested"}],
        "observations": [
            {"attempt": 1, "state": "present"},
            {"attempt": 1, "state": "not_found"},
        ],
    }
    strong_detail["verifier_cleanup"] = cleanup
    strong, issue = _compatible_private_verifier_record(
        {"detail": strong_detail}, adapter, (image, "python3")
    )
    assert issue is None
    assert strong["supervisor_python"] == "python3"
    assert strong["snapshot_recipe_sha256"] == recipe
    assert strong["cleanup"] == cleanup
    assert strong == verifier_isolation_record(
        {"detail": strong_detail}, set(), set(),
        runtime_image=image, supervisor_python="python3",
    )

    incomplete_cleanup, issue = _compatible_private_verifier_record(
        {"detail": {**strong_detail, "verifier_cleanup": {**cleanup, "observations": cleanup["observations"][:1]}}},
        adapter, (image, "python3"),
    )
    assert incomplete_cleanup is None
    assert issue == "has unconfirmed private-verifier cleanup"

    malformed_resource, issue = _compatible_private_verifier_record(
        {
            "detail": {
                **strong_detail,
                "verifier_requested_resource_profile": {"cpu": 2},
            }
        },
        adapter,
        (image, "python3"),
    )
    assert malformed_resource is None
    assert issue == "has a malformed verifier resource profile"

    partial, issue = _compatible_private_verifier_record(
        {"detail": {**base_detail, "verifier_supervisor_python": "python3"}},
        adapter,
        (image, "python3"),
    )
    assert partial is None
    assert issue == "lacks its supervisor-aware snapshot recipe"


def test_nondefault_container_never_accepts_legacy_receipt():
    from capability_pipeline.daytona_policy import verifier_bootstrap_sha256

    adapter = "a" * 64
    result = {
        "detail": {
            "verifier_isolation": "daytona-network-block-all",
            "verifier_adapter_sha256": adapter,
            "verifier_bootstrap_sha256": verifier_bootstrap_sha256(),
            "verifier_sandbox_id": "sandbox-1",
            "verifier_snapshot": "legacy-snapshot",
        }
    }
    record, issue = _compatible_private_verifier_record(
        result,
        adapter,
        ("registry.example/verifier@sha256:" + "b" * 64, "/opt/python/bin/python3"),
    )
    assert record is None
    assert issue == "lacks its supervisor-aware snapshot recipe"


def _incomplete_adversary_report(tmp_path, attempts):
    root = tmp_path / "runtime-trials" / "independent-attack-boundary"
    agent = root / "agent"
    agent.mkdir(parents=True)
    result = {
        "verifier_result": None,
        "exception_info": {
            "exception_type": "RuntimeError",
            "exception_message": "Incomplete GLM solver output: length",
        },
    }
    (root / "result.json").write_text(json.dumps(result))
    with (agent / "glm-requests.jsonl").open("w") as stream:
        for index, limit in enumerate(attempts, 1):
            stream.write(
                json.dumps(
                    {
                        "request_attempt": index,
                        "finish_reason": "length",
                        "usage": {
                            "completion_tokens": limit or 258_932,
                            "completion_tokens_details": {
                                "reasoning_tokens": limit or 258_932
                            },
                        },
                        "request": {"max_tokens": limit},
                    }
                )
                + "\n"
            )
    trial = root / "result.json"
    return {
        "state": "needs_adjudication_or_retry",
        "independent": True,
        "cases": [
            {
                "strategy": strategy,
                "steps": [{"step_index": 0, "step_name": "main"}],
            }
            for strategy in ("injection", "shortcut")
        ]
        + [
            {
                "strategy": "boundary",
                "error": "model_output_truncated",
                "trial_artifact": str(trial.relative_to(tmp_path)),
                "trial_sha256": hashlib.sha256(trial.read_bytes()).hexdigest(),
            }
        ],
    }


def test_bounded_incomplete_adversary_routes_output_exhaustion_without_adjudication(
    tmp_path,
):
    report = _incomplete_adversary_report(tmp_path, [32768, 65536])
    typed = _bounded_incomplete_adversary(report, tmp_path, ["main"])
    assert typed is not None
    assert typed["state"] == "bounded_model_output_truncated"
    assert typed["failed_strategies"] == ["boundary"]
    assert [entry["max_tokens"] for entry in typed["failures"][0]["attempts"]] == [
        32768,
        65536,
    ]
    assert typed["required_follow_up"] == "fresh_independent_adversary_suite_only"
    assert INCOMPLETE_ADVERSARY_ISSUE


def test_bounded_incomplete_adversary_accepts_128k_then_remaining_context_trace(
    tmp_path,
):
    report = _incomplete_adversary_report(tmp_path, [131072, None])
    typed = _bounded_incomplete_adversary(report, tmp_path, ["main"])
    assert typed is not None
    assert [entry["max_tokens"] for entry in typed["failures"][0]["attempts"]] == [
        131072,
        None,
    ]


def test_bounded_incomplete_adversary_accepts_phase_reset_after_planner_fallback(
    tmp_path,
):
    """Planner retries may end in a draft before finalizer attempts restart."""
    report = _incomplete_adversary_report(tmp_path, [131072, None])
    request_log = (
        tmp_path / "runtime-trials/independent-attack-boundary/agent/glm-requests.jsonl"
    )
    entries = [json.loads(line) for line in request_log.read_text().splitlines()]
    entries[:] = [
        {
            **entries[0],
            "phase": "boundary-planning",
            "finish_reason": "length",
        },
        {
            **entries[1],
            "phase": "boundary-planning",
            "finish_reason": "stop",
            "message": {"content": "Submit a duplicate public record."},
        },
        {
            **entries[0],
            "phase": "boundary-finalization",
            "finish_reason": "length",
        },
        {
            **entries[1],
            "phase": "boundary-finalization",
            "finish_reason": "length",
        },
    ]
    request_log.write_text("".join(json.dumps(entry) + "\n" for entry in entries))

    typed = _bounded_incomplete_adversary(report, tmp_path, ["main"])

    assert typed is not None
    failure = typed["failures"][0]
    assert failure["exhausted_phase"] == "boundary-finalization"
    assert [entry["max_tokens"] for entry in failure["attempts"]] == [131072, None]
    assert [
        entry["finish_reason"]
        for entry in failure["phase_history"]["boundary-planning"]
    ] == ["length", "stop"]


def test_bounded_incomplete_adversary_rejects_interleaved_phase_trace(tmp_path):
    report = _incomplete_adversary_report(tmp_path, [131072, None])
    request_log = (
        tmp_path / "runtime-trials/independent-attack-boundary/agent/glm-requests.jsonl"
    )
    entries = [json.loads(line) for line in request_log.read_text().splitlines()]
    entries[0]["phase"] = "boundary-planning"
    entries[0]["finish_reason"] = "stop"
    entries[0]["message"] = {"content": "A draft."}
    entries[1]["phase"] = "boundary-finalization"
    entries.append({**entries[0], "phase": "boundary-planning"})
    request_log.write_text("".join(json.dumps(entry) + "\n" for entry in entries))

    assert _bounded_incomplete_adversary(report, tmp_path, ["main"]) is None


def test_bounded_incomplete_adversary_rejects_nonmonotone_trace(tmp_path):
    report = _incomplete_adversary_report(tmp_path, [131072, 65536])
    assert _bounded_incomplete_adversary(report, tmp_path, ["main"]) is None


def test_bounded_incomplete_adversary_rejects_missing_budget_field(tmp_path):
    report = _incomplete_adversary_report(tmp_path, [131072, None])
    request_log = (
        tmp_path / "runtime-trials/independent-attack-boundary/agent/glm-requests.jsonl"
    )
    entries = [json.loads(line) for line in request_log.read_text().splitlines()]
    del entries[-1]["request"]["max_tokens"]
    request_log.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    assert _bounded_incomplete_adversary(report, tmp_path, ["main"]) is None


def test_adversary_retry_archives_only_output_exhaustion_then_reruns_full_gate(
    tmp_path, monkeypatch
):
    item = accepted()
    key = f"{item['proposal']['capability_id']}:{item['proposal']['slot']}"
    item_root = tmp_path / "items" / f"{_safe_name(key)}-{item['proposal_hash'][:12]}"
    item_root.mkdir(parents=True)
    report = _incomplete_adversary_report(item_root, [32768, 65536])
    report_path = item_root / "independent-adversary.json"
    report_path.write_text(json.dumps(report))
    (item_root / "harbor").mkdir()
    (item_root / "harbor/manifest.json").write_text('{"step_names":["main"]}')
    controls = {
        "cases": [
            {
                "id": "positive",
                "class": "positive",
                "source_author": "builder",
                "category": "known_correct",
                "expect": {"status": "graded", "reward_min": 0.8},
            },
            {
                "id": "negative",
                "class": "negative",
                "source_author": "builder",
                "category": "wrong",
                "expect": {"status": "graded", "reward_max": 0.2},
            },
        ]
    }
    task = item_root / "workspace/task"
    task.mkdir(parents=True)
    (task / "controls.json").write_text(json.dumps(controls))
    evidence_path = item_root / "runtime-evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "attestation": {
                    "adversary_artifact": "independent-adversary.json",
                    "adversary_artifact_sha256": hashlib.sha256(
                        report_path.read_bytes()
                    ).hexdigest(),
                    "oracle": {"authored": True},
                    "solver": {"state": "passed"},
                    "adversarial": {"authored_controls_executed": True},
                },
                "cases": [
                    {
                        "id": "positive",
                        "source_author": "builder",
                        "category": "known_correct",
                        "control_type": "independent_solver",
                        "result": {"status": "graded", "reward": 1.0},
                    },
                    {
                        "id": "negative",
                        "source_author": "builder",
                        "category": "wrong",
                        "control_type": "authored_adversarial_control",
                        "result": {"status": "graded", "reward": 0.0},
                    },
                ],
            }
        )
    )
    typed = _validated_pending_adversary_retry
    incomplete = _bounded_incomplete_adversary(report, item_root, ["main"])
    (item_root / "status.json").write_text(
        json.dumps(
            {
                "state": "failed",
                "issues": [
                    "attestation lacks an independent adversarial attack",
                    "independent adversary report needs adjudication or retry",
                    "adversary boundary lacks exact step coverage",
                ],
            }
        )
    )
    assert typed(item_root) == incomplete

    calls = []

    def rerun(*args, **kwargs):
        calls.append((args, kwargs))
        return {"state": "fresh_full_runtime"}

    monkeypatch.setattr("capability_pipeline.synthesis._synthesize_attempt", rerun)
    result = synthesize_one(
        item,
        tmp_path,
        FakeAgent(),
        None,
        None,
        14_400,
        retry_adversary=True,
    )
    assert result["state"] == "fresh_full_runtime"
    assert result["adversary_revalidation"]["policy"] == (
        "fresh_full_independent_adversary_suite"
    )
    assert len(calls) == 1
    archived = Path(result["adversary_revalidation"]["prior_attempt"])
    assert json.loads((archived / "status.json").read_text())["state"] == "failed"
    assert (archived / "runtime-evidence.json").is_file()
    assert (
        archived / "runtime-trials/independent-attack-boundary/result.json"
    ).is_file()
    repair_root = tmp_path / "repairs"
    assert not repair_root.exists() or not any(repair_root.rglob("*"))


def test_toolchain_commands_isolate_inherited_worker_environment(tmp_path, monkeypatch):
    # Exercise the actual child-process boundary, with a uv stand-in that
    # reports its environment. No dependencies are installed by this test.
    uv = tmp_path / "uv"
    uv.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$UV_PROJECT_ENVIRONMENT" "${VIRTUAL_ENV-unset}"\n'
    )
    uv.chmod(0o755)
    package = tmp_path / "taskcompendium with spaces"
    package.mkdir()
    (package / "uv.lock").write_text("version = 1\n")
    monkeypatch.setenv("VIRTUAL_ENV", "/app/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/app/.venv")
    toolchain = OfficialToolchain(package, str(uv))
    core = _run(toolchain._command("validate-and-lower"))
    runtime = _run(toolchain.runtime_command())
    assert core.returncode == runtime.returncode == 0
    assert core.stdout.splitlines() == [str(package / ".venv-core"), "unset"]
    assert runtime.stdout.splitlines() == [str(package / ".venv-runtime"), "unset"]
    assert os.environ["UV_PROJECT_ENVIRONMENT"] == "/app/.venv"
    assert os.environ["VIRTUAL_ENV"] == "/app/.venv"
    assert "--frozen" in toolchain.runtime_command()
    assert "daytona==0.200.2" in toolchain.runtime_command()
    assert "--prerelease=allow" in toolchain.runtime_command()
    assert "--extra" not in toolchain._command("validate-and-lower")


def test_provider_failure_requires_passed_health_receipt_before_revalidation(tmp_path):
    item_root = tmp_path / "item"
    trial = item_root / "runtime-trials/positive/result.json"
    trial.parent.mkdir(parents=True)
    trial.write_text(
        json.dumps(
            {
                "verifier_result": None,
                "exception_info": {
                    "exception_type": "GradingInfrastructureError",
                    "exception_message": "ProviderRateLimitExhausted",
                },
            }
        )
    )
    failures = _runtime_infrastructure_failures(
        item_root,
        {
            "state": "failed",
            "runtime_validated": None,
            "issues": ["runtime controls failed: provider unavailable"],
        },
    )
    assert failures == [
        {
            "trial": "positive",
            "artifact": str(trial),
            "artifact_sha256": hashlib.sha256(trial.read_bytes()).hexdigest(),
            "exception_type": "GradingInfrastructureError",
            "provider_cause": "ProviderRateLimitExhausted",
        }
    ]

    receipt_path = tmp_path / "health.json"
    receipt = {
        "schema_version": "capability-daytona-health-v1",
        "state": "passed",
        "snapshot": "snapshot-a",
        "snapshot_id": "snapshot-id",
        "sandbox_id": "sandbox-id",
        "network_block_all": True,
        "network_block_all_requested": True,
        "network_block_all_observed": True,
        "provisioning_attempts": [{"attempt": 1, "state": "created"}],
        "deleted": True,
        "lookup_after_delete": "not_found",
        "deletion_lookups": [
            {"elapsed_seconds": 0.0, "state": "present"},
            {"elapsed_seconds": 5.0, "state": "not_found"},
        ],
        "completed_at": datetime.now(UTC).isoformat(),
    }
    receipt_path.write_text(json.dumps(receipt))
    assert _validate_infrastructure_health_receipt(receipt_path) == receipt
    receipt["deleted"] = False
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(SynthesisError, match="did not pass create/delete"):
        _validate_infrastructure_health_receipt(receipt_path)

    receipt["deleted"] = True
    receipt["completed_at"] = (datetime.now(UTC) - timedelta(hours=3)).isoformat()
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(SynthesisError, match="not fresh"):
        _validate_infrastructure_health_receipt(receipt_path)


def test_fresh_failed_task_repairs_and_revalidates_in_same_worker(tmp_path, monkeypatch):
    item = accepted()
    name = f"cap.test-1-{item['proposal_hash'][:12]}"
    item_root = tmp_path / "items" / name
    attempts = []

    def synthesize_attempt(*_args):
        state = "failed" if not attempts else "quality_accepted"
        result = {"state": state, "issues": ["invalid task bundle: bad grader"] if not attempts else [], "item_root": str(item_root)}
        item_root.mkdir(parents=True, exist_ok=True)
        (item_root / "status.json").write_text(json.dumps(result))
        attempts.append(state)
        return result

    def repair(_item_root, repair_root, _agent, _feedback, *, source=None):
        repair_root.mkdir(parents=True)
        result = {"state": "ready_for_validation", "changed_files": ["workspace/task/grader.py"]}
        (repair_root / "result.json").write_text(json.dumps(result))
        return result

    monkeypatch.setenv("CAPABILITY_MAX_REPAIR_ROUNDS", "2")
    monkeypatch.setattr("capability_pipeline.synthesis._synthesize_attempt", synthesize_attempt)
    monkeypatch.setattr("capability_pipeline.synthesis._repair_feedback", lambda *_: {"issues": ["bad grader"]})
    monkeypatch.setattr("capability_pipeline.repair.run_repair", repair)
    result = synthesize_one(item, tmp_path, FakeAgent(), None, None, 30)
    assert result["state"] == "quality_accepted"
    assert attempts == ["failed", "quality_accepted"]
    assert [record["round"] for record in result["repairs"]] == [1]
    failure = Path(result["repairs"][0]["source_failure"]["artifact"])
    assert json.loads(failure.read_text())["issues"] == ["invalid task bundle: bad grader"]
    assert (tmp_path / "repair-history" / name / "attempt-1/status.json").is_file()


def test_failed_repair_invocation_consumes_global_round_budget(tmp_path, monkeypatch):
    item = accepted()
    name = f"cap.test-1-{item['proposal_hash'][:12]}"
    item_root = tmp_path / "items" / name
    calls = []

    def synthesize_attempt(*_args):
        result = {"state": "failed", "issues": ["invalid task bundle: bad grader"], "item_root": str(item_root)}
        item_root.mkdir(parents=True, exist_ok=True)
        (item_root / "status.json").write_text(json.dumps(result))
        return result

    def repair(*_args, **_kwargs):
        calls.append("repair")
        raise RuntimeError("model request failed")

    monkeypatch.setenv("CAPABILITY_MAX_REPAIR_ROUNDS", "1")
    monkeypatch.setattr("capability_pipeline.synthesis._synthesize_attempt", synthesize_attempt)
    monkeypatch.setattr("capability_pipeline.synthesis._repair_feedback", lambda *_: {"issues": ["bad grader"]})
    monkeypatch.setattr("capability_pipeline.repair.run_repair", repair)
    first = synthesize_one(item, tmp_path, FakeAgent(), None, None, 30)
    assert "could not run" in first["repair_issue"]
    assert len(calls) == 1
    assert (tmp_path / "repair-budget" / name / "attempt-1/source-status.json").is_file()
    second = synthesize_one(item, tmp_path, FakeAgent(), None, None, 30)
    assert second["state"] == "failed"
    assert len(calls) == 1


def test_fresh_repair_stops_when_revalidation_hits_ungraded_runtime_failure(tmp_path, monkeypatch):
    item = accepted()
    name = f"cap.test-1-{item['proposal_hash'][:12]}"
    item_root = tmp_path / "items" / name
    attempts = []
    repairs = []

    def synthesize_attempt(*_args):
        issue = "invalid task bundle: bad grader" if not attempts else "runtime controls failed: ProviderRateLimitExhausted"
        result = {"state": "failed", "issues": [issue], "item_root": str(item_root)}
        item_root.mkdir(parents=True, exist_ok=True)
        (item_root / "status.json").write_text(json.dumps(result))
        attempts.append(issue)
        return result

    def repair(_item_root, repair_root, _agent, _feedback, *, source=None):
        repairs.append(repair_root)
        repair_root.mkdir(parents=True)
        result = {"state": "ready_for_validation", "changed_files": ["workspace/task/grader.py"]}
        (repair_root / "result.json").write_text(json.dumps(result))
        return result

    monkeypatch.setenv("CAPABILITY_MAX_REPAIR_ROUNDS", "2")
    monkeypatch.setattr("capability_pipeline.synthesis._synthesize_attempt", synthesize_attempt)
    monkeypatch.setattr("capability_pipeline.synthesis._repair_feedback", lambda *_: {"issues": ["bad grader"]})
    monkeypatch.setattr("capability_pipeline.repair.run_repair", repair)
    result = synthesize_one(item, tmp_path, FakeAgent(), None, None, 30)
    assert result["state"] == "failed"
    assert result["issues"] == ["runtime controls failed: ProviderRateLimitExhausted"]
    assert len(attempts) == 2 and len(repairs) == 1
    assert [record["round"] for record in result["repairs"]] == [1]
    assert not (tmp_path / "repair-budget" / name / "attempt-2").exists()
    second = synthesize_one(item, tmp_path, FakeAgent(), None, None, 30)
    assert second["state"] == "failed"
    assert len(repairs) == 1


@pytest.mark.parametrize("result", [
    {"state": "pending_solver_adjudication", "issues": ["solver missed"]},
    {"state": "pending_adversary_retry", "issues": ["attack output exhausted"]},
    {"state": "failed", "issues": ["runtime controls failed: DaytonaRateLimitError"]},
    {"state": "failed", "issues": ["adversary boundary lacks exact step coverage"]},
    {"state": "pending_quality_review", "quality_review": {"state": "pending"}},
])
def test_fresh_repair_holds_ungraded_and_adjudication_outcomes(result):
    assert _fresh_construction_repair_allowed(result) is False



def test_health_receipt_requires_bounded_hashable_deletion_observations(tmp_path):
    receipt_path = tmp_path / "health.json"
    receipt = {
        "schema_version": "capability-daytona-health-v1",
        "state": "passed",
        "snapshot": "snapshot-a",
        "snapshot_id": "snapshot-id",
        "sandbox_id": "sandbox-id",
        "network_block_all": True,
        "network_block_all_requested": True,
        "network_block_all_observed": True,
        "provisioning_attempts": [{"attempt": 1, "state": "created"}],
        "deleted": True,
        "lookup_after_delete": "not_found",
        "deletion_lookups": [{"elapsed_seconds": 61.0, "state": "not_found"}],
        "completed_at": datetime.now(UTC).isoformat(),
    }
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(SynthesisError, match="observations are invalid"):
        _validate_infrastructure_health_receipt(receipt_path)

    receipt["deletion_lookups"] = []
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(SynthesisError, match="did not pass create/delete"):
        _validate_infrastructure_health_receipt(receipt_path)


def test_infrastructure_revalidation_archives_failure_without_repair(
    tmp_path, monkeypatch
):
    item = accepted()
    name = f"cap.test-1-{item['proposal_hash'][:12]}"
    item_root = tmp_path / "items" / name
    trial = item_root / "runtime-trials/positive/result.json"
    trial.parent.mkdir(parents=True)
    trial.write_text(
        json.dumps(
            {
                "verifier_result": None,
                "exception_info": {
                    "exception_type": "GradingInfrastructureError",
                    "exception_message": "ProviderRateLimitExhausted",
                },
            }
        )
    )
    (item_root / "status.json").write_text(
        json.dumps(
            {
                "state": "failed",
                "issues": ["runtime controls failed: provider failure"],
            }
        )
    )
    for round_ in (1, 2):
        (tmp_path / "repairs" / name / f"attempt-{round_}").mkdir(parents=True)
    health_path = tmp_path / "health.json"
    health_path.write_text("{}")
    health = {"snapshot": "snapshot-a", "sandbox_id": "health-sandbox"}

    def rerun(*_args, **_kwargs):
        return {"state": "pending_quality_review", "item_root": str(item_root)}

    monkeypatch.setattr("capability_pipeline.synthesis._synthesize_attempt", rerun)
    result = synthesize_one(
        item,
        tmp_path,
        FakeAgent(),
        None,
        None,
        30,
        infrastructure_health=health,
        infrastructure_health_path=health_path,
    )

    record = result["infrastructure_revalidation"]
    assert record["attempt"] == 1
    assert record["health_snapshot"] == "snapshot-a"
    history = tmp_path / "infrastructure-history" / name / "revalidation-1"
    assert (history / "runtime-trials/positive/result.json").is_file()
    assert Path(record["prior_failures"][0]["artifact"]).is_file()
    assert record["prior_failures"][0]["original_artifact"] == str(trial)
    assert not (item_root / "runtime-trials").exists()
    assert len(list((tmp_path / "repairs" / name).glob("attempt-*"))) == 2


def test_toolchain_resolve_reports_rejected_candidate_reason(tmp_path):
    candidate = tmp_path / "taskcompendium"
    candidate.mkdir()
    (candidate / "pyproject.toml").write_text("[project]\nname='wrong-source'\n")
    with pytest.raises(SynthesisError) as failure:
        OfficialToolchain.resolve(tmp_path / "out", str(candidate))
    message = str(failure.value)
    assert str(candidate) in message
    assert "TaskCompendium source hash mismatch: pyproject.toml" in message


def test_toolchain_file_set_rejects_appledouble_members(tmp_path):
    package = tmp_path / "taskcompendium"
    files = {
        "pyproject.toml": b"[project]\nname='fixture'\n",
        "uv.lock": b"version = 1\n",
        "src/module.py": b"VALUE = 1\n",
    }
    for name, data in files.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    lock = {
        "files": {
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()
        }
    }
    OfficialToolchain._verify(package, lock)
    (package / "src/._module.py").write_bytes(b"AppleDouble metadata")
    with pytest.raises(SynthesisError, match=r"extra=\['src/\._module\.py'\]"):
        OfficialToolchain._verify(package, lock)


def test_toolchain_overlay_rejects_nonpatched_drift_and_extra_source_members(
    tmp_path, monkeypatch
):
    """The runtime overlay pins the complete source tree, not only its patch hunk."""
    from capability_pipeline import synthesis

    source_lock = tmp_path / "source.lock.json"
    project = tmp_path / "project"
    overlay = tmp_path / "overlay"
    names = {
        "pyproject.toml": b"[project]\nname = 'fixture'\n",
        "uv.lock": b"version = 1\n",
        "src/untouched.py": b"unchanged\n",
        "src/taskcompendium/harbor/verifier.py": b"base verifier\n",
        "src/taskcompendium/harbor/runner.py": b"base runner\n",
        "src/taskcompendium/lowering.py": b"base lowering\n",
    }
    patched = {
        "src/taskcompendium/harbor/verifier.py": b"patched verifier\n",
        "src/taskcompendium/harbor/runner.py": b"patched runner\n",
        "src/taskcompendium/lowering.py": b"patched lowering\n",
    }
    for name, raw in {**names, **patched}.items():
        path = overlay / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    expected = {
        name: {
            "base_sha256": hashlib.sha256(names[name]).hexdigest(),
            "patched_sha256": hashlib.sha256(raw).hexdigest(),
        }
        for name, raw in patched.items()
    }
    source_lock.write_text(
        json.dumps(
            {"files": {name: hashlib.sha256(raw).hexdigest() for name, raw in names.items()}}
        )
    )
    extension = project / "vendor/task_spec"
    extension.mkdir(parents=True)
    patch = project / "overlay.patch"
    patch.write_text("approved patch\n")
    (extension / "composite_extension.lock.json").write_text(
        json.dumps(
            {
                "files": expected,
                "patch": "overlay.patch",
                "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
            }
        )
    )
    monkeypatch.setattr(synthesis, "SOURCE_LOCK", source_lock)
    monkeypatch.setattr(synthesis, "PROJECT_ROOT", project)
    monkeypatch.setattr(
        OfficialToolchain, "_expected_overlay_files", staticmethod(lambda: expected)
    )

    OfficialToolchain._verify_overlay(overlay)
    (overlay / "src/untouched.py").write_text("mutated\n")
    with pytest.raises(SynthesisError, match="source hash mismatch"):
        OfficialToolchain._verify_overlay(overlay)
    (overlay / "src/untouched.py").write_bytes(names["src/untouched.py"])
    (overlay / "src/extra.py").write_text("unexpected\n")
    with pytest.raises(SynthesisError, match="archive file set mismatch"):
        OfficialToolchain._verify_overlay(overlay)


def test_repair_feedback_includes_cross_capability_audit_by_proposal_hash(
    tmp_path, monkeypatch
):
    project = tmp_path / "project"
    audits = project / "docs/audits"
    audits.mkdir(parents=True)
    (project / "docs/task_contract.md").write_text("current contract")
    target = accepted()
    target["proposal_hash"] = "a" * 64
    (audits / "different_prefix_joint_audit.md").write_text(
        f"This audit binds proposal hash `{target['proposal_hash']}`."
    )
    (audits / "unrelated.md").write_text("This audit concerns another proposal.")
    item_root = tmp_path / "item"
    (item_root / "workspace/task").mkdir(parents=True)
    (item_root / "workspace/task/specification.json").write_text("{}")
    monkeypatch.setattr("capability_pipeline.synthesis.PROJECT_ROOT", project)
    feedback = _repair_feedback(
        target,
        {"state": "failed", "issues": ["fixture"], "item_root": str(item_root)},
        1,
        item_root,
    )
    assert [audit["path"] for audit in feedback["measured_audits"]] == [
        "docs/audits/different_prefix_joint_audit.md"
    ]


def test_repair_feedback_rebases_relocated_status_to_current_item(
    tmp_path, monkeypatch
):
    project = tmp_path / "project"
    (project / "docs/audits").mkdir(parents=True)
    (project / "docs/task_contract.md").write_text("current contract")
    item_root = tmp_path / "restored/item"
    task = item_root / "workspace/task"
    task.mkdir(parents=True)
    (task / "specification.json").write_text(
        json.dumps(
            {"steps": [{"verifier": {"kind": "code_answer", "answer_path": "a.json"}}]}
        )
    )
    monkeypatch.setattr("capability_pipeline.synthesis.PROJECT_ROOT", project)
    feedback = _repair_feedback(
        accepted(),
        {
            "state": "failed",
            "issues": ["fixture"],
            "item_root": "/obsolete/remote/pilot/items/item",
        },
        1,
        item_root,
    )
    assert "quality-code-mutations" in feedback["required_quality_conditions"]


def proposal(environment="reasoning", verification="simple", sessions=None):
    return {
        "capability_id": "cap.test",
        "slot": 1,
        "status": "proposed",
        "null_reason": None,
        "title": "A real task",
        "task_family": "analysis",
        "environment": environment,
        "verification": verification,
        "capability_alignment": "Exercises the capability directly.",
        "environment_rationale": "The task needs this interaction surface.",
        "workflow": "Inspect inputs and produce the requested artifact.",
        "task_brief": "Complete the supplied work item.",
        "inputs": ["fixture"],
        "deliverables": ["answer"],
        "constraints": ["preserve evidence"],
        "difficulty_drivers": ["edge cases"],
        "grounding": {"known_facts": [], "research_needed": [], "sources": []},
        "environment_spec": {
            "initial_state": "clean",
            "tools": [],
            "reset": "fresh",
            "dependency_strategy": "pinned",
            "resource_estimate": "measured later",
        },
        "verification_spec": {
            "observable_success": "correct output",
            "grader_design": "exact",
            "positive_controls": ["gold"],
            "negative_controls": ["wrong"],
            "anti_shortcuts": ["private gold"],
            "rubric": ["not applicable: exact"],
        },
        "builder_plan": sessions
        or [
            {
                "session": "s1",
                "depends_on": [],
                "goal": "build fixtures",
                "handoff_artifacts": ["fixtures"],
                "acceptance_checks": ["fixture exists"],
            },
            {
                "session": "s2",
                "depends_on": ["s1"],
                "goal": "build task",
                "handoff_artifacts": ["task bundle"],
                "acceptance_checks": ["bundle exists"],
            },
        ],
        "validation_plan": ["execute positive and negative controls"],
        "risks": [
            {
                "risk": "bad fixture",
                "mitigation": "inspect",
                "abandon_if": "no ground truth",
            }
        ],
        "data_policy": {
            "provenance": "synthetic",
            "license": "CC0",
            "split_group": "fixture-1",
            "contamination_check": "hash",
            "private_evaluator_data": ["gold"],
        },
    }


def accepted(**kwargs):
    value = proposal(**kwargs)
    scores = {
        name: 4
        for name in (
            "realism",
            "alignment",
            "specificity",
            "reward_validity",
            "environment_fit",
            "diversity",
            "source_honesty",
        )
    }
    review = {
        "verdict": "accept",
        "scores": scores,
        "critical_failures": [],
        "required_changes": [],
    }
    record = {
        "capability_id": value["capability_id"],
        "capability": {"id": value["capability_id"]},
    }
    return {
        "proposal": value,
        "review": review,
        "proposal_hash": digest(value),
        "provenance": {
            "catalog_source": {"path": "catalog.json", "sha256": "a" * 64},
            "capability_record": record,
            "capability_record_hash": digest(record),
        },
    }


def test_build_checklist_prefers_exact_proposal_hash(tmp_path, monkeypatch):
    fallback = tmp_path / "docs" / "build_acceptance_001.md"
    exact = tmp_path / "docs" / "build_acceptance" / f"{'a' * 64}.md"
    exact.parent.mkdir(parents=True)
    fallback.write_text("fallback")
    exact.write_text("exact")
    monkeypatch.setattr("capability_pipeline.synthesis.PROJECT_ROOT", tmp_path)
    assert _build_checklist_path("a" * 64) == exact
    assert _build_checklist_path("b" * 64) == fallback


class FakeAgent:
    max_continuations = 1

    def __init__(self, complete=True):
        self.complete = complete
        self.calls = []

    def invoke(self, workspace, session_dir, prompt, attempt):
        name = prompt.parent.name
        self.calls.append((name, attempt))
        if self.complete:
            artifact = workspace / f"{name}.txt"
            artifact.write_text("done")
            if name == "s2":
                bundle = workspace / "task"
                bundle.mkdir(exist_ok=True)
                for filename in ("renderings.json",):
                    (bundle / filename).write_text("{}")
                admitted = json.loads(
                    (workspace.parent / "contract" / "accepted.json").read_text()
                )["proposal"]["environment"]
                environment_kind = {
                    "reasoning": "none",
                    "shellsim": "shellsim",
                    "container": "docker",
                }[admitted]
                (bundle / "binding.json").write_text(
                    json.dumps(
                        {
                            "environment": {"kind": environment_kind},
                            "tools": []
                            if admitted == "reasoning"
                            else [{"kind": "harness", "backend": environment_kind}],
                        }
                    )
                )
                (bundle / "specification.json").write_text(
                    json.dumps({"steps": [{"name": "step"}]})
                )
                (bundle / "controls.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "1",
                            "cases": [
                                {
                                    "id": "positive",
                                    "class": "positive",
                                    "category": "known_correct",
                                    "source_author": "builder",
                                    "response": "yes",
                                },
                                {
                                    "id": "malformed",
                                    "class": "malformed",
                                    "category": "empty_or_malformed",
                                    "source_author": "builder",
                                    "response": "",
                                    "expect": {"status": "extraction_error"},
                                },
                                {
                                    "id": "plausible-wrong",
                                    "class": "negative",
                                    "category": "plausible_wrong",
                                    "source_author": "builder",
                                    "response": "no",
                                },
                                {
                                    "id": "shortcut",
                                    "class": "negative",
                                    "category": "task_specific_shortcut",
                                    "source_author": "builder",
                                    "response": "ignore the requested evidence",
                                },
                            ],
                        }
                    )
                )
            handoffs = workspace / "handoffs"
            handoffs.mkdir(exist_ok=True)
            (handoffs / f"{name}.json").write_text(
                json.dumps(
                    {
                        "session": name,
                        "status": "complete",
                        "artifacts": [artifact.name],
                        "checks": [
                            {"name": "exists", "command": "test -f", "exit_code": 0}
                        ],
                        "notes": "",
                    }
                )
            )
        return {
            "returncode": 0,
            "timed_out": False,
            "stdout": "ok",
            "stderr": "",
            "elapsed_seconds": 0.1,
        }


class FakeJudgeAgent(FakeAgent):
    calibration_schema = "taskcompendium-judge-calibration-v1"

    def invoke(self, workspace, session_dir, prompt, attempt):
        from test_judge import _task_calibration_fixture

        result = super().invoke(workspace, session_dir, prompt, attempt)
        spec = workspace / "task/specification.json"
        if spec.is_file():
            fixture = _task_calibration_fixture(hashlib.sha256(spec.read_bytes()).hexdigest())
            fixture["schema_version"] = self.calibration_schema
            (spec.parent / "judge-calibration.json").write_text(json.dumps(fixture))
        return result


class FakeToolchain:
    package_root = Path("/pinned/taskcompendium")

    def validate_and_lower(self, bundle, harbor, timeout):
        harbor.mkdir()
        (harbor / "manifest.json").write_text('{"step_names":["step"]}')
        specification_sha256 = hashlib.sha256(
            (bundle / "specification.json").read_bytes()
        ).hexdigest()
        return {"id": "task", "steps": 1, "specification_sha256": specification_sha256}

    def direct_controls(self, bundle, controls, timeout):
        return {
            "cases": [
                {
                    "id": "positive",
                    "class": "positive",
                    "result": {"status": "graded", "reward": 1.0},
                },
                {
                    "id": "malformed",
                    "class": "malformed",
                    "result": {"status": "extraction_error", "reward": None},
                },
                {
                    "id": "plausible-wrong",
                    "class": "negative",
                    "result": {"status": "graded", "reward": 0.0},
                },
                {
                    "id": "shortcut",
                    "class": "negative",
                    "result": {"status": "graded", "reward": 0.0},
                },
            ]
        }


def test_load_accepted_checks_proposal_fingerprint(tmp_path):
    item = accepted()
    item["proposal_hash"] = "0" * 64
    path = tmp_path / "accepted.json"
    path.write_text(json.dumps([item]))
    with pytest.raises(SynthesisError, match="hash mismatch"):
        load_accepted(path)


def test_load_accepted_binds_optional_learning_progression(tmp_path):
    item = accepted()
    provenance = item["provenance"]
    provenance["catalog_source"]["catalog_version"] = "1.0"
    context = {
        "catalog_version": "1.0",
        "edges": [{"dependent_id": item["proposal"]["capability_id"], "prerequisite_id": "cap.previous"}],
    }
    provenance["learning_progression"] = context
    provenance["learning_progression_hash"] = digest(context)
    path = tmp_path / "accepted.json"
    path.write_text(json.dumps([item]))
    assert len(load_accepted(path)) == 1
    provenance["learning_progression"]["edges"][0]["dependent_id"] = "other"
    provenance["learning_progression_hash"] = digest(provenance["learning_progression"])
    path.write_text(json.dumps([item]))
    with pytest.raises(SynthesisError, match="learning progression provenance mismatch"):
        load_accepted(path)
    provenance.pop("learning_progression_hash")
    path.write_text(json.dumps([item]))
    with pytest.raises(SynthesisError, match="learning progression provenance mismatch"):
        load_accepted(path)


def test_load_accepted_rejects_revoked_hash_except_for_repair(tmp_path, monkeypatch):
    item = accepted()
    path = tmp_path / "accepted.json"
    path.write_text(json.dumps([item]))
    record = {
        "proposal_hash": item["proposal_hash"],
        "capability_id": item["proposal"]["capability_id"],
        "slot": item["proposal"]["slot"],
        "state": "revoked",
        "reason_code": "audit_failure",
        "reason": "deterministic audit disproved the reward contract",
        "allowed_use": "admission_repair_input_only",
        "evidence": [{"path": "audit.md", "sha256": "a" * 64}],
    }
    monkeypatch.setattr(
        "capability_pipeline.synthesis._revoked_proposals",
        lambda: {item["proposal_hash"]: record},
    )
    with pytest.raises(SynthesisError, match="proposal is revoked"):
        load_accepted(path)
    assert load_accepted(path, allow_revoked=True) == [item]


def test_review005_revocations_block_affected_and_preserve_unaffected(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    source = project_root / "runs/proposal-review-005/terminal-compact/accepted.json"
    entries = json.loads(source.read_text())
    by_hash = {item["proposal_hash"]: item for item in entries}
    revoked = {
        "cf260cfc0cc2d3186a2865a18cc1d53d8de36d6d36868bd7e6b1150b13a9ea78",
        "0f46b1fc0d870903fdb5880b090066981949cb481ad9f43c786f168609279063",
        "96822724f7a6bcd60296ed59b4826bc44798285facd2818dbf307ef1ba2a7ab3",
        "c92389454333089072eb792f8cc612a5f070667a0c779379414ddb3bfd1931b0",
        "bf80328074a70e058c501b778f2052a24ef85dda801e3e0ae562586a93e8caf8",
        "dfa627efaf07d900b751a8b7b449b8ca2ec704d7410a895413ab6f8bfc1ad4b9",
    }
    assert revoked <= by_hash.keys()
    for proposal_hash in sorted(revoked):
        path = tmp_path / f"revoked-{proposal_hash}.json"
        path.write_text(json.dumps([by_hash[proposal_hash]]))
        with pytest.raises(SynthesisError, match="proposal is revoked"):
            load_accepted(path)

    unaffected_hash = "71b7d17bb21cfb58678c6f2c2e625f6e2d4d763f43ef6574bbb6fc4c5815b34a"
    unaffected_path = tmp_path / "unaffected.json"
    unaffected_path.write_text(json.dumps([by_hash[unaffected_hash]]))
    assert load_accepted(unaffected_path) == [by_hash[unaffected_hash]]

    audit = project_root / "docs/audits/proposal_review_005_new_accepts_sample.md"
    audit_sha256 = hashlib.sha256(audit.read_bytes()).hexdigest()
    registry = json.loads((project_root / "data/revocations.json").read_text())
    records = {
        record["proposal_hash"]: record
        for record in registry["records"]
        if record["proposal_hash"] in revoked
    }
    assert records.keys() == revoked
    assert all(
        record["evidence"]
        == [
            {
                "path": "docs/audits/proposal_review_005_new_accepts_sample.md",
                "sha256": audit_sha256,
            }
        ]
        for record in records.values()
    )


def test_c02_clarification_is_repair_only_and_preserves_lineage():
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / "data/c02_slot10_clarification_input.json"
    source_hash = "d5aa409874d71bb4ea4f46bcc7627020ccc8d09948524b2255b22cd29d36c683"
    with pytest.raises(SynthesisError, match="unresolved_rubric_denominator"):
        load_accepted(path)
    items = load_accepted(
        path,
        allow_pending_admission=True,
        allow_revoked=True,
    )
    assert len(items) == 1
    item = items[0]
    assert item["proposal_hash"] == source_hash
    assert item["admission"]["state"] == "pending"
    clarification = item["construction_context"]["score_contract_clarification"]
    assert clarification["normalization"] == "raw_score * 24 / 39"
    assert clarification["minimum_integer_raw_score"] == 33
    assert clarification["judge_protocol"] == {
        "mode": "two_then_third",
        "initial_samples": 2,
        "judge_model_policy_samples": 2,
        "disagreement_tolerance": 0.0,
        "third_trigger": "any criterion disagreement greater than tolerance",
        "adjudicator_samples": 1,
        "adjudicator_scope": "complete thirty-nine-criterion native pass",
        "resolution": "per-criterion median across three binary judgments",
        "no_disagreement": "common initial vector is final",
        "required_result_detail": [
            "initial_judgments",
            "disagreed_indices",
            "adjudicator_judgments",
            "judge_call_count",
            "resolved_criterion_scores",
        ],
        "missing_or_nonmonotone": "invalid_task_with_null_reward",
        "infrastructure_failure": "infra_error_with_null_reward",
        "forbid_total_or_vector_mean": True,
    }
    lineage = item["construction_context"]["clarification_lineage"]
    assert lineage["source_proposal_hash"] == source_hash
    assert len(lineage["source_admission"]["history"]) == 2


def test_load_accepted_checks_frozen_capability_provenance(tmp_path):
    item = accepted()
    item["provenance"]["capability_record"]["capability_id"] = "cap.other"
    path = tmp_path / "accepted.json"
    path.write_text(json.dumps([item]))
    with pytest.raises(SynthesisError, match="record hash mismatch"):
        load_accepted(path)


def test_load_accepted_rejects_score_outside_review_scale(tmp_path):
    item = accepted()
    item["review"]["scores"]["realism"] = 6
    path = tmp_path / "accepted.json"
    path.write_text(json.dumps([item]))
    with pytest.raises(SynthesisError, match="scores do not support acceptance"):
        load_accepted(path)


def test_load_accepted_requires_individual_construction_admission(tmp_path):
    item = accepted()
    item["construction_context"] = {"source_files": [{"sha256": "b" * 64}]}
    item["admission"] = {
        "state": "pending",
        "scope": "individual_construction",
    }
    path = tmp_path / "accepted.json"
    path.write_text(json.dumps([item]))
    with pytest.raises(SynthesisError, match="construction admission"):
        load_accepted(path)
    assert load_accepted(path, allow_pending_admission=True) == [item]
    item["admission"].update(
        {
            "state": "accepted",
            "source_proposal_hash": item["proposal_hash"],
            "portfolio_certified": False,
            "runtime_certified": False,
            "history": [
                {
                    "round": 0,
                    "proposal_hash": item["proposal_hash"],
                    "review": item["review"],
                }
            ],
        }
    )
    path.write_text(json.dumps([item]))
    assert load_accepted(path) == [item]
    item["admission"]["history"][-1]["proposal_hash"] = "0" * 64
    path.write_text(json.dumps([item]))
    with pytest.raises(SynthesisError, match="hash-bound"):
        load_accepted(path)


def test_build_acceptance_requires_exact_hash_bound_checklist(tmp_path):
    item = accepted()
    bundle = tmp_path / "task"
    evidence = bundle / "evidence" / "measurement.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text('{"measured":true}\n')
    checklist = tmp_path / "checklist.md"
    checklist.write_text(
        "# Checks\n\n## cap.test, slot 1 — reasoning / simple\n\n"
        "- `cap-test-measurement`: retain evidence.\n"
    )
    record = {
        "schema_version": "capability-build-acceptance-v1",
        "identity": {
            "capability_id": "cap.test",
            "slot": 1,
            "proposal_hash": item["proposal_hash"],
            "capability_record_hash": item["provenance"]["capability_record_hash"],
            "catalog_sha256": item["provenance"]["catalog_source"]["sha256"],
        },
        "checks": [
            {
                "id": "cap-test-measurement",
                "state": "passed",
                "claim": "the measured behavior meets the bound",
                "artifacts": [
                    {
                        "path": "evidence/measurement.json",
                        "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                    }
                ],
                "summary": "1/1 deterministic measurement passed",
            }
        ],
    }
    (bundle / "build-acceptance.json").write_text(json.dumps(record))
    assert _build_acceptance_issues(item, bundle, checklist) == []
    record["checks"][0]["artifacts"][0]["sha256"] = "0" * 64
    (bundle / "build-acceptance.json").write_text(json.dumps(record))
    assert any(
        "wrong digest" in issue
        for issue in _build_acceptance_issues(item, bundle, checklist)
    )


def test_controls_require_each_adversarial_category_per_step():
    document = {
        "schema_version": "1",
        "cases": [
            {
                "id": "positive",
                "class": "positive",
                "category": "known_correct",
                "source_author": "builder",
                "response": "yes",
            },
            {
                "id": "malformed",
                "class": "malformed",
                "category": "empty_or_malformed",
                "source_author": "builder",
                "response": "",
                "expect": {"status": "extraction_error"},
            },
            {
                "id": "wrong",
                "class": "negative",
                "category": "plausible_wrong",
                "source_author": "builder",
                "response": "no",
            },
        ],
    }
    with pytest.raises(SynthesisError, match="shortcut control category"):
        _validate_controls(document, 1)


@pytest.mark.parametrize("reward", [True, float("nan"), float("inf"), -0.1, 1.1])
def test_controls_reject_non_finite_boolean_and_out_of_range_rewards(reward):
    controls = {
        "cases": [
            {
                "id": "positive",
                "class": "positive",
                "source_author": "builder",
                "category": "known_correct",
            }
        ]
    }
    evidence = {
        "cases": [
            {
                "id": "positive",
                "result": {"status": "graded", "reward": reward},
            }
        ]
    }
    passed, issues = _controls_pass(controls, evidence)
    assert not passed
    assert issues


def test_controls_reject_duplicate_evidence_ids():
    controls = {
        "cases": [
            {
                "id": "positive",
                "class": "positive",
                "source_author": "builder",
                "category": "known_correct",
            }
        ]
    }
    record = {
        "id": "positive",
        "result": {"status": "graded", "reward": 1.0},
    }
    passed, issues = _controls_pass(controls, {"cases": [record, record]})
    assert not passed
    assert "duplicate" in issues[0]


def test_direct_controls_do_not_claim_harbor_validation(tmp_path):
    agent = FakeAgent()
    result = synthesize_one(accepted(), tmp_path, agent, FakeToolchain(), None, 30)
    assert result["state"] == "controls_passed_pending_rollout"
    assert agent.calls == [("s1", 0), ("s2", 0)]
    assert not (tmp_path / "validated").exists()
    evidence = json.loads((Path(result["runtime_evidence"])).read_text())
    assert evidence["cases"][0]["result"]["reward"] == 1.0


def test_custom_image_publication_hold_stops_before_lowering(tmp_path, monkeypatch):
    class NoLowering(FakeToolchain):
        def validate_and_lower(self, *args):
            pytest.fail("image publication hold must stop before lowering")

    monkeypatch.setattr(
        "capability_pipeline.image_pipeline.process_image_construction",
        lambda **kwargs: {"state": "pending_publication", "reason": "isolated publisher receipt absent"},
    )
    result = synthesize_one(accepted(), tmp_path, FakeAgent(), NoLowering(), None, 30)
    assert result["state"] == "pending_image_publication"
    assert result["issues"] == ["isolated publisher receipt absent"]
    assert not (Path(result["item_root"]) / "harbor").exists()
    assert "export" not in result


def test_custom_image_invalid_request_returns_semantic_bundle_feedback(tmp_path, monkeypatch):
    class NoLowering(FakeToolchain):
        def validate_and_lower(self, *args):
            pytest.fail("invalid image request must stop before lowering")

    monkeypatch.setattr(
        "capability_pipeline.image_pipeline.process_image_construction",
        lambda **kwargs: {"state": "repairable", "reason": "invalid_builder_image_request",
                          "issues": ["capture request roles differ from custom task pointers"]},
    )
    result = _synthesize_attempt(accepted(), tmp_path, FakeAgent(), NoLowering(), None, 30)
    assert result["state"] == "failed"
    assert result["issues"] == [
        "invalid task bundle: capture request roles differ from custom task pointers"
    ]
    assert not (Path(result["item_root"]) / "harbor").exists()


def test_custom_image_migrated_pointer_reaches_lowering(tmp_path, monkeypatch):
    image = "registry.example/task@sha256:" + "a" * 64
    observed = []

    class InspectLowering(FakeToolchain):
        def validate_and_lower(self, bundle, harbor, timeout):
            specification = json.loads((bundle / "specification.json").read_text())
            observed.append(specification["requirements"]["state"]["image"])
            return super().validate_and_lower(bundle, harbor, timeout)

    def migrated(**kwargs):
        specification_path = kwargs["item_root"] / "workspace/task/specification.json"
        specification = json.loads(specification_path.read_text())
        specification["requirements"] = {"state": {"image": image}}
        specification_path.write_text(json.dumps(specification))
        return {"state": "ready", "reason": "reviewed_images_migrated"}

    monkeypatch.setattr("capability_pipeline.image_pipeline.process_image_construction", migrated)
    result = synthesize_one(accepted(), tmp_path, FakeAgent(), InspectLowering(), None, 30)
    assert result["state"] == "controls_passed_pending_rollout"
    assert observed == [image]
    assert result["taskcompendium"]["specification_sha256"] == hashlib.sha256(
        (Path(result["item_root"]) / "workspace/task/specification.json").read_bytes()
    ).hexdigest()


@pytest.mark.parametrize("environment", ["none", "shellsim", "docker"])
def test_repeated_diagnostics_uses_base_source_and_staged_helpers(
    tmp_path, monkeypatch, environment,
):
    from capability_pipeline.synthesis import (
        _repeat_and_grading_diagnostics as _repeated_quality_diagnostics,
    )

    root = tmp_path / "item"
    helper = root / "workspace/tools/daytona/dt.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("# fixture only")
    bundle = root / "workspace/task"
    bundle.mkdir()
    (bundle / "binding.json").write_text(
        json.dumps({"environment": {"kind": environment}})
    )
    resources = bundle / "candidate-resources.json"
    resources.write_text('{"cpu":2,"memory_gb":2,"disk_gb":10}')
    base, overlay = tmp_path / "base-source", tmp_path / "patched-overlay"
    toolchain = OfficialToolchain(overlay, "uv", source_package_root=base)
    observed = {}

    def repeated(item_root, source, timeout, **kwargs):
        observed.update(item_root=item_root, source=source, timeout=timeout, **kwargs)
        return {"state": "pending"}

    monkeypatch.setattr("capability_pipeline.diagnostics.run_repeated_diagnostics", repeated)
    assert _repeated_quality_diagnostics(root, toolchain, 30, None) == {"state": "pending"}
    assert observed["source"] == base
    assert observed["daytona_helper"] == helper
    assert observed["candidate_resources"] == (
        resources if environment == "docker" else None
    )


@pytest.mark.parametrize(
    "grading_state,grading_reviewable,expected_state,expected_reviewable",
    [
        ("ready", True, "ready", True),
        ("semantic_failed", True, "pending", True),
        ("pending", False, "pending", False),
        # "unsupported" alone still blocks; only an explicitly unassessed input
        # shape is allowed through (covered by the dedicated test below).
        ("unsupported", False, "pending", False),
        ("pending", True, "pending", False),
    ],
)
def test_fixed_grading_gates_quality_review(
    tmp_path, monkeypatch, grading_state, grading_reviewable,
    expected_state, expected_reviewable,
):
    from capability_pipeline.synthesis import (
        _repeat_and_grading_diagnostics as _repeated_quality_diagnostics,
    )

    root = tmp_path / "item"
    (root / "contract").mkdir(parents=True)
    (root / "contract/accepted.json").write_text(
        json.dumps({"proposal": {"verification": "code"}})
    )
    attempt = root / "diagnostics/attempt-1"
    repeated = {
        "state": "ready", "reviewable": True, "attempt": str(attempt),
        "input_manifest": str(attempt / "inputs/manifest.json"),
        "extra_files": {},
    }
    observed = {}

    def grading(bundle, source, out, **kwargs):
        observed.update(bundle=bundle, source=source, out=out, **kwargs)
        return {
            "state": grading_state, "reviewable": grading_reviewable,
            "extra_files": {"fixed/raw.json": attempt / "fixed-grading/raw.json"},
        }

    monkeypatch.setattr(
        "capability_pipeline.diagnostics.run_repeated_diagnostics", lambda *a, **k: repeated
    )
    monkeypatch.setattr(
        "capability_pipeline.grading_diagnostics.run_grading_diagnostics", grading
    )
    source = tmp_path / "base"
    toolchain = OfficialToolchain(tmp_path / "overlay", "uv", source_package_root=source)
    result = _repeated_quality_diagnostics(root, toolchain, 30, None)
    assert result["state"] == expected_state
    assert result["reviewable"] is expected_reviewable
    assert result["fixed_grading"]["state"] == grading_state
    assert "extra_files" not in result["fixed_grading"]
    assert observed == {
        "bundle": attempt / "inputs", "source": source,
        "out": attempt / "fixed-grading", "timeout": 30,
    }
    assert "fixed/raw.json" in result["extra_files"]


def test_judge_diagnostics_keep_calibration_distinct_from_fixed_grading(tmp_path, monkeypatch):
    from capability_pipeline.synthesis import (
        _repeat_and_grading_diagnostics as _repeated_quality_diagnostics,
    )

    root = tmp_path / "item"
    (root / "contract").mkdir(parents=True)
    (root / "contract/accepted.json").write_text(
        json.dumps({"proposal": {"verification": "judge"}})
    )
    monkeypatch.setattr(
        "capability_pipeline.diagnostics.run_repeated_diagnostics",
        lambda *a, **k: {"state": "ready", "reviewable": True},
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("stochastic judge was routed to deterministic grading")

    monkeypatch.setattr(
        "capability_pipeline.grading_diagnostics.run_grading_diagnostics", unexpected
    )
    result = _repeated_quality_diagnostics(root, OfficialToolchain(tmp_path, "uv"), 30, None)
    assert result["state"] == "ready"
    assert result["fixed_grading"]["state"] == "not_applicable"
    assert result["fixed_grading"]["machine_component_repeatability"] == "unassessed"


@pytest.mark.parametrize("diagnostic_state,reviewable", [("ready", True), ("pending", False), ("pending", True)])
def test_artifact_bound_harbor_attestation_allows_export(tmp_path, monkeypatch, diagnostic_state, reviewable):
    runner = tmp_path / "runner.py"
    runner.write_text("""#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--package'); p.add_argument('--bundle'); p.add_argument('--controls'); p.add_argument('--output')
a=p.parse_args()
h=lambda x: hashlib.sha256(Path(x).read_bytes()).hexdigest()
def tree(root):
 d=hashlib.sha256(); root=Path(root)
 for x in sorted(y for y in root.rglob('*') if y.is_file()):
  d.update(x.relative_to(root).as_posix().encode()); d.update(b'\\0'); d.update(h(x).encode()); d.update(b'\\n')
 return d.hexdigest()
base=Path(a.output).parent
(base/'harbor-run.json').write_text('{"status":"complete"}')
attacks=[]
for strategy in ('injection','shortcut','boundary'):
 trial=base/f"attack-{strategy}-trial.json"; trial.write_text(json.dumps({'strategy':strategy,'kind':'trial'}))
 row={'strategy':strategy,'trial_artifact':trial.name,'trial_sha256':h(trial),'steps':[]}
 step={'step_index':0,'step_name':'step'}
 for kind in ('transcript','grading'):
  path=base/f"attack-{strategy}-{kind}.json"
  content={'strategy':strategy,'kind':kind} if kind == 'transcript' else {'status':'graded','reward':0.0}
  path.write_text(json.dumps(content))
  step[kind+'_artifact']=path.name; step[kind+'_sha256']=h(path)
 step['result']={'status':'graded','reward':0.0}
 row['steps'].append(step)
 attacks.append(row)
(base/'adversary-run.json').write_text(json.dumps({'state':'passed','independent':True,'cases':attacks}))
controls=json.loads(Path(a.controls).read_text())
for case in controls['cases']:
 (base/f"{case['id']}-trial.json").write_text(json.dumps({'case':case['id']}))
positive=next(case for case in controls['cases'] if case['class']=='positive')
(base/'oracle-trial.json').write_text(json.dumps({'case':positive['id'],'kind':'oracle'}))
(base/'oracle-grading.json').write_text(json.dumps({'status':'graded','reward':1.0}))
(base/'authored-oracle.json').write_text(json.dumps({'state':'passed','independent':False,
'source':'authored_reference_controls','cases':[{'case_id':positive['id'],'step_index':0,
'trial_artifact':'oracle-trial.json','trial_sha256':h(base/'oracle-trial.json'),
'grading_artifact':'oracle-grading.json','grading_sha256':h(base/'oracle-grading.json'),
'result':{'status':'graded','reward':1.0},'isolation':{'mechanism':'no-tool-host-boundary'},
'private_verifier':None}]}))
(base/'solver-trial.json').write_text(json.dumps({'case':positive['id'],'kind':'solver'}))
(base/'solver-transcript.json').write_text(json.dumps({'response':'yes'}))
(base/'solver-grading.json').write_text(json.dumps({'status':'graded','reward':1.0}))
(base/'solver-run.json').write_text(json.dumps([{'case_id':positive['id'],'step_index':0,
'attempt':1,'trial_artifact':'solver-trial.json','trial_sha256':h(base/'solver-trial.json'),
'transcript_artifact':'solver-transcript.json','transcript_sha256':h(base/'solver-transcript.json'),
'grading_artifact':'solver-grading.json','grading_sha256':h(base/'solver-grading.json'),
'result':{'status':'graded','reward':1.0},'isolation':{'case_id':positive['id'],
'mechanism':'no-tool-host-boundary'},'private_verifier':None}]))
isolation=[{'case_id':case['id'],'mechanism':'no-tool-host-boundary'} for case in controls['cases']]
e={'attestation': {'kind':'harbor_control_run','harbor_revision':'93147ea9e07b04ec8d2eb5afd2916386f1aacc69',
'specification_sha256':h(Path(a.bundle)/'specification.json'),'package_manifest_sha256':h(Path(a.package)/'manifest.json'),
'package_sha256':tree(a.package),
'controls_sha256':h(a.controls),'isolation':{'network':'blocked','fresh_environment_per_case':True,'cases':isolation},
'solver':{'independent':True,'state':'passed','run_id':'solver-1','retry_limit':2},
'solver_artifact':'solver-run.json',
'solver_artifact_sha256':h(base/'solver-run.json'),
'oracle':{'authored':True,'run_id':'oracle-1'},'oracle_artifact':'authored-oracle.json',
'oracle_artifact_sha256':h(base/'authored-oracle.json'),
'adversarial':{'authored_controls_executed':True,'independent_attack_executed':True},
'adversary_artifact':'adversary-run.json','adversary_artifact_sha256':h(base/'adversary-run.json'),
'run_artifact':'harbor-run.json','run_artifact_sha256':h(base/'harbor-run.json')},
'cases':[]}
for case in controls['cases']:
 artifact=base/'solver-grading.json' if case['class']=='positive' else base/f"{case['id']}-trial.json"
 result={'status':'graded','reward':1.0 if case['class']=='positive' else 0.0}
 if case['class']=='malformed': result={'status':'extraction_error','reward':None}
 artifact.write_text(json.dumps(result))
 e['cases'].append({'id':case['id'],'source_author':case['source_author'],'category':case['category'],
 'control_type':'independent_solver' if case['class']=='positive' else 'authored_adversarial_control',
 'artifact':artifact.name,'artifact_sha256':h(artifact),'result':result})
Path(a.output).write_text(json.dumps(e))
""")
    runner.chmod(0o755)

    def accept_quality(item_root, review_root, agent):
        review_root.mkdir(parents=True, exist_ok=True)
        result = {
            "schema_version": "capability-quality-result-v1",
            "snapshot_hash": "snapshot-1",
            "state": "accept",
        }
        (review_root / "result.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr("capability_pipeline.quality.run_review", accept_quality)
    monkeypatch.setattr(
        "capability_pipeline.synthesis._repeated_quality_diagnostics",
        lambda *args: {"state": diagnostic_state, "reviewable": reviewable, "extra_files": {}},
    )
    result = synthesize_one(
        accepted(), tmp_path / "run", FakeAgent(), FakeToolchain(), runner, 30
    )
    if diagnostic_state == "pending":
        assert result["state"] == "pending_repeated_diagnostics", result
        assert ("quality_review" in result) is reviewable
        assert not (tmp_path / "run" / "validated").exists()
        return
    assert result["state"] == "quality_accepted", result
    assert result["runtime_validated"] is True
    assert result["quality_review"]["state"] == "accept"
    assert (Path(result["export"]) / "manifest.json").is_file()


@pytest.mark.parametrize("diagnostic_state", ["ready", "pending"])
def test_runtime_validated_task_without_quality_acceptance_is_not_exported(
    tmp_path, monkeypatch, diagnostic_state
):
    runner = tmp_path / "runner.py"
    runner.write_text("#!/usr/bin/env python3\n")
    runner.chmod(0o755)

    def controls(*args, **kwargs):
        return {
            "cases": [
                {
                    "id": "positive",
                    "source_author": "builder",
                    "category": "known_correct",
                    "control_type": "independent_solver",
                    "result": {"status": "graded", "reward": 1.0},
                },
                {
                    "id": "malformed",
                    "source_author": "builder",
                    "category": "empty_or_malformed",
                    "control_type": "authored_adversarial_control",
                    "result": {"status": "extraction_error", "reward": None},
                },
                {
                    "id": "plausible-wrong",
                    "source_author": "builder",
                    "category": "plausible_wrong",
                    "control_type": "authored_adversarial_control",
                    "result": {"status": "graded", "reward": 0.0},
                },
                {
                    "id": "shortcut",
                    "source_author": "builder",
                    "category": "task_specific_shortcut",
                    "control_type": "authored_adversarial_control",
                    "result": {"status": "graded", "reward": 0.0},
                },
            ]
        }

    monkeypatch.setattr("capability_pipeline.synthesis._external_controls", controls)
    monkeypatch.setattr(
        "capability_pipeline.synthesis._attestation_issues", lambda *args: []
    )

    def repair_quality(item_root, review_root, agent):
        review_root.mkdir(parents=True, exist_ok=True)
        result = {
            "schema_version": "capability-quality-result-v1",
            "snapshot_hash": "snapshot-2",
            "state": "repair",
        }
        (review_root / "result.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr("capability_pipeline.quality.run_review", repair_quality)
    monkeypatch.setattr(
        "capability_pipeline.synthesis._repeated_quality_diagnostics",
        lambda *args: {"state": diagnostic_state, "reviewable": True, "extra_files": {}},
    )
    result = synthesize_one(
        accepted(), tmp_path / "run", FakeAgent(), FakeToolchain(), runner, 30
    )
    assert result["state"] == "pending_quality_review", result
    assert result["runtime_validated"] is True
    assert "export" not in result
    assert not (tmp_path / "run" / "validated").exists()


def test_bounded_solver_miss_stays_pending_adjudication(tmp_path, monkeypatch):
    runner = tmp_path / "runner.py"
    runner.write_text("#!/usr/bin/env python3\n")
    runner.chmod(0o755)

    def controls(*args, **kwargs):
        evidence = {
            "attestation": {"solver": {"state": "needs_adjudication"}},
            "cases": [],
        }
        for control in json.loads(Path(args[3]).read_text())["cases"]:
            result = {"status": "graded", "reward": 0.0}
            if control["class"] == "malformed":
                result = {"status": "extraction_error", "reward": None}
            evidence["cases"].append(
                {
                    "id": control["id"],
                    "source_author": control["source_author"],
                    "category": control["category"],
                    "control_type": "independent_solver"
                    if control["class"] == "positive"
                    else "authored_adversarial_control",
                    "result": result,
                }
            )
        return evidence

    monkeypatch.setattr("capability_pipeline.synthesis._external_controls", controls)
    monkeypatch.setattr(
        "capability_pipeline.synthesis._attestation_issues",
        lambda *args: ["independent solver needs adjudication after bounded retries"],
    )
    result = synthesize_one(
        accepted(), tmp_path / "run", FakeAgent(), FakeToolchain(), runner, 30
    )
    assert result["state"] == "pending_solver_adjudication"
    assert result["runtime_evidence"].endswith("runtime-evidence.json")
    assert "runtime_validated" not in result
    assert "export" not in result


def test_shellsim_without_harbor_runner_stays_pending(tmp_path):
    result = synthesize_one(
        accepted(environment="shellsim"),
        tmp_path,
        FakeAgent(),
        FakeToolchain(),
        None,
        30,
    )
    assert result["state"] == "pending_runtime"
    assert not (tmp_path / "validated").exists()


@pytest.mark.parametrize("fixture_kind", ["missing", "wrong_schema"])
def test_invalid_authored_judge_fixture_is_repairable_before_runtime_or_export(
    tmp_path, monkeypatch, fixture_kind
):
    monkeypatch.setenv("GLM_BASE_URL", "https://relay.example")
    agent = FakeAgent() if fixture_kind == "missing" else FakeJudgeAgent()
    if fixture_kind == "wrong_schema":
        agent.calibration_schema = "unsupported-schema"
    result = _synthesize_attempt(
        accepted(verification="judge"),
        tmp_path,
        agent,
        FakeToolchain(),
        None,
        30,
    )
    assert result["state"] == "failed"
    assert result["issues"][0].startswith("invalid task bundle:")
    assert _fresh_construction_repair_allowed(result)
    assert "runtime_evidence" not in result
    assert not (tmp_path / "validated").exists()


def test_passing_judge_calibration_binds_raw_specification_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM_BASE_URL", "https://relay.example/")

    def calibrate(args):
        bundle = Path(args.bundle)
        output = Path(args.out) / "judge-calibration.json"
        output.parent.mkdir(parents=True)
        output.write_text(
            json.dumps(
                {
                    "state": "passed",
                    "specification_sha256": hashlib.sha256(
                        (bundle / "specification.json").read_bytes()
                    ).hexdigest(),
                    "fixture_hash": "b" * 64,
                }
            )
        )
        return 0

    monkeypatch.setattr("capability_pipeline.judge.calibrate_task", calibrate)
    result = synthesize_one(
        accepted(verification="judge"),
        tmp_path,
        FakeJudgeAgent(),
        FakeToolchain(),
        None,
        30,
    )
    assert result["state"] == "pending_runtime"
    assert len(result["judge_calibration"]["artifact_sha256"]) == 64
    policy = json.loads(
        (Path(result["item_root"]) / "contract/judge-policy.json").read_text()
    )
    assert policy == {
        "provider": "glm",
        "model": "glm-5.3",
        "base_url": "https://relay.example/v1",
    }
    item_root = Path(result["item_root"])
    assert (item_root / "workspace/task/judge-policy.json").read_bytes() == (
        item_root / "contract/judge-policy.json"
    ).read_bytes()


class FakeJudgeVerifierAgent(FakeAgent):
    """A builder that adds a judge verifier regardless of the proposed grading."""

    def __init__(self, verifier, **kwargs):
        super().__init__(**kwargs)
        self.verifier = verifier

    def invoke(self, workspace, session_dir, prompt, attempt):
        result = super().invoke(workspace, session_dir, prompt, attempt)
        spec = workspace / "task/specification.json"
        if spec.is_file():
            spec.write_text(
                json.dumps({"steps": [{"name": "step", "verifier": self.verifier}]})
            )
        return result


JUDGE_VERIFIER = {
    "kind": "tasktrove",
    "mode": "judge",
    "parameters": {},
    "judge": {"policy": {"provider": "glm", "model": "glm-5.3"}},
}


def test_judge_step_indices_mirror_tasktrove_unwrapping():
    from capability_pipeline.runtime import judge_step_indices

    code = {"kind": "tasktrove", "mode": "pytest", "parameters": {}}
    specification = {
        "steps": [
            {"verifier": JUDGE_VERIFIER},
            {"verifier": code},
            {"verifier": {"kind": "code_answer", "verifier": code, "output_path": "a"}},
            {"verifier": {"kind": "code_answer", "verifier": JUDGE_VERIFIER}},
            {"verifier": {"kind": "instruction_constraints", "mode": "judge"}},
            "malformed",
        ]
    }
    assert judge_step_indices(specification) == (0, 3)
    assert judge_step_indices({"steps": "bad"}) == ()
    assert judge_step_indices(None) == ()


@pytest.mark.parametrize("verification", ["simple", "code"])
def test_non_judge_proposal_with_judge_verifier_gets_frozen_policy(
    tmp_path, monkeypatch, verification
):
    monkeypatch.setenv("GLM_BASE_URL", "https://relay.example/")
    result = _synthesize_attempt(
        accepted(verification=verification),
        tmp_path,
        FakeJudgeVerifierAgent(JUDGE_VERIFIER),
        FakeToolchain(),
        None,
        30,
    )
    item_root = Path(result["item_root"])
    contract_policy = item_root / "contract/judge-policy.json"
    assert json.loads(contract_policy.read_text()) == {
        "provider": "glm",
        "model": "glm-5.3",
        "base_url": "https://relay.example/v1",
    }
    bundle_policy = item_root / "workspace/task/judge-policy.json"
    assert bundle_policy.read_bytes() == contract_policy.read_bytes()
    assert result["state"] != "pending_judge_policy"
    assert not any("judge-policy" in issue for issue in result["issues"])


@pytest.mark.parametrize("verification", ["simple", "code"])
def test_non_judge_proposal_without_judge_verifier_copies_no_policy(
    tmp_path, monkeypatch, verification
):
    monkeypatch.setenv("GLM_BASE_URL", "https://relay.example/")
    result = _synthesize_attempt(
        accepted(verification=verification),
        tmp_path,
        FakeJudgeVerifierAgent({"kind": "tasktrove", "mode": "pytest", "parameters": {}}),
        FakeToolchain(),
        None,
        30,
    )
    item_root = Path(result["item_root"])
    assert (item_root / "contract/judge-policy.json").is_file()
    assert not (item_root / "workspace/task/judge-policy.json").exists()


def test_non_judge_proposal_without_glm_url_is_not_blocked(tmp_path, monkeypatch):
    monkeypatch.delenv("GLM_BASE_URL", raising=False)
    result = _synthesize_attempt(
        accepted(verification="simple"), tmp_path, FakeAgent(), FakeToolchain(), None, 30
    )
    item_root = Path(result["item_root"])
    assert result["state"] != "pending_judge_policy"
    assert not (item_root / "contract/judge-policy.json").exists()
    assert not (item_root / "workspace/task/judge-policy.json").exists()


def test_non_judge_proposal_with_judge_verifier_and_no_glm_url_holds_clearly(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("GLM_BASE_URL", raising=False)
    result = _synthesize_attempt(
        accepted(verification="code"),
        tmp_path,
        FakeJudgeVerifierAgent(JUDGE_VERIFIER),
        FakeToolchain(),
        None,
        30,
    )
    assert result["state"] == "pending_judge_policy"
    assert result["issues"] == [
        "specification uses a judge verifier but no judge policy could be "
        "staged: GLM judge base URL is unavailable"
    ]
    assert not (Path(result["item_root"]) / "workspace/task/judge-policy.json").exists()


def test_judge_proposal_without_glm_url_still_holds_before_build(tmp_path, monkeypatch):
    monkeypatch.delenv("GLM_BASE_URL", raising=False)
    agent = FakeAgent()
    result = _synthesize_attempt(
        accepted(verification="judge"), tmp_path, agent, FakeToolchain(), None, 30
    )
    assert result["state"] == "pending_judge_policy"
    assert result["issues"] == ["GLM judge base URL is unavailable"]
    assert agent.calls == []


def test_judge_calibration_with_mismatched_raw_spec_hash_cannot_export(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GLM_BASE_URL", "https://relay.example")

    def calibrate(args):
        output = Path(args.out) / "judge-calibration.json"
        output.parent.mkdir(parents=True)
        output.write_text(
            json.dumps(
                {
                    "state": "passed",
                    "specification_sha256": "0" * 64,
                    "fixture_hash": "b" * 64,
                }
            )
        )
        return 0

    monkeypatch.setattr("capability_pipeline.judge.calibrate_task", calibrate)
    result = synthesize_one(
        accepted(verification="judge"),
        tmp_path,
        FakeJudgeAgent(),
        FakeToolchain(),
        None,
        30,
    )
    assert result["state"] == "pending_judge_calibration"
    assert not (tmp_path / "validated").exists()


def test_incomplete_session_is_not_treated_as_success(tmp_path):
    one_session = [
        {
            "session": "s1",
            "depends_on": [],
            "goal": "build",
            "handoff_artifacts": ["bundle"],
            "acceptance_checks": ["validate"],
        }
    ]
    result = synthesize_one(
        accepted(sessions=one_session),
        tmp_path,
        FakeAgent(complete=False),
        None,
        None,
        30,
    )
    assert result["state"] == "pending_build"
    assert result["sessions"][0]["status"] == "continuation_required"
    assert len(result["sessions"][0]["attempts"]) == 2


def test_resumed_session_preserves_attempts_and_uses_incremental_prompt(tmp_path):
    one_session = [
        {
            "session": "s1",
            "depends_on": [],
            "goal": "build",
            "handoff_artifacts": ["bundle"],
            "acceptance_checks": ["validate"],
        }
    ]
    item = accepted(sessions=one_session)
    first = FakeAgent(complete=False)
    initial = synthesize_one(item, tmp_path, first, None, None, 30)
    item_root = Path(initial["item_root"])
    attempt_zero = (item_root / "sessions/s1/attempt-0.log").read_bytes()

    resumed = FakeAgent(complete=False)
    result = synthesize_one(item, tmp_path, resumed, None, None, 30)
    assert resumed.calls == [("s1", 2), ("s1", 3)]
    assert (item_root / "sessions/s1/attempt-0.log").read_bytes() == attempt_zero
    assert (item_root / "sessions/s1/continuation-2.md").is_file()
    assert len(result["sessions"][0]["attempts"]) == 4


def test_omp_transcript_totals_surface_length_exhaustion(tmp_path):
    transcript = tmp_path / "session.jsonl"
    events = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "stopReason": "length",
                "usage": {
                    "output": 32768,
                    "reasoningTokens": 32767,
                },
                "content": [{"type": "thinking"}],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "stopReason": "toolUse",
                "usage": {"output": 80, "reasoningTokens": 20},
                "content": [{"type": "toolCall", "name": "write"}],
            },
        },
        {"type": "compaction"},
        {"type": "custom", "customType": "session_exit"},
    ]
    transcript.write_text("".join(json.dumps(event) + "\n" for event in events))
    assert _omp_transcript_totals(tmp_path) == {
        "assistant_messages": 2,
        "length_stops": 1,
        "output_tokens": 32848,
        "reasoning_tokens": 32787,
        "tool_calls": 1,
        "compactions": 1,
        "session_exits": 1,
    }


def test_length_exhaustion_gets_one_incremental_recovery_then_stops(tmp_path):
    class StalledAgent:
        max_continuations = 8
        max_stagnant_attempts = 2

        def __init__(self):
            self.prompts = []

        def invoke(self, workspace, session_dir, prompt, attempt):
            self.prompts.append(prompt.read_text())
            return {
                "returncode": 0,
                "timed_out": False,
                "stdout": "planning",
                "stderr": "",
                "elapsed_seconds": 1,
                "assistant_messages": 3,
                "length_stops": 1,
                "output_tokens": 32768,
                "reasoning_tokens": 32768,
                "tool_calls": 0,
                "compactions": 1,
                "session_exits": 1,
            }

    agent = StalledAgent()
    one_session = [
        {
            "session": "large-generator",
            "depends_on": [],
            "goal": "build thirty fixture files",
            "handoff_artifacts": ["generator/generate.py"],
            "acceptance_checks": ["three deterministic regenerations"],
        }
    ]
    result = synthesize_one(
        accepted(sessions=one_session),
        tmp_path,
        agent,
        None,
        None,
        30,
    )
    session = result["sessions"][0]
    assert result["state"] == "pending_build"
    assert len(agent.prompts) == 2
    assert "create .capability-progress/large-generator.json" in agent.prompts[0]
    assert "Your first action in this continuation must be a write" in agent.prompts[1]
    assert session["continuation_reason"] == (
        "model_output_budget_exhausted_without_workspace_progress"
    )
    assert [attempt["length_stops"] for attempt in session["attempts"]] == [1, 1]
    assert any("output_budget_exhausted" in issue for issue in result["issues"])


def test_invalid_grouped_handoff_gets_specific_repair_prompt(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "handoffs").mkdir(parents=True)
    (workspace / "task").mkdir()
    (workspace / "task" / "specification.json").write_text("{}")
    handoff_path = workspace / "handoffs" / "builder.json"
    handoff = {
        "session": "builder",
        "status": "complete",
        "artifacts": {"task_package": ["task/specification.json"]},
        "checks": [{"name": "exists", "command": "test -f", "exit_code": 0}],
    }
    handoff_path.write_text(json.dumps(handoff))

    assert _handoff(workspace, "builder") is None
    prompt = _continuation_prompt(workspace, {"session": "builder"}, 2, {})
    assert "artifacts must be a non-empty JSON list" in prompt
    assert "Your first action must edit that handoff" in prompt
    assert "do not rerun completed work" in prompt

    handoff["artifacts"] = ["task/specification.json"]
    handoff_path.write_text(json.dumps(handoff))
    assert _handoff(workspace, "builder") == handoff


def test_resolved_rewarded_attack_sidecar_allows_quality_review(tmp_path, monkeypatch):
    runner = tmp_path / "runner.py"
    runner.write_text("#!/usr/bin/env python3\n")
    runner.chmod(0o755)

    def controls(*args, **kwargs):
        document = json.loads(Path(args[3]).read_text())
        cases = []
        for control in document["cases"]:
            result = {
                "status": "graded",
                "reward": 1.0 if control["class"] == "positive" else 0.0,
            }
            if control["class"] == "malformed":
                result = {"status": "extraction_error", "reward": None}
            cases.append(
                {
                    "id": control["id"],
                    "source_author": control["source_author"],
                    "category": control["category"],
                    "control_type": "independent_solver"
                    if control["class"] == "positive"
                    else "authored_adversarial_control",
                    "result": result,
                }
            )
        return {"attestation": {"solver": {"state": "passed"}}, "cases": cases}

    monkeypatch.setattr("capability_pipeline.synthesis._external_controls", controls)
    monkeypatch.setattr(
        "capability_pipeline.synthesis._attestation_issues",
        lambda *args: [
            "attestation lacks an independent adversarial attack",
            "independent adversary report needs adjudication or retry",
        ],
    )

    def adjudicate(item_root, root, agent):
        review = root / "attack-adjudication" / Path(item_root).name / "attempt-1"
        review.mkdir(parents=True)
        for name, contents in {
            "input-manifest.json": "{}",
            "receipt.json": "{}",
            "result.json": '{"state":"resolved"}',
        }.items():
            (review / name).write_text(contents)
        return {"state": "resolved", "snapshot_hash": "snapshot-a"}, review

    monkeypatch.setattr(
        "capability_pipeline.synthesis._attack_adjudication", adjudicate
    )

    def accept_quality(item_root, review_root, agent, *, extra_files):
        assert set(extra_files) == {
            "controller/attack-adjudication/input-manifest.json",
            "controller/attack-adjudication/receipt.json",
            "controller/attack-adjudication/result.json",
        }
        review_root.mkdir(parents=True, exist_ok=True)
        result = {"state": "accept", "snapshot_hash": "snapshot-q"}
        (review_root / "result.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr("capability_pipeline.quality.run_review", accept_quality)
    monkeypatch.setattr(
        "capability_pipeline.synthesis._repeated_quality_diagnostics",
        lambda *args: {"state": "ready", "reviewable": True, "extra_files": {}},
    )
    result = synthesize_one(
        accepted(), tmp_path / "run", FakeAgent(), FakeToolchain(), runner, 30
    )
    assert result["state"] == "quality_accepted"
    assert result["attack_adjudication"]["state"] == "resolved"


def test_unresolved_attack_adjudication_is_not_resampled_for_same_snapshot(
    tmp_path, monkeypatch
):
    from capability_pipeline.quality import sha256, source_files

    item = tmp_path / "item"
    (item / "contract").mkdir(parents=True)
    (item / "contract/accepted.json").write_text('{"proposal_hash":"p"}')
    (item / "independent-adversary.json").write_text(
        '{"independent":true,"max_false_reward":0.2,"cases":[]}'
    )
    review = tmp_path / "run/attack-adjudication/item/attempt-1"
    review.mkdir(parents=True)
    files = {name: sha256(path) for name, path in source_files(item).items()}
    identity = {
        "schema_version": "capability-attack-adjudication-v1-input",
        "files": files,
    }
    manifest = {**identity, "snapshot_hash": digest(identity)}
    (review / "input-manifest.json").write_text(json.dumps(manifest))
    pending = {
        "schema_version": "capability-attack-adjudication-v1-result",
        "snapshot_hash": manifest["snapshot_hash"],
        "state": "needs_repair_or_retry",
        "issues": ["boundary:0: rewarded attack requires independent adjudication"],
    }
    (review / "result.json").write_text(json.dumps(pending))

    monkeypatch.setattr(
        "capability_pipeline.attack_adjudication.run_adjudication",
        lambda *args, **kwargs: pytest.fail("unchanged evidence was resampled"),
    )
    result, reused = _attack_adjudication(item, tmp_path / "run", FakeAgent())
    assert result == pending
    assert reused == review


def _write_image_compatibility_receipt(item_root, bundle, *, parent="runtime-trials", composite=False):
    from capability_pipeline.daytona_policy import verifier_snapshot_recipe

    image = "registry.example/verifier@sha256:" + "a" * 64
    spec = json.loads((bundle / "specification.json").read_text())
    spec["steps"][0]["verifier"] = {"runtime": {"kind": "container", "image": image, "supervisor_python": "python3"}}
    (bundle / "specification.json").write_text(json.dumps(spec))
    receipt = {
        "schema_version": "capability-verifier-image-compatibility-v1",
        "reason": "supervisor_python_too_old", "image": image,
        "supervisor_python": "python3", "snapshot_name": "cap-verifier-test", "snapshot_id": "provider-id",
        "snapshot_recipe_sha256": hashlib.sha256(verifier_snapshot_recipe(image).encode()).hexdigest(),
        "build_log_sha256": "b" * 64, "pinned_requirement": "numpy==2.5.3",
        "required_python": ">=3.12", "observed_python": "3.11",
        "provider_log_excerpt": "cp311; numpy 2.5.3 requires Python >=3.12; no matching distribution",
        "adapter_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "capability_pipeline/daytona_verifier.py").read_bytes()).hexdigest(),
    }
    result = {"status": "infra_error", "reward": None, "detail": {"verifier_image_compatibility": receipt}}
    if composite:
        result = {"status": "infra_error", "reward": None, "detail": {"machine_results": [result]}}
    path = item_root / parent / "oracle-gold/verifier/taskcompendium-result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return path


@pytest.mark.parametrize("parent,composite", [("runtime-trials", False), ("judge-calibration", True)])
def test_image_compatibility_receipt_routes_to_repair(tmp_path, parent, composite):
    bundle = tmp_path / "workspace/task"
    bundle.mkdir(parents=True)
    (bundle / "specification.json").write_text(json.dumps({"steps": [{}]}))
    path = _write_image_compatibility_receipt(tmp_path, bundle, parent=parent, composite=composite)
    failures = _runtime_image_compatibility_failures(tmp_path, bundle)
    assert len(failures) == 1
    assert failures[0]["artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    result = {}
    _record_image_compatibility_failure(result, failures)
    assert _fresh_construction_repair_allowed(result)
    assert result["verifier_image_compatibility_failures"] == failures


@pytest.mark.parametrize("field,value", [
    ("image", "other@sha256:" + "a" * 64),
    ("supervisor_python", "other-python"), ("snapshot_recipe_sha256", "c" * 64),
    ("adapter_sha256", "c" * 64), ("observed_python", "3.12"),
    ("reason", "network_error"), ("build_log_sha256", None),
])
def test_unbound_image_failure_stays_operational(tmp_path, field, value):
    bundle = tmp_path / "workspace/task"
    bundle.mkdir(parents=True)
    (bundle / "specification.json").write_text(json.dumps({"steps": [{}]}))
    path = _write_image_compatibility_receipt(tmp_path, bundle)
    result = json.loads(path.read_text())
    result["detail"]["verifier_image_compatibility"][field] = value
    path.write_text(json.dumps(result))
    assert _runtime_image_compatibility_failures(tmp_path, bundle) == []


def test_runtime_image_failure_preserves_receipt_for_glm_repair(tmp_path, monkeypatch):
    def failed_runtime(command, bundle, harbor, controls, output, timeout):
        _write_image_compatibility_receipt(output.parent, bundle)
        raise SynthesisError("Harbor failed before grading")

    monkeypatch.setattr("capability_pipeline.synthesis._external_controls", failed_runtime)
    result = _synthesize_attempt(accepted(), tmp_path, FakeAgent(), FakeToolchain(), Path("/fake-runner"), 30)
    assert result["state"] == "failed"
    assert result["issues"][0].startswith("invalid task bundle:")
    assert _fresh_construction_repair_allowed(result)
    assert len(result["verifier_image_compatibility_failures"]) == 1
    assert not result.get("runtime_validated")
    assert not (tmp_path / "validated").exists()


def test_composite_image_failure_binds_machine_runtime_not_judge(tmp_path):
    from test_composite_policy import policy

    bundle = tmp_path / "workspace/task"
    bundle.mkdir(parents=True)
    spec_path = bundle / "specification.json"
    spec_path.write_text(json.dumps({"steps": [{}]}))
    path = _write_image_compatibility_receipt(tmp_path, bundle, parent="judge-calibration", composite=True)
    receipt = json.loads(path.read_text())["detail"]["machine_results"][0]["detail"]["verifier_image_compatibility"]
    spec_path.write_text(json.dumps({"steps": [{"verifier": {"kind": "judge"}}]}))
    assert _runtime_image_compatibility_failures(tmp_path, bundle) == []
    root = Path(__file__).resolve().parents[1] / "capability_pipeline"
    step_policy = policy()
    step_policy["machine_checks"][0]["image"] = receipt["image"]
    config = {
        "schema_version": "taskcompendium-composite-verifier-v1",
        "specification_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "implementation": {
            "taskcompendium_revision": "dc6b501c8604bcd2e3c20c1e9947679845fdfef8",
            **{key: hashlib.sha256((root / filename).read_bytes()).hexdigest() for key, filename in {
                "adapter_sha256": "composite_verifier.py",
                "policy_sha256": "composite_policy.py",
                "native_judge_protocol_sha256": "native_judge_protocol.py",
            }.items()},
        },
        "steps": [step_policy],
    }
    (bundle / "composite-verifier.json").write_text(json.dumps(config))
    assert len(_runtime_image_compatibility_failures(tmp_path, bundle)) == 1
    config["specification_sha256"] = "f" * 64
    (bundle / "composite-verifier.json").write_text(json.dumps(config))
    assert _runtime_image_compatibility_failures(tmp_path, bundle) == []


def test_unsupported_fixed_grading_stays_unassessed_and_keeps_the_repeated_verdict(
    tmp_path, monkeypatch
):
    """A final-state task has no fixed response string; that is unassessed, not failed."""
    from capability_pipeline.synthesis import (
        _repeat_and_grading_diagnostics as _repeated_quality_diagnostics,
    )

    root = tmp_path / "item"
    (root / "contract").mkdir(parents=True)
    (root / "contract/accepted.json").write_text(
        json.dumps({"proposal": {"verification": "code"}})
    )
    attempt = root / "diagnostics/attempt-1"
    repeated = {
        "state": "ready", "reviewable": True, "attempt": str(attempt),
        "input_manifest": str(attempt / "inputs/manifest.json"),
        "unassessed_recipe_rows": ["reward_determinism_10_regrades", "judge_calibration"],
        "issues": [],
        "extra_files": {},
    }
    monkeypatch.setattr(
        "capability_pipeline.diagnostics.run_repeated_diagnostics", lambda *a, **k: repeated
    )
    monkeypatch.setattr(
        "capability_pipeline.grading_diagnostics.run_grading_diagnostics",
        lambda *a, **k: {
            "state": "unsupported", "reviewable": False, "unassessed": True,
            "issues": ["no-tool controls require a fixed response string"],
            "extra_files": {},
        },
    )
    source = tmp_path / "base"
    toolchain = OfficialToolchain(tmp_path / "overlay", "uv", source_package_root=source)
    result = _repeated_quality_diagnostics(root, toolchain, 30, None)

    assert result["state"] == "ready" and result["reviewable"] is True
    assert result["fixed_grading"]["state"] == "unsupported"
    assert result["fixed_grading"]["unassessed"] is True
    assert result["fixed_grading"]["issues"] == [
        "no-tool controls require a fixed response string"
    ]
    # The determinism row must remain explicitly unassessed, never silently claimed.
    assert "reward_determinism_10_regrades" in result["unassessed_recipe_rows"]


def test_silo_builder_prompt_drops_the_quota_ceiling_but_keeps_ownership(monkeypatch, tmp_path):
    from capability_pipeline.synthesis import _prompt

    item = accepted()
    session = item["proposal"]["builder_plan"][0] | {"session": "s1"}
    monkeypatch.setenv("CAPABILITY_SANDBOX_PROVIDER", "daytona")
    daytona = _prompt(item, session, False, None, tmp_path, True)
    assert "On quota exhaustion, retain the error and report needs_continuation." in daytona

    monkeypatch.setenv("CAPABILITY_SANDBOX_PROVIDER", "silo")
    silo = _prompt(item, session, False, None, tmp_path, True)
    assert "report needs_continuation" not in silo.split("There is no snapshot quota.")[0][-200:]
    assert "There is no snapshot quota." in silo
    assert "DAYTONA_API_KEY remains in the process environment." not in silo
    assert "SILO_API_TOKEN" in silo
    # ownership rules are unchanged
    assert "Never delete or rename existing" in silo
    assert "Delete only sandboxes created by this task" in silo
