# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a protocol-qualified controller to immutable evidence before dispatch."""

import hashlib
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from finestore.rl.difficulty import PROTOCOL_ID, SOLVER_MODELS, Solver

from experiments.post_training.task_curation.difficulty import __main__ as cli
from experiments.post_training.task_curation.difficulty import controller, executor, runtime
from experiments.post_training.task_curation.difficulty.ledger import append_rows, read_rows
from experiments.post_training.task_curation.difficulty.planner import action_from_payload
from experiments.post_training.task_curation.difficulty.protocols import (
    ProtocolActivation,
    activation_path,
    protocol_supersessions,
    require_protocol_activation,
    write_protocol_activation,
)
from experiments.post_training.task_curation.difficulty.state import ActionFailure, CoveragePolicy, Outcome
from experiments.post_training.task_curation.difficulty_output_budget import LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID as V2
from experiments.post_training.task_curation.difficulty_protocol import PINS
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from experiments.post_training.task_curation.tests.test_difficulty_executor import JobStore
from experiments.post_training.task_curation.tests.test_difficulty_reconciler import SOURCE
from experiments.post_training.task_curation.tests.test_difficulty_solver_protocols import historical_row


def artifact(path, payload):
    data = json.dumps(payload, sort_keys=True).encode()
    path.write_bytes(data)
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(data).hexdigest())


@pytest.fixture
def qualification():
    return {
        "passed": True,
        "native_server_closed": True,
        "compilation_enabled": True,
        "early_stops_are_model_failures": True,
        "protocol_id": V2,
        "model": SOLVER_MODELS[Solver.LARGE],
        "revision": PINS[Solver.LARGE].revision,
        "context_tokens": 65536,
        "safety_margin_tokens": 64,
        "minimum_output_floor_tokens": 16384,
        "tensor_parallel": 16,
        "data_parallel": 2,
        "request_count": 16,
        "completed_at": "2026-11-27T17:00:00+00:00",
        "attempt": "/fixture/canary/0:0",
        "engine_jobs": ["/fixture/canary/inference-one"],
        "completions": [{"index": i, "qualified": True} for i in range(16)],
    }


@pytest.fixture
def cleanup():
    return {
        "jobs": [
            {
                "job_id": "/fixture/canary",
                "state": "JOB_STATE_SUCCEEDED",
                "failures": 0,
                "task_states": ["TASK_STATE_SUCCEEDED"],
            },
            {
                "job_id": "/fixture/canary/inference-one",
                "state": "JOB_STATE_KILLED",
                "failures": 0,
                "task_states": ["TASK_STATE_KILLED"],
            },
        ]
    }


def activation(tmp_path, qualification, cleanup):
    return ProtocolActivation(
        previous_protocol_id=PROTOCOL_ID,
        solver_protocol_id=V2,
        activated_at="2026-11-27T17:01:00+00:00",
        qualification=artifact(tmp_path / "qualification.json", qualification),
        native_cleanup=artifact(tmp_path / "cleanup.json", cleanup),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("passed", False),
        ("native_server_closed", False),
        ("compilation_enabled", False),
        ("revision", "other-revision"),
        ("data_parallel", 1),
        ("minimum_output_floor_tokens", 1),
        ("completions", [{"index": 0, "qualified": True}]),
        ("completions", [{"index": i, "qualified": i != 7} for i in range(16)]),
    ],
)
def test_incomplete_or_unqualified_canary_cannot_activate(tmp_path, qualification, cleanup, field, value):
    qualification[field] = value
    with pytest.raises(ValueError):
        write_protocol_activation(str(tmp_path), activation(tmp_path, qualification, cleanup))
    assert not activation_path(str(tmp_path)).exists()


@pytest.mark.parametrize(
    "root_state,engine_task", [("JOB_STATE_RUNNING", "TASK_STATE_KILLED"), ("JOB_STATE_SUCCEEDED", "TASK_STATE_RUNNING")]
)
def test_native_completion_is_required_beyond_a_passed_probe(tmp_path, qualification, cleanup, root_state, engine_task):
    cleanup["jobs"][0]["state"] = root_state
    cleanup["jobs"][1]["task_states"] = [engine_task]
    with pytest.raises(ValueError):
        write_protocol_activation(str(tmp_path), activation(tmp_path, qualification, cleanup))
    assert not activation_path(str(tmp_path)).exists()


def test_activation_replay_preserves_evidence_and_refuses_legacy_restart_or_digest_drift(
    tmp_path, qualification, cleanup
):
    record = activation(tmp_path, qualification, cleanup)
    write_protocol_activation(str(tmp_path), record)
    wire = activation_path(str(tmp_path)).read_bytes()
    write_protocol_activation(str(tmp_path), record)
    require_protocol_activation(str(tmp_path), V2)
    assert activation_path(str(tmp_path)).read_bytes() == wire
    with pytest.raises(ValueError, match="explicitly select"):
        require_protocol_activation(str(tmp_path), PROTOCOL_ID)
    with pytest.raises(ValueError, match="immutable"):
        write_protocol_activation(str(tmp_path), record.model_copy(update={"activated_at": "2026-11-27T17:02:00+00:00"}))
    (tmp_path / "qualification.json").write_text('{"passed":false}')
    with pytest.raises(ValueError, match="digest mismatch"):
        require_protocol_activation(str(tmp_path), V2)
    assert activation_path(str(tmp_path)).read_bytes() == wire


@pytest.mark.parametrize("pool_protocol,worker_protocol", [(None, V2), (V2, PROTOCOL_ID)])
def test_workers_reject_a_pool_bound_to_the_other_request_policy(pool_protocol, worker_protocol):
    pool = {} if pool_protocol is None else {"solver_protocol_id": pool_protocol}
    with pytest.raises(ActionFailure) as rejected:
        runtime.require_pool_protocol(pool, Solver.LARGE, worker_protocol)
    assert rejected.value.reason.value == "SERVING_RELEASE_MISMATCH"
    runtime.require_pool_protocol({"solver_protocol_id": V2}, Solver.LARGE, V2)
    runtime.require_pool_protocol({}, Solver.SMALL, PROTOCOL_ID)


def test_launch_cli_runs_real_versioned_planner_and_replay_preserves_original_scores(
    tmp_path, monkeypatch, qualification, cleanup
):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    write_protocol_activation(str(ledger), activation(tmp_path, qualification, cleanup))
    imports = ledger / "v1/imports"
    imports.mkdir()
    (imports / "initial.json").write_text("{}")
    original = historical_row()
    append_rows(str(ledger), [original, replace(original, task_id="b", task_sha256="b" * 64, attempt_id="old-b")])
    append_rows(
        str(ledger),
        protocol_supersessions(
            read_rows(str(ledger)),
            solver=Solver.LARGE,
            previous_protocol_id=PROTOCOL_ID,
            replacement_protocol_id=V2,
            evidence_uri=str(activation_path(str(ledger))),
            activated_at="2026-11-27T17:01:00+00:00",
        ),
    )
    pools = tmp_path / "pools"
    pools.mkdir()
    pool = {
        "solver": "large",
        "model": SOLVER_MODELS[Solver.LARGE],
        "revision": PINS[Solver.LARGE].revision,
        "platform": "tpu",
        "region": "us-central2",
        "chip_count": 128,
        "gpu_count": 0,
        "ready_at": datetime.now(UTC).isoformat(),
        "solver_protocol_id": V2,
    }
    (pools / "large.json").write_text(json.dumps(pool))
    backend = JobStore(str(tmp_path / "accepted-jobs"))
    monkeypatch.setattr(controller, "IrisBackend", lambda *args: backend)
    monkeypatch.setattr(controller, "iris_ctx", lambda: SimpleNamespace(client=None, job_id="/fixture/controller"))
    monkeypatch.setattr(controller, "require_secret", lambda name: "fixture-refresh")
    monkeypatch.setattr(controller, "campaign_record", lambda ledger: {"holds": []})
    monkeypatch.setattr(executor, "campaign_record", lambda ledger: {"holds": []})
    monkeypatch.setattr(controller, "served_catalog", lambda *args, **kwargs: [SOURCE])
    monkeypatch.setattr(controller, "available_artifact_uris", lambda cells: frozenset())
    monkeypatch.setattr(controller, "_tpu_pool_is_live", lambda pool: True)
    monkeypatch.setattr(controller, "list_iris_jobs", lambda: [])
    monkeypatch.setattr(controller, "load_credentials", lambda _: SimpleNamespace(edge_refresh_token="fixture"))
    monkeypatch.setattr(controller, "controller_environment", lambda _: {})
    monkeypatch.setattr(cli, "get_iris_ctx", lambda: object())
    monkeypatch.setattr(cli, "configure_coreweave_s3", lambda: None)
    commands = []
    monkeypatch.setattr(controller.subprocess, "run", lambda command, **kwargs: commands.append(command))

    def end_tick(seconds):
        raise StopIteration

    monkeypatch.setattr(controller.time, "sleep", end_tick)
    controller.launch_controller(
        str(ledger),
        "fixture",
        False,
        frozenset(),
        "v2",
        None,
        serving_pool=str(pools),
        excluded_sources=frozenset(),
        large_protocol_id=V2,
    )
    command = commands[0]
    monkeypatch.setattr(sys, "argv", ["difficulty", *command[command.index("reconcile") :]])
    with pytest.raises(StopIteration):
        cli.main()
    first = {p.name: p.read_bytes() for p in (tmp_path / "accepted-jobs").glob("*.json")}
    actions = [action_from_payload(json.loads(payload)) for payload in first.values()]
    assert {action.solver for action in actions} == {Solver.LARGE, Solver.HOSTED}
    assert next(action for action in actions if action.solver == Solver.LARGE).solver_protocol_id == V2
    restored = read_rows(str(ledger))
    assert original in restored
    assert all(
        row.solver_protocol_id == V2
        for row in restored
        if row.solver == Solver.LARGE and row.outcome == Outcome.SUBMITTED
    )
    assert json.loads((ledger / "v1/budget.json").read_text())["limit_gpu_hours"] is None
    late = replace(original, attempt_id="late-old-finish", recorded_at="2026-11-27T17:02:00+00:00")
    append_rows(str(ledger), [late])
    with pytest.raises(StopIteration):
        cli.main()
    assert {p.name: p.read_bytes() for p in (tmp_path / "accepted-jobs").glob("*.json")} == first
    restored = read_rows(str(ledger))
    links = [row for row in restored if row.supersedes_attempt_id == late.attempt_id]
    assert len(links) == 1 and links[0].outcome == Outcome.PROTOCOL_SUPERSEDED
    assert links[0].recorded_at == "2026-11-27T17:01:00+00:00" and links[0].gpu_hours == 0
    assert late in restored
    # Neither controller generation may feed v2 requests to the unversioned pool.
    del pool["solver_protocol_id"]
    (pools / "large.json").write_text(json.dumps(pool))
    assert controller.qualified_tpu_solvers(str(pools), CoveragePolicy(large_protocol_id=V2)) == frozenset(
        {Solver.HOSTED}
    )


def test_v2_opt_in_without_qualification_cannot_reach_the_scheduler(tmp_path):
    # No Iris context or refresh token exists: qualification must fail first.
    with pytest.raises(ValueError, match="no immutable protocol activation"):
        controller.run_controller(
            str(tmp_path), "fixture", False, frozenset(), None, excluded_sources=frozenset(), large_protocol_id=V2
        )
    with pytest.raises(ValueError, match="no immutable protocol activation"):
        controller.launch_controller(
            str(tmp_path), "fixture", False, frozenset(), "v2", None, excluded_sources=frozenset(), large_protocol_id=V2
        )
    assert not list(tmp_path.iterdir())


def test_first_reconcile_refuses_unstaged_initial_supersessions_without_writing_rows(tmp_path, qualification, cleanup):
    write_protocol_activation(str(tmp_path), activation(tmp_path, qualification, cleanup))
    original = historical_row()
    append_rows(str(tmp_path), [original])
    before = {p.name: p.read_bytes() for p in (tmp_path / "v1/rows").glob("*.jsonl")}
    (tmp_path / "v1/imports").mkdir()
    (tmp_path / "v1/imports/initial.json").write_text("{}")
    with pytest.raises(ValueError, match="Stage initial Large supersessions"):
        executor.reconcile_once(
            str(tmp_path),
            [SOURCE],
            CoveragePolicy(large_protocol_id=V2),
            JobStore(str(tmp_path / "jobs")),
            executor.EscalationLog(str(tmp_path)),
            frozenset({Solver.LARGE}),
            frozenset(),
            None,
        )
    assert {p.name: p.read_bytes() for p in (tmp_path / "v1/rows").glob("*.jsonl")} == before
