# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from copy import deepcopy
from typing import Any

import pytest

from experiments.post_training import async_rl_weight_sync_audit as audit_module
from experiments.post_training.async_rl_weight_sync_audit import audit_requests, audit_stale_tokens


def multipart_receipt():
    state = {"requests": [f"{index:032x}" for index in range(256)]}
    payload = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("ascii")
    chunks = [payload[offset : offset + 3072] for offset in range(0, len(payload), 3072)]
    parts = [
        {
            "engine_index": 0,
            "step": 1,
            "moment": "before_pause",
            "receipt_json": chunk.decode("ascii"),
            "receipt_sha256": hashlib.sha256(payload).hexdigest(),
            "receipt_bytes": len(payload),
            "part_count": len(chunks),
            "part_index": index,
        }
        for index, chunk in enumerate(chunks)
    ]
    return state, parts


def test_reassembles_shuffled_receipt_parts_exactly():
    state, parts = multipart_receipt()
    assert audit_module.reassemble_request_receipts(list(reversed(parts))) == [
        {"engine_index": 0, "step": 1, "moment": "before_pause", "state": state}
    ]


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "digest", "metadata", "payload", "bound", "moment"])
def test_rejects_missing_or_corrupt_receipt_parts(corruption):
    _, parts = multipart_receipt()
    if corruption == "missing":
        parts.pop()
    elif corruption == "duplicate":
        parts[-1] = deepcopy(parts[0])
    elif corruption == "digest":
        for part in parts:
            part["receipt_sha256"] = "0" * 64
    elif corruption == "metadata":
        parts[0]["receipt_bytes"] += 1
    elif corruption == "payload":
        parts[0]["receipt_json"] = "x" + parts[0]["receipt_json"][1:]
    elif corruption == "bound":
        for part in parts:
            part["receipt_bytes"] = (1 << 20) + 1
    elif corruption == "moment":
        parts[0]["moment"] = "unknown"
    with pytest.raises(AssertionError):
        audit_module.reassemble_request_receipts(parts)


def receipts() -> list[dict[str, Any]]:
    rows = []
    for engine in range(2):
        boundaries = [[1.0, 0]]
        for index, (moment, step) in enumerate(
            [("initial", 0), ("before_pause", 1), ("after_pause", 1), ("after_resume", 1), ("final", 1)]
        ):
            if moment == "after_resume":
                boundaries.append([4.0, 1])
            accounting = {
                "active_ids": ["request"] if moment == "before_pause" else [],
                "started_ids": ["request"] if moment == "before_pause" else [],
                "terminal": [],
                "pauses": [],
                "pause_count": int(index >= 2),
            }
            if moment == "after_pause":
                accounting["terminal"] = [
                    {
                        "request_id": "request",
                        "reason": "abort",
                        "tokens": 2,
                        "first_token_time": 2.0,
                        "policy_version_at_first_token": 0,
                    }
                ]
                accounting["pauses"] = [
                    {
                        "pause_index": 1,
                        "monotonic_time": 2.5,
                        "active_before": ["request"],
                        "frontend_before": ["request"],
                    }
                ]
            rows.append(
                {
                    "engine_index": engine,
                    "moment": moment,
                    "step": step,
                    "state": {
                        "host": "engine-host",
                        "actor_pid": engine + 1,
                        "core_pids": [engine + 3],
                        "shared_time_and_uts_namespaces": True,
                        "clock_domain": "CLOCK_MONOTONIC",
                        "observed_monotonic": index + 1.1,
                        "terminal_wait_seconds": 0.0,
                        "paused": moment == "after_pause",
                        "policy_version_boundaries": deepcopy(boundaries),
                        "request_accounting": accounting,
                    },
                }
            )
    return rows


def test_exact_native_request_identity_and_first_token_audit():
    result = audit_requests(receipts(), steps=1, engines=2)
    assert result["requests"] == result["cohort_terminal_abort"] == 2
    assert result["native_frontend_cohort_requests"] == result["engine_syncs"] == 2


@pytest.mark.parametrize(
    "corruption", ["lost", "duplicate", "restart", "version", "clock", "unpaused", "missing_engine", "missing_receipt"]
)
def test_native_audit_rejects_missing_or_inconsistent_evidence(corruption):
    rows = receipts()
    state = rows[2]["state"]
    ledger = state["request_accounting"]
    if corruption == "lost":
        ledger["terminal"] = []
    elif corruption == "duplicate":
        ledger["terminal"] *= 2
    elif corruption == "restart":
        state["actor_pid"] += 10
    elif corruption == "version":
        ledger["terminal"][0]["policy_version_at_first_token"] = 1
    elif corruption == "clock":
        state["shared_time_and_uts_namespaces"] = False
    elif corruption == "unpaused":
        state["paused"] = False
    elif corruption == "missing_engine":
        rows = rows[:5]
    elif corruption == "missing_receipt":
        rows.pop()
    with pytest.raises(AssertionError):
        audit_requests(rows, steps=1, engines=2)


def test_shutdown_cancellation_is_reported_as_cancellation_not_completion():
    rows = receipts()
    terminal = rows[2]["state"]["request_accounting"]["terminal"][0]
    terminal.update(reason="CancelledError", tokens=0, first_token_time=None, policy_version_at_first_token=None)
    result = audit_requests(rows, steps=1, engines=2)
    assert result["cohort_terminal_CancelledError"] == 1
    assert result["cohort_terminal_abort"] == 1


def token_receipts():
    calls = [
        {"call_id": str(i), "mode": "async", "outcome": "success", "response_tokens": tokens}
        for i, tokens in enumerate([10, 20, 30])
    ]
    outcomes = [
        {"call_id": str(i), "tokens": tokens, "outcome": outcome}
        for i, (tokens, outcome) in enumerate([(10, "consumed"), (20, "stale_enqueue"), (30, "epoch_discarded")])
    ]
    return calls, outcomes


def test_stale_fraction_uses_all_completed_group_tokens_once():
    result = audit_stale_tokens(*token_receipts())
    assert result == {
        "stale_tokens": 20,
        "completed_group_tokens": 60,
        "stale_fraction": 1 / 3,
        "completed_groups": 3,
        "outcome_groups": {"consumed": 1, "stale_enqueue": 1, "epoch_discarded": 1},
    }


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "tokens"])
def test_stale_fraction_requires_identity_and_token_coverage(corruption):
    calls, outcomes = token_receipts()
    if corruption == "missing":
        outcomes.pop()
    elif corruption == "duplicate":
        outcomes.append(outcomes[0])
    else:
        outcomes[0]["tokens"] += 1
    with pytest.raises(AssertionError):
        audit_stale_tokens(calls, outcomes)


@pytest.mark.parametrize("corruption", [None, "slow", "stale", "missing_pause", "failure", "budget", "recipe"])
@pytest.mark.parametrize("ceiling", [4.0, 16 * 1050 / 3600])
def test_matched_thresholds_reject_incomplete_or_worse_candidate(monkeypatch, corruption, ceiling):
    calls, outcomes = token_receipts()
    baseline: dict[str, Any] = {
        "config_identity": {"batch": 64, "seed": 17},
        "completed_updates": 20,
        "failed_tasks": 0,
        "engine_deaths": 0,
        "preemptions": 0,
        "retries": 0,
        "pause_seconds": [5.1] * 20,
        "task_gpu_hours": ceiling,
        "calls": calls,
        "outcomes": outcomes,
    }
    candidate = deepcopy(baseline)
    candidate["pause_seconds"] = [0.1] * 20
    candidate["request_receipts"] = receipts()
    # This test isolates arm comparison; the same native auditor still runs on
    # the smaller one-sync/two-engine fixture above.
    monkeypatch.setattr(audit_module, "audit_requests", lambda rows: audit_requests(rows, steps=1, engines=2))
    if corruption == "slow":
        candidate["pause_seconds"] = [0.3] * 20
    elif corruption == "stale":
        candidate["outcomes"][0]["outcome"] = "stale"
    elif corruption == "missing_pause":
        candidate["pause_seconds"].pop()
    elif corruption == "failure":
        candidate["failed_tasks"] = 1
    elif corruption == "budget":
        candidate["task_gpu_hours"] = ceiling + 0.01
    elif corruption == "recipe":
        candidate["config_identity"]["seed"] = 18
    if corruption is None:
        result = audit_module.audit_matched_gate(baseline, candidate, task_gpu_hours_ceiling=ceiling)
        assert result["candidate"]["pause_p50"] == 0.1
    else:
        with pytest.raises(AssertionError):
            audit_module.audit_matched_gate(baseline, candidate, task_gpu_hours_ceiling=ceiling)


def test_completed_generation_canceled_at_enqueue_still_counts_in_denominator():
    calls, outcomes = token_receipts()
    outcomes[0]["outcome"] = "cancelled_before_enqueue"
    result = audit_stale_tokens(calls, outcomes)
    assert result["completed_group_tokens"] == 60
    assert result["outcome_groups"]["cancelled_before_enqueue"] == 1


def test_unknown_group_outcome_is_rejected():
    calls, outcomes = token_receipts()
    outcomes[0]["outcome"] = "stale_typo"
    with pytest.raises(AssertionError, match="unknown group outcome"):
        audit_stale_tokens(calls, outcomes)


@pytest.mark.parametrize(
    "corruption", [None, "sequences", "mask", "stop", "group", "lost", "queued", "missing", "abnormal"]
)
def test_stress_work_requires_forced_tokens_and_clean_exporter(corruption):
    arm = {
        "consumed_sequences": 5120,
        "consumed_response_tokens": 5120 * 1024,
        "consumed_loss_tokens": 5120 * 1024,
        "consumed_length_stops": 5120,
        "outcomes": [{"outcome": "consumed", "tokens": 4096} for _ in range(1280)],
        "exporter_terminals": [
            {
                "role": role,
                "export_lost_records": 0,
                "export_queued_records": 0,
                "reason": "normal_exit",
                "status": "completed",
            }
            for role in ("trainer", "driver", "controller", "worker")
        ],
    }
    if corruption == "sequences":
        arm["consumed_sequences"] -= 1
    elif corruption == "mask":
        arm["consumed_loss_tokens"] -= 1
    elif corruption == "stop":
        arm["consumed_length_stops"] -= 1
    elif corruption == "group":
        arm["outcomes"][0]["tokens"] -= 1
    elif corruption == "lost":
        arm["exporter_terminals"][0]["export_lost_records"] = 1
    elif corruption == "queued":
        arm["exporter_terminals"][0]["export_queued_records"] = 1
    elif corruption == "missing":
        arm["exporter_terminals"] = []
    elif corruption == "abnormal":
        arm["exporter_terminals"][0]["reason"] = "exception"
    if corruption is None:
        audit_module.audit_stress_work(arm)
    else:
        with pytest.raises(AssertionError):
            audit_module.audit_stress_work(arm)


def test_positive_timestamp_on_zero_token_abort_is_not_sampled_coverage():
    rows = receipts()
    zero = next(row for row in rows if row["engine_index"] == 0 and row["moment"] == "after_pause")
    terminal = zero["state"]["request_accounting"]["terminal"][0]
    terminal["tokens"] = 0
    terminal["native_first_token_time"] = terminal["first_token_time"]
    result = audit_requests(rows, steps=1, engines=2)
    assert result["requests"] == result["terminal_abort"] == 2
    assert result["requests_with_first_token"] == 1
