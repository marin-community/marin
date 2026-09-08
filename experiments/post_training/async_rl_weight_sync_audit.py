# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Strict receipt checks for the matched twenty-update Qwen weight-sync gate.

Inputs are exact-run normalized native events, not summaries inferred from plots.
The collector retains its SQL, source identity, task receipts and unmodified rows.
"""

import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import pairwise
from statistics import median
from typing import Any


def reassemble_request_receipts(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover exact byte-bounded states, rejecting incomplete or conflicting parts."""
    groups = defaultdict(list)
    moments = {"initial": -1, "before_pause": 0, "after_pause": 1, "after_resume": 2, "final": 3}
    for event in events:
        assert event["moment"] in moments, "unknown receipt moment"
        groups[event["engine_index"], event["step"], event["moment"]].append(event)
    receipts = []
    for (engine, step, moment), parts in groups.items():
        first = parts[0]
        count, size = first["part_count"], first["receipt_bytes"]
        assert type(count) is int and type(size) is int and 0 < size <= 1 << 20, "receipt bound"
        assert count == (size + 3071) // 3072 and len(parts) == count, "receipt part coverage"
        assert {part["part_index"] for part in parts} == set(range(count)), "duplicate or missing receipt part"
        assert all(type(part["part_index"]) is int for part in parts), "receipt part index"
        assert all(
            (part["part_count"], part["receipt_bytes"], part["receipt_sha256"]) == (count, size, first["receipt_sha256"])
            for part in parts
        ), "conflicting receipt metadata"
        ordered = sorted(parts, key=lambda part: part["part_index"])
        chunks = [part["receipt_json"].encode("ascii") for part in ordered]
        assert all(len(chunk) == 3072 for chunk in chunks[:-1]) and 0 < len(chunks[-1]) <= 3072, "chunk bound"
        payload = b"".join(chunks)
        assert len(payload) == size and hashlib.sha256(payload).hexdigest() == first["receipt_sha256"], "receipt digest"
        receipts.append({"engine_index": engine, "step": step, "moment": moment, "state": json.loads(payload)})
    return sorted(receipts, key=lambda receipt: (receipt["engine_index"], receipt["step"], moments[receipt["moment"]]))


def audit_requests(receipts: list[dict[str, Any]], *, steps=20, engines=8):
    """Join every started attempt and pause cohort to a native terminal identity."""
    expected = (
        [("initial", 0)]
        + [(moment, step) for step in range(1, steps + 1) for moment in ("before_pause", "after_pause", "after_resume")]
        + [("final", steps)]
    )
    assert {row["engine_index"] for row in receipts} == set(range(engines)), "engine coverage"
    totals = Counter()
    for engine in range(engines):
        rows = [row for row in receipts if row["engine_index"] == engine]
        assert [(row["moment"], row["step"]) for row in rows] == expected, "receipt sequence"
        started, terminal, active, cohort, frontend_cohort = set(), {}, set(), set(), set()
        pauses = []
        previous_boundaries = []
        origin = None
        previous_clock = 0.0
        for row in rows:
            state = row["state"]
            here = (state["host"], state["actor_pid"], tuple(state["core_pids"]))
            assert all(here) and state["shared_time_and_uts_namespaces"], "engine origin"
            origin = here if origin is None else origin
            assert here == origin, "engine restart"
            assert state["clock_domain"] == "CLOCK_MONOTONIC", "clock domain"
            clock = state["observed_monotonic"]
            assert math.isfinite(clock) and clock > previous_clock, "clock ordering"
            previous_clock = clock
            if row["moment"] == "after_pause":
                assert state["paused"], "scheduler not paused"
            if row["moment"] in ("initial", "after_resume", "final"):
                assert not state["paused"], "scheduler left paused"
            boundaries = [tuple(boundary) for boundary in state["policy_version_boundaries"]]
            assert boundaries[: len(previous_boundaries)] == previous_boundaries, "boundary history changed"
            versions = list(range(row["step"] + (row["moment"] in ("after_resume", "final"))))
            if row["moment"] == "initial":
                versions = [0]
            assert [version for _, version in boundaries] == versions, "installed versions"
            assert all(0 < stamp <= clock for stamp, _ in boundaries), "boundary clock"
            assert all(a[0] < b[0] for a, b in pairwise(boundaries)), "boundary ordering"
            previous_boundaries = boundaries
            wait_seconds = state["terminal_wait_seconds"]
            assert math.isfinite(wait_seconds) and 0 <= wait_seconds <= 30.0, "terminal acknowledgement timing"
            if row["moment"] == "final":
                totals["final_terminal_wait_max_seconds"] = max(totals["final_terminal_wait_max_seconds"], wait_seconds)
            else:
                assert wait_seconds == 0, "unexpected terminal wait during training"
            ledger = state["request_accounting"]
            new = ledger["started_ids"]
            assert len(set(new)) == len(new) and not started.intersection(new), "duplicate starts"
            started.update(new)
            active.update(new)
            for result in ledger["terminal"]:
                request_id = result["request_id"]
                assert request_id in active and request_id not in terminal, "missing or duplicate terminal"
                terminal[request_id] = result
                active.remove(request_id)
                reason, tokens = result["reason"], result["tokens"]
                assert reason in {"stop", "length", "abort", "CancelledError"}, "request failure"
                assert type(tokens) is int and tokens >= 0, "token count"
                stamp, version = result["first_token_time"], result["policy_version_at_first_token"]
                if tokens:
                    assert stamp is not None and math.isfinite(stamp) and 0 < stamp <= clock, "first-token coverage"
                    assigned = next((v for boundary, v in reversed(boundaries) if stamp >= boundary), None)
                    assert assigned is not None and version == assigned, "first-token version mismatch"
                    totals["requests_with_first_token"] += 1
                else:
                    assert stamp is None or (math.isfinite(stamp) and 0 < stamp <= clock), "first-token clock"
                totals["terminal_" + reason] += 1
            assert set(ledger["active_ids"]) == active, "active identity conservation"
            for pause in ledger["pauses"]:
                assert pause["pause_index"] == len(pauses) + 1, "pause sequence"
                assert set(pause["frontend_before"]) <= set(pause["active_before"]) <= started, "pause identity"
                assert 0 < pause["monotonic_time"] <= clock, "pause clock"
                pauses.append(pause)
                cohort.update(pause["active_before"])
                frontend_cohort.update(pause["frontend_before"])
            assert ledger["pause_count"] == len(pauses), "pause receipt loss"
        assert len(pauses) == steps, "pause coverage"
        assert started and started == set(terminal) and not active, "unaccounted final requests"
        assert frontend_cohort, "vacuous abort cohort"
        assert cohort <= set(terminal), "unaccounted pause cohort"
        for request_id in cohort:
            totals["cohort_terminal_" + terminal[request_id]["reason"]] += 1
        totals["requests"] += len(started)
        totals["pause_cohort_requests"] += len(cohort)
        totals["native_frontend_cohort_requests"] += len(frontend_cohort)
        totals["engine_syncs"] += len(pauses)
    assert totals["requests_with_first_token"] > 0 and totals["terminal_abort"] > 0, "no measured abort work"
    return dict(totals)


def audit_stale_tokens(calls, outcomes):
    """Count each completed training group once, matching native call receipts."""
    training = [call for call in calls if call["mode"] == "async"]
    assert len({call["call_id"] for call in training}) == len(training), "duplicate call receipt"
    completed = {call["call_id"]: call for call in training if call["outcome"] == "success"}
    assert completed, "no completed training calls"
    assert len({row["call_id"] for row in outcomes}) == len(outcomes), "duplicate outcome receipt"
    assert {row["call_id"] for row in outcomes} == set(completed), "completed group coverage"
    stale = total = 0
    for row in outcomes:
        # Frozen producer vocabulary: explicit terminals in fully_async_trainer,
        # AdmissionRejection values, and rejected GroupSelectionResult values.
        assert row["outcome"] in {
            "consumed",
            "epoch_discarded",
            "stale_enqueue",
            "shutdown_pending",
            "cancelled_before_enqueue",
            "failed_before_enqueue",
            "duplicate",
            "stale",
            "fully_masked",
            "physical_group_size",
            "below_minimum_group_size",
            "missing_rollout_logprobs",
            "duplicate_uid",
            "dynamic_insufficient_reward_spread",
        }, "unknown group outcome"
        tokens = row["tokens"]
        assert type(tokens) is int and tokens >= 0, "token count"
        assert tokens == completed[row["call_id"]]["response_tokens"], "call/group token disagreement"
        total += tokens
        if row["outcome"] in {"stale", "stale_enqueue"}:
            stale += tokens
    assert total > 0, "zero denominator"
    return {
        "stale_tokens": stale,
        "completed_group_tokens": total,
        "stale_fraction": stale / total,
        "completed_groups": len(completed),
        "outcome_groups": dict(Counter(row["outcome"] for row in outcomes)),
    }


def audit_matched_gate(baseline, candidate, *, task_gpu_hours_ceiling: float):
    """Apply the frozen pause and stale-token thresholds without window exclusions."""
    assert math.isfinite(task_gpu_hours_ceiling) and task_gpu_hours_ceiling > 0, "invalid allocation ceiling"
    assert baseline["config_identity"] == candidate["config_identity"], "unmatched recipe"
    results: dict[str, Any] = {}
    for name, arm in (("baseline", baseline), ("candidate", candidate)):
        assert arm["completed_updates"] == 20 and arm["failed_tasks"] == arm["engine_deaths"] == 0, "incomplete run"
        assert arm["preemptions"] == arm["retries"] == 0, "restarted run"
        assert len(arm["pause_seconds"]) == 20 and all(
            math.isfinite(value) and value >= 0 for value in arm["pause_seconds"]
        ), "pause coverage"
        assert 0 < arm["task_gpu_hours"] <= task_gpu_hours_ceiling, "arm budget"
        results[name] = {"pause_p50": median(arm["pause_seconds"]), **audit_stale_tokens(arm["calls"], arm["outcomes"])}
    assert min(baseline["pause_seconds"]) >= 5.0, "baseline grace absent"
    assert results["candidate"]["pause_p50"] < 0.3, "pause threshold"
    assert (
        results["candidate"]["stale_fraction"] <= results["baseline"]["stale_fraction"]
    ), "stale-token fraction increased"
    results["requests"] = audit_requests(candidate["request_receipts"])
    return results


def audit_stress_work(arm):
    """Reject forced-length mask collapse and incomplete native exporter accounting."""
    assert arm["consumed_sequences"] == 5120, "stress sequence coverage"
    assert arm["consumed_response_tokens"] == arm["consumed_loss_tokens"] == 5120 * 1024, "stress loss-mask coverage"
    assert arm["consumed_length_stops"] == 5120, "stress forced-length coverage"
    consumed = [row for row in arm["outcomes"] if row["outcome"] == "consumed"]
    assert len(consumed) == 1280 and all(row["tokens"] == 4 * 1024 for row in consumed), "stress group coverage"
    terminals = arm["exporter_terminals"]
    assert Counter(row["role"] for row in terminals) == Counter(
        {"trainer": 1, "driver": 1, "controller": 1, "worker": 1}
    ), "missing or duplicate exporter terminal"
    assert all(
        row["export_lost_records"] == row["export_queued_records"] == 0
        and row["reason"] == "normal_exit"
        and row["status"] == "completed"
        for row in terminals
    ), "exporter loss or incomplete drain"


def audit_matched_stress_gate(baseline, candidate, *, task_gpu_hours_ceiling: float):
    """Apply the prospectively matched active-load protocol without weakening the original gate."""
    for arm in (baseline, candidate):
        audit_stress_work(arm)
    results = audit_matched_gate(baseline, candidate, task_gpu_hours_ceiling=task_gpu_hours_ceiling)
    results["baseline_requests"] = audit_requests(baseline["request_receipts"])
    assert results["requests"]["pause_cohort_requests"] >= 64, "insufficient active stress cohort"
    return results
