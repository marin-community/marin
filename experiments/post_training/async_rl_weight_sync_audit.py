# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Strict receipt checks for the matched twenty-update Qwen weight-sync gate.

Inputs are exact-run normalized native events, not summaries inferred from plots.
The collector retains its SQL, source identity, task receipts and unmodified rows.
"""

import math
from collections import Counter
from itertools import pairwise
from statistics import median
from typing import Any


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


def audit_matched_gate(baseline, candidate):
    """Apply the frozen pause and stale-token thresholds without window exclusions."""
    assert baseline["config_identity"] == candidate["config_identity"], "unmatched recipe"
    results: dict[str, Any] = {}
    for name, arm in (("baseline", baseline), ("candidate", candidate)):
        assert arm["completed_updates"] == 20 and arm["failed_tasks"] == arm["engine_deaths"] == 0, "incomplete run"
        assert arm["preemptions"] == arm["retries"] == 0, "restarted run"
        assert len(arm["pause_seconds"]) == 20 and all(
            math.isfinite(value) and value >= 0 for value in arm["pause_seconds"]
        ), "pause coverage"
        assert 0 < arm["task_gpu_hours"] <= 4, "arm budget"
        results[name] = {"pause_p50": median(arm["pause_seconds"]), **audit_stale_tokens(arm["calls"], arm["outcomes"])}
    assert min(baseline["pause_seconds"]) >= 5.0, "baseline grace absent"
    assert results["candidate"]["pause_p50"] < 0.3, "pause threshold"
    assert (
        results["candidate"]["stale_fraction"] <= results["baseline"]["stale_fraction"]
    ), "stale-token fraction increased"
    results["requests"] = audit_requests(candidate["request_receipts"])
    return results
