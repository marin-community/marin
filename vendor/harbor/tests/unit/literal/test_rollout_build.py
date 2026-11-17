"""Unit tests for harbor.literal.rollout_build.

Covers the shared pair->RolloutDetail parity builder and the per-trial id-based
correlation over a SHARED controller proxy log — including the load-bearing
GRPO n>1 identical-seed case (no cross-trial bleed, no empty-from-ambiguity) and
the opencode auto-compaction boundary (ordering by arrival within a trial id).
"""

import json
import logging

from harbor.literal.rollout_build import (
    build_rollout_details_for_trial,
    build_rollout_details_from_pairs,
    read_literal_pairs,
)


def test_read_literal_pairs_reports_and_skips_malformed_records(tmp_path, caplog):
    log_path = tmp_path / "literal.jsonl"
    pair = {"completion_token_ids": [1], "logprobs": [-0.1]}
    log_path.write_text(f"not-json\n{json.dumps({'literal': pair})}\n")

    with caplog.at_level(logging.WARNING):
        assert read_literal_pairs(log_path) == [pair]

    assert "Skipped 1 malformed RecordProxy literal record" in caplog.text


# --------------------------------------------------------------------------- #
# build_rollout_details_from_pairs — parity guard
# --------------------------------------------------------------------------- #
def test_from_pairs_happy_path_and_parity():
    pairs = [
        {
            "prompt_token_ids": [1, 2],
            "completion_token_ids": [3, 4],
            "logprobs": [-0.5, -0.6],
        },
        {
            "prompt_token_ids": [5, 6, 7],
            "completion_token_ids": [8, 9],
            "logprobs": [-0.7],  # length mismatch (1 vs 2)
        },
    ]
    details = build_rollout_details_from_pairs(pairs)
    assert len(details) == 1
    rd = details[0]
    assert rd["prompt_token_ids"] == [[1, 2], [5, 6, 7]]
    assert rd["completion_token_ids"] == [[3, 4], [8, 9]]
    assert rd["logprobs"] == [[-0.5, -0.6], []]  # mismatch turn -> []
    assert len(rd["logprobs"]) == len(rd["completion_token_ids"])


def test_from_pairs_missing_logprobs_records_empty():
    details = build_rollout_details_from_pairs(
        [{"prompt_token_ids": [1], "completion_token_ids": [2, 3]}]
    )
    assert details[0]["completion_token_ids"] == [[2, 3]]
    assert details[0]["logprobs"] == [[]]


def test_from_pairs_empty_when_no_token_data():
    assert build_rollout_details_from_pairs([{}]) == []
    assert build_rollout_details_from_pairs([]) == []


# --------------------------------------------------------------------------- #
# build_rollout_details_for_trial — id-based correlation over a shared log
# --------------------------------------------------------------------------- #
def _entry(trial_id, ts, pids, cids, lps, status=200):
    return {
        "timestamp": ts,
        "status_code": status,
        "trial_id": trial_id,
        "request": {"messages": []},
        "literal": {
            "prompt_token_ids": pids,
            "completion_token_ids": cids,
            "logprobs": lps,
        },
    }


def test_for_trial_selects_only_its_own_turns_in_order():
    entries = [
        _entry("A", 1.0, [1], [10], [-0.1]),
        _entry("B", 1.1, [1], [20], [-0.2]),
        _entry("A", 2.0, [1, 10], [11], [-0.3]),
    ]
    a = build_rollout_details_for_trial(entries, "A")
    assert a[0]["completion_token_ids"] == [[10], [11]]
    assert a[0]["logprobs"] == [[-0.1], [-0.3]]
    b = build_rollout_details_for_trial(entries, "B")
    assert b[0]["completion_token_ids"] == [[20]]


def test_for_trial_orders_by_timestamp_not_log_order():
    # Out-of-order arrival in the log; must be sorted by timestamp.
    entries = [
        _entry("A", 3.0, [1, 1], [30], [-0.3]),
        _entry("A", 1.0, [1], [10], [-0.1]),
        _entry("A", 2.0, [1, 1], [20], [-0.2]),
    ]
    rd = build_rollout_details_for_trial(entries, "A")[0]
    assert rd["completion_token_ids"] == [[10], [20], [30]]


def test_for_trial_n_gt_1_identical_seed_no_bleed():
    """The GRPO case content-correlation FAILS: n=3 rollouts of the IDENTICAL
    prompt, interleaved in one shared log. Id correlation gives each its OWN
    turns with no cross-rollout bleed and no empty-from-ambiguity."""
    seed = [{"role": "user", "content": "same task"}]
    entries = []
    ts = 0.0
    # Interleave 3 trials, each with 2 turns; identical seed messages.
    for turn in range(2):
        for name, base in (("A", 100), ("B", 200), ("C", 300)):
            ts += 1.0
            e = _entry(name, ts, [1] * (turn + 1), [base + turn], [-0.1 * (turn + 1)])
            e["request"]["messages"] = seed  # identical seed => content-ambiguous
            entries.append(e)
    for name, base in (("A", 100), ("B", 200), ("C", 300)):
        rd = build_rollout_details_for_trial(entries, name)[0]
        assert rd["completion_token_ids"] == [[base + 0], [base + 1]]
        # No other trial's tokens leaked in.
        flat = [t for turn in rd["completion_token_ids"] for t in turn]
        for other in (100, 200, 300):
            if other != base:
                assert other not in flat and other + 1 not in flat


def test_for_trial_spans_compaction_boundary():
    """S4: opencode auto-compaction starts a NEW session mid-trial. The trial id
    is stable across the boundary, so all pre+post-compaction turns (interleaved
    with another trial) form ONE continuous per-turn list ordered by arrival."""
    entries = [
        # session 1 (pre-compaction)
        _entry("A", 1.0, [1], [10], [-0.1]),
        _entry("B", 1.5, [1], [99], [-0.9]),
        _entry("A", 2.0, [1, 10], [11], [-0.2]),
        # compaction fires -> session 2 (new prompt = summary; ids discontinuous)
        _entry("A", 3.0, [7, 7, 7], [12], [-0.3]),
        _entry("A", 4.0, [7, 7, 7, 12], [13], [-0.4]),
    ]
    rd = build_rollout_details_for_trial(entries, "A")[0]
    assert rd["completion_token_ids"] == [[10], [11], [12], [13]]
    assert rd["logprobs"] == [[-0.1], [-0.2], [-0.3], [-0.4]]
    assert len(rd["logprobs"]) == len(rd["completion_token_ids"])


def test_for_trial_skips_non_200_and_empty_completion():
    entries = [
        _entry("A", 1.0, [1], [10], [-0.1]),
        _entry("A", 2.0, [1], None, None, status=500),  # error turn
        {"timestamp": 3.0, "status_code": 200, "trial_id": "A", "literal": None},
        _entry("A", 4.0, [1, 10], [], []),  # empty completion
        _entry("A", 5.0, [1, 10], [11], [-0.2]),
    ]
    rd = build_rollout_details_for_trial(entries, "A")[0]
    assert rd["completion_token_ids"] == [[10], [11]]


def test_for_trial_empty_when_no_match():
    entries = [_entry("A", 1.0, [1], [10], [-0.1])]
    assert build_rollout_details_for_trial(entries, "Z") == []
    assert build_rollout_details_for_trial(entries, "") == []
