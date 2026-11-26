# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the registered 0.75 correctness multiplier receipts."""

from datetime import UTC, datetime

import pytest

from experiments.post_training.mochi_loop_guards.user_overrides import (
    CLOCK_RECEIPT_NAME,
    CONSERVATION_RECEIPT_NAME,
    GATE_RECEIPT_NAME,
    override_receipts,
)

NOW = datetime(2026, 11, 26, 17, 0, tzinfo=UTC)


def cohort(name, passes, correctness_min_raw, ub_count=None, ub_cap=None):
    row = {"cohort": name, "passes": passes, "correctness_min_raw": correctness_min_raw}
    if ub_count is not None:
        row.update({"action_loop_UB_count": ub_count, "primary_UB_cap_raw": ub_cap})
    return row


def gate_receipt(cohorts, passes):
    return {"cohorts": cohorts, "passes": passes}


# The registered S1 numbers: correctness raw minimums 40.65 / 56 / 104.2, pooled 222.54;
# passes 59/69/101 (stop) and 61/70/103 (rp); pooled 229/234.
STOP_RECEIPT = gate_receipt(
    [
        cohort("tb2", 59, 40.65),
        cohort("tblite", 69, 56),
        cohort("swesmith-ho", 101, 104.2),
    ],
    229,
)
RP_RECEIPT = gate_receipt(
    [
        cohort("tb2", 61, 40.65, ub_count=46, ub_cap=17),
        cohort("tblite", 70, 56, ub_count=16, ub_cap=10),
        cohort("swesmith-ho", 103, 104.2, ub_count=24, ub_cap=6.5),
    ],
    234,
)


def test_override_recomputes_unrounded_thresholds_and_keeps_the_fail_verdict():
    receipts = override_receipts(STOP_RECEIPT, RP_RECEIPT, now=NOW)
    gate = receipts[GATE_RECEIPT_NAME]
    stop, rp = (result["under_user_override"] for result in gate["results"])

    tb2, tblite, held_out = stop["cohorts"]
    assert tb2["correctness_min_raw"] == pytest.approx(30.4875)
    assert tb2["correctness_min_integer"] == 31
    assert tblite["correctness_min_raw"] == pytest.approx(42.0)
    assert tblite["correctness_min_integer"] == 42
    assert held_out["correctness_min_raw"] == pytest.approx(78.15)
    assert held_out["correctness_min_integer"] == 79
    assert all(row["cohort_correctness_pass"] for row in stop["cohorts"])
    assert stop["pooled_correctness_min_raw"] == pytest.approx(166.905)
    assert stop["pooled_correctness_min_integer"] == 167
    assert stop["pooled_correctness_pass"] is True

    assert stop["final_guard_verdict"] == "FAIL"
    assert stop["failure_reasons"] == ["TBLite repeat-turn share exceeds unchanged I1 cap"]
    assert rp["failure_reasons"] == ["All three action UB counts exceed unchanged I2 caps"]
    assert not stop["repeat_triggered"] and not rp["repeat_triggered"]

    # The originals stay on record side by side with the recomputed verdicts.
    assert gate["results"][0]["original"] == STOP_RECEIPT
    assert gate["results"][0]["under_user_override"] is not STOP_RECEIPT
    assert gate["correctness_multiplier"] == 0.75
    assert gate["repeat_trigger_uses_unrounded_thresholds"] is True
    assert sorted(receipts) == sorted((CLOCK_RECEIPT_NAME, GATE_RECEIPT_NAME, CONSERVATION_RECEIPT_NAME))


def test_override_refuses_receipts_when_a_correctness_repeat_window_fires():
    # passes 32 sits 1.5125 above the overridden TB2 threshold 30.4875, inside ±2.
    boundary = gate_receipt([cohort("tb2", 32, 40.65), cohort("tblite", 69, 56), cohort("swesmith-ho", 101, 104.2)], 229)
    with pytest.raises(ValueError, match="repeat window"):
        override_receipts(boundary, RP_RECEIPT, now=NOW)


def test_override_refuses_receipts_when_an_rp_ub_window_fires():
    # 12 counts sits 2 above the unchanged TBLite UB cap, inside ±2.
    boundary = gate_receipt(
        [
            cohort("tb2", 61, 40.65, ub_count=46, ub_cap=17),
            cohort("tblite", 70, 56, ub_count=12, ub_cap=10),
            cohort("swesmith-ho", 103, 104.2, ub_count=24, ub_cap=6.5),
        ],
        234,
    )
    with pytest.raises(ValueError, match="repeat window"):
        override_receipts(STOP_RECEIPT, boundary, now=NOW)
