# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from experiments.post_training.rl_tests.fixtures import (
    KL,
    compare,
    controls,
)

pytestmark = [pytest.mark.usefixtures("source_provenance"), pytest.mark.timeout(180)]


def test_compare_leave_one_out_writes_the_printed_report(recorded_runs):
    controls(recorded_runs, "speed-medium")
    report = compare("speed-medium", "--leave-out", "r0")
    assert (report["verdict"], report["effect"]) == ("pass", "no change"), report
    artifact = Path(report.pop("report"))
    assert json.loads(artifact.read_text()) == report and artifact.with_suffix(".md").exists()
    assert {row["role"] for row in report["comparisons"]} >= {"objective", "guard", "reported"}
    assert all(row["comparator"] is not None for row in report["invariants"])
    for name, speed, kl, verdict, effect in (
        ("improvement", 90, 0.1, "pass", "improves"),
        ("regression", 150, 0.1, "fail", "regression"),
        ("trade-off", 90, 0.2, "fail", "trade-off"),
    ):
        candidate = "users/test/" + name
        recorded_runs("speed-medium", candidate, speed=speed, metrics={KL: kl})
        result = compare("speed-medium", "--candidate", candidate)
        assert (result["verdict"], result["effect"]) == (verdict, effect), result
