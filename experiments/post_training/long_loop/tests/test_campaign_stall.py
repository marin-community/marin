# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

from experiments.post_training.long_loop.campaign_stall_e2e import run_proof


def test_authored_campaign_stop_keeps_final_state_grading():
    receipt = asyncio.run(run_proof())
    assert [case["verifier_reward"] for case in receipt["cases"]] == [1.0, 0.0]
    assert all(case["stop_reason"] == "repeat_stall" and case["turns"] == 3 for case in receipt["cases"])
