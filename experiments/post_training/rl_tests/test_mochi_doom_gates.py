# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy

from experiments.post_training.mochi_doom_ftpo.gates import campaign_result


def test_paired_gates_require_every_domain_and_pooled_correctness():
    manifest = {"cohorts": dict.fromkeys(("math16k", "math32k", "if32k", "code32k"))}
    baseline = {
        "domains": {
            "math16k": {"N": 512, "confirmed": 12, "passes": 390},
            "math32k": {"N": 128, "confirmed": 20, "passes": 100},
            "if32k": {"N": 256, "confirmed": 10, "passes": 200},
            "code32k": {"N": 256, "confirmed": 9, "passes": 200},
        },
        "cohort_hashes": {name: name for name in manifest["cohorts"]},
        "target_status": {name: "non-target" if name == "code32k" else "target" for name in manifest["cohorts"]},
    }
    candidate = copy.deepcopy(baseline)
    for name, row in candidate["domains"].items():
        row["UB"] = 6 if name == "math16k" else (12 if name == "code32k" else row["confirmed"] // 2)
    candidate["domains"]["math16k"]["passes"] = 365
    candidate["domains"]["if32k"]["passes"] = 188
    result = campaign_result(manifest, baseline, candidate)
    assert result["meets_success"] is True
    assert result["repeat_final_greedy"] is True
    assert result["domains"]["math16k"]["passes_min"] == 365
    candidate["domains"]["math32k"]["passes"] = 99
    result = campaign_result(manifest, baseline, candidate)
    assert all(row["correctness_ok"] for row in result["domains"].values())
    assert result["panel"]["correctness_ok"] is False
    assert result["meets_success"] is False
    candidate["domains"]["math32k"]["passes"] = 100
    candidate["domains"]["code32k"]["UB"] = 13
    result = campaign_result(manifest, baseline, candidate)
    assert result["domains"]["code32k"]["loops_ok"] is False
    assert result["meets_success"] is False
    candidate["domains"]["code32k"]["UB"] = 12
    candidate["domains"]["math16k"]["passes"] = 364
    result = campaign_result(manifest, baseline, candidate)
    assert result["domains"]["math16k"]["correctness_ok"] is False
    assert result["meets_success"] is False
