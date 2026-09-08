# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests against vacuous or incomplete correction-screen receipts."""

import pytest

from experiments.post_training.async_rl_correction_audit import (
    CORRECTION_FIELDS,
    audit_completed_noise_band,
    audit_correction_history,
    audit_source_control,
)


def receipt(correction="regular_mask", minibatches=1):
    algorithm = {
        "policy_loss_type": "behavior_clip" if correction == "behavior_clip" else "regular",
        "require_rollout_logprobs": True,
        "grad_cosine": {"enabled": True, "store": "gpu_fp32"},
        "offpolicy_mask": {
            "enabled": correction == "regular_mask",
            "ratio": "mismatch",
            "low": 0.5,
            "high": 5.0,
            "veto_ratio": 1e-5,
            "renormalize": False,
        },
        "m2_mask": {
            "enabled": correction == "regular_m2",
            "ratio": "stale",
            "tau": 0.04,
            "mode": "mask",
            "renormalize": False,
        },
    }
    rows = []
    for step in range(1, 8 // minibatches + 1):
        row = {
            "global_step": step,
            "policy/policy_update_steps": minibatches,
            "policy/updates_attempted": step * minibatches,
            "policy/updates_completed": step * minibatches,
            "policy/updates_completed_valid": 1,
        }
        row.update({"policy/" + field: 0.0 for field in CORRECTION_FIELDS})
        for index in range(minibatches):
            prefix = f"policy/by_update/{index}/"
            row.update({prefix + field: 0.0 for field in CORRECTION_FIELDS})
            row.update(
                {
                    prefix + key: value
                    for key, value in {
                        "update_index": index,
                        "update_age": index,
                        "optimizer_step_succeeded": 1,
                        "grad_norm_valid": 1,
                        "raw_grad_norm": 2.0,
                        "grad_norm_reduced": 1.0,
                        "stale/selected_tokens": 1024,
                    }.items()
                }
            )
        rows.append(row)
    return rows, algorithm


def test_dormant_masks_require_enabled_formula_and_nonzero_gradients():
    rows, algorithm = receipt()
    result = audit_correction_history(rows, algorithm=algorithm, correction="regular_mask", minibatches=1)
    assert result["successful_updates"] == 8
    assert result["minimum_raw_gradient_norm"] == 2
    assert result["update_field_checks"] == 112
    algorithm["offpolicy_mask"]["enabled"] = False
    with pytest.raises(ValueError, match="Effective off-policy mask"):
        audit_correction_history(rows, algorithm=algorithm, correction="regular_mask", minibatches=1)


@pytest.mark.parametrize(
    "key,value",
    [
        ("raw_grad_norm", 0),
        ("grad_norm_reduced", 0),
        ("optimizer_step_succeeded", 0),
        ("offpolicy_mask/masked_fraction", 0.011),
        ("offpolicy_mask/vetoed_sequence_fraction", 0.001),
        ("entropy_mean_selected", float("nan")),
    ],
)
def test_bad_native_update_prevents_gate(key, value):
    rows, algorithm = receipt()
    rows[5]["policy/by_update/0/" + key] = value
    with pytest.raises(ValueError):
        audit_correction_history(rows, algorithm=algorithm, correction="regular_mask", minibatches=1)


def test_missing_update_entropy_cannot_be_replaced_by_aggregate():
    rows, algorithm = receipt()
    del rows[2]["policy/by_update/0/m2_mask/masked_entropy_mean"]
    with pytest.raises(ValueError, match="Missing or nonfinite"):
        audit_correction_history(rows, algorithm=algorithm, correction="regular_mask", minibatches=1)


def test_m2_diagnostic_reports_crossing_without_requiring_it():
    rows, algorithm = receipt("regular_m2", 8)
    result = audit_correction_history(rows, algorithm=algorithm, correction="regular_m2", minibatches=8)
    assert result["m2_threshold_crossing_indices"] == []
    rows[0]["policy/by_update/4/m2_mask/m2_before"] = 0.05
    result = audit_correction_history(rows, algorithm=algorithm, correction="regular_m2", minibatches=8)
    assert result["m2_threshold_crossing_indices"] == [4]


def test_m2_n1_and_disabled_objectives_have_distinct_dormancy_contracts():
    rows, algorithm = receipt("regular_m2")
    rows[0]["policy/by_update/0/m2_mask/m2_before"] = 1e-7
    with pytest.raises(ValueError, match="not dormant"):
        audit_correction_history(rows, algorithm=algorithm, correction="regular_m2", minibatches=1)
    rows, algorithm = receipt("behavior_clip")
    rows[0]["policy/by_update/0/entropy_mean_selected"] = 0.1
    with pytest.raises(ValueError, match="fourteen zero"):
        audit_correction_history(rows, algorithm=algorithm, correction="behavior_clip", minibatches=1)


def test_noise_band_uses_completed_scores_and_equality_boundary():
    baseline = [1] * 10 + [0] * 118
    repeat = [1] * 12 + [0] * 116
    candidate = [1] * 8 + [0] * 120
    assert audit_completed_noise_band(baseline, repeat, candidate)["within_noise_band"]
    candidate[7] = 0
    assert not audit_completed_noise_band(baseline, repeat, candidate)["within_noise_band"]


def test_source_control_checks_batches_before_their_union():
    reference = {step: [str(index) for index in range((step - 1) * 64, step * 64)] for step in range(1, 9)}
    reordered = {step: list(reversed(values)) for step, values in reference.items()}
    assert audit_source_control(reference, reordered)["matched_batches"] == 8
    combined = {1: [uid for values in reference.values() for uid in values]}
    assert audit_source_control(reference, combined)["source_groups"] == 512
    reordered[1][0], reordered[2][0] = reordered[2][0], reordered[1][0]
    with pytest.raises(ValueError, match="Per-batch"):
        audit_source_control(reference, reordered)
    combined[1][-1] = combined[1][0]
    with pytest.raises(ValueError, match="distinct"):
        audit_source_control(reference, combined)
