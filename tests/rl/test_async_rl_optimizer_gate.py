# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy

import pytest

from experiments.post_training.async_rl_optimizer_gate import (
    GRADIENT_STATISTICS,
    RATIO_STATISTICS,
    audit_consumed_age_events,
    audit_optimizer_history,
    audit_policy_update_events,
    audit_source_order_events,
)


def history():
    rows = []
    for step in (1, 2):
        row = {
            "global_step": step,
            "policy/policy_update_steps": 1,
            "policy/updates_completed": step,
            "policy/updates_attempted": step,
            "policy/updates_completed_valid": 1,
        }
        for bucket in ("pooled", "age0", "age1", "age2", "age3", "age4-7", "age8+"):
            prefix = f"policy/mismatch/{bucket}/"
            row[prefix + "selected_tokens"] = 8 if bucket in {"pooled", "age0"} else 0
            if bucket in {"pooled", "age0"}:
                row.update({prefix + key: 0.0 for key in RATIO_STATISTICS})
                row.update(
                    {
                        prefix + "selected_tokens": 8,
                        prefix + "finite_tokens": 8,
                        prefix + "finite_fraction": 1,
                        prefix + "ess_fraction": 1,
                    }
                )
        update = {"stale/" + key: 0.0 for key in RATIO_STATISTICS}
        update.update(
            {
                "stale/" + key: 1.0
                for key in ("finite_fraction", "statistics_valid", "p999_valid", "quantiles_valid", "ess_fraction")
            }
        )
        update.update(
            {
                "stale/selected_tokens": 8,
                "stale/finite_tokens": 8,
                "stale/quantiles_overflow": 0,
                "grad_cosine": float(step == 2),
                "grad_cosine_valid": float(step == 2),
                "grad_norm_reduced": 1.0,
                "grad_norm_valid": 1,
                "grad_dot": float(step == 2),
                "raw_grad_norm": 2.0,
                "ppo_clip_ratio": 0.0,
                "update_index": 0,
                "update_age": 0,
                "optimizer_step_succeeded": 1,
            }
        )
        row.update({"policy/by_update/0/" + key: value for key, value in update.items()})
        rows.append(row)
    return rows


def test_complete_two_update_schema_has_valid_consecutive_gradients():
    receipt = audit_optimizer_history(history(), minibatches=1, rollout_batches=2, synchronous=True)
    assert receipt["optimizer_updates"] == 2
    assert receipt["maximum_relative_gradient_norm_error"] == 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("stale/abs_log_ratio_p95", None),
        ("grad_norm_reduced", 2.0),
        ("grad_cosine_valid", 0.5),
        ("stale/quantiles_overflow", 1),
        ("update_age", 1),
        ("stale/finite_fraction", 0.99),
    ],
)
def test_missing_new_fields_or_invalid_measurements_cannot_pass(key, value):
    rows = deepcopy(history())
    rows[1]["policy/by_update/0/" + key] = value
    with pytest.raises(ValueError):
        audit_optimizer_history(rows, minibatches=1, rollout_batches=2, synchronous=True)


def test_native_event_requires_complete_coverage_and_exact_metric_parity():
    rows = history()
    events = []
    stale_names = (
        "log_ratio_mean",
        "mean_squared_log_ratio",
        "abs_log_ratio_mean",
        "abs_log_ratio_max",
        "abs_log_ratio_p99",
        "p99_approximate",
        "abs_log_ratio_p999",
        "frac_outside_0.5_2",
        "frac_below_1e-5",
        "ess_fraction",
        "kl_k1",
        "kl_k3",
        "chi2",
        "statistics_valid",
        "selected_tokens",
        "p999_valid",
    )
    for row in rows:
        prefix = "policy/by_update/0/"
        row[prefix + "stale/p99_approximate"] = 1
        for position in ("first256", "last256", "middle"):
            for metric in ("selected_tokens", "abs_log_ratio_mean", "frac_outside_0.5_2"):
                row[prefix + f"stale/pos_{position}/{metric}"] = 0
        keys = ["update_index", "update_age", "raw_grad_norm", "ppo_clip_ratio", *GRADIENT_STATISTICS]
        keys += ["stale/" + name for name in stale_names]
        keys += [
            f"stale/pos_{position}/{metric}"
            for position in ("first256", "last256", "middle")
            for metric in ("selected_tokens", "abs_log_ratio_mean", "frac_outside_0.5_2")
        ]
        body = {key: row[prefix + key] for key in keys}
        body.update({key: body["stale/" + key] for key in ("abs_log_ratio_mean", "ess_fraction", "frac_outside_0.5_2")})
        events.append({"step": row["global_step"], "body": body})
    assert audit_policy_update_events(rows, events)["scalar_comparisons"] == 74
    with pytest.raises(ValueError, match="coverage"):
        audit_policy_update_events(rows, events[:1])
    events[1]["body"]["raw_grad_norm"] += 1
    with pytest.raises(ValueError, match="differs"):
        audit_policy_update_events(rows, events)


def test_native_source_order_matches_across_batch_geometry_and_rejects_duplicates():
    def events(minibatches):
        return [
            {
                "step": 1 + block // minibatches,
                "body": {
                    "prompt_offset": 64 * block,
                    "uids_json": json.dumps([str(index) for index in range(64 * block, 64 * (block + 1))]),
                },
            }
            for block in range(8)
        ]

    single = audit_source_order_events(events(1), minibatches=1, rollout_batches=8)
    multiple = audit_source_order_events(list(reversed(events(4))), minibatches=4, rollout_batches=2)
    assert single["ordered_uids"] == multiple["ordered_uids"] == [str(index) for index in range(512)]
    duplicate = events(4)
    duplicate[1]["body"]["uids_json"] = duplicate[0]["body"]["uids_json"]
    with pytest.raises(ValueError, match="unique"):
        audit_source_order_events(duplicate, minibatches=4, rollout_batches=2)
    with pytest.raises(ValueError, match="coverage"):
        audit_source_order_events(events(1)[:-1], minibatches=1, rollout_batches=8)


def test_native_consumed_age_uses_token_weights_and_excludes_padding():
    bodies = [{"age": age, "groups": 64, "sequences": 256, "response_tokens": 100 * (age + 1)} for age in range(4)]
    events = [{"step": 1, "body": body} for body in bodies]
    work = {1: {"sequences": 1024, "response_tokens": 1000}}
    receipt = audit_consumed_age_events(events, work, minibatches=4, synchronous=True)
    assert receipt["by_step"][1]["token_weighted_age_mean"] == 2
    assert receipt["by_step"][1]["groups"] == 256
    work[1]["response_tokens"] += 7
    with pytest.raises(ValueError, match="unpadded"):
        audit_consumed_age_events(events, work, minibatches=4, synchronous=True)


def test_native_async_age_requires_the_full_group_count():
    events = [
        {"step": 1, "body": {"age": index % 4, "groups": 1, "sequences": 4, "response_tokens": 10}}
        for index in range(64)
    ]
    work = {1: {"sequences": 256, "response_tokens": 640}}
    assert (
        audit_consumed_age_events(events, work, minibatches=1, synchronous=False)["by_step"][1][
            "token_weighted_age_mean"
        ]
        == 1.5
    )
    with pytest.raises(ValueError, match="geometry"):
        audit_consumed_age_events(events + events[:1], work, minibatches=1, synchronous=False)
