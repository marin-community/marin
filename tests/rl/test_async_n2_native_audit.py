# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json

import pytest

from experiments.post_training.async_n2_native_audit import RATIO_STATISTICS, audit_fresh_events


def fixture():
    events, scalars, rows = [], [], []

    def event(name, step, body):
        events.append(dict(name=name, attributes_json=json.dumps(dict(step=str(step))), body_json=json.dumps(body)))

    for step in range(9):
        event("policy_weights_published", step, dict(completed_update=step, duration=1.0))
    for step in range(1, 9):
        index = (step - 1) % 2
        admission = step - index
        if not index:
            event("cohort_prepared", step, dict(admission_step=step, groups=128, sequences=512, updates=2, dp_size=4))
        uids = list(map(str, range((step - 1) * 64, step * 64)))
        digest = int(hashlib.sha256(json.dumps(sorted(uids)).encode()).hexdigest()[:13], 16)
        event("consumed_source_order", step, dict(prompt_offset=(step - 1) * 64, uids_json=json.dumps(uids), epoch=0))
        for uid in uids:
            event(
                "cohort_consumption",
                step,
                dict(
                    uid=uid,
                    admission_step=admission,
                    admission_model_step=admission,
                    admission_age=0,
                    consume_age=index,
                    within_cohort_lag=index,
                ),
            )
            event("consumed_age", step, dict(groups=1, sequences=4, response_tokens=8, age=index))
        row = {
            "global_step": step,
            "policy/policy_update_steps": 1,
            "policy/updates_attempted": step,
            "policy/updates_completed": step,
            "policy/updates_completed_valid": 1,
            "consumed/uid_digest_u52": digest,
        }
        if not index:
            for bucket in ("pooled", "age0", "age1", "age2", "age3", "age4-7", "age8+"):
                for key in RATIO_STATISTICS:
                    row[f"async/cohort_preparation/policy/mismatch/{bucket}/{key}"] = 0
                population = 1024 if bucket in ("pooled", f"age{index}") else 0
                row[f"async/cohort_preparation/policy/mismatch/{bucket}/selected_tokens"] = population
                row[f"async/cohort_preparation/policy/mismatch/{bucket}/finite_fraction"] = 1
                row[f"async/cohort_preparation/policy/mismatch/{bucket}/ess_fraction"] = 1
        update = {"stale/" + key: 0 for key in RATIO_STATISTICS}
        for key in ("finite_fraction", "statistics_valid", "p999_valid", "quantiles_valid", "ess_fraction"):
            update["stale/" + key] = 1
        for key in ("selected_tokens", "finite_tokens"):
            update["stale/" + key] = 512
        update.update(
            {
                "stale/p99_approximate": 0,
                "stale/quantiles_overflow": 0,
                "grad_cosine": 0,
                "grad_cosine_valid": int(step > 1),
                "grad_norm_valid": 1,
                "grad_norm_reduced": 0.5,
                "grad_dot": 0,
                "raw_grad_norm": 0.5,
                "ppo_clip_ratio": 0,
                "update_index": index,
                "update_age": index,
                "optimizer_step_succeeded": 1,
            }
        )
        for position in ("first256", "last256", "middle"):
            for metric in ("selected_tokens", "abs_log_ratio_mean", "frac_outside_0.5_2"):
                update[f"stale/pos_{position}/{metric}"] = 0
        row.update({f"policy/by_update/{index}/" + key: value for key, value in update.items()})
        keys = (
            "update_index",
            "update_age",
            "raw_grad_norm",
            "ppo_clip_ratio",
            "grad_cosine",
            "grad_cosine_valid",
            "grad_norm_reduced",
            "grad_norm_valid",
            "grad_dot",
        )
        body = {key: update[key] for key in keys}
        keys = (
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
        body.update({"stale/" + key: update["stale/" + key] for key in keys})
        body.update({key: value for key, value in update.items() if key.startswith("stale/pos_")})
        body.update(
            {key: update["stale/" + key] for key in ("abs_log_ratio_mean", "ess_fraction", "frac_outside_0.5_2")}
        )
        assert len(body) == 37
        event("policy_update", step, body)
        rows.append(row)
        scalars.extend(
            dict(step=str(step), metric=key, value=value) for key, value in row.items() if key != "global_step"
        )
    return dict(results=dict(events=events, scalars=scalars, work=[])), rows


def test_native_n2_clock_and_component_scope():
    capture, rows = fixture()
    result = audit_fresh_events(capture, rows)
    assert result["optimizer_updates"] == 8 and result["publications"] == 9
    assert result["event_scalar_joins"] == 296 and result["consumed_groups"] == 512
    assert result["numerical"]["optimizer_updates"] == 8
    assert "saved checkpoint7 and checkpoint8 bytes" in result["remaining_proofs"]


@pytest.mark.parametrize(
    "defect", ["reset_index", "clipped_age", "reprepare", "source_digest", "duplicate_metric", "half_clock"]
)
def test_native_n2_gate_rejects_corruption(defect):
    capture, rows = fixture()
    if defect == "duplicate_metric":
        rows.append({"global_step": 1, "policy/updates_completed": 2})
    elif defect == "source_digest":
        rows[0]["consumed/uid_digest_u52"] += 1
    elif defect == "half_clock":
        rows[1]["policy/updates_completed"] = 1
    elif defect == "reprepare":
        item = copy.deepcopy(next(item for item in capture["results"]["events"] if item["name"] == "cohort_prepared"))
        capture["results"]["events"].append(item)
    else:
        name = "policy_update" if defect == "reset_index" else "cohort_consumption"
        item = next(
            item
            for item in capture["results"]["events"]
            if item["name"] == name and json.loads(item["attributes_json"])["step"] == "2"
        )
        body = json.loads(item["body_json"])
        body["update_index" if defect == "reset_index" else "consume_age"] = 0
        item["body_json"] = json.dumps(body)
    with pytest.raises(ValueError):
        audit_fresh_events(capture, rows)


@pytest.mark.parametrize("step", [1.25, True, "1.25", "1e0", "01"])
def test_reject_invalid_native_step_before_conversion(step):
    capture, rows = fixture()
    capture["results"]["scalars"][0]["step"] = step
    with pytest.raises(ValueError):
        audit_fresh_events(capture, rows)


@pytest.mark.parametrize("defect", ["reprepared_even", "missing_odd", "wrong_population", "stale_admission"])
def test_preparation_metrics_are_bound_once_per_cohort(defect):
    capture, rows = fixture()
    prefix = "async/cohort_preparation/policy/mismatch/"
    if defect == "reprepared_even":
        rows[1][prefix + "pooled/selected_tokens"] = 1024
    elif defect == "missing_odd":
        del rows[2][prefix + "pooled/selected_tokens"]
    elif defect == "wrong_population":
        rows[0][prefix + "pooled/selected_tokens"] = 512
        rows[0][prefix + "age0/selected_tokens"] = 512
    else:
        rows[0][prefix + "age1/selected_tokens"] = 1
    with pytest.raises(ValueError):
        audit_fresh_events(capture, rows)
