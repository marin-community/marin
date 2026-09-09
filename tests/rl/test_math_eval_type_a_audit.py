# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from experiments.post_training.math_eval.type_a_audit import audit_async_groups, audit_async_history


def population():
    rows = {"events": [], "scalars": [], "work": []}

    def event(name, body, step, **attrs):
        rows["events"].append(
            {
                "name": name,
                "body_json": json.dumps(body),
                "attributes_json": json.dumps({"role": "trainer", "step": str(step), **attrs}),
            }
        )

    for step in [1, 2, 3]:
        uids = [str(index) for index in reversed(range(64))]
        event(
            "consumed_source_order",
            {"uids_json": json.dumps(uids), "prompt_offset": (step - 1) * 64, "epoch": step - 1},
            step,
        )
        digest = int(hashlib.sha256(json.dumps(sorted(set(uids))).encode()).hexdigest()[:13], 16)
        rows["scalars"].extend(
            [
                {"step": step, "metric": "consumed/uid_digest_u52", "value": digest},
                {"step": step, "metric": "policy/by_update/0/stale/selected_tokens", "value": 640},
            ]
        )
        rows["work"].extend(
            {"step": step, "work_kind": kind, "value": value}
            for kind, value in [("consumed_response_token", 640), ("consumed_loss_token", 640), ("consumed_sample", 256)]
        )
        for group in range(64):
            call_id = f"call-{step}-{group}"
            event(
                "rollout_admission_stamp",
                {
                    "submission_model_step": 0,
                    "first_token_model_step": 1,
                    "admission_model_step": 1,
                    "sampled_tokens": 10,
                    "first_token_evidence_complete": True,
                    "first_token_admission": True,
                },
                step,
                call_id=call_id,
            )
            event("rollout_group_outcome", {"call_id": call_id, "tokens": 10}, step, outcome="consumed")
            event("consumed_age", {"age": step - 1, "groups": 1, "sequences": 4, "response_tokens": 10}, step)
    # An unconsumed completion is not part of the learner population.
    event("rollout_group_outcome", {"call_id": "stale-unused", "tokens": 200}, 3, outcome="stale")
    return {"results": rows}


def test_actual_async_population_joins_first_token_not_submission_age():
    result = audit_async_groups(population(), staleness_limit=16, updates=3, dataset_rows=64)
    assert result["groups"] == result["nonempty_first_token_joins"] == 192
    assert result["first_token_newer_than_submission"] == 192
    assert result["by_step"][3]["age_counts"] == {2: 64}
    assert result["by_step"][3]["token_weighted_age_mean"] == 2
    assert result["integer_digest_joins"] == 3


@pytest.mark.parametrize(
    "defect",
    [
        "limit",
        "source",
        "digest",
        "mask",
        "missing_stamp",
        "submission_stamp",
        "duplicate_terminal",
        "epoch_duplicate",
        "wrong_first_token",
    ],
)
def test_rejects_missing_or_wrong_native_evidence(defect):
    capture = population()
    limit = 16
    if defect == "limit":
        limit = 1
    elif defect == "digest":
        capture["results"]["scalars"][0]["value"] += 0.5
    elif defect == "duplicate_terminal":
        capture["results"]["events"].append(
            next(row for row in capture["results"]["events"] if row["name"] == "rollout_group_outcome")
        )
    else:
        name = {
            "source": "consumed_source_order",
            "epoch_duplicate": "consumed_source_order",
            "mask": "consumed_age",
            "missing_stamp": "rollout_admission_stamp",
            "submission_stamp": "rollout_admission_stamp",
            "wrong_first_token": "rollout_admission_stamp",
        }[defect]
        event = next(
            row
            for row in capture["results"]["events"]
            if row["name"] == name and (defect != "epoch_duplicate" or json.loads(row["attributes_json"])["step"] == "2")
        )
        body = json.loads(event["body_json"])
        if defect == "source":
            body["uids_json"] = json.dumps(["999", *map(str, range(63))])
        elif defect == "epoch_duplicate":
            body["epoch"] = 0
        elif defect == "mask":
            body["response_tokens"] = 11
        elif defect == "missing_stamp":
            body["first_token_evidence_complete"] = False
        elif defect == "wrong_first_token":
            body["first_token_model_step"] = 0
        else:
            body["admission_model_step"] = 0
        event["body_json"] = json.dumps(body)
    with pytest.raises(ValueError):
        audit_async_groups(capture, staleness_limit=limit, updates=3, dataset_rows=64)


def test_async_history_pools_rollout_ages_within_one_optimizer_update():
    selected, scalars, events = [], [], []
    for step in [1, 2]:
        fields = {
            "policy/by_update/0/" + key: 1
            for key in [
                "optimizer_step_succeeded",
                "grad_norm_valid",
                "stale/statistics_valid",
                "stale/finite_fraction",
                "stale/quantiles_valid",
                "stale/p999_valid",
                "raw_grad_norm",
                "grad_norm_reduced",
            ]
        }
        fields.update(
            {
                "policy/by_update/0/stale/quantiles_overflow": 0,
                "policy/by_update/0/grad_cosine_valid": int(step > 1),
                "policy/by_update/0/stale/selected_tokens": 10,
                "consumed/uid_digest_u52": 3970228113034015,
            }
        )
        selected.append({"global_step": step, **fields})
        scalars.extend({"step": step, "metric": key, "value": value} for key, value in fields.items())
        for age, tokens in [(0, 7), (step - 1, 3)]:
            events.append(
                {
                    "name": "consumed_age",
                    "body_json": json.dumps({"age": age, "response_tokens": tokens}),
                    "attributes_json": json.dumps({"step": str(step)}),
                }
            )
    capture = {"results": {"events": events, "scalars": scalars}}
    result = audit_async_history(capture, selected, updates=2)
    assert result["response_mask_tokens"] == 20
    assert result["optimizer_updates"] == result["exact_integer_digest_joins"] == 2
    event = events[-1]
    event["body_json"] = json.dumps({"age": 1, "response_tokens": 4})
    with pytest.raises(ValueError, match="Async mask-token total"):
        audit_async_history(capture, selected, updates=2)
