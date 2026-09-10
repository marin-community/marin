# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import random
import zlib

import pytest

from experiments.post_training.async_rl_admission_audit import audit_delayed_admission


def native_receipts():
    events, sources = [], {}
    for step in range(1, 9):
        sources[step] = []
        index = 0
        while len(sources[step]) < 64:
            uid = f"{step}:{index}"
            call = f"call-{uid}"
            delay = random.Random(17 ^ zlib.crc32(uid.encode())).randint(0, 3)
            index += 1
            if delay >= step:
                continue
            stamp = step - delay
            sources[step].append(uid)
            events.extend(
                [
                    {
                        "name": "rollout_admission_stamp",
                        "attributes": {"call_id": call},
                        "body": {
                            "first_token_admission": True,
                            "first_token_evidence_complete": True,
                            "sampled_tokens": 4,
                            "admission_model_step": stamp,
                            "first_token_model_step": stamp,
                        },
                    },
                    {
                        "name": "rollout_group_outcome",
                        "attributes": {"step": str(step), "outcome": "consumed"},
                        "body": {
                            "uid": uid,
                            "call_id": call,
                            "injected_delay_steps": delay,
                            "admission_model_step": stamp,
                            "release_step": step,
                        },
                    },
                ]
            )
    return events, sources


def test_native_delay_source_and_stamp_joins_are_order_independent():
    events, sources = native_receipts()
    random.Random(91).shuffle(events)
    result = audit_delayed_admission(events, sources, seed=17, maximum_delay=3, age_limit=6)
    assert result["groups"] == 512
    assert result["nonvacuous_extended_support"] is True


@pytest.mark.parametrize("defect", ["delay", "release", "source", "stamp", "duplicate"])
def test_native_delay_audit_rejects_conflicting_evidence(defect):
    events, sources = native_receipts()
    if defect == "delay":
        events[1]["body"]["injected_delay_steps"] += 1
    elif defect == "release":
        events[1]["body"]["release_step"] += 1
    elif defect == "source":
        sources[1][0] = "different-uid"
    elif defect == "stamp":
        events[0]["body"]["first_token_model_step"] += 1
    else:
        events.append(copy.deepcopy(events[1]))
    with pytest.raises(ValueError):
        audit_delayed_admission(events, sources, seed=17, maximum_delay=3, age_limit=6)
