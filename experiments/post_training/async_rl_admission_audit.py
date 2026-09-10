# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Join delayed native group outcomes to source UIDs and sampled-token stamps."""

import random
import zlib
from collections import Counter


def audit_delayed_admission(events, source_uids, *, seed, maximum_delay, age_limit):
    if maximum_delay <= 0 or set(source_uids) != set(range(1, 9)):
        raise ValueError("Expected the declared eight-update delayed capability")
    stamps = {}
    consumed = {}
    delays, ages = Counter(), Counter()
    for event in events:
        if event["name"] == "rollout_admission_stamp":
            call = event["attributes"]["call_id"]
            if not call or call in stamps:
                raise ValueError("Duplicate or missing native call identity")
            stamps[call] = event["body"]
    for event in events:
        if event["name"] != "rollout_group_outcome" or event["attributes"]["outcome"] != "consumed":
            continue
        step = int(event["attributes"]["step"])
        body = event["body"]
        uid, call = body["uid"], body["call_id"]
        expected_delay = random.Random(seed ^ zlib.crc32(uid.encode())).randint(0, maximum_delay)
        if type(body["injected_delay_steps"]) is not int or body["injected_delay_steps"] != expected_delay:
            raise ValueError("Native delay differs from seeded UID formula")
        stamp = body["admission_model_step"]
        release = body["release_step"]
        if type(stamp) is not int or stamp < 1 or type(release) is not int or release != stamp + expected_delay:
            raise ValueError("Native release differs from admission stamp plus delay")
        if release > step or not 0 <= step - stamp <= age_limit:
            raise ValueError("Consumed before release or outside true age bound")
        native_stamp = stamps.get(call, {})
        if not (
            native_stamp.get("first_token_admission") is True
            and native_stamp.get("first_token_evidence_complete") is True
            and native_stamp.get("sampled_tokens", 0) > 0
            and native_stamp.get("admission_model_step") == stamp
            and native_stamp.get("first_token_model_step") == stamp
        ):
            raise ValueError("Missing or conflicting native sampled-token stamp")
        identities = consumed.setdefault(step, {})
        if uid in identities or any(call in values.values() for values in consumed.values()):
            raise ValueError("Duplicate consumed UID or call")
        identities[uid] = call
        delays[expected_delay] += 1
        ages[step - stamp] += 1
    if set(consumed) != set(source_uids):
        raise ValueError("Missing consumed update coverage")
    for step, values in source_uids.items():
        if len(values) != 64 or len(set(values)) != 64 or set(values) != set(consumed[step]):
            raise ValueError("Consumed event UIDs differ from native source vector")
    return {
        "status": "DELAYED_ADMISSION_IDENTITY_PASS",
        "groups": sum(ages.values()),
        "delay_histogram": dict(sorted(delays.items())),
        "realized_age_histogram": dict(sorted(ages.items())),
        "realized_age_mean": sum(age * count for age, count in ages.items()) / sum(ages.values()),
        "realized_age_above_two_groups": sum(count for age, count in ages.items() if age > 2),
        "nonvacuous_extended_support": any(age > 2 for age in ages),
    }
