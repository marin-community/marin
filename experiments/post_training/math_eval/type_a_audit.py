# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Join measured asynchronous source, first-token admission and mask-token ages."""

import collections
import hashlib
import json

from experiments.post_training import async_rl_audit as audit
from experiments.post_training import async_rl_optimizer_gate as optimizer


def audit_async_groups(capture, *, staleness_limit, updates=96, dataset_rows=1918):
    """Audit actual consumed groups without assuming a synchronous sampler order.

    This proves the group/source/age subset. Policy fields, terminal losses,
    saved successful updates, native attempts and evaluation need separate joins.
    """
    if type(staleness_limit) is not int or staleness_limit < 0:
        raise ValueError("Admission limit must be a nonnegative integer")
    events = []
    stamps = {}
    outcomes = {}
    for row in capture["results"]["events"]:
        name = row["name"]
        if name not in {"consumed_source_order", "consumed_age", "rollout_admission_stamp", "rollout_group_outcome"}:
            continue
        body = json.loads(row["body_json"])
        attrs = json.loads(row["attributes_json"])
        if attrs["role"] != "trainer":
            raise ValueError("Group evidence must originate from the trainer")
        step = int(attrs["step"])
        if name in {"consumed_source_order", "consumed_age"}:
            events.append({"name": name, "step": step, "body": body})
            continue
        call_id = attrs.get("call_id") if name == "rollout_admission_stamp" else body["call_id"]
        target = stamps if name == "rollout_admission_stamp" else outcomes
        if not call_id or call_id in target:
            raise ValueError("Missing or repeated rollout call identity")
        target[call_id] = (step, body, attrs)
    source_events = [row for row in events if row["name"] == "consumed_source_order"]
    source = optimizer.audit_source_order_events(source_events, minibatches=1, rollout_batches=updates)
    valid_uids = {str(index) for index in range(dataset_rows)}
    if any(uid not in valid_uids for uid in source["ordered_uids"]):
        raise ValueError("Consumed UID is outside the actual filtered dataset")
    epochs = {}
    seen = collections.defaultdict(set)
    for row in sorted(source_events, key=lambda row: row["step"]):
        epoch = row["body"]["epoch"]
        if type(epoch) is not int or epoch < 0 or (epochs and epoch < max(epochs.values())):
            raise ValueError("Invalid or decreasing source epoch")
        uids = json.loads(row["body"]["uids_json"])
        if seen[epoch].intersection(uids):
            raise ValueError("Consumed question duplicated within an epoch")
        seen[epoch].update(uids)
        epochs[row["step"]] = epoch
    scalars = {}
    for row in capture["results"]["scalars"]:
        key = (int(row["step"]), row["metric"])
        if key in scalars:
            raise ValueError("Duplicate native scalar")
        scalars[key] = row["value"]
    work = collections.defaultdict(dict)
    for row in capture["results"]["work"]:
        step, kind = int(row["step"]), row["work_kind"]
        if kind in work[step]:
            raise ValueError("Duplicate consumed-work counter")
        work[step][kind] = row["value"]
    ages = collections.defaultdict(list)
    for row in events:
        if row["name"] == "consumed_age":
            ages[row["step"]].append(row["body"])
    consumed = collections.defaultdict(list)
    empty_groups = 0
    newer_stamps = 0
    for call_id, (step, body, attrs) in outcomes.items():
        if attrs["outcome"] != "consumed":
            continue
        if call_id not in stamps:
            raise ValueError("Consumed group lacks its admission stamp")
        _, stamp, _ = stamps[call_id]
        tokens = body["tokens"]
        if type(tokens) is not int or tokens < 0 or tokens != stamp["sampled_tokens"]:
            raise ValueError("Consumed tokens differ from sampled admission evidence")
        if stamp["first_token_admission"] is not True:
            raise ValueError("First-token admission was disabled")
        if tokens:
            if stamp["first_token_evidence_complete"] is not True:
                raise ValueError("Nonempty consumed group lacks complete first-token evidence")
            if stamp["admission_model_step"] != stamp["first_token_model_step"]:
                raise ValueError("Admission used a stamp other than the earliest first token")
            newer_stamps += stamp["first_token_model_step"] > stamp["submission_model_step"]
        else:
            empty_groups += 1
        selected = stamp["admission_model_step"]
        if type(selected) is not int or not 0 <= step - selected <= staleness_limit:
            raise ValueError("Consumed admission age exceeds the configured limit")
        consumed[step].append((step - selected, tokens))
    expected_steps = set(range(1, updates + 1))
    if set(ages) != expected_steps or set(consumed) != expected_steps or set(work) != expected_steps:
        raise ValueError("Incomplete asynchronous group/age/work coverage")
    by_step = {}
    for step in sorted(expected_steps):
        bodies = ages[step]
        if len(bodies) != 64 or len(consumed[step]) != 64:
            raise ValueError("Each update must consume exactly 64 groups")
        for body in bodies:
            if any(
                type(body[key]) is not int or body[key] < 0 for key in ["age", "groups", "sequences", "response_tokens"]
            ):
                raise ValueError("Age population requires nonnegative integer counts")
            if body["groups"] != 1 or body["sequences"] != 4:
                raise ValueError("Admitted group geometry differs")
        mask_population = collections.Counter((b["age"], b["response_tokens"]) for b in bodies)
        if mask_population != collections.Counter(consumed[step]):
            raise ValueError("First-token ages/tokens differ from actual masked admitted groups")
        tokens = sum(b["response_tokens"] for b in bodies)
        if not tokens or any(work[step][key] != tokens for key in ["consumed_response_token", "consumed_loss_token"]):
            raise ValueError("Consumed response/loss work differs from mask-token ages")
        if work[step]["consumed_sample"] != 256 or scalars[step, "policy/by_update/0/stale/selected_tokens"] != tokens:
            raise ValueError("Selected worker population differs from actual consumed work")
        uids = source["ordered_uids"][(step - 1) * 64 : step * 64]
        digest = int(hashlib.sha256(json.dumps(sorted(set(uids))).encode()).hexdigest()[:13], 16)
        if scalars[step, "consumed/uid_digest_u52"] != digest:
            raise ValueError("UID digest must match the exact integer, without tolerance")
        by_step[step] = {
            "epoch": epochs[step],
            "tokens": tokens,
            "age_counts": dict(collections.Counter(b["age"] for b in bodies)),
            "token_weighted_age_mean": sum(b["age"] * b["response_tokens"] for b in bodies) / tokens,
        }
    return {
        "status": "ASYNC_CONSUMED_GROUP_EVIDENCE_PASS",
        "groups": updates * 64,
        "source_order_sha256": audit.canonical_sha(source["ordered_uids"]),
        "integer_digest_joins": updates,
        "nonempty_first_token_joins": updates * 64 - empty_groups,
        "empty_groups_without_sampled_token_requirement": empty_groups,
        "first_token_newer_than_submission": newer_stamps,
        "by_step": by_step,
        "scope": (
            "Source/admission/mask population only; separate numerical, terminal, "
            "saved-state and evaluation audits remain required."
        ),
    }
