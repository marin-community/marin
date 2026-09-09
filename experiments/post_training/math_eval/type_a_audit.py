# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Join measured asynchronous source, first-token admission and mask-token ages."""

import collections
import hashlib
import json
import math

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


def audit_async_history(capture, selected_history, *, updates=96, require_nonzero_gradients=False):
    """Cross-check native updates against W&B, retaining strict capability defaults.

    Exploratory calibration may explicitly permit finite zero-gradient updates.
    Their norms must be exactly zero and adjacent cosine comparisons invalid;
    successful-update and all other numerical coverage requirements still apply.
    """
    minibatches = 1
    steps = updates
    combined = {}
    for row in selected_history:
        step = row.get("global_step", row.get("trainer/global_step"))
        if step is None:
            continue
        target = combined.setdefault(int(step), {})
        for key, value in row.items():
            if key in target and target[key] != value:
                raise ValueError("Conflicting selected W&B rows")
            target[key] = value
    scalars = {(row["step"], row["metric"]): row["value"] for row in capture["results"]["scalars"]}
    comparisons = 0
    for (step, metric), value in scalars.items():
        if not metric.startswith("policy/by_update/") and metric != "consumed/uid_digest_u52":
            continue
        reference = combined[step][metric]
        if metric == "consumed/uid_digest_u52":
            if type(reference) not in (int, float) or reference != int(reference) or reference != value:
                raise ValueError("W&B and native UID digests must match as exact integers")
        elif not math.isfinite(reference) or not math.isclose(value, reference, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Native scalar differs from its actual W&B update")
        comparisons += 1
    maximum_norm_error = 0.0
    previous_norm = 0.0
    zero_gradient_updates = 0
    for step in range(1, steps + 1):
        for index in range(minibatches):
            prefix = f"policy/by_update/{index}/"
            for key in (
                "optimizer_step_succeeded",
                "grad_norm_valid",
                "stale/statistics_valid",
                "stale/finite_fraction",
                "stale/quantiles_valid",
                "stale/p999_valid",
            ):
                if scalars[step, prefix + key] != 1:
                    raise ValueError("Invalid successful-update or numerical coverage flag")
            if scalars[step, prefix + "stale/quantiles_overflow"] != 0:
                raise ValueError("Exact quantiles overflowed")
            raw, norm = (scalars[step, prefix + key] for key in ("raw_grad_norm", "grad_norm_reduced"))
            if not all(math.isfinite(value) and value >= 0 for value in (raw, norm)):
                raise ValueError("Invalid gradient norm")
            target = min(raw, 1.0)
            if target == 0 and require_nonzero_gradients:
                raise ValueError("Observed run cannot qualify nonzero gradient coverage")
            expected_cosine_valid = int(previous_norm > 0 and norm > 0)
            if scalars[step, prefix + "grad_cosine_valid"] != expected_cosine_valid:
                raise ValueError("Gradient cosine validity differs from consecutive nonzero gradients")
            if target == 0:
                if norm != 0:
                    raise ValueError("Zero raw gradient has nonzero post-clip norm")
                zero_gradient_updates += 1
                error = 0.0
            else:
                error = abs(norm - target) / target
            if error >= 1e-3:
                raise ValueError("Post-clip gradient coverage differs")
            previous_norm = norm
            maximum_norm_error = max(maximum_norm_error, error)
    mask_tokens = collections.Counter()
    for row in capture["results"]["events"]:
        if row["name"] == "consumed_age":
            event_body, attrs = json.loads(row["body_json"]), json.loads(row["attributes_json"])
            mask_tokens[int(attrs["step"])] += event_body["response_tokens"]
    if set(mask_tokens) != set(range(1, steps + 1)):
        raise ValueError("Missing actual async mask-token population")
    for step, count in mask_tokens.items():
        if count != scalars[step, "policy/by_update/0/stale/selected_tokens"]:
            raise ValueError("Async mask-token total differs from selected worker population")
    tokens = sum(mask_tokens.values())
    training = [combined[step] for step in range(1, steps + 1)]
    step_walls = [row.get("timing/step") for row in training]
    has_step_walls = all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in step_walls)
    result = {
        "native_wandb_scalar_joins": comparisons,
        "optimizer_updates": updates,
        "maximum_postclip_relative_norm_error": maximum_norm_error,
        "response_mask_tokens": tokens,
        "exact_integer_digest_joins": steps,
        "summed_training_step_wall_seconds": sum(step_walls) if has_step_walls else None,
        "step_wall_scope": (
            "Recorded timing/step when present; use native async_step_window for independent core timing."
        ),
        "dedicated_core_seconds_available": any(any(key.endswith("core_seconds") for key in row) for row in training),
        "selected_history_sha256": audit.canonical_sha(selected_history),
    }
    if not require_nonzero_gradients:
        result["zero_gradient_updates"] = zero_gradient_updates
        result["nonzero_gradient_coverage_required"] = False
    return result


def audit_native_capture(capture, *, staleness_limit):
    """Join actual update fields, source vectors, integer hashes and mask-token ages."""
    minibatches = 1
    steps = 96
    group_proof = audit_async_groups(capture, staleness_limit=staleness_limit)
    rows = capture["results"]
    parsed = []
    terminals = {}
    for row in rows["events"]:
        body, attrs = json.loads(row["body_json"]), json.loads(row["attributes_json"])
        if row["name"] == "terminal":
            role = attrs["role"]
            if (
                role in terminals
                or body["status"] != "completed"
                or any(body[key] != 0 for key in ("export_lost_records", "export_queued_records"))
            ):
                raise ValueError("Native terminal is duplicated, incomplete or lost telemetry")
            terminals[role] = body
        else:
            parsed.append({"name": row["name"], "step": int(attrs["step"]), "body": body})
    if set(terminals) != {"trainer", "driver", "worker", "controller"}:
        raise ValueError("Incomplete native role terminal coverage")
    scalars = {}
    for row in rows["scalars"]:
        key = (row["step"], row["metric"])
        if key in scalars or not math.isfinite(row["value"]):
            raise ValueError("Duplicated or nonfinite native scalar")
        scalars[key] = row["value"]
    history = []
    for step in range(1, steps + 1):
        history.append(
            {
                "global_step": step,
                "policy/policy_update_steps": minibatches,
                **{metric: value for (index, metric), value in scalars.items() if index == step},
            }
        )
        if (
            scalars[step, "policy/updates_completed"] != step * minibatches
            or scalars[step, "policy/updates_completed_valid"] != 1
        ):
            raise ValueError("Native successful optimizer update count differs")
    update_proof = optimizer.audit_policy_update_events(
        history, [row for row in parsed if row["name"] == "policy_update"]
    )
    source = optimizer.audit_source_order_events(
        [row for row in parsed if row["name"] == "consumed_source_order"],
        minibatches=minibatches,
        rollout_batches=steps,
    )
    if len(source["ordered_uids"]) != 6144:
        raise ValueError("Actual source vector has incomplete consumed coverage")
    for step in range(1, steps + 1):
        uids = source["ordered_uids"][(step - 1) * 64 * minibatches : step * 64 * minibatches]
        digest = int(hashlib.sha256(json.dumps(sorted(set(uids))).encode()).hexdigest()[:13], 16)
        value = scalars[step, "consumed/uid_digest_u52"]
        if type(value) not in (int, float) or value != int(value) or value != digest:
            raise ValueError("Native UID hash must equal the exact integer without a numeric tolerance")
    work = {}
    loss_tokens = {}
    for row in rows["work"]:
        step, kind, value = row["step"], row["work_kind"], row["value"]
        if kind == "consumed_loss_token":
            if step in loss_tokens:
                raise ValueError("Duplicated loss-token counter")
            loss_tokens[step] = value
        else:
            key = {"consumed_sample": "sequences", "consumed_response_token": "response_tokens"}[kind]
            if key in work.setdefault(step, {}):
                raise ValueError("Duplicated consumed-work counter")
            work[step][key] = value
    if set(work) != set(range(1, steps + 1)) or set(loss_tokens) != set(work):
        raise ValueError("Native work coverage differs")
    age = {
        "by_step": {
            step: {
                "response_tokens": row["tokens"],
                "groups": 64,
                "token_weighted_age_mean": row["token_weighted_age_mean"],
            }
            for step, row in group_proof["by_step"].items()
        }
    }
    if any(loss_tokens[step] != values["response_tokens"] for step, values in work.items()):
        raise ValueError("Non-agentic loss and response mask counts differ")
    tokens = sum(row["response_tokens"] for row in work.values())
    return {
        "schema": "math_eval_type_a_native_audit_v1",
        "group_evidence": group_proof,
        "minibatches": minibatches,
        "optimizer": update_proof,
        "native_successful_updates": 96,
        "source_order_sha256": audit.canonical_sha(source["ordered_uids"]),
        "source_groups": 6144,
        "unique_question_indices": len(set(source["ordered_uids"])),
        "exposure_histogram": dict(collections.Counter(collections.Counter(source["ordered_uids"]).values())),
        "integer_digest_joins": steps,
        "consumed_age": age,
        "terminals": terminals,
        "response_mask_tokens": tokens,
        "loss_mask_tokens": sum(loss_tokens.values()),
        "token_weighted_age_mean": (
            sum(row["response_tokens"] * row["token_weighted_age_mean"] for row in age["by_step"].values()) / tokens
        ),
        "capture_sha256": audit.canonical_sha(capture),
    }
