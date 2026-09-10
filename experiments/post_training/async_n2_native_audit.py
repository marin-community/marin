# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded native-event and scalar checks for the fresh eight-update N2 gate.

This is one component of qualification. Saved checkpoint bytes, finalized tensor
captures, actual runtime/attempt identity and controller lifecycle are separate
required inputs to the final gate; this module does not synthesize those proofs.
"""

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence


def _integer(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
        raise ValueError("Expected an exact finite integer")
    return int(value)


def _equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"Unexpected {label}")


def merge_history(rows):
    """Merge actual W&B fragments, rejecting conflicting duplicate metrics."""
    merged = {}
    for row in rows:
        if "global_step" not in row:
            continue
        step = _integer(row["global_step"])
        target = merged.setdefault(step, {})
        for name, value in row.items():
            if name.startswith("_"):
                continue
            if name in target and target[name] != value:
                raise ValueError(f"Conflicting W&B metric at {step}: {name}")
            target[name] = value
    training = {step: row for step, row in merged.items() if "policy/policy_update_steps" in row}
    _equal(set(training), set(range(1, 9)), "eight-update history coverage")
    return training


def audit_fresh_events(capture, history):
    """Join actual optimizer-clock events to source vectors and W&B scalars."""
    rows = merge_history(history)
    numerical = audit_async_n2_history(list(rows.values()))
    events = {}
    for item in capture["results"]["events"]:
        attributes = json.loads(item["attributes_json"])
        body = json.loads(item["body_json"])
        step = _integer(int(attributes.get("step", -1)))
        events.setdefault(item["name"], []).append((step, body, attributes))

    prepared = events.get("cohort_prepared", [])
    _equal([step for step, _, _ in prepared], [1, 3, 5, 7], "one preparation per cohort")
    for step, body, _ in prepared:
        _equal(body, dict(admission_step=step, groups=128, sequences=512, updates=2, dp_size=4), "cohort geometry")

    source = {}
    for step, body, _ in events.get("consumed_source_order", []):
        if step in source or step not in range(1, 9):
            raise ValueError("Duplicate or unexpected source-vector step")
        uids = json.loads(body["uids_json"])
        _equal(len(uids), 64, "per-update group count")
        _equal(body["prompt_offset"], (step - 1) * 64, "optimizer-clock source offset")
        if not all(isinstance(uid, str) and uid for uid in uids):
            raise ValueError("Invalid source UID")
        source[step] = uids
    _equal(set(source), set(range(1, 9)), "source-vector coverage")
    all_uids = [uid for step in range(1, 9) for uid in source[step]]
    _equal(len(set(all_uids)), 512, "unique admitted/consumed groups")

    consumed = {step: {} for step in range(1, 9)}
    for step, body, _ in events.get("cohort_consumption", []):
        if step not in consumed or body["uid"] in consumed[step]:
            raise ValueError("Duplicate or unexpected cohort consumption")
        lag = (step - 1) % 2
        admission = step - lag
        _equal(
            body,
            dict(
                uid=body["uid"],
                admission_step=admission,
                admission_model_step=admission,
                admission_age=0,
                consume_age=lag,
                within_cohort_lag=lag,
            ),
            "unclipped cohort ages",
        )
        consumed[step][body["uid"]] = body
    for step in range(1, 9):
        _equal(set(consumed[step]), set(source[step]), "cohort/source UID union")

    age_counts = Counter()
    response_tokens = Counter()
    for step, body, _ in events.get("consumed_age", []):
        if step not in source:
            raise ValueError("Unexpected consumed-age step")
        _equal(body["groups"], 1, "age-event group count")
        _equal(body["sequences"], 4, "responses per group")
        _equal(body["age"], (step - 1) % 2, "actual consumption age")
        tokens = _integer(body["response_tokens"])
        if tokens < 0:
            raise ValueError("Negative response-token count")
        response_tokens[step] += tokens
        age_counts[step] += 1
    _equal(age_counts, Counter({step: 64 for step in range(1, 9)}), "age-event coverage")

    published = events.get("policy_weights_published", [])
    _equal([step for step, _, _ in published], list(range(9)), "initial plus C1 publications")
    for step, body, _ in published:
        _equal(body["completed_update"], step, "publication clock")
        if not math.isfinite(body["duration"]) or body["duration"] <= 0:
            raise ValueError("Invalid publication duration")

    update_events = events.get("policy_update", [])
    _equal([step for step, _, _ in update_events], list(range(1, 9)), "optimizer-event coverage")
    event_joins = 0
    for step, body, _ in update_events:
        index = (step - 1) % 2
        _equal(body["update_index"], index, "preserved minibatch index")
        _equal(body["update_age"], index, "within-cohort update lag")
        _equal(len(body), 37, "native optimizer schema")
        if not math.isfinite(body["raw_grad_norm"]) or body["raw_grad_norm"] <= 0:
            raise ValueError("Nonfinite or zero native gradient")
        for key, value in body.items():
            target = "stale/" + key if key in ("abs_log_ratio_mean", "ess_fraction", "frac_outside_0.5_2") else key
            metric = f"policy/by_update/{index}/{target}"
            if not math.isfinite(value) or not math.isclose(value, rows[step][metric], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"Native optimizer mismatch: {step} {key}")
            event_joins += 1

    scalar_joins = 0
    scalar_seen = {}
    for scalar in capture["results"]["scalars"]:
        step = _integer(int(scalar["step"]))
        if step not in rows:
            continue
        metric, value = scalar["metric"], scalar["value"]
        identity = (step, metric)
        if identity in scalar_seen and scalar_seen[identity] != value:
            raise ValueError("Conflicting duplicate native scalar")
        scalar_seen[identity] = value
        expected = rows[step].get(metric)
        if expected is None or not math.isfinite(value) or not math.isfinite(expected):
            raise ValueError(f"Missing/nonfinite scalar join: {identity}")
        if metric == "consumed/uid_digest_u52":
            _equal(_integer(value), _integer(expected), "exact UID integer digest join")
        elif not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"Native/W&B scalar mismatch: {identity}")
        scalar_joins += 1
    for step in range(1, 9):
        row = rows[step]
        for key, expected in (
            ("policy/updates_completed", step),
            ("policy/updates_completed_valid", 1),
            ("policy/policy_update_steps", 1),
        ):
            _equal(row[key], expected, "successful optimizer clock")
            if (step, key) not in scalar_seen:
                raise ValueError("Missing native optimizer clock scalar")
        digest = int(hashlib.sha256(json.dumps(sorted(set(source[step]))).encode()).hexdigest()[:13], 16)
        _equal(_integer(row["consumed/uid_digest_u52"]), digest, "source-vector digest")
        if (step, "consumed/uid_digest_u52") not in scalar_seen:
            raise ValueError("Missing native UID digest")
    return dict(
        status="ASYNC_N2_FRESH_NATIVE_EVENT_COMPONENT_PASS",
        optimizer_updates=8,
        cohorts=4,
        consumed_groups=512,
        consumed_sequences=2048,
        admitted_age0=512,
        consumed_age0=256,
        consumed_age1=256,
        publications=9,
        event_scalar_joins=event_joins,
        native_scalar_joins=scalar_joins,
        response_tokens_by_step=dict(response_tokens),
        source_uids_by_step=source,
        numerical=numerical,
        remaining_proofs=[
            "finalized tensors and source masks",
            "saved checkpoint7 and checkpoint8 bytes",
            "actual source/entrypoint and all-attempt lifecycle",
            "work-token conservation",
        ],
    )


RATIO_STATISTICS = (
    "selected_tokens",
    "finite_tokens",
    "finite_fraction",
    "log_ratio_mean",
    "mean_squared_log_ratio",
    "abs_log_ratio_mean",
    "abs_log_ratio_p50",
    "abs_log_ratio_p95",
    "abs_log_ratio_p99",
    "abs_log_ratio_p999",
    "abs_log_ratio_max",
    "lower_clip_pressure",
    "upper_clip_pressure",
    "frac_outside_0.5_2",
    "frac_below_1e-5",
    "ess_fraction",
    "kl_k1",
    "kl_k3",
    "chi2",
)
GRADIENT_STATISTICS = ("grad_cosine", "grad_cosine_valid", "grad_norm_reduced", "grad_norm_valid", "grad_dot")


def _number(row: Mapping, key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing or nonfinite metric: {key}")
    return value


def audit_async_n2_history(rows: Sequence[Mapping]) -> dict:
    """Check full numerical coverage at the async optimizer clock, retaining N2 indices."""
    rollout_batches = 8
    minibatches = 1
    training = [row for row in rows if "policy/policy_update_steps" in row]
    steps = [_number(row, "global_step") for row in training]
    if sorted(steps) != list(range(1, rollout_batches + 1)):
        raise ValueError(f"Training step coverage differs: {steps}")
    updates, first_stale, monotone_batches, norm_errors = [], [], [], []
    for row in sorted(training, key=lambda item: item["global_step"]):
        step = row["global_step"]
        if _number(row, "policy/policy_update_steps") != minibatches:
            raise ValueError("Unexpected optimizer-update count")
        if (
            _number(row, "policy/updates_attempted") != step * minibatches
            or _number(row, "policy/updates_completed_valid") != 1
        ):
            raise ValueError("Attempted/successful update coverage differs")
        if _number(row, "policy/updates_completed") != step * minibatches:
            raise ValueError("Optimizer-update x-axis differs")
        pooled = _number(row, "policy/mismatch/pooled/selected_tokens")
        if pooled <= 0 or pooled != sum(
            _number(row, f"policy/mismatch/{bucket}/selected_tokens")
            for bucket in ("age0", "age1", "age2", "age3", "age4-7", "age8+")
        ):
            raise ValueError("Mismatch age populations do not partition the consumed tokens")
        for bucket in ("pooled", "age0", "age1", "age2", "age3", "age4-7", "age8+"):
            prefix = f"policy/mismatch/{bucket}/"
            population = _number(row, prefix + "selected_tokens")
            if population:
                for key in RATIO_STATISTICS:
                    _number(row, prefix + key)
                if not 0 < _number(row, prefix + "ess_fraction") <= 1.0000001:
                    raise ValueError("Invalid mismatch token ESS fraction")
                if _number(row, prefix + "finite_fraction") != 1:
                    raise ValueError("Incomplete mismatch finite coverage")
        batch_stale = []
        for index in [(int(step) - 1) % 2]:
            prefix = f"policy/by_update/{index}/"
            update = {key.removeprefix(prefix): value for key, value in row.items() if key.startswith(prefix)}
            for key in (*RATIO_STATISTICS, "statistics_valid", "p999_valid", "quantiles_valid", "quantiles_overflow"):
                _number(update, "stale/" + key)
            for key in (
                *GRADIENT_STATISTICS,
                "raw_grad_norm",
                "ppo_clip_ratio",
                "update_index",
                "update_age",
                "optimizer_step_succeeded",
            ):
                _number(update, key)
            if update["optimizer_step_succeeded"] != 1:
                raise ValueError("Native optimizer update did not succeed")
            if update["update_index"] != index or update["update_age"] != index:
                raise ValueError("Optimizer update identity or age differs")
            if any(
                update["stale/" + key] != 1
                for key in ("finite_fraction", "statistics_valid", "p999_valid", "quantiles_valid")
            ):
                raise ValueError("Invalid ratio statistic or finite coverage")
            if update["stale/quantiles_overflow"] != 0:
                raise ValueError("Exact quantile storage overflow")
            if update["stale/selected_tokens"] <= 0 or update["stale/finite_tokens"] != update["stale/selected_tokens"]:
                raise ValueError("Worker finite token population differs")
            if not 0 < update["stale/ess_fraction"] <= 1.0000001:
                raise ValueError("Invalid token ESS fraction")
            if (
                not 0
                <= update["stale/abs_log_ratio_p50"]
                <= update["stale/abs_log_ratio_p95"]
                <= update["stale/abs_log_ratio_max"]
            ):
                raise ValueError("Invalid exact absolute quantile ordering")
            expected_valid = float(bool(updates))
            if update["grad_cosine_valid"] != expected_valid or update["grad_norm_valid"] != 1:
                raise ValueError("Gradient validity does not match consecutive successful updates")
            if not -1 <= update["grad_cosine"] <= 1:
                raise ValueError("Gradient cosine lies outside [-1,1]")
            raw = update["raw_grad_norm"]
            error = abs(update["grad_norm_reduced"] - min(raw, 1.0)) / raw if raw > 0 else math.inf
            if error >= 1e-3:
                raise ValueError(f"Gradient parameter-coverage norm check failed: {error}")
            norm_errors.append(error)
            batch_stale.append(update["stale/abs_log_ratio_mean"])
            updates.append(update)
        first_stale.append(batch_stale[0])
        monotone_batches.append(batch_stale == sorted(batch_stale))
    if max(first_stale[::2]) > 1e-4:
        raise ValueError("First cohort partitions differ from their freshly prepared policy log probabilities")
    return {
        "status": "ASYNC_N2_NUMERICAL_HISTORY_PASS",
        "rollout_batches": rollout_batches,
        "optimizer_updates": len(updates),
        "maximum_relative_gradient_norm_error": max(norm_errors),
        "first_update_stale_absolute_means": first_stale,
        "monotone_stale_by_batch": monotone_batches,
        "exact_quantiles_valid": True,
    }
