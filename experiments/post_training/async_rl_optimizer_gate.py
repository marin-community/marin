# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit complete optimizer telemetry schemas before qualifying native runs."""

import math
from collections.abc import Mapping, Sequence

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


def audit_optimizer_history(
    rows: Sequence[Mapping], *, minibatches: int, rollout_batches: int, synchronous: bool
) -> dict:
    """Check complete W&B update records; native parity and durable scoring are separate."""
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
        for index in range(minibatches):
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
    if synchronous and max(first_stale) > 1e-4:
        raise ValueError("First synchronous update differs from consumed-forward log probabilities")
    return {
        "status": "OPTIMIZER_HISTORY_PASS",
        "rollout_batches": rollout_batches,
        "optimizer_updates": len(updates),
        "maximum_relative_gradient_norm_error": max(norm_errors),
        "first_update_stale_absolute_means": first_stale,
        "monotone_stale_by_batch": monotone_batches,
        "exact_quantiles_valid": True,
    }


def audit_policy_update_events(rows: Sequence[Mapping], events: Sequence[Mapping]) -> dict:
    """Require one native event per retained optimizer update and scalar parity."""
    expected = {}
    for row in rows:
        if "policy/policy_update_steps" not in row:
            continue
        for index in range(int(row["policy/policy_update_steps"])):
            prefix = f"policy/by_update/{index}/"
            expected[(int(row["global_step"]), index)] = {
                key.removeprefix(prefix): value for key, value in row.items() if key.startswith(prefix)
            }
    observed, comparisons = set(), 0
    for event in events:
        body = event["body"]
        identity = (int(event["step"]), int(_number(body, "update_index")))
        if identity in observed or identity not in expected:
            raise ValueError("Duplicate or unexpected native optimizer event")
        observed.add(identity)
        reference = expected[identity]
        if len(body) != 37:
            raise ValueError("Native optimizer event field count differs from the bounded schema")
        for key in ("raw_grad_norm", "ppo_clip_ratio", *GRADIENT_STATISTICS):
            _number(body, key)
        for key in body:
            target = "stale/" + key if key in ("abs_log_ratio_mean", "ess_fraction", "frac_outside_0.5_2") else key
            if not math.isclose(_number(body, key), _number(reference, target), rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"Native optimizer event scalar differs: {identity} {key}")
            comparisons += 1
    if observed != set(expected):
        raise ValueError("Native optimizer event coverage is incomplete")
    return {"status": "OPTIMIZER_EVENTS_PASS", "events": len(observed), "scalar_comparisons": comparisons}
