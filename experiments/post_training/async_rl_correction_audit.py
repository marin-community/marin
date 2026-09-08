# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed checks for the bounded correction screen, separate from model quality claims."""

import math
from collections.abc import Mapping, Sequence

CORRECTION_FIELDS = (
    "offpolicy_mask/masked_fraction",
    "offpolicy_mask/masked_fraction_low",
    "offpolicy_mask/masked_fraction_high",
    "offpolicy_mask/vetoed_sequence_fraction",
    "offpolicy_mask/masked_advantage_abs_mean",
    "offpolicy_mask/masked_entropy_mean",
    "m2_mask/m2_before",
    "m2_mask/m2_after",
    "m2_mask/masked_fraction",
    "m2_mask/candidate_fraction",
    "m2_mask/unsatisfied",
    "m2_mask/tau_effective",
    "m2_mask/masked_entropy_mean",
    "entropy_mean_selected",
)


def number(row: Mapping, key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"Missing or nonfinite metric: {key}")
    return float(value)


def audit_correction_history(rows: Sequence[Mapping], *, algorithm: Mapping, correction: str, minibatches: int) -> dict:
    """Audit eight successful updates; correction fields describe the final microbatch/rank mean.

    Native event/scalar joins, source identities, receipt integrity, and exporter-loss counters
    must also pass. This function does not claim token-pooled correction fractions.
    """
    if correction not in {"behavior_clip", "regular_mask", "regular_m2"} or minibatches not in {1, 8}:
        raise ValueError("Unexpected screen mode")
    if minibatches == 8 and correction != "regular_m2":
        raise ValueError("Only the separately held M2 diagnostic uses N8")
    active = correction != "behavior_clip"
    if algorithm.get("policy_loss_type") != ("regular" if active else "behavior_clip"):
        raise ValueError("Effective objective differs")
    if active and algorithm.get("require_rollout_logprobs") is not True:
        raise ValueError("Active correction requires rollout log probabilities")
    off, m2 = algorithm.get("offpolicy_mask", {}), algorithm.get("m2_mask", {})
    expected_off = {
        "enabled": True,
        "ratio": "mismatch",
        "low": 0.5,
        "high": 5.0,
        "veto_ratio": 1e-5,
        "renormalize": False,
    }
    expected_m2 = {"enabled": True, "ratio": "stale", "tau": 0.04, "mode": "mask", "renormalize": False}
    if correction == "regular_mask" and any(off.get(k) != v for k, v in expected_off.items()):
        raise ValueError("Effective off-policy mask differs")
    if correction == "regular_m2" and any(m2.get(k) != v for k, v in expected_m2.items()):
        raise ValueError("Effective M2 mask differs")
    if bool(off.get("enabled", False)) != (correction == "regular_mask") or bool(m2.get("enabled", False)) != (
        correction == "regular_m2"
    ):
        raise ValueError("Transform activation differs")
    if algorithm.get("grad_cosine") != {"enabled": True, "store": "gpu_fp32"}:
        raise ValueError("Gradient coverage must be explicitly enabled")
    training = sorted(
        (row for row in rows if "policy/policy_update_steps" in row), key=lambda row: number(row, "global_step")
    )
    if [number(row, "global_step") for row in training] != list(range(1, 8 // minibatches + 1)):
        raise ValueError("Missing or duplicate rollout batches")
    retained, aggregate_checks = [], 0
    for row in training:
        step = int(number(row, "global_step"))
        for key, expected in {
            "policy/policy_update_steps": minibatches,
            "policy/updates_attempted": step * minibatches,
            "policy/updates_completed": step * minibatches,
            "policy/updates_completed_valid": 1,
        }.items():
            if number(row, key) != expected:
                raise ValueError("Attempted and successful update counts differ")
        for field in CORRECTION_FIELDS:
            number(row, "policy/" + field)
            aggregate_checks += 1
        for index in range(minibatches):
            prefix = f"policy/by_update/{index}/"
            values = {field: number(row, prefix + field) for field in CORRECTION_FIELDS}
            for key, expected in {
                "update_index": index,
                "update_age": index,
                "optimizer_step_succeeded": 1,
                "grad_norm_valid": 1,
            }.items():
                if number(row, prefix + key) != expected:
                    raise ValueError("Update identity or gradient validity differs")
            raw = number(row, prefix + "raw_grad_norm")
            reduced = number(row, prefix + "grad_norm_reduced")
            selected = number(row, prefix + "stale/selected_tokens")
            if min(raw, reduced, selected) <= 0:
                raise ValueError("Vacuous update: require selected tokens and nonzero gradients")
            if abs(reduced - min(raw, 1.0)) / min(raw, 1.0) >= 1e-3:
                raise ValueError("Clipped-gradient norm coverage differs")
            if not active and any(values.values()):
                raise ValueError("Disabled transforms must emit all fourteen zero fields")
            if correction == "regular_mask":
                if not 0 <= values["offpolicy_mask/masked_fraction"] <= 0.01:
                    raise ValueError("Off-policy masked fraction exceeds the N1 band")
                if values["offpolicy_mask/vetoed_sequence_fraction"] != 0:
                    raise ValueError("Unexpected vetoed sequence")
            if correction == "regular_m2" and minibatches == 1:
                if not 0 <= values["m2_mask/m2_before"] <= 1e-8 or values["m2_mask/masked_fraction"] != 0:
                    raise ValueError("M2 N1 is not dormant within the frozen tolerance")
            retained.append(values | {"raw_grad_norm": raw, "update_index": index})
    crossing = [
        int(row["update_index"]) for row in retained if row["update_index"] >= 4 and row["m2_mask/m2_before"] > 0.04
    ]
    return {
        "status": "CORRECTION_HISTORY_PASS",
        "correction": correction,
        "minibatches": minibatches,
        "successful_updates": len(retained),
        "aggregate_field_checks": aggregate_checks,
        "update_field_checks": len(retained) * len(CORRECTION_FIELDS),
        "minimum_raw_gradient_norm": min(row["raw_grad_norm"] for row in retained),
        "m2_threshold_crossing_indices": crossing,
        "scope": "last microbatch and rank mean; no token-pooled correction-fraction claim",
    }


def audit_completed_noise_band(baseline: Sequence[float], repeat: Sequence[float], candidate: Sequence[float]) -> dict:
    """Apply the frozen absolute final-score band; paired CIs require a separate UID-paired audit."""
    if len(baseline) != 128 or len(repeat) != 128 or len(candidate) != 128:
        raise ValueError("Expected the fixed 128-question development battery")
    if any(value not in (0, 1) for values in (baseline, repeat, candidate) for value in values):
        raise ValueError("The historical screen uses binary completed correctness")
    noise = abs(sum(repeat) - sum(baseline)) / 128
    difference = abs(sum(candidate) - sum(baseline)) / 128
    return {"within_noise_band": difference <= noise, "absolute_difference": difference, "noise_band": noise}


def audit_source_control(reference: Mapping[int, Sequence[str]], candidate: Mapping[int, Sequence[str]]) -> dict:
    """Compare native UID vectors recovered from complete consumed_source_order events.

    N1 requires the same set at every rollout batch. The N8 diagnostic compares its
    one 512-group batch with the union of the eight N1 batches. Sequence order is not
    part of this gate; the caller retains the original vectors for inspection.
    """
    for batches in (reference, candidate):
        if set(batches) not in (set(range(1, 9)), {1}):
            raise ValueError("Incomplete source batch coverage")
        width = 512 // len(batches)
        flattened = []
        for values in batches.values():
            if len(values) != width or any(not isinstance(value, str) or not value for value in values):
                raise ValueError("Incomplete source UID vector")
            flattened.extend(values)
        if len(flattened) != 512 or len(set(flattened)) != 512:
            raise ValueError("The qualification requires 512 distinct source groups")
    if len(reference) != 8:
        raise ValueError("The source reference must be N1")
    if len(candidate) == 8:
        if any(set(reference[step]) != set(candidate[step]) for step in reference):
            raise ValueError("Per-batch consumed UID sets differ")
    elif set(candidate[1]) != {uid for values in reference.values() for uid in values}:
        raise ValueError("N8 consumed UID set differs from the N1 union")
    return {"source_groups": 512, "matched_batches": len(candidate), "sequence_order_claim": False}
