# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exploratory next-evaluation discrimination with strictly preceding feature windows."""

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

FEATURES = {
    "legacy_ess_risk": ("policy/behavior_drift/token_weight_ess_fraction", -1),
    "legacy_abs_log_ratio": ("policy/behavior_drift/abs_log_ratio_mean", 1),
    "legacy_abs_log_ratio_p99": ("policy/behavior_drift/abs_log_ratio_p99", 1),
    "legacy_mean_squared_log_ratio": ("policy/behavior_drift/mean_squared_log_ratio", 1),
    "legacy_lower_clip_pressure": ("policy/behavior_drift/lower_clip_pressure", 1),
    "legacy_upper_clip_pressure": ("policy/behavior_drift/upper_clip_pressure", 1),
    "low_entropy": ("policy/policy_entropy", -1),
    "raw_gradient_norm": ("policy/raw_grad_norm", 1),
    "worker_abs_log_ratio": ("policy/log_ratio_abs_mean", 1),
    "ppo_clip_ratio": ("policy/ppo_clip_ratio", 1),
    "mean_admitted_age": ("async/staleness_mean", 1),
    "max_admitted_age": ("async/staleness_max", 1),
}
MISSING_INDICATORS = ("split_stale_ess", "gradient_cosine", "split_mismatch_ess", "legacy_abs_log_ratio_p999")


def auroc(labels: Sequence[int], risks: Sequence[float]) -> float | None:
    """Return pairwise AUROC with half credit for ties, or None without both classes."""
    y, scores = np.asarray(labels), np.asarray(risks, dtype=np.float64)
    if y.shape != scores.shape or y.ndim != 1 or not np.isfinite(scores).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid binary-label/risk observations")
    positive, negative = scores[y == 1], scores[y == 0]
    if not len(positive) or not len(negative):
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))


def preceding_windows(records: Sequence[Mapping], completed: Mapping[int, float], *, width: int = 20) -> list[dict]:
    """Join mean features from (t-width,t] to the score change from t to t+width.

    Every observed metric must cover the full feature window. Missing entire metrics
    remain missing; a partial window raises instead of silently changing exposure.
    """
    values = {}
    for row in records:
        step, metric, value = row["step"], row["metric"], row["value"]
        if not isinstance(step, int) or not math.isfinite(value):
            raise ValueError("Nonintegral step or nonfinite native scalar")
        if (step, metric) in values:
            raise ValueError("Duplicate native step/metric")
        values[step, metric] = value
    windows = []
    for anchor in sorted(completed):
        if anchor < width or anchor + width not in completed:
            continue
        window = range(anchor - width + 1, anchor + 1)
        features, per_step_flags = {}, {}
        for name, (metric, direction) in FEATURES.items():
            observed = [values[step, metric] for step in window if (step, metric) in values]
            if not observed:
                continue
            if len(observed) != width:
                raise ValueError(f"Incomplete preceding window: {anchor} {metric}")
            features[name] = direction * float(np.mean(observed))
            if name == "legacy_ess_risk":
                per_step_flags[name] = any(value < 0.5 for value in observed)
            elif name == "legacy_abs_log_ratio":
                per_step_flags[name] = any(value > 0.03 for value in observed)
        windows.append(
            {
                "anchor": anchor,
                "feature_start": anchor - width + 1,
                "feature_end": anchor,
                "next_eval": anchor + width,
                "completed_before": completed[anchor],
                "completed_after": completed[anchor + width],
                "drop": completed[anchor] - completed[anchor + width],
                "features": features,
                "per_step_flags": per_step_flags,
            }
        )
    return windows


def run_cluster_auc_interval(windows: Sequence[Mapping], feature: str, threshold: float) -> dict:
    """Resample whole runs, retaining their dependent windows; this is not seed uncertainty."""
    groups = sorted({row["run"] for row in windows})
    by_run = {group: [row for row in windows if row["run"] == group] for group in groups}
    rng, draws = np.random.default_rng(20260909), []
    for _ in range(5000):
        selected = [row for group in rng.choice(groups, size=len(groups), replace=True) for row in by_run[group]]
        value = auroc([int(row["drop"] > threshold) for row in selected], [row["features"][feature] for row in selected])
        if value is not None:
            draws.append(value)
    return {
        "run_cluster_percentile_interval_95": np.quantile(draws, [0.025, 0.975]).tolist() if draws else None,
        "valid_draws": len(draws),
        "draws": 5000,
        "independent_seed_interval": False,
    }


def indicator_report(windows: Sequence[Mapping]) -> dict:
    """Report availability, fixed-direction discrimination and existing flags without fitting thresholds."""
    result = {"observations": len(windows), "run_clusters": len({row["run"] for row in windows}), "indicators": {}}
    for feature in (*FEATURES, *MISSING_INDICATORS):
        available = [row for row in windows if feature in row["features"]]
        risks = [row["features"][feature] for row in available]
        if len(available) != len(windows):
            result["indicators"][feature] = {"status": "untestable_missing", "available": len(available)}
            continue
        if len(set(risks)) <= 1:
            result["indicators"][feature] = {"status": "untestable_constant", "value": risks[0] if risks else None}
            continue
        row = {"status": "exploratory", "risk_min": min(risks), "risk_max": max(risks), "labels": {}}
        for label, threshold in (("any_drop", 0.0), ("drop_gt_2pp", 0.02)):
            labels = [int(item["drop"] > threshold) for item in available]
            row["labels"][label] = {
                "positives": sum(labels),
                "negatives": len(labels) - sum(labels),
                "auroc": auroc(labels, risks),
                **run_cluster_auc_interval(available, feature, threshold),
            }
            flag = None
            if feature == "legacy_ess_risk":
                flag = [value > -0.5 for value in risks]
            elif feature == "legacy_abs_log_ratio":
                flag = [value > 0.03 for value in risks]
            if flag is not None:
                negatives = len(labels) - sum(labels)
                row["labels"][label]["window_mean_flag_false_alarm_rate"] = (
                    sum(fired and not target for fired, target in zip(flag, labels, strict=True)) / negatives
                    if negatives
                    else None
                )
                row["labels"][label]["window_mean_flag_count"] = sum(flag)
                if all(feature in item.get("per_step_flags", {}) for item in available):
                    native_flags = [item["per_step_flags"][feature] for item in available]
                    row["labels"][label]["any_step_flag_count"] = sum(native_flags)
                    row["labels"][label]["any_step_flag_false_alarm_rate"] = (
                        sum(fired and not target for fired, target in zip(native_flags, labels, strict=True)) / negatives
                        if negatives
                        else None
                    )
        result["indicators"][feature] = row
    result["held_out_seed_gate"] = "untestable: only one historical training seed"
    result["operating_decision"] = "No predictive alert adopted; plan ladders around the empirical quality knee."
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--audited", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native, audited = json.loads(args.native.read_text()), json.loads(args.audited.read_text())
    windows = []
    for label, run in native.items():
        evidence = audited["runs"][label]
        if run["run_id"] != evidence["run_id"] or not evidence["clean_end_to_end"]:
            raise ValueError("Native history differs from the audited run")
        completed = {
            row["step"]: row["metrics"]["eval/all/completed_stop_score_contribution"] for row in evidence["eval_dumps"]
        }
        windows.extend({"run": label, **row} for row in preceding_windows(run["rows"], completed))
    report = indicator_report(windows)
    report["windows"] = windows
    report["inputs_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.native, args.audited)
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"INDICATOR_PILOT_COMPLETE windows={len(windows)} runs={report['run_clusters']} held_out_seed_gate=untestable")


if __name__ == "__main__":
    main()
