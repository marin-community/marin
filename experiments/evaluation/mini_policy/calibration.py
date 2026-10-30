# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nested family calibration of subset predictions; empirical coverage, not a guarantee."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from rigging.filesystem.buckets import filesystem_for
from sklearn.model_selection import GroupKFold

from experiments.evaluation.mini_policy.experiment import (
    CONTINUOUS,
    SEED,
    choose_dimension,
    read_bytes,
    retained_benchmark_indices,
    tournament,
)
from experiments.evaluation.mini_policy.greedy import greedy_cv_order
from experiments.evaluation.mini_policy.methods import (
    estimate_ability,
    information_order,
    predict_panel,
    random_order,
    ridge_prediction,
    supervised_order,
)

STUDY = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/study-20261030-v6-21x21"
DEST = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/calibration-20261030-v6-21x21"
METHODS = ("random", "regularized", "greedy_cv", "mirt_static", "mirt_adaptive")


def publish(name, value):
    payload = json.dumps(value, indent=2, allow_nan=False).encode()
    output = Path(os.environ["IRIS_OUTPUT_DIR"]) / name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    fs, key = filesystem_for(DEST + "/" + name)
    fs.pipe_file(key, payload)


def tournament_bounds(lower, upper, sem_lower, sem_upper):
    """Bound each tournament score over the supplied score and SEM rectangles."""
    n = len(lower)
    low = np.zeros(n)
    high = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            least = 1.96 * np.hypot(sem_lower[i], sem_lower[j]) / np.sqrt(0.9)
            greatest = 1.96 * np.hypot(sem_upper[i], sem_upper[j]) / np.sqrt(0.9)
            diff_low = lower[i] - upper[j]
            diff_high = upper[i] - lower[j]
            low[i] += np.where(diff_low < -least, 0, np.where(diff_low > greatest, 1, 0.5)).mean() / (n - 1)
            high[i] += np.where(diff_high > least, 1, np.where(diff_high < -greatest, 0, 0.5)).mean() / (n - 1)
    return low, high


def main():
    parser = argparse.ArgumentParser()
    global DEST
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--study", default=STUDY)
    parser.add_argument("--destination", default=DEST)
    args = parser.parse_args()
    DEST = args.destination
    settings = json.loads(read_bytes(args.study + "/settings.json"))
    source = settings["source"]
    split = settings["splits"][args.fold]
    outer_train = np.asarray(split["train"])
    outer_test = np.asarray(split["test"])
    metadata = json.loads(read_bytes(source + "/matrix/tasks.json"))
    arrays = np.load(io.BytesIO(read_bytes(source + "/matrix/responses.npz")))
    response, counts = arrays["responses"], arrays["counts"]
    primary = retained_benchmark_indices(metadata["benchmarks"])
    if settings["target_benchmarks"] != [metadata["benchmarks"][j] for j in primary]:
        raise ValueError("Study target scope differs from the current 21-benchmark scope")
    target, sems = arrays["targets"][:, primary], arrays["sems"][:, primary]
    benchmarks = np.asarray(metadata["benchmarks"])[primary]
    tasks = metadata["tasks"]
    task_benchmarks = np.asarray([task["benchmark"] for task in tasks])
    continuous = np.isin(task_benchmarks, list(CONTINUOUS))
    groups = np.asarray(settings["split_groups"])
    candidates = np.asarray(
        [
            j
            for j, task in enumerate(tasks)
            if task["benchmark"] not in metadata["predictor_exclusions"]
            and (task["benchmark"] != "mrcr" or int(task["strata"]["mrcr_bin_upper"]) <= 16384)
        ]
    )
    contexts = {
        row["model"]: int(row["serving"]["max_model_len"])
        for row in json.loads(read_bytes(source + "/assembly_audit.json"))
        if row["benchmark"] == "mrcr"
    }
    eligibility = np.ones(response.shape, dtype=bool)
    for i, model in enumerate(metadata["models"]):
        for j, task in enumerate(tasks):
            if task["benchmark"] == "mrcr":
                eligibility[i, j] = int(task["strata"]["mrcr_bin_upper"]) <= contexts[model] // 2
    predictions = {
        (method, size): np.full((len(outer_train), len(primary)), np.nan) for method in METHODS for size in (210, 840)
    }
    predicted_sems = np.full((len(outer_train), len(primary)), np.nan)
    fits = []
    # Inner held-out families do not enter selection, preprocessing, dimensionality
    # selection, or predictor fitting. Outer test outcomes are read only for scoring.
    for inner, (train_pos, valid_pos) in enumerate(GroupKFold(3).split(outer_train, groups=groups[outer_train])):
        train, valid = outer_train[train_pos], outer_train[valid_pos]
        pool = candidates[np.mean(np.isfinite(response[train][:, candidates]), axis=0) >= 0.75]
        predicted_sems[valid_pos] = np.median(sems[train], axis=0)
        orders = {
            "random": random_order(pool, tasks, SEED),
            "regularized": supervised_order(response[train], target[train], pool, 840, SEED + inner, "regularized"),
            "greedy_cv": greedy_cv_order(
                response[train], target[train], sems[train], groups[train], pool, 840, SEED + inner, np.ones(len(tasks))
            ),
        }
        for method, order in orders.items():
            for size in (210, 840):
                predictions[method, size][valid_pos] = ridge_prediction(
                    response[train], response[valid], target[train], order[:size]
                )
        mirt, dimensionality = choose_dimension(
            response[train], counts[train], task_benchmarks, continuous, SEED + inner
        )
        fixed, _ = information_order(mirt, pool, 840, SEED + inner)
        fits.append({"inner": inner, "train": train.tolist(), "valid": valid.tolist(), "dimension_cv": dimensionality})
        stream = io.BytesIO()
        np.savez_compressed(
            stream,
            ability=mirt.ability,
            discrimination=mirt.discrimination,
            intercept=mirt.intercept,
            continuous=mirt.continuous,
            expected_counts=mirt.expected_counts,
            pool=pool,
            train=train,
            valid=valid,
        )
        fs, key = filesystem_for(DEST + f"/fold-{args.fold}/inner-{inner}-irt.npz")
        fs.pipe_file(key, stream.getvalue())
        for position, i in zip(valid_pos, valid, strict=True):
            _, checkpoints = information_order(mirt, pool, 840, SEED + inner, response[i], counts[i])
            for size in (210, 840):
                ability = estimate_ability(mirt, fixed[:size], response[i], counts[i])
                predictions["mirt_static", size][position] = predict_panel(
                    mirt, ability, task_benchmarks, benchmarks, eligibility[i]
                )
                predictions["mirt_adaptive", size][position] = predict_panel(
                    mirt, checkpoints[size], task_benchmarks, benchmarks, eligibility[i]
                )
        print(json.dumps({"outer_fold": args.fold, "inner_complete": inner}), flush=True)
    results = []
    for (method, size), predicted in predictions.items():
        if not np.isfinite(predicted).all():
            raise ValueError("Incomplete inner out-of-family predictions")
        residual = target[outer_train] - predicted
        sem_residual = sems[outer_train] - predicted_sems
        families = sorted(set(groups[outer_train]))
        # Family maxima prevent six related checkpoints from behaving like six
        # independent calibration families. Quantiles are empirical CV estimates;
        # cross-fitted residuals do not provide a split-conformal guarantee.
        error_by_family = np.stack(
            [np.max(np.abs(residual[groups[outer_train] == group]), axis=0) for group in families]
        )
        sem_error_by_family = np.stack(
            [np.max(np.abs(sem_residual[groups[outer_train] == group]), axis=0) for group in families]
        )
        radius = np.quantile(error_by_family, 0.9, axis=0, method="higher")
        sem_radius = np.quantile(sem_error_by_family, 0.9, axis=0, method="higher")
        outer = json.loads(read_bytes(args.study + f"/fold-{args.fold}-{method}-0-{size}.json"))
        point = np.asarray(outer["predicted"])
        lower, upper = np.clip(point - radius, 0, 1), np.clip(point + radius, 0, 1)
        sem_point = np.median(sems[outer_train], axis=0)
        sem_lower = np.broadcast_to(np.maximum(0, sem_point - sem_radius), point.shape)
        sem_upper = np.broadcast_to(np.minimum(0.5, sem_point + sem_radius), point.shape)
        # Rank against the same fixed, fully observed training anchors in every
        # draw. The outer held-out models also compete against one another.
        low, high = tournament_bounds(
            np.vstack([target[outer_train], lower]),
            np.vstack([target[outer_train], upper]),
            np.vstack([sems[outer_train], sem_lower]),
            np.vstack([sems[outer_train], sem_upper]),
        )
        truth = tournament(
            np.vstack([target[outer_train], target[outer_test]]), np.vstack([sems[outer_train], sems[outer_test]])
        )
        low, high, truth = low[len(outer_train) :], high[len(outer_train) :], truth[len(outer_train) :]
        covered = (target[outer_test] >= lower) & (target[outer_test] <= upper)
        results.append(
            {
                "method": method,
                "budget": size,
                "families": families,
                "benchmark_coverage": float(covered.mean()),
                "all_benchmarks_covered": covered.all(axis=1).tolist(),
                "benchmark_mean_interval_width": float(np.mean(upper - lower)),
                "tournament_covered": ((truth >= low) & (truth <= high)).tolist(),
                "tournament_lower": low.tolist(),
                "tournament_upper": high.tolist(),
                "tournament_actual": truth.tolist(),
                "inner_predictions": predicted.tolist(),
                "inner_actual": target[outer_train].tolist(),
                "benchmark_radius": radius.tolist(),
                "sem_radius": sem_radius.tolist(),
            }
        )
    publish(
        f"fold-{args.fold}/calibration.json",
        {
            "outer_fold": args.fold,
            "test_models": [metadata["models"][i] for i in outer_test],
            "train_models": [metadata["models"][i] for i in outer_train],
            "benchmarks": benchmarks.tolist(),
            "method": (
                "Training-only three-fold family CV residuals; per-family maxima and "
                "empirical 90th-percentile benchmark bounds; conservative rectangular "
                "propagation to tournament scores."
            ),
            "limitation": (
                "Empirical coverage audit, not distribution-free 90% coverage. Historical "
                "residuals do not verify fresh-generation or harness-shift coverage."
            ),
            "fits": fits,
            "results": results,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_settings_sha256": hashlib.sha256(read_bytes(args.study + "/settings.json")).hexdigest(),
        },
    )
    print("CALIBRATION_COMPLETE", args.fold, flush=True)


if __name__ == "__main__":
    main()
