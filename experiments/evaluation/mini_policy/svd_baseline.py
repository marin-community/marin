# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Family-held-out PCA reconstruction using the same random Micro/Mini probes."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from rigging.filesystem.buckets import filesystem_for
from sklearn.model_selection import GroupKFold

from experiments.evaluation.mini_policy.experiment import evaluate, read_bytes, retained_benchmark_indices
from experiments.evaluation.mini_policy.greedy import prediction_loss
from experiments.evaluation.mini_policy.methods import random_order, standardized

DIMENSIONS = (1, 2, 3, 5)
SEED = 20261030


def fit_pca(response, pool, task_benchmarks, dimension):
    z, mean, scale = standardized(response[:, pool])
    _, inverse, counts = np.unique(task_benchmarks[pool], return_inverse=True, return_counts=True)
    weights = 1 / np.sqrt(counts[inverse])
    weighted = z * weights
    eigenvalues, vectors = np.linalg.eigh(weighted @ weighted.T)
    if eigenvalues[-dimension] <= 1e-10:
        raise ValueError("PCA dimension exceeds supported calibration rank")
    ability = vectors[:, -dimension:] * np.sqrt(len(response))
    loading = z.T @ ability / len(response)
    return mean, scale, loading


def abilities(response, selected_positions, fit):
    mean, scale, loading = fit
    chosen = loading[selected_positions]
    result = []
    for row in response:
        observed = np.isfinite(row[selected_positions])
        a = chosen[observed]
        z = (row[selected_positions][observed] - mean[selected_positions][observed]) / scale[selected_positions][
            observed
        ]
        result.append(np.linalg.solve(a.T @ a + np.eye(a.shape[1]), a.T @ z))
    return np.stack(result)


def predict(train, new, target, pool, selected, fit):
    positions = {column: j for j, column in enumerate(pool)}
    subset = np.asarray([positions[column] for column in selected])
    train_ability = abilities(train[:, pool], subset, fit)
    new_ability = abilities(new[:, pool], subset, fit)
    center = target.mean(axis=0)
    coefficient = np.linalg.solve(
        train_ability.T @ train_ability + 10 * np.eye(train_ability.shape[1]), train_ability.T @ (target - center)
    )
    return np.clip(center + new_ability @ coefficient, 0, 1)


def publish(destination, output, name, document):
    payload = json.dumps(
        document, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value)
    ).encode()
    (output / name).write_bytes(payload)
    fs, key = filesystem_for(destination + "/" + name)
    fs.pipe_file(key, payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    settings = json.loads(read_bytes(args.study + "/settings.json"))
    source = settings["source"]
    metadata = json.loads(read_bytes(source + "/matrix/tasks.json"))
    arrays = np.load(io.BytesIO(read_bytes(source + "/matrix/responses.npz")))
    target_columns = retained_benchmark_indices(metadata["benchmarks"])
    if settings["target_benchmarks"] != [metadata["benchmarks"][j] for j in target_columns]:
        raise ValueError("Study target scope differs from the current 21-benchmark scope")
    target, sems = arrays["targets"][:, target_columns], arrays["sems"][:, target_columns]
    response = arrays["responses"]
    benchmarks = np.asarray([row["benchmark"] for row in metadata["tasks"]])
    candidates = np.asarray(
        [
            j
            for j, task in enumerate(metadata["tasks"])
            if task["benchmark"] not in metadata["predictor_exclusions"]
            and (task["benchmark"] != "mrcr" or int(task["strata"]["mrcr_bin_upper"]) <= 16384)
        ]
    )
    groups = np.asarray(settings["split_groups"])
    predicted = {budget: np.full(target.shape, np.nan) for budget in (210, 840)}
    predicted_sem = np.full(sems.shape, np.nan)
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    evaluated = []
    for fold, split in enumerate(settings["splits"]):
        train, test = np.asarray(split["train"]), np.asarray(split["test"])
        evaluated.extend(test.tolist())
        losses = {budget: {dimension: [] for dimension in DIMENSIONS} for budget in (210, 840)}
        for fit_positions, valid_positions in GroupKFold(3).split(train, groups=groups[train]):
            fit_ids, valid_ids = train[fit_positions], train[valid_positions]
            pool = candidates[np.mean(np.isfinite(response[fit_ids][:, candidates]), axis=0) >= 0.75]
            selected = random_order(pool, metadata["tasks"], SEED)[:840]
            for dimension in DIMENSIONS:
                fit = fit_pca(response[fit_ids], pool, benchmarks, dimension)
                for budget in (210, 840):
                    point = predict(
                        response[fit_ids], response[valid_ids], target[fit_ids], pool, selected[:budget], fit
                    )
                    value = prediction_loss(
                        point[None], target[valid_ids], sems[valid_ids], np.median(sems[fit_ids], axis=0)
                    )[0]
                    losses[budget][dimension].append(float(value))
        pool = candidates[np.mean(np.isfinite(response[train][:, candidates]), axis=0) >= 0.75]
        selected = random_order(pool, metadata["tasks"], SEED)[:840]
        predicted_sem[test] = np.median(sems[train], axis=0)
        for budget in (210, 840):
            dimension = min(DIMENSIONS, key=lambda value: np.mean(losses[budget][value]))
            fit = fit_pca(response[train], pool, benchmarks, dimension)
            point = predict(response[train], response[test], target[train], pool, selected[:budget], fit)
            predicted[budget][test] = point
            result = {
                "fold": fold,
                "budget": budget,
                "dimension": dimension,
                "dimension_cv": losses[budget],
                "train_models": [metadata["models"][i] for i in train],
                "test_models": [metadata["models"][i] for i in test],
                "selected": selected[:budget].tolist(),
                "predicted": point.tolist(),
                "actual": target[test].tolist(),
                "metrics": evaluate(point, target[test], predicted_sem[test], sems[test]),
            }
            publish(args.destination, output, f"fold-{fold}-{budget}.json", result)
            stream = io.BytesIO()
            np.savez_compressed(
                stream, mean=fit[0], scale=fit[1], loading=fit[2], pool=pool, train=train, selected=selected[:budget]
            )
            fs, key = filesystem_for(args.destination + f"/fold-{fold}-{budget}.npz")
            fs.pipe_file(key, stream.getvalue())
        print(json.dumps({"fold_complete": fold}), flush=True)
    evaluated = np.asarray(sorted(set(evaluated)))
    results = []
    for budget in (210, 840):
        if not np.isfinite(predicted[budget][evaluated]).all():
            raise ValueError("Incomplete outer predictions")
        results.append(
            {
                "method": "svd_random",
                "budget": budget,
                "metrics": evaluate(
                    predicted[budget][evaluated], target[evaluated], predicted_sem[evaluated], sems[evaluated]
                ),
            }
        )
    publish(
        args.destination,
        output,
        "summary.json",
        {
            "study": args.study,
            "matrix_source": source,
            "source_settings_sha256": hashlib.sha256(read_bytes(args.study + "/settings.json")).hexdigest(),
            "method": (
                "Equal-benchmark-weighted PCA on training families only; inner family CV chooses "
                "1/2/3/5 dimensions separately for each budget; identical random seed-0 task "
                "manifests; ridge target prediction."
            ),
            "results": results,
        },
    )
    print("SVD_BASELINE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
