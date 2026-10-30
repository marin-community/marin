# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen-cohort, family-held-out mini/micro subset experiment on RNO2A."""

import argparse
import hashlib
import io
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from rigging.filesystem.buckets import filesystem_for
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from experiments.evaluation.mini_policy.costs import policy_costs
from experiments.evaluation.mini_policy.greedy import greedy_cv_order
from experiments.evaluation.mini_policy.methods import (
    estimate_ability,
    fit_mirt,
    information_order,
    predict_panel,
    random_order,
    ridge_prediction,
    supervised_order,
)

SOURCE = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/inventory-20261030-v5"
DEST = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/study-20261030-v6-21x21"
SEED = 20261030
EXCLUDED_TARGET = {"financebench", "piqa", "winogrande", "boolq", "SOTOPIA-hard"}
CONTINUOUS = {"truthfulqa", "mrcr", "SOTOPIA-hard"}


def read_bytes(url):
    fs, path = filesystem_for(url)
    return fs.cat(path)


def save(name, data, output):
    payload = json.dumps(
        data, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value)
    ).encode()
    (output / name).write_bytes(payload)
    fs, path = filesystem_for(DEST + "/" + name)
    fs.pipe_file(path, payload)


def tournament(scores, sems):
    n, benchmarks = scores.shape
    half_points = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            difference = scores[i] - scores[j]
            threshold = 1.96 * np.hypot(sems[i], sems[j]) / np.sqrt(0.9)
            half_points[i] += np.where(difference > threshold, 2, np.where(difference < -threshold, 0, 1)).sum()
    # Every candidate has the same denominator. Integer accumulation preserves
    # exact ties and makes ranks independent of candidate input order.
    return half_points / (2 * (n - 1) * benchmarks)


def evaluate(predicted, actual, predicted_sems, actual_sems):
    reference = tournament(actual, actual_sems)
    estimated = tournament(predicted, predicted_sems)
    decisive = []
    tie_correct = []
    for i in range(len(actual)):
        for j in range(i):
            d = actual[i] - actual[j]
            pd = predicted[i] - predicted[j]
            threshold = 1.96 * np.hypot(actual_sems[i], actual_sems[j]) / np.sqrt(0.9)
            pt = 1.96 * np.hypot(predicted_sems[i], predicted_sems[j]) / np.sqrt(0.9)
            label = np.where(d > threshold, 1, np.where(d < -threshold, -1, 0))
            plabel = np.where(pd > pt, 1, np.where(pd < -pt, -1, 0))
            decisive.extend((plabel[label != 0] == label[label != 0]).tolist())
            tie_correct.extend((plabel[label == 0] == 0).tolist())
    winner_set = set(np.flatnonzero(reference >= reference.max() - 1e-12).tolist())
    top3 = set(np.argsort(-estimated, kind="stable")[: min(3, len(estimated))].tolist())
    return {
        "score_mae": float(np.mean(np.abs(predicted - actual))),
        "tournament_spearman": float(spearmanr(reference, estimated).statistic),
        "decisive_agreement": float(np.mean(decisive)) if decisive else None,
        "tie_agreement": float(np.mean(tie_correct)) if tie_correct else None,
        "winner_retained_top3": bool(winner_set <= top3),
        "actual_tournament_scores": reference.tolist(),
        "predicted_tournament_scores": estimated.tolist(),
    }


def retained_benchmark_indices(benchmarks):
    """Select the same 21 benchmarks for predictors and outcomes."""
    columns = [j for j, benchmark in enumerate(benchmarks) if benchmark not in EXCLUDED_TARGET]
    if len(columns) != 21 or not EXCLUDED_TARGET <= set(benchmarks):
        raise ValueError("Expected 26 benchmarks with the five agreed exclusions")
    return columns


def retained_responses(response, task_benchmarks):
    """Remove excluded benchmark outcomes from latent-model fitting."""
    retained = ~np.isin(task_benchmarks, list(EXCLUDED_TARGET))
    return np.where(retained[None, :], response, np.nan)


def choose_dimension(response, counts, task_benchmarks, continuous, seed):
    response = retained_responses(response, task_benchmarks)
    rng = np.random.default_rng(seed)
    # Entire outer-held-out families are absent here. Masked-entry validation
    # chooses dimension using only the remaining calibration models.
    observed = np.isfinite(response)
    validation = observed & (rng.random(response.shape) < 0.2)
    fitmask = observed & ~validation
    records = []
    for dimension in [1, 2, 3, 5]:
        model = fit_mirt(response, counts, task_benchmarks, continuous, dimension, seed, mask=fitmask, maxiter=6000)
        p = expit(model.ability @ model.discrimination.T + model.intercept)
        error = []
        for b in sorted(set(task_benchmarks)):
            mask = validation & (task_benchmarks[None, :] == b)
            if mask.any():
                error.append(float(np.mean((p[mask] - response[mask]) ** 2)))
        if not model.converged:
            raise ArithmeticError(f"Unconverged IRT candidate: dimension={dimension}, iterations={model.iterations}")
        record = {
            "dimension": dimension,
            "validation_mse": float(np.mean(error)),
            "converged": model.converged,
            "iterations": model.iterations,
            "loss": model.loss,
            "gradient_error": model.gradient_error,
        }
        records.append(record)
        print(json.dumps({"dimension_cv": record}), flush=True)
    best = min(records, key=lambda row: row["validation_mse"])["dimension"]
    model = fit_mirt(response, counts, task_benchmarks, continuous, best, seed, maxiter=6000)
    if not model.converged:
        raise ArithmeticError("Unconverged full IRT fit")
    return model, records


def main(mode="grouped"):
    global DEST
    if mode not in {"grouped", "chronological"}:
        raise ValueError(mode)
    if mode == "chronological":
        DEST += "-chronological"
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(exist_ok=True, parents=True)
    raw_tasks = read_bytes(SOURCE + "/matrix/tasks.json")
    raw_matrix = read_bytes(SOURCE + "/matrix/responses.npz")
    metadata = json.loads(raw_tasks)
    arrays = np.load(io.BytesIO(raw_matrix))
    models = metadata["models"]
    tasks = metadata["tasks"]
    benchmarks = metadata["benchmarks"]
    response = arrays["responses"]
    counts = arrays["counts"]
    targets = arrays["targets"]
    sems = arrays["sems"]
    costs, domains, cost_description = policy_costs(tasks)
    primary = retained_benchmark_indices(benchmarks)
    target_benchmarks = np.asarray([benchmarks[j] for j in primary])
    targets = targets[:, primary]
    sems = sems[:, primary]
    unresolved = [row for row in metadata["unresolved_cells"] if row["benchmark"] not in EXCLUDED_TARGET]
    if unresolved:
        raise ValueError(f"Unresolved primary target cells: {unresolved}")
    if not np.isfinite(targets).all() or not np.isfinite(sems).all():
        raise ValueError("Incomplete reference target")
    task_benchmarks = np.asarray([t["benchmark"] for t in tasks])
    continuous = np.isin(task_benchmarks, list(CONTINUOUS))
    exclusions = set(metadata["predictor_exclusions"])
    if exclusions != EXCLUDED_TARGET:
        raise ValueError("Predictor and target benchmark exclusions must agree")
    candidates = np.asarray(
        [
            j
            for j, t in enumerate(tasks)
            if t["benchmark"] not in exclusions
            and (t["benchmark"] != "mrcr" or int(t["strata"]["mrcr_bin_upper"]) <= 16384)
        ]
    )
    if len(set(task_benchmarks[candidates])) != 21:
        raise ValueError("Pool must cover 21 benchmarks")
    expected_counts = {
        "NUPA": 5000,
        "aime24": 30,
        "cruxeval": 800,
        "humanevalplus": 164,
        "triviaqa": 17944,
        "swebench-verified": 500,
        "bixbench-pi": 205,
    }
    for b, expected in expected_counts.items():
        actual = int(np.sum(task_benchmarks == b))
        if actual != expected:
            raise ValueError(f"Identity audit failed: {b} {actual} != {expected}")
    groups = np.array(
        [
            "snowball_grug" if ("snowball" in m.lower() or "grug" in m.lower()) else m.split("/")[0].lower()
            for m in models
        ]
    )
    splits = list(GroupKFold(n_splits=4).split(response, groups=groups))
    if mode == "chronological":
        # Fixed baseline anchors plus base and Antidoom calibrate later dated SFT/RL/merge checkpoints.
        later = [
            "DARE-TIES-02-2026.10.10",
            "Datakit-SFT-262K-2026.10.05",
            "Smartie-RL-2026.10.15.32-step14",
            "Repro-262K-Step222-2026.10.25",
        ]
        test = np.asarray([i for i, m in enumerate(models) if any(label in m for label in later)])
        if len(test) != 4:
            raise ValueError("Chronological checkpoint manifest changed")
        train = np.asarray([i for i in range(len(models)) if i not in test])
        splits = [(train, test)]
    evaluated = np.unique(np.concatenate([test for _, test in splits]))
    data_hash = hashlib.sha256(raw_tasks + raw_matrix).hexdigest()
    for name, payload in [("tasks.json", raw_tasks), ("responses.npz", raw_matrix)]:
        fs, path = filesystem_for(DEST + "/inputs/" + name)
        fs.pipe_file(path, payload)
    settings = {
        "validation_mode": mode,
        "evaluated_models": [models[i] for i in evaluated],
        "data_sha256": data_hash,
        "models": models,
        "seed": SEED,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "source": SOURCE,
        "destination": DEST,
        "pool_count": len(candidates),
        "target_benchmarks": target_benchmarks.tolist(),
        "excluded_benchmarks": sorted(EXCLUDED_TARGET),
        "scope": "21 retained benchmarks predict the same 21 outcomes",
        "budgets": [210, 840],
        "cost_proxy": cost_description,
        "mrcr_pool_rule": (
            "Only bins <=16384, eligible under every cohort model native context; full "
            "target retains each model policy bins."
        ),
        "split_groups": groups.tolist(),
        "splits": [{"train": train.tolist(), "test": test.tolist()} for train, test in splits],
        "code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("experiments/evaluation/mini_policy").glob("*.py")
        },
    }
    save("settings.json", settings, output)
    (output / "code").mkdir(exist_ok=True)
    for file in Path("experiments/evaluation/mini_policy").iterdir():
        if file.suffix not in {".py", ".json", ".md"}:
            continue
        content = file.read_bytes()
        (output / "code" / file.name).write_bytes(content)
        fs, path = filesystem_for(DEST + "/code/" + file.name)
        fs.pipe_file(path, content)
    audit = json.loads(read_bytes(SOURCE + "/assembly_audit.json"))
    contexts = {}
    for row in audit:
        if row["benchmark"] == "mrcr":
            context = (row.get("serving") or {}).get("max_model_len")
            if context is None:
                raise ValueError(f'Missing MRCR context for {row["model"]}')
            contexts[row["model"]] = int(context)
    eligibility = np.ones(response.shape, dtype=bool)
    for i, model in enumerate(models):
        for j, task in enumerate(tasks):
            if task["benchmark"] == "mrcr":
                eligibility[i, j] = int(task["strata"]["mrcr_bin_upper"]) <= contexts[model] // 2
    predictions = {}
    details = []
    started = time.monotonic()

    def record(method, seed, fold, test, size, points, selection, train):
        key = f"{method}:{seed}:{size}"
        if key not in predictions:
            predictions[key] = {"predicted": np.full(targets.shape, np.nan), "sems": np.full(sems.shape, np.nan)}
        predicted_sem = np.repeat(np.median(sems[train], axis=0)[None, :], len(test), axis=0)
        predictions[key]["predicted"][test] = points
        predictions[key]["sems"][test] = predicted_sem
        per_model = []
        for position, i in enumerate(test):
            selected = selection[position] if isinstance(selection, list) else selection
            fraction = float(np.isfinite(response[i, selected]).mean())
            per_model.append({"model": models[i], "selected": selected.tolist(), "observed_task_fraction": fraction})
        result = {
            "method": method,
            "seed": seed,
            "fold": fold,
            "budget": size,
            "test_models": [models[i] for i in test],
            "metrics": evaluate(points, targets[test], predicted_sem, sems[test]),
            "probe_manifests": per_model,
            "predicted": points.tolist(),
            "actual": targets[test].tolist(),
            "elapsed_seconds": time.monotonic() - started,
        }
        details.append(result)
        save(f"fold-{fold}-{method}-{seed}-{size}.json", result, output)
        print(
            json.dumps(
                {"completed": {k: result[k] for k in ["method", "seed", "fold", "budget", "metrics", "elapsed_seconds"]}}
            ),
            flush=True,
        )

    for fold, (train, test) in enumerate(splits):
        train_response = response[train]
        train_counts = counts[train]
        # This coverage eligibility is learned only from calibration models.
        pool = candidates[np.mean(np.isfinite(train_response[:, candidates]), axis=0) >= 0.75]
        for seed in [0, 1, 2]:
            order = random_order(pool, tasks, SEED + seed)
            for size in [210, 840]:
                chosen = order[:size]
                points = ridge_prediction(train_response, response[test], targets[train], chosen)
                record("random", seed, fold, test, size, points, chosen, train)
        for method in ["regularized", "mi_mrmr"]:
            order = supervised_order(train_response, targets[train], pool, 840, SEED + fold, method)
            for size in [210, 840]:
                chosen = order[:size]
                points = ridge_prediction(train_response, response[test], targets[train], chosen)
                record(method, 0, fold, test, size, points, chosen, train)
        for method, weight, coverage in [
            ("greedy_cv", np.ones(len(tasks)), None),
            ("greedy_cv_cost", costs, None),
            ("greedy_cv_cost_coverage", costs, domains),
        ]:
            order = greedy_cv_order(
                train_response, targets[train], sems[train], groups[train], pool, 840, SEED + fold, weight, coverage
            )
            for size in [210, 840]:
                chosen = order[:size]
                points = ridge_prediction(train_response, response[test], targets[train], chosen)
                record(method, 0, fold, test, size, points, chosen, train)
        mirt, cv = choose_dimension(train_response, train_counts, task_benchmarks, continuous, SEED + fold)
        save(
            f"fold-{fold}-irt-fit.json",
            {
                "dimension_cv": cv,
                "chosen_dimension": mirt.ability.shape[1],
                "converged": mirt.converged,
                "iterations": mirt.iterations,
                "loss": mirt.loss,
                "gradient_error": mirt.gradient_error,
            },
            output,
        )
        order, _ = information_order(mirt, pool, 840, SEED + fold)
        for size in [210, 840]:
            chosen = order[:size]
            points = np.stack(
                [
                    predict_panel(
                        mirt,
                        estimate_ability(mirt, chosen, response[i], counts[i]),
                        task_benchmarks,
                        target_benchmarks,
                        eligibility[i],
                    )
                    for i in test
                ]
            )
            record("mirt_static", 0, fold, test, size, points, chosen, train)
        adaptive = []
        abilities = []
        for i in test:
            order, checkpoint = information_order(mirt, pool, 840, SEED + fold, response[i], counts[i])
            adaptive.append(order)
            abilities.append(checkpoint)
        for size in [210, 840]:
            points = np.stack(
                [
                    predict_panel(mirt, abilities[position][size], task_benchmarks, target_benchmarks, eligibility[i])
                    for position, i in enumerate(test)
                ]
            )
            record("mirt_adaptive", 0, fold, test, size, points, [order[:size] for order in adaptive], train)
        for suffix, coverage in [("cost", None), ("cost_coverage", domains)]:
            order, _ = information_order(mirt, pool, 840, SEED + fold, costs=costs, domains=coverage)
            for size in [210, 840]:
                chosen = order[:size]
                points = np.stack(
                    [
                        predict_panel(
                            mirt,
                            estimate_ability(mirt, chosen, response[i], counts[i]),
                            task_benchmarks,
                            target_benchmarks,
                            eligibility[i],
                        )
                        for i in test
                    ]
                )
                record("mirt_static_" + suffix, 0, fold, test, size, points, chosen, train)
            adaptive = []
            abilities = []
            for i in test:
                order, checkpoint = information_order(
                    mirt, pool, 840, SEED + fold, response[i], counts[i], costs=costs, domains=coverage
                )
                adaptive.append(order)
                abilities.append(checkpoint)
            for size in [210, 840]:
                points = np.stack(
                    [
                        predict_panel(
                            mirt, abilities[position][size], task_benchmarks, target_benchmarks, eligibility[i]
                        )
                        for position, i in enumerate(test)
                    ]
                )
                record(
                    "mirt_adaptive_" + suffix, 0, fold, test, size, points, [order[:size] for order in adaptive], train
                )
    summaries = []
    for key, data in predictions.items():
        method, seed, budget = key.split(":")
        if not np.isfinite(data["predicted"][evaluated]).all():
            raise ValueError("Incomplete out-of-fold predictions")
        summary = {
            "method": method,
            "seed": int(seed),
            "budget": int(budget),
            "metrics": evaluate(
                data["predicted"][evaluated], targets[evaluated], data["sems"][evaluated], sems[evaluated]
            ),
        }
        summaries.append(summary)
        save(
            f"oof-{method}-{seed}-{budget}.json",
            {
                "models": models,
                "benchmarks": target_benchmarks.tolist(),
                "predicted": data["predicted"].tolist(),
                "actual": targets.tolist(),
                "predicted_sem": data["sems"].tolist(),
                "actual_sem": sems.tolist(),
            },
            output,
        )
    save("summary.json", {"results": summaries, "elapsed_seconds": time.monotonic() - started}, output)
    print("EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["grouped", "chronological"], nargs="?", default="grouped")
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--destination", default=DEST)
    arguments = parser.parse_args()
    SOURCE = arguments.source
    DEST = arguments.destination
    main(arguments.mode)
