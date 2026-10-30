# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exploratory benchmark-level AutoML analysis for the Marin evaluation cohort.

Prepare the score matrix locally, then run the analysis on an Iris CPU node:

    python cohort_automl.py prepare TRACKER.md scores.csv
    python cohort_automl.py analyze scores.csv OUTPUT_DIR

The 26 benchmark aggregates are provisional features, not trial-level items.
"""

import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GroupKFold

EXCLUDED_TOURNAMENT = {"financebench", "piqa", "triviaqa"}
SIZES = (2, 4, 6, 8, 12)
SEED = 20261030
RIDGE_ALPHA = 10.0
METHODS = ("variance", "mutual_info", "mrmr", "pca_leverage", "svd_logit", "irt_information", "lasso", "extra_trees")


def family(model):
    lower = model.lower()
    if "campaign4" in lower:
        return "campaign4"
    if "merge-campaign" in lower or "merge-nupa" in lower:
        return "early_merges"
    if "datakit-sft-repro" in lower:
        return "datakit_repro"
    if "datakit-sft" in lower:
        return "datakit"
    if "antidoom" in lower:
        return "antidoom"
    if "bfcl" in lower and "grug" in lower:
        return "grug_bfcl"
    if "smartie" in lower:
        return "smartie"
    if "snowball" in lower or "grug" in lower:
        return "snowball_other"
    if lower.startswith("qwen/"):
        return "qwen"
    if lower.startswith("nvidia/"):
        return "nvidia"
    return model.split("/", 1)[0].lower()


def prepare(tracker, destination):
    lines = Path(tracker).read_text().splitlines()
    table = [line for line in lines if line.startswith("| ")]
    headers = [cell.strip() for cell in table[0].split("|")[1:-1]]
    assert headers[0] == "Model", headers[0]
    assert len(headers) == 27, len(headers)
    rows = []
    for line in table[3:]:
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        assert len(cells) == len(headers), cells[0]
        values = []
        for cell in cells[1:]:
            match = re.match(r"^(0\.\d+|1\.0+|1)(?:\s|$)", cell)
            if not match:
                raise ValueError(f"Non-score cell for {cells[0]}: {cell[:100]}")
            values.append(float(match.group(1)))
        rows.append([cells[0], family(cells[0]), *values])
    assert len(rows) == 42, len(rows)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "family", *headers[1:]])
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "benchmarks": len(headers) - 1,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        )
    )


def read_scores(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames[2:]
        rows = list(reader)
    models = [row["model"] for row in rows]
    groups = np.array([row["family"] for row in rows])
    values = np.array([[float(row[name]) for name in names] for row in rows])
    assert values.shape == (42, 26), values.shape
    assert np.isfinite(values).all()
    return models, groups, names, values


def standardize(train, test):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (train - mean) / std, (test - mean) / std


def relevance(x, y):
    return mutual_info_regression(x, y, random_state=SEED, n_neighbors=3)


def irt_proxy_information(x):
    """2PL-style information from fractional benchmark scores.

    PC1 estimates model ability. Regression of logit(score) on that ability
    gives benchmark discrimination and intercept; this is not item-level IRT.
    """
    pc = PCA(n_components=1).fit_transform(x).ravel()
    if np.corrcoef(pc, x.mean(axis=1))[0, 1] < 0:
        pc = -pc
    pc = (pc - pc.mean()) / max(pc.std(), 1e-6)
    # x here is standardized; map each benchmark back to its empirical rank
    # probability to keep logits finite and comparable across metrics.
    ranks = rankdata(x, axis=0, method="average")
    probabilities = (ranks - 0.5) / len(x)
    logit = np.log(probabilities / (1 - probabilities))
    design = np.column_stack([pc, np.ones(len(pc))])
    slopes, intercepts = np.linalg.lstsq(design, logit, rcond=None)[0]
    slopes = np.maximum(slopes, 0)
    predicted = 1 / (1 + np.exp(-(pc[:, None] * slopes + intercepts)))
    info = np.mean(slopes**2 * predicted * (1 - predicted), axis=0)
    return info, slopes, intercepts


def feature_order(method, x, target, raw):
    n = x.shape[1]
    if method == "variance":
        score = np.var(raw, axis=0)
    elif method == "mutual_info":
        score = relevance(x, target)
    elif method == "mrmr":
        rel = relevance(x, target)
        corr = np.nan_to_num(np.corrcoef(x, rowvar=False))
        selected = []
        remaining = set(range(n))
        while remaining:
            best = max(remaining, key=lambda j: rel[j] - (np.abs(corr[j, selected]).mean() if selected else 0))
            selected.append(best)
            remaining.remove(best)
        return selected
    elif method == "pca_leverage":
        pca = PCA(n_components=min(3, x.shape[0] - 1)).fit(x)
        score = ((pca.components_**2) * pca.explained_variance_ratio_[:, None]).sum(axis=0)
    elif method == "svd_logit":
        # A monotonic transform stresses near-boundary differences unlike PCA.
        bounded = np.clip(0.5 + x / 8, 0.01, 0.99)
        transformed = np.log(bounded / (1 - bounded))
        _, singular, right = np.linalg.svd(transformed - transformed.mean(axis=0), full_matrices=False)
        score = ((right[:3] ** 2) * singular[:3, None] ** 2).sum(axis=0)
    elif method == "irt_information":
        score, _, _ = irt_proxy_information(x)
    elif method == "lasso":
        fit = Lasso(alpha=0.025, max_iter=10000).fit(x, target)
        score = np.abs(fit.coef_)
    elif method == "extra_trees":
        fit = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=1).fit(x, target)
        score = fit.feature_importances_
    else:
        raise ValueError(method)
    return list(np.argsort(-score, kind="stable"))


def selected_prediction(x_train, x_test, target_indices, chosen):
    fit = Ridge(alpha=RIDGE_ALPHA).fit(x_train[:, chosen], x_train[:, target_indices])
    return fit.predict(x_test[:, chosen])


def metrics(predicted, actual):
    error = float(np.abs(predicted - actual).mean())
    predicted_composite = predicted.mean(axis=1)
    actual_composite = actual.mean(axis=1)
    rho = float(spearmanr(predicted_composite, actual_composite).statistic) if len(actual) >= 3 else float("nan")
    return error, rho


def evaluate(models, groups, names, values, output):
    output.mkdir(parents=True, exist_ok=True)
    target_indices = np.array([j for j, name in enumerate(names) if name not in EXCLUDED_TOURNAMENT])
    assert len(target_indices) == 23
    splits = list(GroupKFold(n_splits=5).split(values, groups=groups))
    results = []
    selections = []
    for fold, (train, test) in enumerate(splits):
        x_train, x_test = standardize(values[train], values[test])
        target = x_train[:, target_indices].mean(axis=1)
        for method in METHODS:
            order = feature_order(method, x_train, target, values[train])
            selections.append({"fold": fold, "method": method, "ordered_benchmarks": [names[j] for j in order]})
            for size in SIZES:
                chosen = order[:size]
                predicted = selected_prediction(x_train, x_test, target_indices, chosen)
                error, rho = metrics(predicted, x_test[:, target_indices])
                results.append(
                    {
                        "fold": fold,
                        "method": method,
                        "size": size,
                        "mae_z": error,
                        "spearman": rho,
                        "test_models": len(test),
                    }
                )
        for repeat in range(20):
            order = np.random.default_rng(SEED + fold * 100 + repeat).permutation(len(names))
            for size in SIZES:
                predicted = selected_prediction(x_train, x_test, target_indices, order[:size])
                error, rho = metrics(predicted, x_test[:, target_indices])
                results.append(
                    {
                        "fold": fold,
                        "method": "random",
                        "size": size,
                        "mae_z": error,
                        "spearman": rho,
                        "test_models": len(test),
                    }
                )

    with (output / "cross_validation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    (output / "selected_features.json").write_text(json.dumps(selections, indent=2))

    x, _ = standardize(values, values)
    pca = PCA().fit(x)
    rng = np.random.default_rng(SEED)
    null_eigenvalues = []
    for _ in range(250):
        permuted = np.column_stack([rng.permutation(x[:, j]) for j in range(x.shape[1])])
        null_eigenvalues.append(PCA().fit(permuted).explained_variance_)
    null_threshold = np.quantile(null_eigenvalues, 0.95, axis=0)
    information, slopes, intercepts = irt_proxy_information(x)
    with (output / "irt_benchmark_proxy.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["benchmark", "discrimination", "intercept", "mean_information"])
        writer.writerows(zip(names, slopes, intercepts, information, strict=True))

    summary = []
    for method in (*METHODS, "random"):
        for size in SIZES:
            subset = [row for row in results if row["method"] == method and row["size"] == size]
            weights = np.array([row["test_models"] for row in subset])
            summary.append(
                {
                    "method": method,
                    "size": size,
                    "mae_z": float(np.average([row["mae_z"] for row in subset], weights=weights)),
                    "mean_fold_spearman": float(np.nanmean([row["spearman"] for row in subset])),
                }
            )
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    axes[0].plot(np.arange(1, 27), pca.explained_variance_, "o-", label="Observed")
    axes[0].plot(np.arange(1, 27), null_threshold, "--", label="95% shuffled null")
    axes[0].set(xlabel="Component", ylabel="Eigenvalue", title="PCA parallel analysis")
    axes[0].legend(fontsize=8)
    for method in (*METHODS, "random"):
        subset = [row for row in summary if row["method"] == method]
        axes[1].plot(SIZES, [row["mae_z"] for row in subset], marker="o", label=method)
        axes[2].plot(SIZES, [row["mean_fold_spearman"] for row in subset], marker="o", label=method)
    axes[1].set(
        xlabel="Benchmarks observed", ylabel="Held-out mean absolute error (SD units)", title="Scorecard reconstruction"
    )
    axes[2].set(xlabel="Benchmarks observed", ylabel="Mean fold Spearman rho", title="Composite rank fidelity")
    axes[1].legend(fontsize=7, ncol=2)
    fig.savefig(output / "cohort_methods.png", dpi=180)
    fig.savefig(output / "cohort_methods.svg")
    plt.close(fig)

    coordinates = pca.transform(x)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    is_snowball = np.array(["grug" in model.lower() or "snowball" in model.lower() for model in models])
    ax.scatter(coordinates[~is_snowball, 0], coordinates[~is_snowball, 1], label="Baseline", alpha=0.8)
    ax.scatter(coordinates[is_snowball, 0], coordinates[is_snowball, 1], label="Snowball / Grug", alpha=0.8)
    for j in np.argsort(np.linalg.norm(coordinates[:, :2], axis=1))[-5:]:
        ax.annotate(models[j].split("/")[-1][:24], coordinates[j, :2], fontsize=7)
    ax.set(
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%})",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%})",
        title="Model score profiles across 26 benchmarks",
    )
    ax.legend()
    fig.savefig(output / "cohort_pca.png", dpi=180)
    fig.savefig(output / "cohort_pca.svg")
    plt.close(fig)

    corr = np.corrcoef(x, rowvar=False)
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_title("Benchmark score correlations across models")
    fig.colorbar(image, ax=ax, fraction=0.04)
    fig.savefig(output / "benchmark_correlations.png", dpi=180)
    fig.savefig(output / "benchmark_correlations.svg")
    plt.close(fig)

    report = {
        "rows": len(models),
        "benchmarks": len(names),
        "target_benchmarks": len(target_indices),
        "family_counts": {group: int(np.sum(groups == group)) for group in sorted(set(groups))},
        "pca_explained_first_five": pca.explained_variance_ratio_[:5].tolist(),
        "pca_cumulative_first_five": np.cumsum(pca.explained_variance_ratio_[:5]).tolist(),
        "parallel_analysis_components": int(np.sum(pca.explained_variance_ > null_threshold)),
        "top_irt_information": [names[j] for j in np.argsort(-information)[:8]],
        "best_by_size": {
            str(size): min(
                (row for row in summary if row["size"] == size and row["method"] != "random"),
                key=lambda row: row["mae_z"],
            )
            for size in SIZES
        },
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "prepare":
        prepare(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == "analyze":
        models, groups, names, values = read_scores(sys.argv[2])
        output = Path(os.environ["IRIS_OUTPUT_DIR"]) if sys.argv[3] == "@iris-output" else Path(sys.argv[3])
        evaluate(models, groups, names, values, output)
    else:
        raise SystemExit("usage: cohort_automl.py prepare TRACKER.md scores.csv | analyze scores.csv OUTPUT_DIR")


if __name__ == "__main__":
    main()
