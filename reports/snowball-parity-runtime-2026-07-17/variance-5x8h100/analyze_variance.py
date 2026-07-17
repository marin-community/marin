from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
PREFIX = "LOGPROB_VARIANCE "
TOLERANCE = 0.075
RUNS = [
    (1, "7ff3af48"),
    (2, "9af1f63f"),
    (3, "c8ab3bb9"),
    (4, "0373bc39"),
    (5, "b3e2493b"),
]
INFRA_FAILURE = "f6b61bf9"


def read_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    decoder = json.JSONDecoder()
    for line in path.read_text().splitlines():
        position = 0
        while (marker := line.find(PREFIX, position)) >= 0:
            payload_start = marker + len(PREFIX)
            record, consumed = decoder.raw_decode(line[payload_start:])
            records.append(record)
            position = payload_start + consumed
    return records


def percentile(values: pd.Series, q: float) -> float:
    return float(np.percentile(values.to_numpy(), q))


def main() -> None:
    observations: list[dict[str, object]] = []
    tokens: list[dict[str, object]] = []

    for run, job_id in RUNS:
        records = read_records(ROOT / f"job-{job_id}.log")
        if len(records) != 72:
            raise ValueError(f"run {run} ({job_id}) has {len(records)} records, expected 72")
        for record in records:
            scores = record.pop("scores")
            observation = {"run": run, "job_id": job_id, **record}
            observations.append(observation)
            for golden_rank, score in enumerate(scores):
                actual_logprob = float(score["actual_logprob"])
                golden_logprob = float(score["golden_logprob"])
                tokens.append(
                    {
                        "run": run,
                        "job_id": job_id,
                        "request_kind": record["request_kind"],
                        "case_id": record["case_id"],
                        "backend_rank": record["rank"],
                        "golden_rank": golden_rank,
                        "token_id": score["token_id"],
                        "actual_logprob": actual_logprob,
                        "golden_logprob": golden_logprob,
                        "actual_probability": math.exp(actual_logprob),
                        "golden_probability": math.exp(golden_logprob),
                        "absolute_probability_error": abs(math.exp(actual_logprob) - math.exp(golden_logprob)),
                    }
                )

    obs = pd.DataFrame(observations).sort_values(["run", "request_kind", "case_id", "rank"])
    token = pd.DataFrame(tokens).sort_values(
        ["run", "request_kind", "case_id", "backend_rank", "golden_rank"]
    )
    obs.to_csv(ROOT / "observations.csv", index=False)
    token.to_csv(ROOT / "token-observations.csv", index=False)

    run_rows: list[dict[str, object]] = []
    for run, job_id in RUNS:
        run_obs = obs[obs.run == run]
        rep = run_obs[run_obs.request_kind == "representative"]
        sentinel = run_obs[run_obs.request_kind == "sentinel"]
        rep_worst = rep.loc[rep.max_probability_error.idxmax()]
        sentinel_worst = sentinel.loc[sentinel.max_probability_error.idxmax()]
        run_rows.append(
            {
                "run": run,
                "job_id": job_id,
                "passed_0_075_gate": bool(run_obs.max_probability_error.max() <= TOLERANCE),
                "representative_max_probability_error": float(rep_worst.max_probability_error),
                "representative_worst_case": rep_worst.case_id,
                "representative_worst_rank": int(rep_worst["rank"]),
                "sentinel_max_probability_error": float(sentinel_worst.max_probability_error),
                "sentinel_worst_rank": int(sentinel_worst["rank"]),
                "sentinel_min_probability_error": float(sentinel.max_probability_error.min()),
                "sentinel_median_probability_error": float(sentinel.max_probability_error.median()),
            }
        )
    run_summary = pd.DataFrame(run_rows)
    run_summary.to_csv(ROOT / "run-summary.csv", index=False)

    group_keys = ["request_kind", "case_id", "backend_rank", "golden_rank", "token_id"]
    cross_run = (
        token.groupby(group_keys, as_index=False)
        .agg(
            run_count=("run", "nunique"),
            golden_probability=("golden_probability", "first"),
            actual_probability_mean=("actual_probability", "mean"),
            actual_probability_std=("actual_probability", "std"),
            actual_probability_min=("actual_probability", "min"),
            actual_probability_max=("actual_probability", "max"),
            actual_logprob_min=("actual_logprob", "min"),
            actual_logprob_max=("actual_logprob", "max"),
        )
    )
    if not (cross_run.run_count == len(RUNS)).all():
        raise ValueError("not every exact request/token tuple appears in all five runs")
    cross_run["actual_probability_range"] = (
        cross_run.actual_probability_max - cross_run.actual_probability_min
    )
    cross_run["actual_logprob_range"] = cross_run.actual_logprob_max - cross_run.actual_logprob_min
    cross_run.to_csv(ROOT / "cross-run-token-variance.csv", index=False)

    worst_token = cross_run.loc[cross_run.actual_probability_range.idxmax()]
    request_metric_variance = (
        obs.groupby(["request_kind", "case_id", "rank"], as_index=False)
        .agg(
            run_count=("run", "nunique"),
            max_error_min=("max_probability_error", "min"),
            max_error_max=("max_probability_error", "max"),
            max_error_mean=("max_probability_error", "mean"),
            max_error_std=("max_probability_error", "std"),
        )
    )
    request_metric_variance["max_error_range"] = (
        request_metric_variance.max_error_max - request_metric_variance.max_error_min
    )
    request_metric_variance.to_csv(ROOT / "cross-run-request-variance.csv", index=False)
    worst_request = request_metric_variance.loc[request_metric_variance.max_error_range.idxmax()]

    kind_stats: dict[str, dict[str, object]] = {}
    for kind, rows in cross_run.groupby("request_kind"):
        worst = rows.loc[rows.actual_probability_range.idxmax()]
        kind_stats[kind] = {
            "exact_token_tuples": int(len(rows)),
            "median_probability_range": float(rows.actual_probability_range.median()),
            "p95_probability_range": percentile(rows.actual_probability_range, 95),
            "max_probability_range": float(rows.actual_probability_range.max()),
            "median_logprob_range": float(rows.actual_logprob_range.median()),
            "p95_logprob_range": percentile(rows.actual_logprob_range, 95),
            "max_logprob_range": float(rows.actual_logprob_range.max()),
            "largest_probability_range_tuple": {
                "case_id": worst.case_id,
                "backend_rank": int(worst.backend_rank),
                "golden_rank": int(worst.golden_rank),
                "token_id": int(worst.token_id),
                "golden_probability": float(worst.golden_probability),
                "actual_probability_min": float(worst.actual_probability_min),
                "actual_probability_max": float(worst.actual_probability_max),
                "actual_probability_range": float(worst.actual_probability_range),
                "actual_logprob_min": float(worst.actual_logprob_min),
                "actual_logprob_max": float(worst.actual_logprob_max),
                "actual_logprob_range": float(worst.actual_logprob_range),
            },
        }

    summary = {
        "commit": "4b32fcfb0747dbb83ef4c0e8a13ffa7d4863218d",
        "valid_numerical_runs": len(RUNS),
        "passing_runs": int(run_summary.passed_0_075_gate.sum()),
        "failing_runs": int((~run_summary.passed_0_075_gate).sum()),
        "excluded_infrastructure_failure_job": INFRA_FAILURE,
        "tolerance": TOLERANCE,
        "records_per_run": 72,
        "representative_records_per_run": 64,
        "sentinel_records_per_run": 8,
        "top_tokens_per_record": 25,
        "runs": run_rows,
        "sentinel_max_across_runs": float(run_summary.sentinel_max_probability_error.max()),
        "sentinel_min_of_run_maxima": float(run_summary.sentinel_max_probability_error.min()),
        "representative_max_across_runs": float(run_summary.representative_max_probability_error.max()),
        "token_variance_by_request_kind": kind_stats,
        "largest_exact_token_probability_range": {
            key: int(worst_token[key])
            if key in {"backend_rank", "golden_rank", "token_id"}
            else float(worst_token[key])
            if key in {
                "golden_probability",
                "actual_probability_min",
                "actual_probability_max",
                "actual_probability_range",
                "actual_logprob_min",
                "actual_logprob_max",
                "actual_logprob_range",
            }
            else worst_token[key]
            for key in [
                "request_kind",
                "case_id",
                "backend_rank",
                "golden_rank",
                "token_id",
                "golden_probability",
                "actual_probability_min",
                "actual_probability_max",
                "actual_probability_range",
                "actual_logprob_min",
                "actual_logprob_max",
                "actual_logprob_range",
            ]
        },
        "largest_request_metric_range": {
            "request_kind": worst_request.request_kind,
            "case_id": worst_request.case_id,
            "rank": int(worst_request["rank"]),
            "max_error_min": float(worst_request.max_error_min),
            "max_error_max": float(worst_request.max_error_max),
            "max_error_range": float(worst_request.max_error_range),
        },
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    save_run_maxima(run_summary)
    save_sentinel_error_heatmap(obs)
    save_top_token_probability_heatmap(token)
    save_token_range_distribution(cross_run)

    print(json.dumps(summary, indent=2))


def save_run_maxima(run_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(len(run_summary))
    width = 0.34
    ax.bar(
        x - width / 2,
        run_summary.representative_max_probability_error,
        width,
        color="#4C78A8",
        label="64 representative cases",
    )
    sentinel_bars = ax.bar(
        x + width / 2,
        run_summary.sentinel_max_probability_error,
        width,
        color=["#F58518" if passed else "#E45756" for passed in run_summary.passed_0_075_gate],
        label="8-rank sentinel",
    )
    ax.axhline(TOLERANCE, color="#222222", linestyle="--", linewidth=1.3, label="0.075 gate")
    for bar, value in zip(sentinel_bars, run_summary.sentinel_max_probability_error):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, [f"Run {run}" for run in run_summary.run])
    ax.set_ylabel("Maximum absolute probability error")
    ax.set_title("Three of five cold vLLM runs exceed the sentinel gate")
    ax.set_ylim(0, max(0.105, run_summary.sentinel_max_probability_error.max() * 1.17))
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "run-maxima.png", dpi=180)
    plt.close(fig)


def save_sentinel_error_heatmap(obs: pd.DataFrame) -> None:
    sentinel = obs[obs.request_kind == "sentinel"]
    matrix = sentinel.pivot(index="run", columns="rank", values="max_probability_error").sort_index()
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    image = ax.imshow(matrix.to_numpy(), cmap="YlOrRd", vmin=0, vmax=0.10, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iat[row, column]
            ax.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if value >= 0.065 else "#222222")
    ax.set_xticks(np.arange(matrix.shape[1]), [f"Rank {rank}" for rank in matrix.columns])
    ax.set_yticks(np.arange(matrix.shape[0]), [f"Run {run}" for run in matrix.index])
    ax.set_title("Sentinel maximum probability error varies by cold run and rank")
    ax.set_xlabel("vLLM tensor-parallel rank targeted by the request")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Maximum absolute probability error")
    fig.tight_layout()
    fig.savefig(ROOT / "sentinel-error-heatmap.png", dpi=180)
    plt.close(fig)


def save_top_token_probability_heatmap(token: pd.DataFrame) -> None:
    sentinel_top = token[(token.request_kind == "sentinel") & (token.golden_rank == 0)]
    matrix = sentinel_top.pivot(index="run", columns="backend_rank", values="actual_probability").sort_index()
    golden = float(sentinel_top.golden_probability.iloc[0])
    low = min(float(matrix.min().min()), golden)
    high = max(float(matrix.max().max()), golden)
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    image = ax.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=low, vmax=high, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix.iat[row, column]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(np.arange(matrix.shape[1]), [f"Rank {rank}" for rank in matrix.columns])
    ax.set_yticks(np.arange(matrix.shape[0]), [f"Run {run}" for run in matrix.index])
    ax.set_title(f"Sentinel top-token probability (golden = {golden:.3f})")
    ax.set_xlabel("vLLM tensor-parallel rank targeted by the request")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Observed probability for token 423")
    fig.tight_layout()
    fig.savefig(ROOT / "sentinel-top-token-probability.png", dpi=180)
    plt.close(fig)


def save_token_range_distribution(cross_run: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    values = [
        cross_run[cross_run.request_kind == "representative"].actual_probability_range.to_numpy(),
        cross_run[cross_run.request_kind == "sentinel"].actual_probability_range.to_numpy(),
    ]
    ax.boxplot(values, tick_labels=["Representative\n(1,600 tuples)", "Sentinel\n(200 tuples)"], showfliers=False)
    rng = np.random.default_rng(7)
    for index, group in enumerate(values, start=1):
        x = rng.normal(index, 0.055, size=len(group))
        ax.scatter(x, np.maximum(group, 1e-8), s=8, alpha=0.18, color=["#4C78A8", "#F58518"][index - 1])
    ax.set_yscale("log")
    ax.set_ylabel("Across-run range of actual token probability (log scale)")
    ax.set_title("Exact request/token outputs vary across identical cold deployments")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "token-probability-range.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
