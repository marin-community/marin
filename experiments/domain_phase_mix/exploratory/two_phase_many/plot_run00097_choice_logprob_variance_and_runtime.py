# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot choice-logprob mean/std and active runtime across compute settings."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from marin.speedrun.speedrun import get_step_times_from_wandb

from experiments.domain_phase_mix.launch_two_phase_many_run_00097_seed_study import (
    N_SEED_SWEEP_RUNS,
    SEED_SWEEP_START,
)

OUTPUT_DIR = Path(__file__).resolve().parent
SUMMARY_JSON_PATH = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_compute_scaling/"
    "compute_scaling_noise_report-0ebf16/compute_scaling_noise_summary.json"
)
COMPUTE_RESULTS_CSV_PATH = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_compute_scaling/"
    "compute_scaling_noise_report-0ebf16/results.csv"
)
SEED_STUDY_CHECKPOINT_PREFIX = (
    "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_seed_study"
)
FIXED_SUBSET_CHECKPOINT_PREFIX = (
    "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_fixed_subset_study"
)
SWARM_RESULTS_CSV_PATH = OUTPUT_DIR / "two_phase_many.csv"
SETTING_ORDER = ("30M 3B", "60M 1.2B", "60M 1.2B Fixed Subset", "60M 1.2B Swarm", "60M 6B")
SETTING_LABELS = {
    "baseline_1x": "60M 1.2B",
    "fixed_subset_1x": "60M 1.2B Fixed Subset",
    "olmo3_30m_3b": "30M 3B",
    "regmix60m_6b": "60M 6B",
    "swarm_241": "60M 1.2B Swarm",
}
TARGET_METRIC = "lm_eval/mmlu_5shot/choice_logprob"
OUTPUT_PNG = OUTPUT_DIR / "run00097_choice_logprob_variance_and_runtime.png"
OUTPUT_CSV = OUTPUT_DIR / "run00097_choice_logprob_variance_and_runtime.csv"
RUNTIME_CACHE_JSON = OUTPUT_DIR / "run_active_compute_hours_cache.json"
RUNTIME_FETCH_WORKERS = 6


@dataclass(frozen=True)
class SettingSummary:
    setting: str
    cohort: str
    metric_mean: float
    metric_std: float
    metric_n: int
    active_compute_hours_mean: float
    active_compute_hours_std: float
    runtime_n: int


def _load_summary_rows() -> pd.DataFrame:
    with fsspec.open(SUMMARY_JSON_PATH) as f:
        payload = json.load(f)
    return pd.DataFrame(payload["rows"])


def _extract_seed_study_run_ids() -> list[str]:
    fs, _, _ = fsspec.get_fs_token_paths(SEED_STUDY_CHECKPOINT_PREFIX)
    status_paths = sorted(fs.glob(SEED_STUDY_CHECKPOINT_PREFIX + "/**/.executor_status"))
    expected_prefixes = [f"trainer_seed_{SEED_SWEEP_START + offset}" for offset in range(N_SEED_SWEEP_RUNS)]
    run_ids: list[str] = []
    for path in status_paths:
        run_id = path.split("ngd3dm2_run00097_seed_study/", 1)[1].split("/.executor_status", 1)[0]
        if any(run_id.startswith(prefix) for prefix in expected_prefixes):
            run_ids.append(run_id)
    if len(run_ids) != N_SEED_SWEEP_RUNS:
        raise ValueError(f"Expected {N_SEED_SWEEP_RUNS} seed-study run IDs, found {len(run_ids)}: {run_ids}")
    return sorted(run_ids)


def _extract_fixed_subset_run_ids() -> list[str]:
    fs, _, _ = fsspec.get_fs_token_paths(FIXED_SUBSET_CHECKPOINT_PREFIX)
    status_paths = sorted(fs.glob(FIXED_SUBSET_CHECKPOINT_PREFIX + "/**/.executor_status"))
    expected_prefixes = [f"trainer_seed_{SEED_SWEEP_START + offset}" for offset in range(N_SEED_SWEEP_RUNS)]
    run_ids: list[str] = []
    for path in status_paths:
        run_id = path.split("ngd3dm2_run00097_fixed_subset_study/", 1)[1].split("/.executor_status", 1)[0]
        if any(run_id.startswith(prefix) for prefix in expected_prefixes):
            run_ids.append(run_id)
    if len(run_ids) != N_SEED_SWEEP_RUNS:
        raise ValueError(f"Expected {N_SEED_SWEEP_RUNS} fixed-subset run IDs, found {len(run_ids)}: {run_ids}")
    return sorted(run_ids)


def _extract_compute_run_ids() -> dict[str, list[str]]:
    df = pd.read_csv(COMPUTE_RESULTS_CSV_PATH, usecols=["ladder", "status", "wandb_run_id"])
    completed = df[(df["status"] == "completed") & df["wandb_run_id"].notna()].copy()
    by_ladder = {
        ladder: sorted(completed.loc[completed["ladder"] == ladder, "wandb_run_id"].astype(str).tolist())
        for ladder in ("olmo3_30m_3b", "regmix60m_6b")
    }
    for ladder, run_ids in by_ladder.items():
        if len(run_ids) != 10:
            raise ValueError(f"Expected 10 completed run IDs for {ladder}, found {len(run_ids)}: {run_ids}")
    return by_ladder


def _extract_swarm_run_ids_and_metric_stats() -> tuple[list[str], float, float, int]:
    df = pd.read_csv(
        SWARM_RESULTS_CSV_PATH,
        usecols=["wandb_run_id", "status", TARGET_METRIC],
    )
    completed = df[(df["status"] == "completed") & df["wandb_run_id"].notna()].copy()
    metric = pd.to_numeric(completed[TARGET_METRIC], errors="coerce").dropna()
    run_ids = completed.loc[completed[TARGET_METRIC].notna(), "wandb_run_id"].astype(str).tolist()
    if len(run_ids) != 241:
        raise ValueError(f"Expected 241 completed swarm run IDs, found {len(run_ids)}")
    return run_ids, float(metric.mean()), float(metric.std(ddof=1)), len(metric)


def _load_runtime_cache() -> dict[str, float]:
    if not RUNTIME_CACHE_JSON.exists():
        return {}
    return json.loads(RUNTIME_CACHE_JSON.read_text())


def _save_runtime_cache(cache: dict[str, float]) -> None:
    RUNTIME_CACHE_JSON.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _fetch_active_compute_hours(run_id: str) -> float:
    return float(sum(get_step_times_from_wandb(run_id=run_id)) / 3600.0)


def _fetch_metric_values(run_ids: list[str], metric: str) -> np.ndarray:
    import wandb

    api = wandb.Api()
    values: list[float] = []
    for run_id in run_ids:
        run = api.run(f"marin-community/marin/{run_id}")
        value = run.summary.get(metric)
        if value is None:
            raise ValueError(f"Missing {metric} for run {run_id}")
        values.append(float(value))
    return np.asarray(values, dtype=float)


def _active_compute_hours(run_ids: list[str]) -> tuple[float, float]:
    cache = _load_runtime_cache()
    missing_run_ids = [run_id for run_id in run_ids if run_id not in cache]

    if missing_run_ids:
        with ThreadPoolExecutor(max_workers=RUNTIME_FETCH_WORKERS) as pool:
            future_by_run_id = {pool.submit(_fetch_active_compute_hours, run_id): run_id for run_id in missing_run_ids}
            for future in as_completed(future_by_run_id):
                run_id = future_by_run_id[future]
                cache[run_id] = float(future.result())
        _save_runtime_cache(cache)

    durations = np.array([cache[run_id] for run_id in run_ids], dtype=float)
    return float(durations.mean()), float(durations.std(ddof=1))


def build_setting_summaries() -> list[SettingSummary]:
    rows = _load_summary_rows()
    metric_rows = rows[(rows["metric"] == TARGET_METRIC) & rows["cohort"].isin(SETTING_LABELS)].copy()

    seed_run_ids = _extract_seed_study_run_ids()
    fixed_subset_run_ids = _extract_fixed_subset_run_ids()
    compute_run_ids = _extract_compute_run_ids()
    swarm_run_ids, swarm_mean, swarm_std, swarm_n = _extract_swarm_run_ids_and_metric_stats()
    fixed_subset_metric_values = _fetch_metric_values(fixed_subset_run_ids, TARGET_METRIC)

    runtime_by_cohort = {
        "baseline_1x": _active_compute_hours(seed_run_ids),
        "fixed_subset_1x": _active_compute_hours(fixed_subset_run_ids),
        "olmo3_30m_3b": _active_compute_hours(compute_run_ids["olmo3_30m_3b"]),
        "regmix60m_6b": _active_compute_hours(compute_run_ids["regmix60m_6b"]),
        "swarm_241": _active_compute_hours(swarm_run_ids),
    }

    summaries: list[SettingSummary] = []
    for cohort in ("olmo3_30m_3b", "baseline_1x", "fixed_subset_1x", "swarm_241", "regmix60m_6b"):
        runtime_mean, runtime_std = runtime_by_cohort[cohort]
        if cohort == "swarm_241":
            summaries.append(
                SettingSummary(
                    setting=SETTING_LABELS[cohort],
                    cohort=cohort,
                    metric_mean=swarm_mean,
                    metric_std=swarm_std,
                    metric_n=swarm_n,
                    active_compute_hours_mean=runtime_mean,
                    active_compute_hours_std=runtime_std,
                    runtime_n=len(swarm_run_ids),
                )
            )
            continue
        if cohort == "fixed_subset_1x":
            summaries.append(
                SettingSummary(
                    setting=SETTING_LABELS[cohort],
                    cohort=cohort,
                    metric_mean=float(fixed_subset_metric_values.mean()),
                    metric_std=float(fixed_subset_metric_values.std(ddof=1)),
                    metric_n=int(fixed_subset_metric_values.size),
                    active_compute_hours_mean=runtime_mean,
                    active_compute_hours_std=runtime_std,
                    runtime_n=len(fixed_subset_run_ids),
                )
            )
            continue

        row = metric_rows.loc[metric_rows["cohort"] == cohort].iloc[0]
        summaries.append(
            SettingSummary(
                setting=SETTING_LABELS[cohort],
                cohort=cohort,
                metric_mean=float(row["mean"]),
                metric_std=float(row["std"]),
                metric_n=int(row["n"]),
                active_compute_hours_mean=runtime_mean,
                active_compute_hours_std=runtime_std,
                runtime_n=10,
            )
        )
    return summaries


def plot_summaries(summaries: list[SettingSummary]) -> None:
    df = pd.DataFrame([s.__dict__ for s in summaries])
    df["setting"] = pd.Categorical(df["setting"], categories=SETTING_ORDER, ordered=True)
    df = df.sort_values("setting")

    cmap = plt.get_cmap("RdYlGn_r")
    colors = [cmap(value) for value in np.linspace(0.12, 0.88, len(df))]
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 4.8))

    panels = [
        ("metric_mean", "Choice Logprob Mean", "higher is better"),
        ("metric_std", "Choice Logprob Std", "lower is better"),
        ("active_compute_hours_mean", "Active Compute Time (hours)", "lower is better"),
    ]

    for ax, (column, title, note) in zip(axes, panels, strict=True):
        if column == "active_compute_hours_mean":
            ax.bar(
                x,
                df[column],
                yerr=df["active_compute_hours_std"],
                capsize=6,
                color=colors,
                edgecolor="black",
                linewidth=0.8,
            )
        else:
            ax.bar(x, df[column], color=colors, edgecolor="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(df["setting"], rotation=18, ha="right")
        ax.set_title(title, fontsize=11, weight="bold")
        ax.text(0.98, 0.96, note, transform=ax.transAxes, ha="right", va="top", fontsize=9, color="dimgray")
        ax.grid(axis="y", alpha=0.2)
        for xi, value in zip(x, df[column], strict=True):
            if column == "active_compute_hours_mean":
                label = f"{value:.2f}h"
            elif column == "metric_std":
                label = f"{value:.4f}"
            else:
                label = f"{value:.3f}"
            ax.text(xi, value, label, ha="center", va="bottom", fontsize=9)

    fig.suptitle("MMLU Choice-Logprob and Active Runtime Across Compute Settings", fontsize=13, weight="bold")
    fig.text(
        0.5,
        0.01,
        "Active runtime sums throughput/duration across steps. Swarm cohort uses 241 completed runs.",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")


def main() -> None:
    summaries = build_setting_summaries()
    df = pd.DataFrame([s.__dict__ for s in summaries])
    df.to_csv(OUTPUT_CSV, index=False)
    plot_summaries(summaries)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
