# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "pandas"]
# ///
"""Build 300M signal-to-noise estimates for every shared numeric eval metric."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import fsspec
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
METRICS_WIDE_CSV = SCRIPT_DIR / "metric_registry" / "metrics_wide.csv"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "eval_signal_to_noise_all_metrics_300m_current.csv"
DEFAULT_SUMMARY_JSON = SCRIPT_DIR / "eval_signal_to_noise_all_metrics_300m_current_summary.json"
DEFAULT_TASK_SUMMARY_CSV = SCRIPT_DIR / "eval_signal_to_noise_all_metrics_300m_current_task_summary.csv"
DEFAULT_KEEP_DROP_CSV = SCRIPT_DIR / "eval_signal_to_noise_all_metrics_300m_current_keep_drop.csv"
DEFAULT_EXTRA_RESULTS_CSVS = (
    SCRIPT_DIR.parent / "paper_plots" / "img" / "baseline_scaling_downstream_eval_metrics_merged.csv",
    SCRIPT_DIR / "metric_registry" / "300m_gsm8k_humaneval_completion" / "300m_gsm8k_humaneval_eval_results.csv",
    SCRIPT_DIR / "metric_registry" / "300m_english_lite_completion" / "300m_english_lite_eval_results_merged.csv",
    SCRIPT_DIR
    / "metric_registry"
    / "300m_generative_smooth_proxy_completion"
    / "300m_generative_smooth_proxy_eval_results.csv",
    SCRIPT_DIR / "metric_registry" / "300m_mcq_smooth_proxy_completion" / "300m_mcq_smooth_proxy_eval_results.csv",
)
RUN00097_300M_FIXED_SUBSET_RESULTS_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_run00097_300m_6b_fixed_subset/collect_results-605e6a/results.csv"
)
RUN00097_300M_FIXED_SUBSET_CHECKPOINT_PREFIX = (
    "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_300m_6b_fixed_subset"
)

METRIC_PREFIXES = ("eval/", "lm_eval/", "teacher_forced/", "mcq_smooth/")
MMLU_SOURCE_PATH = SCRIPT_DIR.parents[2] / "evals" / "olmo_base_easy_overlap.py"
MMLU_CATEGORY_ORDER = ("stem", "humanities", "social_sciences", "other")
MMLU_CATEGORY_SMOOTH_LEAVES = (
    "choice_prob_norm",
    "choice_logprob_norm",
    "choice_logprob",
    "logprob",
    "bpb",
)
MIN_SIGNAL_ROWS = 2
MIN_NOISE_ROWS = 2
FALLBACK_KIND_PRIORITY = (
    "choice_prob_norm",
    "choice_logprob_norm",
    "bpb",
    "nll",
    "perplexity",
    "loss",
    "choice_logprob",
    "logprob",
    "acc_norm",
    "acc",
    "exact_match",
    "pass_at_1",
)
SMOOTH_PROXY_KINDS = (
    "choice_prob_norm",
    "choice_logprob_norm",
    "bpb",
    "nll",
    "perplexity",
    "loss",
    "choice_logprob",
    "logprob",
)
SMOOTHNESS_WEIGHTS = {
    "choice_prob_norm": 1.0,
    "choice_logprob_norm": 1.0,
    "bpb": 1.0,
    "nll": 1.0,
    "loss": 1.0,
    "perplexity": 0.9,
    "choice_logprob": 0.8,
    "logprob": 0.7,
}
ACCURACY_KIND_PRIORITY = ("acc", "acc_norm", "exact_match", "pass_at_1")
CORRELATION_KIND_PRIORITY = (
    "choice_prob_norm",
    "choice_logprob_norm",
    "bpb",
    "nll",
    "perplexity",
    "loss",
    "choice_logprob",
    "logprob",
)
PARENT_TASK_BY_PROXY_TASK = {
    "gsm8k_5shot_answer_hash": "gsm8k",
    "gsm8k_5shot_gold_solution": "gsm8k",
    "humaneval_10shot_canonical_solution": "humaneval",
}
MIN_KEEP_PROXY_SNR = 2.0
MIN_KEEP_PROXY_ABS_SPEARMAN = 0.40
MIN_DOWNWEIGHT_PROXY_SNR = 1.0
MIN_DOWNWEIGHT_PROXY_ABS_SPEARMAN = 0.25
LM_EVAL_ARTIFACT_MARKER = "/lm_eval_artifacts/lm_eval_harness_results"
LM_EVAL_METRIC_SUFFIX = ",none"


def _read_csv(path_or_uri: str | Path) -> pd.DataFrame:
    path_string = str(path_or_uri)
    if path_string.startswith("gs://"):
        with fsspec.open(path_string, "rt") as handle:
            return pd.read_csv(handle, low_memory=False)
    return pd.read_csv(path_or_uri, low_memory=False)


def _metric_kind(metric: str) -> str:
    lowered = metric.lower()
    if "stderr" in lowered:
        return "stderr"
    if lowered.endswith("/bpb") or lowered.endswith("_bpb") or lowered == "eval/bpb":
        return "bpb"
    if lowered.endswith("/loss") or lowered.endswith("_loss") or lowered == "eval/loss":
        return "loss"
    if lowered.endswith("/nll") or lowered.endswith("_nll"):
        return "nll"
    if lowered.endswith("/perplexity"):
        return "perplexity"
    if "choice_prob_norm" in lowered:
        return "choice_prob_norm"
    if "choice_logprob_norm" in lowered:
        return "choice_logprob_norm"
    if "choice_logprob" in lowered:
        return "choice_logprob"
    if lowered.endswith("/logprob") or lowered.endswith("_logprob"):
        return "logprob"
    if "exact_match" in lowered:
        return "exact_match"
    if "pass@1" in lowered:
        return "pass_at_1"
    if lowered.endswith("/acc_norm") or lowered.endswith("_acc_norm"):
        return "acc_norm"
    if lowered.endswith("/acc") or lowered.endswith("_acc"):
        return "acc"
    return "other"


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(column for column in frame.columns if column.startswith(METRIC_PREFIXES))


def _metric_task(metric: str) -> str:
    parts = metric.split("/")
    if len(parts) >= 3 and parts[0] == "lm_eval":
        return parts[1]
    if len(parts) >= 3 and parts[0] == "teacher_forced":
        return parts[1]
    if len(parts) >= 3 and parts[0] == "mcq_smooth":
        return parts[1]
    if len(parts) >= 2 and parts[0] == "eval":
        return "eval"
    return metric.rsplit("/", maxsplit=1)[0]


def _optimization_task(metric: str) -> str:
    return PARENT_TASK_BY_PROXY_TASK.get(_metric_task(metric), _metric_task(metric))


def _metric_leaf(metric: str) -> str:
    return metric.rsplit("/", maxsplit=1)[-1]


def _priority_index(kind: str, priority: tuple[str, ...]) -> int:
    try:
        return priority.index(kind)
    except ValueError:
        return len(priority)


def _overlay_metrics(base: pd.DataFrame, overlay: pd.DataFrame, *, key_column: str) -> pd.DataFrame:
    if overlay.empty:
        return base
    if key_column not in overlay.columns:
        raise ValueError(f"Overlay frame is missing {key_column!r}")
    overlay = _enrich_with_referenced_lm_eval_artifacts(overlay)
    overlay = overlay.drop_duplicates(subset=[key_column], keep="last").copy()
    metric_columns = _metric_columns(overlay)
    if not metric_columns:
        return base

    merged = base.merge(
        overlay[[key_column, *metric_columns]],
        on=key_column,
        how="left",
        suffixes=("", "__overlay"),
    )
    for column in metric_columns:
        overlay_column = f"{column}__overlay"
        if overlay_column not in merged.columns:
            continue
        if column in merged.columns:
            merged[column] = merged[overlay_column].where(merged[overlay_column].notna(), merged[column])
        else:
            merged[column] = merged[overlay_column]
        merged = merged.drop(columns=[overlay_column])
    return merged


def _flatten_lm_eval_payload(payload: dict[str, object]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    results = payload.get("results", {})
    if isinstance(results, dict):
        for task_key, task_metrics in results.items():
            if not isinstance(task_metrics, dict):
                continue
            for metric_key, value in task_metrics.items():
                metric_name = str(metric_key).removesuffix(LM_EVAL_METRIC_SUFFIX)
                if isinstance(value, int | float):
                    metrics[f"lm_eval/{task_key}/{metric_name}"] = float(value)
    averages = payload.get("averages", {})
    if isinstance(averages, dict):
        for metric_key, value in averages.items():
            metric_name = str(metric_key).removesuffix(LM_EVAL_METRIC_SUFFIX)
            if isinstance(value, int | float):
                metrics[f"lm_eval/averages/{metric_name}"] = float(value)
    return metrics


def _artifact_paths_from_value(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    paths: list[str] = []
    for part in str(value).split(";"):
        path = part.strip()
        if LM_EVAL_ARTIFACT_MARKER in path and path.endswith(".json"):
            paths.append(path)
    return paths


def _referenced_lm_eval_artifact_paths(row: pd.Series) -> list[str]:
    path_columns = [column for column in row.index if column == "source_path" or str(column).endswith("__source_path")]
    paths: list[str] = []
    for column in path_columns:
        paths.extend(_artifact_paths_from_value(row.get(column)))
    return sorted(set(paths))


def _read_lm_eval_artifact_metrics(path: str, cache: dict[str, dict[str, float]]) -> dict[str, float]:
    if path not in cache:
        with fsspec.open(path, "rt") as handle:
            cache[path] = _flatten_lm_eval_payload(json.load(handle))
    return cache[path]


def _enrich_with_referenced_lm_eval_artifacts(frame: pd.DataFrame) -> pd.DataFrame:
    if "checkpoint_root" not in frame.columns:
        return frame
    cache: dict[str, dict[str, float]] = {}
    out = frame.copy()
    artifact_rows: dict[int, dict[str, float]] = {}
    for row_index, row in out.iterrows():
        for path in _referenced_lm_eval_artifact_paths(row):
            try:
                metrics = _read_lm_eval_artifact_metrics(path, cache)
            except (OSError, json.JSONDecodeError):
                continue
            artifact_rows.setdefault(row_index, {}).update(metrics)
    if not artifact_rows:
        return out

    artifact_frame = pd.DataFrame.from_dict(artifact_rows, orient="index").reindex(out.index)
    new_columns: dict[str, pd.Series] = {}
    for metric in artifact_frame.columns:
        values = artifact_frame[metric]
        if metric in out.columns:
            out[metric] = out[metric].where(out[metric].notna(), values)
        else:
            new_columns[metric] = values
    if new_columns:
        out = pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1)
    return out


def _ensure_fixed_seed_checkpoint_roots(frame: pd.DataFrame) -> pd.DataFrame:
    if "checkpoint_root" in frame.columns and frame["checkpoint_root"].notna().all():
        return frame
    if "wandb_run_id" not in frame.columns:
        raise ValueError("Fixed-seed noise frame has neither checkpoint_root nor wandb_run_id")
    out = frame.copy()
    out["checkpoint_root"] = out["wandb_run_id"].map(
        lambda run_id: f"{RUN00097_300M_FIXED_SUBSET_CHECKPOINT_PREFIX}/{run_id}"
    )
    return out


def _load_mmlu_subject_to_category() -> dict[str, str]:
    """Load the canonical MMLU subject grouping without importing eval configs."""
    module = ast.parse(MMLU_SOURCE_PATH.read_text())
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MMLU_SUBJECT_TO_CATEGORY":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, dict):
                        raise ValueError(f"MMLU_SUBJECT_TO_CATEGORY in {MMLU_SOURCE_PATH} is not a dict")
                    return {str(subject): str(category) for subject, category in value.items()}
    raise ValueError(f"Could not find MMLU_SUBJECT_TO_CATEGORY in {MMLU_SOURCE_PATH}")


def _add_mmlu_category_smooth_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive MMLU category smooth proxies from subject-level smooth metrics."""
    subject_to_category = _load_mmlu_subject_to_category()
    new_columns: dict[str, pd.Series] = {}
    for category in MMLU_CATEGORY_ORDER:
        subjects = [subject for subject, subject_category in subject_to_category.items() if subject_category == category]
        for leaf in MMLU_CATEGORY_SMOOTH_LEAVES:
            output_column = f"lm_eval/mmlu_{category}_5shot/{leaf}"
            if output_column in frame.columns:
                continue
            subject_columns = [f"lm_eval/mmlu_{subject}_5shot/{leaf}" for subject in subjects]
            if not all(column in frame.columns for column in subject_columns):
                continue
            values = frame[subject_columns].apply(pd.to_numeric, errors="coerce")
            new_columns[output_column] = values.mean(axis=1, skipna=False)
    if not new_columns:
        return frame
    return pd.concat([frame, pd.DataFrame(new_columns, index=frame.index)], axis=1)


def _load_signal_frame(extra_results: list[str]) -> pd.DataFrame:
    if not METRICS_WIDE_CSV.exists():
        raise FileNotFoundError(f"Missing metric registry {METRICS_WIDE_CSV}")
    frame = pd.read_csv(METRICS_WIDE_CSV, low_memory=False)
    signal = frame[frame["scale"].eq("300m_6b")].copy()
    if signal.empty:
        raise ValueError("No 300m_6b signal rows found in metric registry")
    for path in extra_results:
        signal = _overlay_metrics(signal, _read_csv(path), key_column="checkpoint_root")
    signal = _add_mmlu_category_smooth_metrics(signal)
    return signal.reset_index(drop=True)


def _load_noise_frame(extra_results: list[str]) -> pd.DataFrame:
    frame = _read_csv(RUN00097_300M_FIXED_SUBSET_RESULTS_URI)
    if "cohort" not in frame.columns:
        raise ValueError(f"Noise CSV is missing cohort column: {RUN00097_300M_FIXED_SUBSET_RESULTS_URI}")
    noise = frame[frame["cohort"].eq("seed_sweep")].copy()
    if len(noise) != 10:
        raise ValueError(f"Expected 10 fixed-seed noise rows, found {len(noise)}")
    noise = _ensure_fixed_seed_checkpoint_roots(noise)
    for path in extra_results:
        noise = _overlay_metrics(noise, _read_csv(path), key_column="checkpoint_root")
    noise = _add_mmlu_category_smooth_metrics(noise)
    return noise.reset_index(drop=True)


def _snr_row(metric: str, signal_values: pd.Series, noise_values: pd.Series) -> dict[str, float | int | str] | None:
    signal = pd.to_numeric(signal_values, errors="coerce").dropna()
    noise = pd.to_numeric(noise_values, errors="coerce").dropna()
    if len(signal) < MIN_SIGNAL_ROWS or len(noise) < MIN_NOISE_ROWS:
        return None
    signal_scale = float(signal.std(ddof=1))
    noise_scale = float(noise.std(ddof=1))
    if noise_scale == 0 or pd.isna(noise_scale):
        return None
    return {
        "metric": metric,
        "task": _metric_task(metric),
        "metric_leaf": _metric_leaf(metric),
        "primary_metric_kind": _metric_kind(metric),
        "signal_n": len(signal),
        "noise_n": len(noise),
        "signal_mean": float(signal.mean()),
        "signal_min": float(signal.min()),
        "signal_max": float(signal.max()),
        "noise_mean": float(noise.mean()),
        "noise_min": float(noise.min()),
        "noise_max": float(noise.max()),
        "signal_scale": signal_scale,
        "noise_scale": noise_scale,
        "signal_range": float(signal.max() - signal.min()),
        "noise_range": float(noise.max() - noise.min()),
        "signal_to_noise": signal_scale / noise_scale,
    }


def build_signal_to_noise_table(extra_results: list[str]) -> pd.DataFrame:
    """Return SNR rows for every numeric metric shared by signal and noise frames."""
    signal = _load_signal_frame(extra_results)
    noise = _load_noise_frame(extra_results)
    shared_metrics = sorted(set(_metric_columns(signal)) & set(_metric_columns(noise)))
    rows = [row for metric in shared_metrics if (row := _snr_row(metric, signal[metric], noise[metric])) is not None]
    if not rows:
        raise ValueError("No shared numeric eval metrics had enough signal/noise rows")
    frame = pd.DataFrame(rows).sort_values("signal_to_noise", ascending=False).reset_index(drop=True)
    frame.attrs["signal_rows"] = len(signal)
    frame.attrs["noise_rows"] = len(noise)
    frame.attrs["shared_metric_count"] = len(shared_metrics)
    return frame


def _sorted_task_metrics(task_frame: pd.DataFrame, priority: tuple[str, ...]) -> pd.DataFrame:
    out = task_frame.copy()
    out["_priority"] = out["primary_metric_kind"].map(lambda kind: _priority_index(str(kind), priority))
    return out.sort_values(["_priority", "signal_to_noise"], ascending=[True, False]).drop(columns=["_priority"])


def _select_task_metric(task_frame: pd.DataFrame, priority: tuple[str, ...]) -> pd.Series | None:
    candidates = _sorted_task_metrics(task_frame, priority)
    if candidates.empty:
        return None
    return candidates.iloc[0]


def _snr_shrink(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or numeric < 0:
        return None
    return float(numeric / (numeric + 1.0))


def _safe_corr(left: pd.Series, right: pd.Series, *, method: str) -> float | None:
    joined = pd.concat(
        [
            pd.to_numeric(left, errors="coerce").rename("left"),
            pd.to_numeric(right, errors="coerce").rename("right"),
        ],
        axis=1,
    ).dropna()
    if len(joined) < 3:
        return None
    if joined["left"].nunique() < 2 or joined["right"].nunique() < 2:
        return None
    value = joined["left"].corr(joined["right"], method=method)
    if pd.isna(value):
        return None
    return float(value)


def _proxy_score(
    *,
    proxy_snr: float,
    accuracy_snr: float | None,
    spearman: float,
    proxy_kind: str,
) -> float:
    proxy_shrink = _snr_shrink(proxy_snr)
    accuracy_shrink = _snr_shrink(accuracy_snr)
    if proxy_shrink is None:
        return 0.0
    if accuracy_shrink is None:
        accuracy_shrink = 1.0
    return abs(spearman) * proxy_shrink * accuracy_shrink * SMOOTHNESS_WEIGHTS.get(proxy_kind, 0.0)


def _smooth_proxy_candidates(task_frame: pd.DataFrame) -> pd.DataFrame:
    return task_frame[task_frame["primary_metric_kind"].isin(SMOOTH_PROXY_KINDS)].copy()


def _select_smooth_proxy(
    *,
    signal: pd.DataFrame,
    task_frame: pd.DataFrame,
    accuracy_metric: str | None,
    accuracy_snr: float | None,
) -> pd.Series | None:
    if accuracy_metric is None or accuracy_metric not in signal.columns:
        return None
    rows: list[dict[str, object]] = []
    for _, candidate in _smooth_proxy_candidates(task_frame).iterrows():
        proxy_metric = str(candidate["metric"])
        if proxy_metric == accuracy_metric or proxy_metric not in signal.columns:
            continue
        spearman = _safe_corr(signal[accuracy_metric], signal[proxy_metric], method="spearman")
        pearson = _safe_corr(signal[accuracy_metric], signal[proxy_metric], method="pearson")
        if spearman is None:
            continue
        proxy_kind = str(candidate["primary_metric_kind"])
        score = _proxy_score(
            proxy_snr=float(candidate["signal_to_noise"]),
            accuracy_snr=accuracy_snr,
            spearman=spearman,
            proxy_kind=proxy_kind,
        )
        row = candidate.to_dict()
        row["accuracy_pearson"] = pearson
        row["accuracy_spearman"] = spearman
        row["accuracy_abs_spearman"] = abs(spearman)
        row["proxy_score"] = score
        row["proxy_direction"] = "maximize" if spearman >= 0 else "minimize"
        row["smoothness_weight"] = SMOOTHNESS_WEIGHTS.get(proxy_kind, 0.0)
        rows.append(row)
    if not rows:
        return None
    candidates = pd.DataFrame(rows)
    candidates["_kind_priority"] = candidates["primary_metric_kind"].map(
        lambda kind: _priority_index(str(kind), SMOOTH_PROXY_KINDS)
    )
    candidates = candidates.sort_values(
        ["proxy_score", "accuracy_abs_spearman", "signal_to_noise", "_kind_priority"],
        ascending=[False, False, False, True],
    )
    return candidates.iloc[0].drop(labels=["_kind_priority"])


def _correlations_for_task(
    signal: pd.DataFrame,
    task_metrics: list[str],
    accuracy_metric: str | None,
) -> dict[str, float | str]:
    if accuracy_metric is None or accuracy_metric not in signal.columns:
        return {}
    rows: dict[str, float | str] = {}
    by_kind: dict[str, tuple[float, dict[str, float | str]]] = {}
    for metric in task_metrics:
        if metric not in signal.columns:
            continue
        kind = _metric_kind(metric)
        if kind not in CORRELATION_KIND_PRIORITY:
            continue
        pearson = _safe_corr(signal[accuracy_metric], signal[metric], method="pearson")
        spearman = _safe_corr(signal[accuracy_metric], signal[metric], method="spearman")
        if pearson is None and spearman is None:
            continue
        metric_rows: dict[str, float | str] = {f"acc_vs_{kind}_metric": metric}
        if pearson is not None:
            metric_rows[f"acc_vs_{kind}_pearson"] = pearson
        if spearman is not None:
            metric_rows[f"acc_vs_{kind}_spearman"] = spearman
        rank_value = abs(spearman) if spearman is not None else 0.0
        if kind not in by_kind or rank_value > by_kind[kind][0]:
            by_kind[kind] = (rank_value, metric_rows)
    for _, metric_rows in by_kind.values():
        rows.update(metric_rows)
    return rows


def build_task_summary_table(extra_results: list[str], snr_frame: pd.DataFrame) -> pd.DataFrame:
    """Return one task-level summary row with a smooth proxy when available."""
    signal = _load_signal_frame(extra_results)
    task_snr = snr_frame.copy()
    task_snr["optimization_task"] = task_snr["metric"].map(_optimization_task)
    task_snr["source_task"] = task_snr["metric"].map(_metric_task)
    rows: list[dict[str, float | int | str | None]] = []
    for task, task_frame in task_snr.groupby("optimization_task", sort=True):
        fallback = _select_task_metric(task_frame, FALLBACK_KIND_PRIORITY)
        accuracy = _select_task_metric(
            task_frame[task_frame["primary_metric_kind"].isin(ACCURACY_KIND_PRIORITY)],
            ACCURACY_KIND_PRIORITY,
        )
        accuracy_metric = None if accuracy is None else str(accuracy["metric"])
        accuracy_snr = None if accuracy is None else float(accuracy["signal_to_noise"])
        proxy = _select_smooth_proxy(
            signal=signal,
            task_frame=task_frame,
            accuracy_metric=accuracy_metric,
            accuracy_snr=accuracy_snr,
        )
        source_tasks = sorted(set(str(value) for value in task_frame["source_task"]))
        available_metric_kinds = sorted(set(str(value) for value in task_frame["primary_metric_kind"]))
        row: dict[str, float | int | str | None] = {
            "task": task,
            "source_tasks": ";".join(source_tasks),
            "metric_count": len(task_frame),
            "accuracy_metric_count": int(task_frame["primary_metric_kind"].isin(ACCURACY_KIND_PRIORITY).sum()),
            "smooth_proxy_metric_count": int(task_frame["primary_metric_kind"].isin(SMOOTH_PROXY_KINDS).sum()),
            "available_metric_kinds": ";".join(available_metric_kinds),
        }
        if proxy is not None:
            row.update(
                {
                    "selected_proxy_metric": proxy["metric"],
                    "selected_proxy_kind": proxy["primary_metric_kind"],
                    "selected_proxy_score": float(proxy["proxy_score"]),
                    "selected_proxy_accuracy_pearson": proxy.get("accuracy_pearson"),
                    "selected_proxy_accuracy_spearman": float(proxy["accuracy_spearman"]),
                    "selected_proxy_accuracy_abs_spearman": float(proxy["accuracy_abs_spearman"]),
                    "selected_proxy_direction": proxy["proxy_direction"],
                    "selected_proxy_smoothness_weight": float(proxy["smoothness_weight"]),
                    "selected_proxy_snr": float(proxy["signal_to_noise"]),
                    "selected_proxy_signal_mean": float(proxy["signal_mean"]),
                    "selected_proxy_signal_min": float(proxy["signal_min"]),
                    "selected_proxy_signal_max": float(proxy["signal_max"]),
                    "selected_proxy_noise_scale": float(proxy["noise_scale"]),
                    "selected_proxy_signal_range": float(proxy["signal_range"]),
                }
            )
        if fallback is not None:
            row.update(
                {
                    "fallback_metric": fallback["metric"],
                    "fallback_kind": fallback["primary_metric_kind"],
                    "fallback_snr": float(fallback["signal_to_noise"]),
                }
            )
        if accuracy is not None:
            row.update(
                {
                    "accuracy_metric": accuracy["metric"],
                    "accuracy_kind": accuracy["primary_metric_kind"],
                    "accuracy_snr": float(accuracy["signal_to_noise"]),
                    "accuracy_signal_mean": float(accuracy["signal_mean"]),
                    "accuracy_signal_min": float(accuracy["signal_min"]),
                    "accuracy_signal_max": float(accuracy["signal_max"]),
                    "accuracy_noise_scale": float(accuracy["noise_scale"]),
                    "accuracy_signal_range": float(accuracy["signal_range"]),
                }
            )
        row.update(_correlations_for_task(signal, list(task_frame["metric"]), accuracy_metric))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("selected_proxy_score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def _keep_drop_recommendation(row: pd.Series) -> tuple[str, str]:
    selected_snr = pd.to_numeric(pd.Series([row.get("selected_proxy_snr")]), errors="coerce").iloc[0]
    selected_abs_spearman = pd.to_numeric(
        pd.Series([row.get("selected_proxy_accuracy_abs_spearman")]), errors="coerce"
    ).iloc[0]
    signal_range = pd.to_numeric(pd.Series([row.get("selected_proxy_signal_range")]), errors="coerce").iloc[0]
    noise_scale = pd.to_numeric(pd.Series([row.get("selected_proxy_noise_scale")]), errors="coerce").iloc[0]
    accuracy_max = pd.to_numeric(pd.Series([row.get("accuracy_signal_max")]), errors="coerce").iloc[0]
    accuracy_range = pd.to_numeric(pd.Series([row.get("accuracy_signal_range")]), errors="coerce").iloc[0]

    if pd.isna(pd.to_numeric(pd.Series([row.get("accuracy_snr")]), errors="coerce").iloc[0]):
        return "report_only", "missing_accuracy_target"
    if pd.isna(selected_snr):
        return "report_only", "missing_smooth_proxy"
    if pd.isna(selected_abs_spearman):
        return "report_only", "missing_proxy_accuracy_correlation"
    noise_dominated = pd.notna(signal_range) and pd.notna(noise_scale) and signal_range <= 2.0 * noise_scale
    floor_like = pd.notna(accuracy_max) and pd.notna(accuracy_range) and accuracy_max <= 0.05 and accuracy_range <= 0.02
    if floor_like:
        return "report_only", "accuracy_at_floor"
    if noise_dominated:
        return "report_only", "range_noise_dominated"
    if selected_snr >= MIN_KEEP_PROXY_SNR and selected_abs_spearman >= MIN_KEEP_PROXY_ABS_SPEARMAN:
        return "keep", "smooth_proxy_snr_and_corr_pass"
    if selected_snr >= MIN_DOWNWEIGHT_PROXY_SNR and selected_abs_spearman >= MIN_DOWNWEIGHT_PROXY_ABS_SPEARMAN:
        return "downweight", "smooth_proxy_snr_or_corr_marginal"
    if selected_snr < MIN_DOWNWEIGHT_PROXY_SNR:
        return "report_only", "smooth_proxy_snr_lt_1"
    return "report_only", "smooth_proxy_corr_lt_0p25"


def build_keep_drop_table(task_summary: pd.DataFrame) -> pd.DataFrame:
    """Return the default task basket recommendation table."""
    if task_summary.empty:
        return task_summary.copy()
    rows: list[dict[str, object]] = []
    for _, row in task_summary.iterrows():
        recommendation, reason = _keep_drop_recommendation(row)
        record = row.to_dict()
        record["optimization_recommendation"] = recommendation
        record["recommendation_reason"] = reason
        record["default_weight"] = {"keep": 1.0, "downweight": 0.5, "report_only": 0.0}[recommendation]
        rows.append(record)
    out = pd.DataFrame(rows)
    recommendation_order = {"keep": 0, "downweight": 1, "report_only": 2}
    out["_recommendation_order"] = out["optimization_recommendation"].map(recommendation_order)
    return out.sort_values(
        ["_recommendation_order", "selected_proxy_score"],
        ascending=[True, False],
        na_position="last",
    ).drop(columns=["_recommendation_order"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extra-results-csv",
        action="append",
        default=[],
        help=(
            "Additional collected eval CSV to overlay by checkpoint_root. Can be repeated. "
            "Default local overlay CSVs are included unless --no-default-extra-results is set."
        ),
    )
    parser.add_argument("--no-default-extra-results", action="store_true")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--task-summary-csv", default=str(DEFAULT_TASK_SUMMARY_CSV))
    parser.add_argument("--keep-drop-csv", default=str(DEFAULT_KEEP_DROP_CSV))
    return parser.parse_args()


def _default_extra_results_csvs() -> list[str]:
    return [str(path) for path in DEFAULT_EXTRA_RESULTS_CSVS if path.exists()]


def _extra_results_csvs(args: argparse.Namespace) -> list[str]:
    paths: list[str] = [] if args.no_default_extra_results else _default_extra_results_csvs()
    paths.extend(args.extra_results_csv)
    deduped: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def main() -> None:
    args = _parse_args()
    extra_results_csv = _extra_results_csvs(args)
    frame = build_signal_to_noise_table(extra_results_csv)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)
    task_summary_csv = Path(args.task_summary_csv)
    keep_drop_csv = Path(args.keep_drop_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    task_summary = build_task_summary_table(extra_results_csv, frame)
    keep_drop = build_keep_drop_table(task_summary)
    task_summary.to_csv(task_summary_csv, index=False)
    keep_drop.to_csv(keep_drop_csv, index=False)
    summary = {
        "extra_results_csv": extra_results_csv,
        "keep_drop_csv": str(keep_drop_csv),
        "n_metrics": len(frame),
        "noise_rows": int(frame.attrs["noise_rows"]),
        "shared_metric_count": int(frame.attrs["shared_metric_count"]),
        "signal_rows": int(frame.attrs["signal_rows"]),
        "task_summary_csv": str(task_summary_csv),
        "top_metrics": (
            frame.head(20)[["metric", "task", "primary_metric_kind", "signal_n", "noise_n", "signal_to_noise"]].to_dict(
                orient="records"
            )
        ),
        "task_recommendation_counts": (
            keep_drop["optimization_recommendation"].value_counts().to_dict() if not keep_drop.empty else {}
        ),
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nWrote {output_csv}")
    print(f"Wrote {task_summary_csv}")
    print(f"Wrote {keep_drop_csv}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
