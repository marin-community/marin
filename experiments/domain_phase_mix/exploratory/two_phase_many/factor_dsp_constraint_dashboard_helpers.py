# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Data helpers for the factor-DSP constraint dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

PROPORTIONAL_RUN_NAME = "baseline_proportional"
PHASE_PREFIXES = ("phase_0_", "phase_1_")
CACHE_SOURCE_INCLUDE_DEFAULT = frozenset({"sobol_logit_trust", "canonical_dsp_path", "dsp_endpoint_path"})
TASK_SLIDER_MIN = -10.0
TASK_SLIDER_MAX = 10.0
TASK_SLIDER_STEP = 0.5


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with low-memory disabled for wide metric matrices."""
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet table, failing clearly if the cache is absent."""
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def centered_task_slider_steps(
    *,
    min_value: float = TASK_SLIDER_MIN,
    max_value: float = TASK_SLIDER_MAX,
    step: float = TASK_SLIDER_STEP,
) -> list[float]:
    """Return symmetric task-slider tick values centered at zero."""
    if min_value >= 0.0 or max_value <= 0.0:
        raise ValueError("task slider range must straddle zero")
    if not np.isclose(abs(min_value), abs(max_value)):
        raise ValueError("task slider range must be symmetric around zero")
    if step <= 0.0:
        raise ValueError("task slider step must be positive")
    count = round((max_value - min_value) / step)
    steps = [round(min_value + idx * step, 10) for idx in range(count + 1)]
    if not np.isclose(steps[-1], max_value):
        raise ValueError("task slider step does not land on max_value")
    if 0.0 not in steps:
        raise ValueError("task slider steps must include zero")
    return steps


def active_task_thresholds_from_controls(
    *,
    slider_values: dict[str, float],
    lock_values: dict[str, bool],
    activation_epsilon: float = 1e-9,
) -> dict[str, float]:
    """Convert slider and lock controls into active per-task constraints."""
    active: dict[str, float] = {}
    for task, threshold_value in slider_values.items():
        is_locked = bool(lock_values.get(task, False))
        threshold = float(threshold_value)
        is_moved = abs(threshold) > activation_epsilon
        if not is_locked and not is_moved:
            continue
        if np.isfinite(threshold):
            active[task] = threshold
    return active


def selected_candidate_for_dashboard(
    *,
    recommended_candidate: str,
    manual_candidate: str,
    follow_recommendation: bool,
) -> str:
    """Select the dashboard candidate according to recommendation mode."""
    return recommended_candidate if follow_recommendation else manual_candidate


def phase_weight_columns(frame: pd.DataFrame) -> list[str]:
    """Return phase-weight columns in stable phase/domain order."""
    return sorted(col for col in frame.columns if col.startswith(PHASE_PREFIXES))


def phase_domain_names(frame: pd.DataFrame) -> list[str]:
    """Return domains with both phase-weight columns present."""
    phase0 = {col.removeprefix("phase_0_") for col in frame.columns if col.startswith("phase_0_")}
    phase1 = {col.removeprefix("phase_1_") for col in frame.columns if col.startswith("phase_1_")}
    domains = sorted(phase0.intersection(phase1))
    if not domains:
        raise ValueError("no complete two-phase domain weight columns found")
    return domains


def phase_weight_feature_matrix(frame: pd.DataFrame, domains: list[str]) -> np.ndarray:
    """Return a two-phase weight feature matrix in stable domain order."""
    columns = [f"{phase}_{domain}" for phase in ("phase_0", "phase_1") for domain in domains]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing phase-weight columns: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return frame.loc[:, columns].to_numpy(dtype=np.float64)


def phase_weight_long_from_signal(signal_frame: pd.DataFrame, run_name: str) -> pd.DataFrame:
    """Extract two-phase domain weights for one observed signal row."""
    rows = signal_frame.loc[signal_frame["run_name"].eq(run_name)]
    if rows.empty:
        raise ValueError(f"run_name not found in signal frame: {run_name}")
    if len(rows) > 1:
        raise ValueError(f"run_name is not unique in signal frame: {run_name}")
    return phase_weight_long_from_wide(rows, candidate_column="run_name")


def phase_weight_long_from_wide(frame: pd.DataFrame, *, candidate_column: str) -> pd.DataFrame:
    """Convert wide phase-weight columns to candidate/domain/phase/weight rows."""
    phase_cols = phase_weight_columns(frame)
    if candidate_column not in frame.columns:
        raise ValueError(f"candidate column missing: {candidate_column}")
    if not phase_cols:
        raise ValueError("no phase-weight columns found")

    records: list[dict[str, object]] = []
    for _, row in frame.loc[:, [candidate_column, *phase_cols]].iterrows():
        candidate = str(row[candidate_column])
        for column in phase_cols:
            if column.startswith("phase_0_"):
                phase = "phase_0"
                domain = column.removeprefix("phase_0_")
            else:
                phase = "phase_1"
                domain = column.removeprefix("phase_1_")
            records.append(
                {
                    "candidate": candidate,
                    "domain": domain,
                    "phase": phase,
                    "weight": float(row[column]),
                }
            )
    return normalize_weight_table(pd.DataFrame.from_records(records))


def normalize_candidate_cache_summary(
    frame: pd.DataFrame,
    *,
    include_sources: set[str] | frozenset[str] = CACHE_SOURCE_INCLUDE_DEFAULT,
) -> pd.DataFrame:
    """Normalize precomputed candidate-cache rows to dashboard summary columns."""
    if frame.empty:
        return pd.DataFrame()
    required = {
        "candidate_id",
        "candidate_source",
        "predicted_y_factor",
        "predicted_y_factor_gain_vs_proportional",
        "nearest_observed_tv",
        "max_phase_weight",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"candidate cache summary missing columns: {sorted(missing)}")

    out = frame.loc[frame["candidate_source"].astype(str).isin(include_sources)].copy()
    if out.empty:
        return pd.DataFrame()
    out["candidate"] = out["candidate_id"].astype(str)
    out["source"] = out["candidate_source"].astype(str)
    out["score_kind"] = "dsp_prediction"
    out["target_score"] = pd.to_numeric(out["predicted_y_factor"], errors="raise")
    out["target_gain"] = pd.to_numeric(out["predicted_y_factor_gain_vs_proportional"], errors="raise")
    if "predicted_y_factor_gain_lcb" in out.columns:
        out["target_gain_lcb"] = pd.to_numeric(out["predicted_y_factor_gain_lcb"], errors="raise")
    else:
        out["target_gain_lcb"] = np.nan
    out["has_task_deltas"] = False
    if "passes_basic_dashboard_gate" in out.columns:
        out["deployability_flag"] = out["passes_basic_dashboard_gate"].astype(bool) & out["target_gain"].ge(0.0)
    else:
        out["deployability_flag"] = (
            out["target_gain"].ge(0.0) & out["nearest_observed_tv"].le(0.45) & out["max_phase_weight"].le(0.50)
        )
    return out


def load_dashboard_candidate_cache(
    *,
    candidate_summary_path: Path,
    endpoint_path_summary_path: Path,
) -> pd.DataFrame:
    """Load safe-library and endpoint-path candidate summaries for the dashboard."""
    tables = []
    if candidate_summary_path.exists():
        tables.append(
            normalize_candidate_cache_summary(
                load_parquet(candidate_summary_path),
                include_sources=frozenset({"sobol_logit_trust", "canonical_dsp_path"}),
            )
        )
    if endpoint_path_summary_path.exists():
        tables.append(
            normalize_candidate_cache_summary(
                load_parquet(endpoint_path_summary_path),
                include_sources=frozenset({"dsp_endpoint_path"}),
            )
        )
    non_empty = [table for table in tables if table is not None and not table.empty]
    if not non_empty:
        return pd.DataFrame()
    combined = pd.concat(non_empty, ignore_index=True, sort=False)
    if combined["candidate"].duplicated().any():
        duplicates = combined.loc[combined["candidate"].duplicated(), "candidate"].unique().tolist()
        raise ValueError(f"duplicate cached candidate names: {duplicates[:10]}")
    return combined


def load_selected_candidate_weights(
    candidate: str,
    *,
    eager_weights: pd.DataFrame,
    parquet_weight_paths: list[Path],
) -> pd.DataFrame:
    """Load one candidate's phase weights from eager rows or cached wide Parquet."""
    eager = normalize_weight_table(eager_weights)
    eager_match = eager.loc[eager["candidate"].eq(candidate)].copy()
    if not eager_match.empty:
        return eager_match

    for path in parquet_weight_paths:
        if not path.exists():
            continue
        try:
            row = pd.read_parquet(path, filters=[("candidate_id", "==", candidate)])
        except TypeError:
            row = pd.read_parquet(path)
            row = row.loc[row["candidate_id"].astype(str).eq(candidate)].copy()
        if not row.empty:
            if len(row) > 1:
                raise ValueError(f"candidate is not unique in cached weights: {candidate}")
            return phase_weight_long_from_wide(row, candidate_column="candidate_id")
    raise ValueError(f"candidate weights not found: {candidate}")


def sample_frontier_for_plot(
    frame: pd.DataFrame,
    *,
    max_rows: int = 20_000,
    top_rows: int = 2_000,
    seed: int = 5416,
) -> pd.DataFrame:
    """Bound Plotly scatter size while always retaining top target-gain rows."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if top_rows < 0:
        raise ValueError("top_rows must be nonnegative")
    if len(frame) <= max_rows:
        return frame.copy()
    required = {"candidate", "target_gain"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frontier frame missing columns: {sorted(missing)}")

    top = frame.nlargest(min(top_rows, max_rows, len(frame)), "target_gain")
    remaining_n = max_rows - len(top)
    if remaining_n <= 0:
        return top.reset_index(drop=True)
    rest = frame.loc[~frame["candidate"].isin(set(top["candidate"]))].copy()
    if len(rest) > remaining_n:
        rest = rest.sample(n=remaining_n, random_state=seed)
    return pd.concat([top, rest], ignore_index=True).reset_index(drop=True)


def candidate_selector_options(
    filtered_candidates: pd.DataFrame,
    *,
    preferred: str,
    limit: int = 2_000,
) -> list[str]:
    """Return a bounded candidate selector list ranked by gain then trust."""
    if filtered_candidates.empty:
        return [preferred]
    if limit <= 0:
        raise ValueError("limit must be positive")
    required = {"candidate", "target_gain", "nearest_observed_tv"}
    missing = required.difference(filtered_candidates.columns)
    if missing:
        raise ValueError(f"filtered candidates missing columns: {sorted(missing)}")
    ranked = filtered_candidates.sort_values(["target_gain", "nearest_observed_tv"], ascending=[False, True])
    options = ranked["candidate"].astype(str).head(limit).tolist()
    if preferred in set(filtered_candidates["candidate"].astype(str)) and preferred not in options:
        options = [preferred, *options[:-1]]
    elif preferred in options:
        options = [preferred, *[option for option in options if option != preferred]]
    return options


def label_phase_weights_long(mixture_weights: pd.DataFrame) -> pd.DataFrame:
    """Convert canonical DSP labeled mixture weights to long phase rows."""
    required = {"label", "domain", "phase_0_weight", "phase_1_weight"}
    missing = required.difference(mixture_weights.columns)
    if missing:
        raise ValueError(f"mixture weight table missing columns: {sorted(missing)}")

    records: list[dict[str, object]] = []
    for row in mixture_weights.itertuples(index=False):
        records.append(
            {
                "candidate": str(row.label),
                "domain": str(row.domain),
                "phase": "phase_0",
                "weight": float(row.phase_0_weight),
            }
        )
        records.append(
            {
                "candidate": str(row.label),
                "domain": str(row.domain),
                "phase": "phase_1",
                "weight": float(row.phase_1_weight),
            }
        )
    return normalize_weight_table(pd.DataFrame.from_records(records))


def epoch_scale_table_from_mixture_weights(
    mixture_weights: pd.DataFrame,
    *,
    baseline_label: str = "proportional",
) -> pd.DataFrame:
    """Recover materialized-epoch scales from canonical DSP mixture metadata."""
    required = {"label", "domain", "phase_0_weight", "phase_1_weight", "phase_0_epochs", "phase_1_epochs"}
    missing = required.difference(mixture_weights.columns)
    if missing:
        raise ValueError(f"mixture weight table missing epoch columns: {sorted(missing)}")
    rows = mixture_weights.loc[mixture_weights["label"].astype(str).eq(baseline_label)].copy()
    if rows.empty:
        raise ValueError(f"baseline label not found in mixture weights: {baseline_label}")

    records: list[dict[str, object]] = []
    for row in rows.itertuples(index=False):
        for phase in ("phase_0", "phase_1"):
            weight = float(getattr(row, f"{phase}_weight"))
            epochs = float(getattr(row, f"{phase}_epochs"))
            if not np.isfinite(weight) or weight <= 0.0:
                raise ValueError(f"nonpositive baseline weight for {row.domain} {phase}")
            records.append(
                {
                    "domain": str(row.domain),
                    "phase": phase,
                    "epoch_per_unit_weight": epochs / weight,
                    "proportional_epochs": epochs,
                }
            )
    return pd.DataFrame.from_records(records)


def add_materialized_epochs(weights: pd.DataFrame, epoch_scales: pd.DataFrame) -> pd.DataFrame:
    """Attach materialized epoch diagnostics to long phase-weight rows."""
    weight_table = normalize_weight_table(weights)
    extra_columns = [column for column in weights.columns if column not in {"candidate", "domain", "phase", "weight"}]
    if extra_columns:
        extras = weights.loc[:, ["candidate", "domain", "phase", *extra_columns]].copy()
        extras["candidate"] = extras["candidate"].astype(str)
        extras["domain"] = extras["domain"].astype(str)
        extras["phase"] = extras["phase"].astype(str)
        extras = extras.drop_duplicates(["candidate", "domain", "phase"])
        weight_table = weight_table.merge(extras, on=["candidate", "domain", "phase"], how="left")
    required = {"domain", "phase", "epoch_per_unit_weight", "proportional_epochs"}
    missing = required.difference(epoch_scales.columns)
    if missing:
        raise ValueError(f"epoch scale table missing columns: {sorted(missing)}")
    merged = weight_table.merge(
        epoch_scales.loc[:, ["domain", "phase", "epoch_per_unit_weight", "proportional_epochs"]],
        on=["domain", "phase"],
        how="left",
    )
    if merged["epoch_per_unit_weight"].isna().any():
        missing_rows = merged.loc[merged["epoch_per_unit_weight"].isna(), ["domain", "phase"]].head().to_dict("records")
        raise ValueError(f"missing epoch scales for rows: {missing_rows}")
    merged["materialized_epochs"] = merged["weight"] * merged["epoch_per_unit_weight"]
    merged["epoch_delta_vs_proportional"] = merged["materialized_epochs"] - merged["proportional_epochs"]
    return merged


def normalize_weight_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize long phase-weight rows."""
    if frame.empty:
        return pd.DataFrame(columns=["candidate", "domain", "phase", "weight"])
    required = {"candidate", "domain", "phase", "weight"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"weight table missing columns: {sorted(missing)}")

    out = frame.loc[:, ["candidate", "domain", "phase", "weight"]].copy()
    out["candidate"] = out["candidate"].astype(str)
    out["domain"] = out["domain"].astype(str)
    out["phase"] = out["phase"].astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="raise")

    invalid_phase = sorted(set(out["phase"]).difference({"phase_0", "phase_1"}))
    if invalid_phase:
        raise ValueError(f"invalid phases: {invalid_phase}")
    if (out["weight"] < -1e-12).any():
        raise ValueError("negative phase weights found")

    phase_sums = out.groupby(["candidate", "phase"], observed=True)["weight"].sum()
    bad = phase_sums.loc[~np.isclose(phase_sums.to_numpy(), 1.0, atol=1e-6)]
    if not bad.empty:
        raise ValueError(f"phase weights do not sum to 1: {bad.to_dict()}")
    return out.sort_values(["candidate", "phase", "domain"]).reset_index(drop=True)


def combine_phase_weights(*tables: pd.DataFrame) -> pd.DataFrame:
    """Combine long weight tables and validate the result."""
    non_empty = [table for table in tables if table is not None and not table.empty]
    if not non_empty:
        return pd.DataFrame(columns=["candidate", "domain", "phase", "weight"])
    return normalize_weight_table(pd.concat(non_empty, ignore_index=True))


def summarize_phase_weights(weights: pd.DataFrame) -> pd.DataFrame:
    """Compute compact deployability diagnostics from long phase weights."""
    if weights.empty:
        return pd.DataFrame()
    eps = 1e-12
    grouped = weights.groupby(["candidate", "phase"], observed=True)
    summary = grouped["weight"].agg(
        phase_support_gt_1e3=lambda series: int((series > 1e-3).sum()),
        phase_max_weight="max",
        phase_entropy=lambda series: float(-(series.clip(lower=eps) * np.log(series.clip(lower=eps))).sum()),
        phase_effective_support=lambda series: float(
            np.exp(-(series.clip(lower=eps) * np.log(series.clip(lower=eps))).sum())
        ),
    )
    return summary.reset_index()


def candidate_weight_diagnostics(weights: pd.DataFrame) -> pd.DataFrame:
    """Summarize each candidate across both phases."""
    phase_summary = summarize_phase_weights(weights)
    if phase_summary.empty:
        return pd.DataFrame()
    diagnostics = phase_summary.groupby("candidate", observed=True).agg(
        max_phase_weight=("phase_max_weight", "max"),
        min_phase_support_gt_1e3=("phase_support_gt_1e3", "min"),
        mean_phase_entropy=("phase_entropy", "mean"),
        mean_phase_effective_support=("phase_effective_support", "mean"),
    )
    return diagnostics.reset_index()


def nearest_observed_tv(named_weights: pd.DataFrame, observed_weights: pd.DataFrame) -> pd.DataFrame:
    """Compute each named candidate's nearest average phase TV to observed rows."""
    named = normalize_weight_table(named_weights)
    observed = normalize_weight_table(observed_weights)
    if named.empty or observed.empty:
        return pd.DataFrame(columns=["candidate", "nearest_observed_run", "nearest_observed_tv"])

    named_candidates = sorted(named["candidate"].unique())
    observed_candidates = sorted(observed["candidate"].unique())
    records: list[dict[str, object]] = []
    for candidate in named_candidates:
        candidate_weights = named.loc[named["candidate"].eq(candidate), ["domain", "phase", "weight"]]
        best_name = ""
        best_tv = np.inf
        for observed_candidate in observed_candidates:
            observed_candidate_weights = observed.loc[
                observed["candidate"].eq(observed_candidate), ["domain", "phase", "weight"]
            ]
            merged = candidate_weights.merge(
                observed_candidate_weights,
                on=["domain", "phase"],
                how="outer",
                suffixes=("_candidate", "_observed"),
            ).fillna(0.0)
            phase_tv = (
                merged.assign(abs_delta=(merged["weight_candidate"] - merged["weight_observed"]).abs())
                .groupby("phase", observed=True)["abs_delta"]
                .sum()
                .mul(0.5)
            )
            tv = float(phase_tv.mean())
            if tv < best_tv:
                best_tv = tv
                best_name = observed_candidate
        records.append(
            {
                "candidate": candidate,
                "nearest_observed_run": best_name,
                "nearest_observed_tv": best_tv,
            }
        )
    return pd.DataFrame.from_records(records)


def oriented_task_delta_table(
    signal_frame: pd.DataFrame,
    selected_tasks: pd.DataFrame,
    proportional_run_name: str = PROPORTIONAL_RUN_NAME,
    noise_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build observed current-task deltas against proportional in standardized units."""
    required = {"task", "sign"}
    missing = required.difference(selected_tasks.columns)
    if missing:
        raise ValueError(f"selected task table missing columns: {sorted(missing)}")

    tasks = selected_tasks.loc[:, ["task", "sign"]].copy()
    tasks["task"] = tasks["task"].astype(str)
    tasks["sign"] = pd.to_numeric(tasks["sign"], errors="raise")
    missing_tasks = sorted(set(tasks["task"]).difference(signal_frame.columns))
    if missing_tasks:
        raise ValueError(f"selected task columns missing from signal frame: {missing_tasks}")

    proportional_rows = signal_frame.loc[signal_frame["run_name"].eq(proportional_run_name)]
    if proportional_rows.empty:
        raise ValueError(f"proportional run not found: {proportional_run_name}")
    if len(proportional_rows) > 1:
        raise ValueError(f"proportional run is not unique: {proportional_run_name}")
    proportional = proportional_rows.iloc[0]

    noise_frame = pd.DataFrame() if noise_frame is None else noise_frame
    signal_scales = signal_frame.loc[:, tasks["task"].tolist()].std(axis=0, ddof=1, skipna=True)
    noise_scales = pd.Series(dtype=float)
    noise_task_cols = [task for task in tasks["task"] if task in noise_frame.columns]
    if noise_task_cols:
        noise_scales = noise_frame.loc[:, noise_task_cols].std(axis=0, ddof=1, skipna=True)

    records: list[dict[str, object]] = []
    for _, row in signal_frame.iterrows():
        candidate = str(row["run_name"])
        for task, sign in tasks.itertuples(index=False):
            value = float(row[task])
            baseline = float(proportional[task])
            oriented_delta = float(sign) * (value - baseline)
            noise_scale = float(noise_scales.get(task, np.nan))
            signal_scale = float(signal_scales.get(task, np.nan))
            if np.isfinite(noise_scale) and noise_scale > 0.0:
                scale = noise_scale
                scale_source = "noise_std"
            elif np.isfinite(signal_scale) and signal_scale > 0.0:
                scale = signal_scale
                scale_source = "signal_std_fallback"
            else:
                scale = np.nan
                scale_source = "missing"
            records.append(
                {
                    "candidate": candidate,
                    "task_column": task,
                    "oriented_delta": oriented_delta,
                    "task_scale": scale,
                    "task_delta_standardized": oriented_delta / scale if np.isfinite(scale) and scale > 0.0 else np.nan,
                    "scale_source": scale_source,
                }
            )

    return pd.DataFrame.from_records(records)


def ridge_task_delta_prediction_wide(
    signal_frame: pd.DataFrame,
    selected_tasks: pd.DataFrame,
    candidate_weights_wide: pd.DataFrame,
    *,
    proportional_run_name: str = PROPORTIONAL_RUN_NAME,
    noise_frame: pd.DataFrame | None = None,
    alpha: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict standardized task deltas for candidate weights with a local ridge surrogate.

    The surrogate is deliberately lightweight and dashboard-scoped: it maps
    centered two-phase weights to observed task deltas. It is not the canonical
    DSP model and should be presented as a local task-response approximation.
    Observed candidate rows are overwritten with empirical deltas when present.
    """
    if alpha < 0.0:
        raise ValueError("alpha must be nonnegative")
    if "candidate_id" not in candidate_weights_wide.columns:
        raise ValueError("candidate_weights_wide must include candidate_id")
    domains = phase_domain_names(signal_frame)
    missing_candidate_domains = [
        column
        for phase in ("phase_0", "phase_1")
        for domain in domains
        for column in [f"{phase}_{domain}"]
        if column not in candidate_weights_wide.columns
    ]
    if missing_candidate_domains:
        raise ValueError(f"candidate weights missing phase columns: {missing_candidate_domains[:5]}")

    task_deltas = oriented_task_delta_table(
        signal_frame,
        selected_tasks,
        proportional_run_name,
        noise_frame=noise_frame,
    )
    y_table = task_deltas.pivot_table(
        index="candidate",
        columns="task_column",
        values="task_delta_standardized",
        aggfunc="mean",
        observed=True,
    )

    proportional_rows = signal_frame.loc[signal_frame["run_name"].astype(str).eq(proportional_run_name)]
    if len(proportional_rows) != 1:
        raise ValueError(f"expected exactly one proportional row, found {len(proportional_rows)}")
    baseline_features = phase_weight_feature_matrix(proportional_rows, domains)[0]
    x_train = phase_weight_feature_matrix(signal_frame, domains) - baseline_features[None, :]
    x_candidate = phase_weight_feature_matrix(candidate_weights_wide, domains) - baseline_features[None, :]

    predictions = pd.DataFrame(
        {
            "candidate": candidate_weights_wide["candidate_id"].astype(str).to_numpy(),
            "prediction_source": "local_weight_ridge",
        }
    )
    if "candidate_source" in candidate_weights_wide.columns:
        predictions["candidate_source"] = candidate_weights_wide["candidate_source"].astype(str).to_numpy()

    metrics: list[dict[str, object]] = []
    train_candidates = signal_frame["run_name"].astype(str).to_numpy()
    for task in selected_tasks["task"].astype(str):
        if task not in y_table.columns:
            predictions[task] = np.nan
            metrics.append(
                {
                    "task_column": task,
                    "prediction_source": "local_weight_ridge",
                    "train_n": 0,
                    "train_rmse": np.nan,
                    "train_pearson": np.nan,
                    "alpha": float(alpha),
                }
            )
            continue
        y = y_table.reindex(train_candidates)[task].to_numpy(dtype=np.float64)
        valid = np.isfinite(y)
        if valid.sum() < 2:
            predictions[task] = np.nan
            train_rmse = np.nan
            train_pearson = np.nan
        else:
            model = Ridge(alpha=float(alpha), fit_intercept=False)
            model.fit(x_train[valid], y[valid])
            predicted = model.predict(x_candidate)
            observed_by_candidate = pd.Series(y[valid], index=train_candidates[valid])
            observed_mask = predictions["candidate"].isin(observed_by_candidate.index)
            if observed_mask.any():
                predictions.loc[observed_mask, task] = (
                    predictions.loc[observed_mask, "candidate"].map(observed_by_candidate).to_numpy(dtype=np.float64)
                )
                missing_mask = ~observed_mask
                predictions.loc[missing_mask, task] = predicted[missing_mask]
            else:
                predictions[task] = predicted
            train_pred = model.predict(x_train[valid])
            train_rmse = float(np.sqrt(np.mean((train_pred - y[valid]) ** 2)))
            train_pearson = float(np.corrcoef(train_pred, y[valid])[0, 1]) if valid.sum() > 2 else np.nan
        metrics.append(
            {
                "task_column": task,
                "prediction_source": "local_weight_ridge",
                "train_n": int(valid.sum()),
                "train_rmse": train_rmse,
                "train_pearson": train_pearson,
                "alpha": float(alpha),
            }
        )

    return predictions, pd.DataFrame.from_records(metrics)


def candidate_names_satisfying_task_thresholds_wide(
    task_prediction_wide: pd.DataFrame,
    thresholds: dict[str, float],
    *,
    keep_missing: bool,
    all_candidates: set[str],
) -> set[str]:
    """Return candidates satisfying every active per-task threshold from a wide prediction table."""
    active = {task: threshold for task, threshold in thresholds.items() if np.isfinite(threshold)}
    if not active:
        return set(all_candidates)
    if task_prediction_wide.empty or "candidate" not in task_prediction_wide.columns:
        return set(all_candidates) if keep_missing else set()

    table = task_prediction_wide.set_index("candidate", drop=False)
    satisfying: set[str] = set()
    for candidate in all_candidates:
        if candidate not in table.index:
            if keep_missing:
                satisfying.add(candidate)
            continue
        row = table.loc[candidate]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        passes = True
        for task, threshold in active.items():
            if task not in row.index:
                passes = keep_missing
                break
            value = float(row[task]) if pd.notna(row[task]) else np.nan
            if not np.isfinite(value):
                passes = keep_missing
                if not passes:
                    break
                continue
            if value < threshold:
                passes = False
                break
        if passes:
            satisfying.add(candidate)
    return satisfying


def selected_task_prediction_long(
    task_prediction_wide: pd.DataFrame,
    candidate: str,
    *,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Return one selected candidate's predicted task deltas in long form."""
    if task_prediction_wide.empty or "candidate" not in task_prediction_wide.columns:
        return pd.DataFrame(
            columns=[
                "task_column",
                "predicted_task_delta_standardized",
                "locked",
                "target_threshold",
                "meets_target",
            ]
        )
    rows = task_prediction_wide.loc[task_prediction_wide["candidate"].astype(str).eq(candidate)]
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "task_column",
                "predicted_task_delta_standardized",
                "locked",
                "target_threshold",
                "meets_target",
            ]
        )
    row = rows.iloc[0]
    metadata_cols = {"candidate", "candidate_id", "candidate_source", "prediction_source"}
    task_columns = [column for column in task_prediction_wide.columns if column not in metadata_cols]
    records: list[dict[str, object]] = []
    for task in task_columns:
        value = float(row[task]) if pd.notna(row[task]) else np.nan
        threshold = float(thresholds[task]) if task in thresholds else np.nan
        is_locked = task in thresholds
        records.append(
            {
                "task_column": task,
                "predicted_task_delta_standardized": value,
                "locked": is_locked,
                "target_threshold": threshold,
                "meets_target": (value >= threshold) if is_locked and np.isfinite(value) else (not is_locked),
            }
        )
    return pd.DataFrame.from_records(records)


def candidate_names_satisfying_task_thresholds(
    task_deltas: pd.DataFrame,
    thresholds: dict[str, float],
    *,
    keep_missing: bool,
    all_candidates: set[str],
) -> set[str]:
    """Return candidates satisfying every active per-task threshold."""
    active = {task: threshold for task, threshold in thresholds.items() if np.isfinite(threshold)}
    if not active:
        return set(all_candidates)

    pivot = task_deltas.pivot_table(
        index="candidate",
        columns="task_column",
        values="task_delta_standardized",
        aggfunc="mean",
        observed=True,
    )
    satisfying: set[str] = set()
    for candidate in all_candidates:
        if candidate not in pivot.index:
            if keep_missing:
                satisfying.add(candidate)
            continue
        row = pivot.loc[candidate]
        passes = True
        for task, threshold in active.items():
            if task not in row.index or pd.isna(row[task]) or float(row[task]) < threshold:
                passes = False
                break
        if passes:
            satisfying.add(candidate)
    return satisfying


def filter_candidate_summary(
    candidate_summary: pd.DataFrame,
    *,
    min_target_gain: float,
    max_nearest_tv: float,
    max_phase_weight: float,
) -> pd.DataFrame:
    """Filter candidate rows using the clean-slate dashboard constraints."""
    required = {"target_gain", "nearest_observed_tv", "max_phase_weight"}
    missing = required.difference(candidate_summary.columns)
    if missing:
        raise ValueError(f"candidate summary missing columns: {sorted(missing)}")

    mask = (
        candidate_summary["target_gain"].ge(min_target_gain)
        & candidate_summary["nearest_observed_tv"].le(max_nearest_tv)
        & candidate_summary["max_phase_weight"].le(max_phase_weight)
    )
    return (
        candidate_summary.loc[mask]
        .sort_values(["target_gain", "nearest_observed_tv"], ascending=[False, True])
        .reset_index(drop=True)
    )
