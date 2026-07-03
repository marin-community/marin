# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "fsspec", "gcsfs", "numpy", "pandas", "plotly", "scipy", "scikit-learn"]
# ///
"""Evaluate Table-9 models on extra 300M diagnostic heldouts.

The original OLMoBaseEval Easy Table-9 fits used the 280-row
deletion-augmented panel: qsplit signal rows plus the 39 full-domain deletion
rows. This script evaluates those saved/refittable models on later 300M
diagnostic checkpoints that were not used in fitting:

* +5pp domain bumps,
* +5pp family bumps,
* high/low quality swaps,
* the proportional-gradient validation point,
* the 240 single-phase 300M qsplit checkpoints.

It also runs grouped 5-fold cross-validation over the expanded diagnostic panel
using the same convention as the Table-9 DSP decision scripts: keep the
nonlinear DSP geometry fixed and refit linear heads inside each fold.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_paper_faithful_olmix_300m as paper_olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_per_component_dsp_kl_sweep_300m as per_component_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_top_level_dsp_300m as top_level_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_table9_dsp_validation_mixtures_300m as materialize_table9,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "olmo_base_easy_extra_300m_heldout_eval_20260630"
DEFAULT_PROBE = (
    REFERENCE_OUTPUTS
    / "table9_extra_checkpoint_eval_20260629"
    / "table9_extra_checkpoint_eval_result_probe_refresh_latest.csv"
)
DEFAULT_AGGREGATE_DSP_MODEL = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_table9_macro_dsp_300m_20260625"
    / "dsp_effective_exposure"
    / "table9_macro_bpb"
    / "linear_reg_0.0001"
    / "model.json"
)
DEFAULT_AGGREGATE_DSP_SUMMARY = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_table9_macro_dsp_300m_20260625"
    / "effective_exposure_table9_macro_fit_summary.csv"
)
DEFAULT_PER_COMPONENT_DSP_DIR = REFERENCE_OUTPUTS / "olmo_base_easy_per_component_dsp_kl_sweep_300m_20260628"
DEFAULT_PER_COMPONENT_DSP_SUMMARY = DEFAULT_PER_COMPONENT_DSP_DIR / "per_component_dsp_kl_sweep_summary.csv"
DEFAULT_OLMIX_SUMMARY = REFERENCE_OUTPUTS / "olmo_base_easy_paper_faithful_olmix_300m_20260625" / "summary.csv"
DEFAULT_SINGLE_PHASE_WIDE = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_one_phase_parity_panel_300m_20260628"
    / "single_phase_table9_wide.csv"
)
DEFAULT_SINGLE_PHASE_PANEL = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_one_phase_parity_panel_300m_20260628"
    / "one_phase_augmented_fit_panel.csv"
)

COMPONENT_PREFIX = "olmo_base_eval/easy_bpb/"
COMPONENT_SUFFIX = "/bpb"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
LOWER_TAIL_FRAC = 0.15
N_SPLITS = 5
CV_SEED = 0
MACRO_TARGET = "table9_macro_bpb"


@dataclass(frozen=True)
class ModelPrediction:
    model_name: str
    prediction: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--aggregate-dsp-model", type=Path, default=DEFAULT_AGGREGATE_DSP_MODEL)
    parser.add_argument("--aggregate-dsp-summary", type=Path, default=DEFAULT_AGGREGATE_DSP_SUMMARY)
    parser.add_argument("--per-component-dsp-dir", type=Path, default=DEFAULT_PER_COMPONENT_DSP_DIR)
    parser.add_argument("--per-component-dsp-summary", type=Path, default=DEFAULT_PER_COMPONENT_DSP_SUMMARY)
    parser.add_argument("--olmix-summary", type=Path, default=DEFAULT_OLMIX_SUMMARY)
    parser.add_argument("--single-phase-wide", type=Path, default=DEFAULT_SINGLE_PHASE_WIDE)
    parser.add_argument("--single-phase-panel", type=Path, default=DEFAULT_SINGLE_PHASE_PANEL)
    parser.add_argument("--olmix-huber-delta", type=float, default=0.01)
    parser.add_argument("--olmix-fit-n-starts", type=int, default=12)
    return parser.parse_args()


def component_short_name(component: str) -> str:
    if component.startswith(COMPONENT_PREFIX) and component.endswith(COMPONENT_SUFFIX):
        return component.removeprefix(COMPONENT_PREFIX).removesuffix(COMPONENT_SUFFIX)
    return component


def load_json(path: str) -> dict[str, Any]:
    with fsspec.open(path, "r") as f:
        return json.load(f)


def load_extra_probe(path: Path) -> pd.DataFrame:
    probe = pd.read_csv(path)
    out = probe[probe["scale"].eq("300m_6b")].copy()
    incomplete = out[~out["has_result"].astype(bool)]
    if not incomplete.empty:
        missing = incomplete[["run_name", "eval_name", "method"]].to_dict(orient="records")
        raise ValueError(f"300M extra heldout panel is incomplete: {missing}")
    if out["run_name"].duplicated().any():
        dupes = sorted(out.loc[out["run_name"].duplicated(), "run_name"].unique().tolist())
        raise ValueError(f"Duplicate 300M run names in probe: {dupes}")
    return out.reset_index(drop=True)


def manifest_row(manifest_path: str, run_name: str, columns: list[str]) -> pd.Series:
    manifest = pd.read_csv(manifest_path, low_memory=False)
    matches = manifest[manifest["run_name"].eq(run_name)]
    if "scale" in matches.columns:
        matches = matches[matches["scale"].eq("300m_6b")]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {run_name} in {manifest_path}, found {len(matches)}")
    row = matches.iloc[0]
    missing_columns = sorted(set(columns).difference(manifest.columns))
    if missing_columns:
        raise ValueError(f"{manifest_path} is missing phase columns: {missing_columns[:10]}")
    return row


def load_heldout_panel(probe: pd.DataFrame, columns: list[str], components: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    short_names = [component_short_name(component) for component in components]
    for probe_row in probe.itertuples(index=False):
        result = load_json(str(probe_row.result_path))
        table9 = result["table9_components"]
        row = manifest_row(str(probe_row.source_manifest), str(probe_row.run_name), columns)
        values: dict[str, Any] = {
            "run_name": str(probe_row.run_name),
            "panel": str(probe_row.panel),
            "method": str(probe_row.method),
            "source_experiment": str(probe_row.source_experiment),
            "source_manifest": str(probe_row.source_manifest),
            "eval_name": str(probe_row.eval_name),
            "result_path": str(probe_row.result_path),
            "table9_macro_bpb": float(result["table9_macro_bpb"]),
            "diagnostic_group": f"extra_300m_{probe_row.method}",
            "diagnostic_family": "extra_300m_interventions",
        }
        values.update({column: float(row[column]) for column in columns})
        for component, short_name in zip(components, short_names, strict=True):
            if short_name not in table9:
                raise ValueError(f"{probe_row.run_name} missing Table-9 component {short_name}")
            values[component] = float(table9[short_name])
        component_macro = float(np.mean([values[component] for component in components]))
        if not math.isclose(component_macro, values["table9_macro_bpb"], rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(
                f"Macro mismatch for {probe_row.run_name}: result={values['table9_macro_bpb']} "
                f"components={component_macro}"
            )
        rows.append(values)
    return pd.DataFrame(rows)


def load_single_phase_heldout(
    *,
    wide_path: Path,
    panel_path: Path,
    columns: list[str],
    components: list[str],
) -> pd.DataFrame:
    wide = pd.read_csv(wide_path, low_memory=False)
    panel = pd.read_csv(panel_path, low_memory=False)
    weights = panel[panel["panel_source"].eq("single_phase_qsplit_signal")].copy()
    if len(wide) != 240 or len(weights) != 240:
        raise ValueError(f"Expected 240 single-phase rows, found wide={len(wide)} weights={len(weights)}")
    if wide["run_name"].duplicated().any() or weights["run_name"].duplicated().any():
        raise ValueError("Single-phase rows have duplicate run_name values")
    missing = sorted(set(components).difference(wide.columns))
    if missing:
        raise ValueError(f"Single-phase wide is missing Table-9 components: {missing[:12]}")
    keep = [
        "run_name",
        "source_experiment",
        "source_run_name",
        "source_panel",
        *columns,
    ]
    merged = wide[
        [
            "run_name",
            "eval_source_run_name",
            "eval_target_name",
            "wandb_run_id",
            "wandb_url",
            "table9_macro_bpb",
            *components,
        ]
    ].merge(weights[keep], on="run_name", how="inner", validate="one_to_one")
    if len(merged) != 240:
        raise ValueError(f"Expected 240 merged single-phase rows, found {len(merged)}")
    merged["panel"] = "single_phase_qsplit"
    merged["method"] = "single_phase_exposure_average"
    merged["source_manifest"] = str(panel_path)
    merged["eval_name"] = merged["eval_target_name"]
    merged["result_path"] = merged["wandb_url"]
    merged["diagnostic_group"] = "single_phase_300m_qsplit"
    merged["diagnostic_family"] = "single_phase_300m"
    component_macro = merged[components].astype(float).mean(axis=1)
    macro_delta = (component_macro - merged["table9_macro_bpb"].astype(float)).abs()
    if float(macro_delta.max()) > 1e-8:
        row = merged.iloc[int(macro_delta.to_numpy().argmax())]
        raise ValueError(
            f"Single-phase macro mismatch for {row['run_name']}: "
            f"stored={row['table9_macro_bpb']} components={component_macro.loc[row.name]}"
        )
    return merged[
        [
            "run_name",
            "panel",
            "method",
            "source_experiment",
            "source_manifest",
            "eval_name",
            "result_path",
            "diagnostic_group",
            "diagnostic_family",
            "source_run_name",
            "source_panel",
            "table9_macro_bpb",
            *columns,
            *components,
        ]
    ].copy()


def prepare_old_fit_panel(fit_panel: pd.DataFrame, columns: list[str], components: list[str]) -> pd.DataFrame:
    out = fit_panel[
        [
            "run_name",
            "source_experiment",
            "panel_source",
            MACRO_TARGET,
            *columns,
            *components,
        ]
    ].copy()
    out["panel"] = "old_fit_panel"
    out["method"] = out["panel_source"]
    out["source_manifest"] = "paper_faithful_olmix_fit_panel"
    out["eval_name"] = out["run_name"]
    out["result_path"] = ""
    out["diagnostic_group"] = "old_280_" + out["panel_source"].astype(str)
    out["diagnostic_family"] = "old_280_fit_panel"
    out["source_run_name"] = out["run_name"]
    out["source_panel"] = out["panel_source"]
    return out[
        [
            "run_name",
            "panel",
            "method",
            "source_experiment",
            "source_manifest",
            "eval_name",
            "result_path",
            "diagnostic_group",
            "diagnostic_family",
            "source_run_name",
            "source_panel",
            MACRO_TARGET,
            *columns,
            *components,
        ]
    ].copy()


def regression_summary(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rmse, mae, pearson, spearman = base.regression_metrics(actual, pred)
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "pearson": float(pearson),
        "spearman": float(spearman),
    }


def selection_summary(actual: np.ndarray, pred: np.ndarray, run_names: np.ndarray, *, prefix: str = "") -> dict[str, Any]:
    order = np.argsort(pred)
    best_actual_idx = int(np.argmin(actual))
    out: dict[str, Any] = {
        f"{prefix}best_observed_run_name": str(run_names[best_actual_idx]),
        f"{prefix}best_observed_value": float(actual[best_actual_idx]),
        f"{prefix}selected_at_1_run_name": str(run_names[order[0]]),
        f"{prefix}selected_at_1_actual": float(actual[order[0]]),
        f"{prefix}selected_at_1_predicted": float(pred[order[0]]),
        f"{prefix}optimism_at_1": float(actual[order[0]] - pred[order[0]]),
    }
    for k in (1, 3, 5):
        top = order[: min(k, len(order))]
        out[f"{prefix}regret_at_{k}"] = float(np.min(actual[top]) - actual[best_actual_idx])
        out[f"{prefix}top{k}_best_run_name"] = str(run_names[top[np.argmin(actual[top])]])
        out[f"{prefix}top{k}_best_actual"] = float(np.min(actual[top]))
    tail_count = max(5, int(np.ceil(LOWER_TAIL_FRAC * len(actual))))
    tail = order[:tail_count]
    residual = pred[tail] - actual[tail]
    out[f"{prefix}lower_tail_optimism"] = float(np.mean(np.maximum(actual[tail] - pred[tail], 0.0)))
    out[f"{prefix}low_tail_rmse"] = float(np.sqrt(np.mean(residual * residual)))
    return out


def stratified_folds(groups: pd.Series, *, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    fold_id = np.full(len(groups), -1, dtype=int)
    for _group, indices in groups.groupby(groups, sort=True).indices.items():
        shuffled = np.asarray(list(indices), dtype=int)
        rng.shuffle(shuffled)
        for position, row_idx in enumerate(shuffled):
            fold_id[row_idx] = position % n_splits
    if np.any(fold_id < 0):
        raise ValueError("Failed to assign all rows to CV folds")
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(len(groups))
    for fold in range(n_splits):
        test = np.flatnonzero(fold_id == fold)
        train = np.setdiff1d(all_idx, test, assume_unique=True)
        if len(test) == 0 or len(train) == 0:
            raise ValueError(f"Empty CV fold {fold}")
        folds.append((train, test))
    return folds


def fold_regrets(
    actual: np.ndarray,
    pred: np.ndarray,
    run_names: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    subset_idx: np.ndarray,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    subset_set = set(int(idx) for idx in subset_idx)
    for k in (1, 3, 5):
        regrets: list[float] = []
        selected_rows: list[str] = []
        for _train, test in folds:
            fold_idx = np.asarray([idx for idx in test if int(idx) in subset_set], dtype=int)
            if len(fold_idx) == 0:
                continue
            order = fold_idx[np.argsort(pred[fold_idx])]
            selected = order[: min(k, len(order))]
            regrets.append(float(np.min(actual[selected]) - np.min(actual[fold_idx])))
            selected_rows.append(str(run_names[order[0]]))
        out[f"fold_mean_regret_at_{k}"] = float(np.mean(regrets)) if regrets else float("nan")
        out[f"fold_selected_at_{k}_run_names"] = ";".join(selected_rows)
    return out


def model_metrics(
    *,
    model_name: str,
    actual: np.ndarray,
    pred: np.ndarray,
    run_names: np.ndarray,
    method: str,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "subset": method,
        "n_rows": int(len(actual)),
        **regression_summary(actual, pred),
        **selection_summary(actual, pred, run_names),
    }


def cv_model_metrics(
    *,
    model_name: str,
    actual: np.ndarray,
    pred: np.ndarray,
    run_names: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    subset_name: str,
    subset_idx: np.ndarray,
) -> dict[str, Any]:
    sub_actual = actual[subset_idx]
    sub_pred = pred[subset_idx]
    sub_names = run_names[subset_idx]
    return {
        "model_name": model_name,
        "subset": subset_name,
        "n_rows": int(len(subset_idx)),
        **regression_summary(sub_actual, sub_pred),
        **selection_summary(sub_actual, sub_pred, sub_names),
        **fold_regrets(actual, pred, run_names, folds, subset_idx),
    }


def load_aggregate_dsp_prediction(model_path: Path, weights: np.ndarray) -> np.ndarray:
    model = dsp.model_from_json(json.loads(model_path.read_text()))
    return dsp.predict(model, weights)


def load_per_component_dsp_prediction(model_dir: Path, weights: np.ndarray) -> np.ndarray:
    models = materialize_table9.load_per_component_models(model_dir)
    return np.mean(per_component_dsp.predict_component_matrix(models, weights), axis=1)


def dsp_linear_head_oof(
    *,
    model: dsp.FittedDSPModel,
    panel: pd.DataFrame,
    columns: list[str],
    domains: list[str],
    token_counts: np.ndarray,
    target_name: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    packet = top_level_dsp.build_dsp_packet(panel, columns, domains, token_counts, target_name)
    oof = np.zeros(len(panel), dtype=float)
    for train_idx, test_idx in folds:
        fold_model = dsp.fit_linear_head(
            packet.w[train_idx],
            packet.y[train_idx],
            packet,
            model.variant,
            model.params,
        )
        oof[test_idx] = dsp.predict(fold_model, packet.w[test_idx])
    return oof


def per_component_dsp_linear_head_oof(
    *,
    model_dir: Path,
    panel: pd.DataFrame,
    columns: list[str],
    domains: list[str],
    token_counts: np.ndarray,
    components: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    models = materialize_table9.load_per_component_models(model_dir)
    selected = pd.read_csv(model_dir / "selected_component_l2_summary.csv")
    selected_components = selected["component"].tolist()
    if selected_components != components:
        raise ValueError("Saved per-component DSP model order does not match Table-9 component order")
    predictions = np.zeros((len(panel), len(components)), dtype=float)
    for idx, (component, model) in enumerate(zip(components, models, strict=True), start=1):
        predictions[:, idx - 1] = dsp_linear_head_oof(
            model=model,
            panel=panel,
            columns=columns,
            domains=domains,
            token_counts=token_counts,
            target_name=component,
            folds=folds,
        )
        if idx % 10 == 0 or idx == len(components):
            print(f"  per-component DSP CV {idx}/{len(components)}", flush=True)
    return predictions.mean(axis=1)


def fit_olmix_prediction(
    *,
    fit_panel: pd.DataFrame,
    heldout: pd.DataFrame,
    columns: list[str],
    domains: list[str],
    components: list[str],
    huber_delta: float,
    fit_n_starts: int,
) -> np.ndarray:
    train_features = paper_olmix.feature_tensor(fit_panel, columns, domains, "two_phase_adapted")
    heldout_features = paper_olmix.feature_tensor(heldout, columns, domains, "two_phase_adapted")
    targets = fit_panel[components].astype(float).to_numpy()
    predictions = np.zeros((len(heldout), len(components)), dtype=float)
    for component_idx, component in enumerate(components, start=1):
        log_c, coef, _loss = base.fit_olmix_loglinear(
            train_features,
            targets[:, component_idx - 1],
            delta=float(huber_delta),
            seed=paper_olmix.FIT_SEED + component_idx,
            n_starts=int(fit_n_starts),
            verbose=False,
        )
        predictions[:, component_idx - 1] = base.predict(log_c, coef, heldout_features)
        if component_idx % 10 == 0 or component_idx == len(components):
            print(f"  OLMix component {component_idx}/{len(components)}", flush=True)
    return predictions.mean(axis=1)


def fit_olmix_oof_prediction(
    *,
    panel: pd.DataFrame,
    columns: list[str],
    domains: list[str],
    components: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    huber_delta: float,
    fit_n_starts: int,
) -> np.ndarray:
    features = paper_olmix.feature_tensor(panel, columns, domains, "two_phase_adapted")
    targets = panel[components].astype(float).to_numpy()
    predictions = np.zeros((len(panel), len(components)), dtype=float)
    print("Cross-validating paper-faithful OLMix", flush=True)
    for component_idx, component in enumerate(components, start=1):
        y = targets[:, component_idx - 1]
        for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
            log_c, coef, _loss = base.fit_olmix_loglinear(
                features[train_idx],
                y[train_idx],
                delta=float(huber_delta),
                seed=paper_olmix.FIT_SEED + component_idx * 100 + fold_idx,
                n_starts=int(fit_n_starts),
                verbose=False,
            )
            predictions[test_idx, component_idx - 1] = base.predict(log_c, coef, features[test_idx])
        if component_idx % 10 == 0 or component_idx == len(components):
            print(f"  OLMix CV component {component_idx}/{len(components)}", flush=True)
    return predictions.mean(axis=1)


def old_metric_rows(
    *,
    aggregate_summary_path: Path,
    per_component_summary_path: Path,
    olmix_summary_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    aggregate = pd.read_csv(aggregate_summary_path)
    aggregate_row = aggregate[np.isclose(aggregate["hyperparameter_value"].astype(float), 0.0001)].iloc[0]
    rows.append(
        {
            "model_name": "aggregate_effective_exposure_dsp_l2_0p0001",
            "old_eval": "280-row OOF",
            "old_rmse": float(aggregate_row["oof_rmse"]),
            "old_spearman": float(aggregate_row["oof_spearman"]),
            "old_regret_at_1": float(aggregate_row["fold_mean_regret_at_1"]),
            "old_lower_tail_optimism": float(aggregate_row["lower_tail_optimism"]),
            "old_low_tail_rmse": float(aggregate_row["low_tail_rmse"]),
        }
    )
    per_component = pd.read_csv(per_component_summary_path).iloc[0]
    rows.append(
        {
            "model_name": "per_component_effective_exposure_dsp_selected_l2",
            "old_eval": "280-row OOF",
            "old_rmse": float(per_component["macro_oof_rmse"]),
            "old_spearman": float(per_component["macro_oof_spearman"]),
            "old_regret_at_1": float(per_component["macro_fold_mean_regret_at_1"]),
            "old_lower_tail_optimism": float(per_component["macro_lower_tail_optimism"]),
            "old_low_tail_rmse": float(per_component["macro_low_tail_rmse"]),
        }
    )
    olmix = pd.read_csv(olmix_summary_path)
    olmix_row = olmix[olmix["variant"].eq("two_phase_adapted") & np.isclose(olmix["huber_delta"], 0.01)].iloc[0]
    rows.append(
        {
            "model_name": "paper_faithful_olmix_two_phase_delta_0p01",
            "old_eval": "280-row OOF",
            "old_rmse": float(olmix_row["oof_macro_rmse"]),
            "old_spearman": float(olmix_row["oof_macro_spearman"]),
            "old_regret_at_1": float(olmix_row["fold_mean_regret_at_1"]),
            "old_lower_tail_optimism": float(olmix_row["lower_tail_optimism"]),
            "old_low_tail_rmse": float(olmix_row["low_tail_rmse"]),
        }
    )
    return pd.DataFrame(rows)


def write_prediction_plot(path: Path, predictions: pd.DataFrame) -> None:
    models = list(dict.fromkeys(predictions["model_name"].tolist()))
    fig = make_subplots(rows=1, cols=len(models), subplot_titles=models, shared_xaxes=False, shared_yaxes=False)
    for idx, model_name in enumerate(models, start=1):
        frame = predictions[predictions["model_name"].eq(model_name)]
        fig.add_trace(
            go.Scatter(
                x=frame["table9_macro_bpb"],
                y=frame["predicted_table9_macro_bpb"],
                mode="markers",
                marker={"size": 9, "color": frame["method_code"], "colorscale": "RdYlGn_r", "showscale": idx == len(models)},
                text=frame["run_name"],
                customdata=np.stack([frame["method"], frame["panel"]], axis=1),
                hovertemplate=(
                    "run=%{text}<br>method=%{customdata[0]}<br>panel=%{customdata[1]}"
                    "<br>observed=%{x:.6f}<br>predicted=%{y:.6f}<extra></extra>"
                ),
                name=model_name,
                showlegend=False,
            ),
            row=1,
            col=idx,
        )
        lo = float(min(frame["table9_macro_bpb"].min(), frame["predicted_table9_macro_bpb"].min()))
        hi = float(max(frame["table9_macro_bpb"].max(), frame["predicted_table9_macro_bpb"].max()))
        fig.add_trace(
            go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"dash": "dash", "color": "#444"}, showlegend=False),
            row=1,
            col=idx,
        )
        fig.update_xaxes(title_text="observed heldout macro BPB", row=1, col=idx)
        fig.update_yaxes(title_text="predicted macro BPB", row=1, col=idx)
    fig.update_layout(
        title="Extra 300M intervention heldouts: predicted vs observed Table-9 macro BPB",
        template="plotly_white",
        width=max(1050, 430 * len(models)),
        height=520,
    )
    fig.write_html(path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_regret_plot(path: Path, summary: pd.DataFrame) -> None:
    frame = summary[summary["subset"].eq("all")].copy()
    fig = go.Figure()
    for metric in ("regret_at_1", "regret_at_3", "regret_at_5"):
        fig.add_trace(go.Bar(x=frame["model_name"], y=frame[metric], name=metric))
    fig.update_layout(
        title="Extra heldout post-selection regret within the 56-row intervention panel",
        xaxis_title="Model",
        yaxis_title="Observed BPB regret vs best heldout row",
        template="plotly_white",
        barmode="group",
        width=1050,
        height=600,
    )
    fig.write_html(path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_cv_plot(path: Path, summary: pd.DataFrame) -> None:
    frame = summary[summary["subset"].isin(["all", "old_280_fit_panel", "extra_300m_interventions", "single_phase_300m"])]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["OOF RMSE", "OOF Spearman"])
    for metric, col in [("rmse", 1), ("spearman", 2)]:
        for model_name, group in frame.groupby("model_name", sort=False):
            fig.add_trace(
                go.Bar(x=group["subset"], y=group[metric], name=model_name, showlegend=col == 1),
                row=1,
                col=col,
            )
    fig.update_layout(
        title="Expanded diagnostic panel 5-fold CV by subset",
        template="plotly_white",
        barmode="group",
        width=1250,
        height=560,
    )
    fig.write_html(path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _signal, columns, domains, _natural = base.load_raw_signal_panel()
    token_counts = base.load_domain_token_counts(domains)
    components = paper_olmix.table9_component_order()
    fit_panel, fit_metadata = paper_olmix.build_fit_panel(columns)

    probe = load_extra_probe(args.probe)
    extra_heldout = load_heldout_panel(probe, columns, components)
    single_phase_heldout = load_single_phase_heldout(
        wide_path=args.single_phase_wide,
        panel_path=args.single_phase_panel,
        columns=columns,
        components=components,
    )
    heldout = pd.concat([extra_heldout, single_phase_heldout], ignore_index=True)
    heldout.to_csv(args.output_dir / "combined_300m_table9_heldout_panel.csv", index=False)
    # Preserve the original filename for downstream notebooks from the first pass.
    heldout.to_csv(args.output_dir / "extra_300m_table9_heldout_panel.csv", index=False)

    weights = heldout[columns].astype(float).to_numpy().reshape(len(heldout), 2, len(domains))
    print("Scoring aggregate effective-exposure DSP", flush=True)
    aggregate_model = dsp.model_from_json(json.loads(args.aggregate_dsp_model.read_text()))
    aggregate_pred = dsp.predict(aggregate_model, weights)
    print("Scoring per-component effective-exposure DSP", flush=True)
    per_component_pred = load_per_component_dsp_prediction(args.per_component_dsp_dir, weights)
    print("Refitting/scoring paper-faithful two-phase OLMix", flush=True)
    olmix_pred = fit_olmix_prediction(
        fit_panel=fit_panel,
        heldout=heldout,
        columns=columns,
        domains=domains,
        components=components,
        huber_delta=float(args.olmix_huber_delta),
        fit_n_starts=int(args.olmix_fit_n_starts),
    )

    model_predictions = [
        ModelPrediction("aggregate_effective_exposure_dsp_l2_0p0001", aggregate_pred),
        ModelPrediction("per_component_effective_exposure_dsp_selected_l2", per_component_pred),
        ModelPrediction("paper_faithful_olmix_two_phase_delta_0p01", olmix_pred),
    ]
    prediction_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    actual = heldout["table9_macro_bpb"].to_numpy(dtype=float)
    run_names = heldout["run_name"].to_numpy(dtype=str)
    method_codes = {method: idx for idx, method in enumerate(sorted(heldout["method"].unique()))}
    for model in model_predictions:
        pred_frame = heldout[
            [
                "run_name",
                "panel",
                "method",
                "source_experiment",
                "source_manifest",
                "result_path",
                "table9_macro_bpb",
            ]
        ].copy()
        pred_frame["model_name"] = model.model_name
        pred_frame["predicted_table9_macro_bpb"] = model.prediction
        pred_frame["residual_pred_minus_actual"] = model.prediction - actual
        pred_frame["method_code"] = pred_frame["method"].map(method_codes).astype(int)
        prediction_frames.append(pred_frame)
        summary_rows.append(
            model_metrics(
                model_name=model.model_name,
                actual=actual,
                pred=model.prediction,
                run_names=run_names,
                method="all",
            )
        )
        for method, group in heldout.groupby("method"):
            idx = group.index.to_numpy(dtype=int)
            summary_rows.append(
                model_metrics(
                    model_name=model.model_name,
                    actual=actual[idx],
                    pred=model.prediction[idx],
                    run_names=run_names[idx],
                    method=str(method),
                )
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    diagnostic_panel = pd.concat(
        [prepare_old_fit_panel(fit_panel, columns, components), heldout],
        ignore_index=True,
    )
    diagnostic_panel.to_csv(args.output_dir / "expanded_300m_table9_diagnostic_panel.csv", index=False)
    folds = stratified_folds(diagnostic_panel["diagnostic_group"], n_splits=N_SPLITS, seed=CV_SEED)
    actual_cv = diagnostic_panel[MACRO_TARGET].to_numpy(dtype=float)
    run_names_cv = diagnostic_panel["run_name"].to_numpy(dtype=str)
    subset_indices: dict[str, np.ndarray] = {
        "all": np.arange(len(diagnostic_panel)),
        "old_280_fit_panel": np.flatnonzero(diagnostic_panel["diagnostic_family"].eq("old_280_fit_panel").to_numpy()),
        "extra_300m_interventions": np.flatnonzero(
            diagnostic_panel["diagnostic_family"].eq("extra_300m_interventions").to_numpy()
        ),
        "single_phase_300m": np.flatnonzero(diagnostic_panel["diagnostic_family"].eq("single_phase_300m").to_numpy()),
    }
    for group in sorted(diagnostic_panel["diagnostic_group"].unique()):
        subset_indices[group] = np.flatnonzero(diagnostic_panel["diagnostic_group"].eq(group).to_numpy())

    print("Cross-validating aggregate effective-exposure DSP linear head", flush=True)
    aggregate_cv = dsp_linear_head_oof(
        model=aggregate_model,
        panel=diagnostic_panel,
        columns=columns,
        domains=domains,
        token_counts=token_counts,
        target_name=MACRO_TARGET,
        folds=folds,
    )
    print("Cross-validating per-component effective-exposure DSP linear heads", flush=True)
    per_component_cv = per_component_dsp_linear_head_oof(
        model_dir=args.per_component_dsp_dir,
        panel=diagnostic_panel,
        columns=columns,
        domains=domains,
        token_counts=token_counts,
        components=components,
        folds=folds,
    )
    olmix_cv = fit_olmix_oof_prediction(
        panel=diagnostic_panel,
        columns=columns,
        domains=domains,
        components=components,
        folds=folds,
        huber_delta=float(args.olmix_huber_delta),
        fit_n_starts=int(args.olmix_fit_n_starts),
    )
    cv_predictions = pd.DataFrame(
        {
            "run_name": diagnostic_panel["run_name"],
            "diagnostic_family": diagnostic_panel["diagnostic_family"],
            "diagnostic_group": diagnostic_panel["diagnostic_group"],
            "table9_macro_bpb": actual_cv,
            "aggregate_effective_exposure_dsp_l2_0p0001": aggregate_cv,
            "per_component_effective_exposure_dsp_selected_l2": per_component_cv,
            "paper_faithful_olmix_two_phase_delta_0p01": olmix_cv,
        }
    )
    cv_predictions.to_csv(args.output_dir / "expanded_300m_table9_cv_predictions.csv", index=False)
    cv_summary_rows: list[dict[str, Any]] = []
    for model_name, pred in [
        ("aggregate_effective_exposure_dsp_l2_0p0001", aggregate_cv),
        ("per_component_effective_exposure_dsp_selected_l2", per_component_cv),
        ("paper_faithful_olmix_two_phase_delta_0p01", olmix_cv),
    ]:
        for subset_name, subset_idx in subset_indices.items():
            cv_summary_rows.append(
                cv_model_metrics(
                    model_name=model_name,
                    actual=actual_cv,
                    pred=pred,
                    run_names=run_names_cv,
                    folds=folds,
                    subset_name=subset_name,
                    subset_idx=subset_idx,
                )
            )
    cv_summary = pd.DataFrame(cv_summary_rows)
    cv_summary.to_csv(args.output_dir / "expanded_300m_table9_cv_summary.csv", index=False)

    old = old_metric_rows(
        aggregate_summary_path=args.aggregate_dsp_summary,
        per_component_summary_path=args.per_component_dsp_summary,
        olmix_summary_path=args.olmix_summary,
    )
    old_vs_heldout = old.merge(
        summary[summary["subset"].eq("all")],
        on="model_name",
        how="inner",
        validate="one_to_one",
        suffixes=("_old", "_extra"),
    )
    predictions.to_csv(args.output_dir / "extra_300m_table9_heldout_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "extra_300m_table9_heldout_model_summary.csv", index=False)
    old_vs_heldout.to_csv(args.output_dir / "old_oof_vs_extra_300m_heldout_summary.csv", index=False)
    with (args.output_dir / "metadata.json").open("w") as f:
        json.dump(
            {
                "n_extra_300m_rows": int(len(heldout)),
                "heldout_rows_by_family": heldout["diagnostic_family"].value_counts().sort_index().astype(int).to_dict(),
                "heldout_rows_by_group": heldout["diagnostic_group"].value_counts().sort_index().astype(int).to_dict(),
                "method_counts": heldout["method"].value_counts().sort_index().astype(int).to_dict(),
                "expanded_cv_rows": int(len(diagnostic_panel)),
                "expanded_cv_splits": int(N_SPLITS),
                "fit_panel_rows": int(len(fit_panel)),
                "fit_panel_metadata": fit_metadata,
                "components": components,
                "olmix_huber_delta": float(args.olmix_huber_delta),
                "olmix_fit_n_starts": int(args.olmix_fit_n_starts),
                "probe": str(args.probe),
            },
            f,
            indent=2,
            sort_keys=True,
        )
    write_prediction_plot(args.output_dir / "extra_300m_heldout_predicted_vs_observed.html", predictions)
    write_regret_plot(args.output_dir / "extra_300m_heldout_regret.html", summary)
    write_cv_plot(args.output_dir / "expanded_300m_cv_summary.html", cv_summary)

    print("Wrote", args.output_dir)
    print(summary[summary["subset"].eq("all")].to_string(index=False))
    print(cv_summary[cv_summary["subset"].isin(["all", "old_280_fit_panel", "extra_300m_interventions", "single_phase_300m"])].to_string(index=False))


if __name__ == "__main__":
    main()
