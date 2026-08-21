# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy"]
# ///
"""Benchmark paper-faithful OLMix baselines on OLMoBaseEval Easy at 300M.

This script differs from the top-level OLMoBaseEval Easy sweeps: it follows the
OLMix paper's granularity by fitting one log-linear model per BPB component and
optimizing the mean predicted component BPB. Per Table 9, subtasks are standalone
tasks, except MMLU, which is collapsed to four category averages.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    fit_olmix_reference_deletion_augmented_300m as base,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OLMO_FULL_WIDE = (
    SCRIPT_DIR
    / "reference_outputs"
    / "olmo_base_easy_300m_full_results_20260625"
    / "olmo_base_easy_300m_full_results_wide.csv"
)
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "reference_outputs" / "olmo_base_easy_paper_faithful_olmix_300m_20260625"
)

ADAPTIVE_OLMIX_RUN_NAME = base.ADAPTIVE_OLMIX_RUN_NAME
PHASE_FRACTIONS = base.PHASE_FRACTIONS
REPETITION_FACTOR = base.REPETITION_FACTOR
KL_REG = 0.05
FIT_SEED = 0
CV_SEED = 0
N_SPLITS = 5
LOWER_TAIL_FRAC = 0.15
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

MINERVA_SUBTASKS = (
    "minerva_math_algebra",
    "minerva_math_counting_and_probability",
    "minerva_math_geometry",
    "minerva_math_intermediate_algebra",
    "minerva_math_number_theory",
    "minerva_math_prealgebra",
    "minerva_math_precalculus",
)
BASIC_SKILLS_SUBTASKS = (
    "basic_skills_arithmetic",
    "basic_skills_coding",
    "basic_skills_common_knowledge",
    "basic_skills_logical_reasoning",
    "basic_skills_pattern",
    "basic_skills_string_operations",
)
MT_MBPP_SUBTASKS = (
    "mt_mbpp_bash",
    "mt_mbpp_c",
    "mt_mbpp_cpp",
    "mt_mbpp_csharp",
    "mt_mbpp_go",
    "mt_mbpp_haskell",
    "mt_mbpp_java",
    "mt_mbpp_javascript",
    "mt_mbpp_matlab",
    "mt_mbpp_php",
    "mt_mbpp_python",
    "mt_mbpp_r",
    "mt_mbpp_ruby",
    "mt_mbpp_rust",
    "mt_mbpp_scala",
    "mt_mbpp_swift",
    "mt_mbpp_typescript",
)
STANDALONE_TASKS = (
    "arc_easy",
    "arc_challenge",
    "codex_humaneval",
    "mbpp",
    "csqa",
    "hellaswag",
    "winogrande",
    "socialiqa",
    "piqa",
    "coqa",
    "drop",
    "jeopardy",
    "naturalqs",
    "squad",
    "sciq",
    "lambada",
    "medmcqa",
)

# Copied from OLMix's aggregate_mmlu implementation so this analysis remains
# standalone and does not depend on importing the external reference repo.
MMLU_CATEGORY_WEIGHTS: dict[str, dict[str, float]] = {
    "mmlu_stem": {
        "mmlu_abstract_algebra": 0.03313452617627568,
        "mmlu_astronomy": 0.05036447978793903,
        "mmlu_college_biology": 0.04771371769383698,
        "mmlu_college_chemistry": 0.03313452617627568,
        "mmlu_college_computer_science": 0.03313452617627568,
        "mmlu_college_mathematics": 0.03313452617627568,
        "mmlu_college_physics": 0.033797216699801194,
        "mmlu_computer_security": 0.03313452617627568,
        "mmlu_conceptual_physics": 0.07786613651424784,
        "mmlu_electrical_engineering": 0.04804506295559974,
        "mmlu_elementary_mathematics": 0.12524850894632206,
        "mmlu_high_school_biology": 0.10271703114645461,
        "mmlu_high_school_chemistry": 0.06726308813783963,
        "mmlu_high_school_computer_science": 0.03313452617627568,
        "mmlu_high_school_mathematics": 0.08946322067594434,
        "mmlu_high_school_physics": 0.050033134526176276,
        "mmlu_high_school_statistics": 0.07157057654075547,
        "mmlu_machine_learning": 0.03711066931742876,
    },
    "mmlu_other": {
        "mmlu_anatomy": 0.04164096236890808,
        "mmlu_business_ethics": 0.030845157310302282,
        "mmlu_clinical_knowledge": 0.08173966687230105,
        "mmlu_college_medicine": 0.05336212214682295,
        "mmlu_global_facts": 0.030845157310302282,
        "mmlu_human_aging": 0.06878470080197409,
        "mmlu_management": 0.03177051202961135,
        "mmlu_marketing": 0.07217766810610735,
        "mmlu_medical_genetics": 0.030845157310302282,
        "mmlu_miscellaneous": 0.24151758173966686,
        "mmlu_nutrition": 0.09438618136952498,
        "mmlu_professional_accounting": 0.08698334361505243,
        "mmlu_professional_medicine": 0.08389882788402221,
        "mmlu_virology": 0.05120296113510179,
    },
    "mmlu_social_sciences": {
        "mmlu_econometrics": 0.03704907377315567,
        "mmlu_high_school_geography": 0.06434839129021774,
        "mmlu_high_school_government_and_politics": 0.06272343191420214,
        "mmlu_high_school_macroeconomics": 0.12674683132921677,
        "mmlu_high_school_microeconomics": 0.07734806629834254,
        "mmlu_high_school_psychology": 0.17712057198570036,
        "mmlu_human_sexuality": 0.04257393565160871,
        "mmlu_professional_psychology": 0.19889502762430938,
        "mmlu_public_relations": 0.03574910627234319,
        "mmlu_security_studies": 0.07962300942476438,
        "mmlu_sociology": 0.0653233669158271,
        "mmlu_us_foreign_policy": 0.032499187520311994,
    },
    "mmlu_humanities": {
        "mmlu_formal_logic": 0.026780021253985122,
        "mmlu_high_school_european_history": 0.03506907545164718,
        "mmlu_high_school_us_history": 0.04335812964930925,
        "mmlu_high_school_world_history": 0.050371944739638685,
        "mmlu_international_law": 0.0257173219978746,
        "mmlu_jurisprudence": 0.022954303931987247,
        "mmlu_logical_fallacies": 0.034643995749202974,
        "mmlu_moral_disputes": 0.07353878852284804,
        "mmlu_moral_scenarios": 0.1902231668437832,
        "mmlu_philosophy": 0.06609989373007438,
        "mmlu_prehistory": 0.06886291179596174,
        "mmlu_professional_law": 0.32603613177470775,
        "mmlu_world_religions": 0.03634431455897981,
    },
}


@dataclass(frozen=True)
class ComponentFit:
    variant: str
    huber_delta: float
    component: str
    n_rows: int
    fit_log_c: float
    fit_huber_loss: float
    train_rmse: float
    train_spearman: float
    oof_rmse: float
    oof_spearman: float


@dataclass(frozen=True)
class VariantSummary:
    variant: str
    huber_delta: float
    n_components: int
    n_rows: int
    train_macro_rmse: float
    train_macro_spearman: float
    oof_macro_rmse: float
    oof_macro_spearman: float
    fold_mean_regret_at_1: float
    lower_tail_optimism: float
    low_tail_rmse: float
    cvxpy_status: str
    kl_reg: float
    repetition_factor: float
    predicted_macro_bpb: float
    regularized_objective: float
    proportional_macro_bpb: float
    proportional_predicted_macro_bpb: float
    best_observed_run_name: str
    best_observed_macro_bpb: float
    nearest_observed_run_name: str
    nearest_observed_macro_bpb: float
    nearest_observed_mean_phase_tv: float
    mean_phase_tv_to_proportional: float
    max_epoch_multiplier: float
    q95_epoch_multiplier: float
    max_simulated_epoch: float
    q95_simulated_epoch: float
    max_repetition_cap_violation: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fit-n-starts", type=int, default=12)
    parser.add_argument("--huber-deltas", type=float, nargs="+", default=[0.02])
    parser.add_argument("--variants", nargs="+", default=["single_tied", "two_phase_adapted"])
    return parser.parse_args()


def metric_key(task: str) -> str:
    return f"olmo_base_eval/easy_bpb/{task}/bpb"


def mmlu_metric_key(task: str) -> str:
    return metric_key(f"{task}_rc")


def table9_component_order() -> list[str]:
    return [
        *[metric_key(task) for task in MINERVA_SUBTASKS],
        metric_key("codex_humaneval"),
        metric_key("mbpp"),
        *[metric_key(task) for task in MT_MBPP_SUBTASKS],
        metric_key("arc_easy"),
        metric_key("arc_challenge"),
        "mmlu_stem",
        "mmlu_humanities",
        "mmlu_social_sciences",
        "mmlu_other",
        *[
            metric_key(task)
            for task in (
                "csqa",
                "hellaswag",
                "winogrande",
                "socialiqa",
                "piqa",
                "coqa",
                "drop",
                "jeopardy",
                "naturalqs",
                "squad",
                "sciq",
            )
        ],
        *[metric_key(task) for task in BASIC_SKILLS_SUBTASKS],
        metric_key("lambada"),
        metric_key("medmcqa"),
    ]


def load_olmo_wide_with_table9_components() -> pd.DataFrame:
    wide = pd.read_csv(OLMO_FULL_WIDE, low_memory=False)
    if wide["run_name"].duplicated().any():
        dupes = wide.loc[wide["run_name"].duplicated(), "run_name"].head(20).tolist()
        raise ValueError(f"Duplicate run_name rows in OLMoBaseEval full wide export: {dupes}")
    category_columns: dict[str, np.ndarray] = {}
    for category, weights in MMLU_CATEGORY_WEIGHTS.items():
        if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"MMLU category weights for {category} do not sum to 1")
        columns = [mmlu_metric_key(task) for task in weights]
        missing = sorted(set(columns).difference(wide.columns))
        if missing:
            raise ValueError(f"Missing MMLU columns for {category}: {missing}")
        weight = np.asarray([weights[task] for task in weights], dtype=float)
        category_columns[category] = wide[columns].astype(float).to_numpy() @ weight
    wide = pd.concat([wide, pd.DataFrame(category_columns, index=wide.index)], axis=1)

    components = table9_component_order()
    if len(components) != 51:
        raise ValueError(f"Expected 51 Table-9 components, found {len(components)}")
    missing = sorted(set(components).difference(wide.columns))
    if missing:
        raise ValueError(f"Missing Table-9 components: {missing}")
    null_counts = wide[components].isna().sum()
    if int(null_counts.sum()) != 0:
        raise ValueError(f"Missing Table-9 BPB values:\n{null_counts[null_counts.gt(0)]}")
    return wide[["run_name", "panel", *components]].copy()


def build_fit_panel(columns: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    signal, _, _, _ = base.load_raw_signal_panel()
    signal = signal.copy()
    signal["panel_source"] = "qsplit_signal"
    deletion_weights = base.load_deletion_weights(columns)
    deletion_weights = deletion_weights[deletion_weights["intervention_type"].eq("domain_deletion")].copy()
    deletion_weights["panel_source"] = "domain_deletion"
    fit_rows = pd.concat(
        [
            signal[["run_name", "source_experiment", "panel_source", *columns]],
            deletion_weights[["run_name", "source_experiment", "panel_source", *columns]],
        ],
        ignore_index=True,
    )

    olmo = load_olmo_wide_with_table9_components()
    components = table9_component_order()
    proportional_reference = olmo[
        olmo["run_name"].eq("baseline_proportional") | olmo["panel"].eq("proportional_noise")
    ][components]
    if len(proportional_reference) != 11:
        raise ValueError(f"Expected 11 proportional rows, found {len(proportional_reference)}")
    component_means = proportional_reference.mean(axis=0)
    component_stds = proportional_reference.std(axis=0, ddof=1)

    panel = fit_rows.merge(olmo[["run_name", *components]], on="run_name", how="inner", validate="one_to_one")
    missing_fit_rows = sorted(set(fit_rows["run_name"]).difference(panel["run_name"]))
    if missing_fit_rows:
        raise ValueError(f"Missing OLMoBaseEval rows for fit runs: {missing_fit_rows[:20]}")
    panel.loc[panel["run_name"].eq("baseline_proportional"), components] = component_means.to_numpy()
    panel["table9_macro_bpb"] = panel[components].mean(axis=1)
    if len(panel) != 280:
        raise ValueError(f"Expected 280 fit rows, found {len(panel)}")
    if int(panel["panel_source"].eq("qsplit_signal").sum()) != 241:
        raise ValueError("Expected 241 qsplit signal rows")
    if int(panel["panel_source"].eq("domain_deletion").sum()) != 39:
        raise ValueError("Expected 39 domain-deletion rows")
    metadata = {
        "components": components,
        "source_metric_wide": str(OLMO_FULL_WIDE),
        "n_proportional_reference_rows": int(len(proportional_reference)),
        "proportional_reference_macro_mean": float(component_means.mean()),
        "proportional_reference_component_means": {key: float(component_means[key]) for key in components},
        "proportional_reference_component_stds": {key: float(component_stds[key]) for key in components},
        "excluded_adaptive_run_name": ADAPTIVE_OLMIX_RUN_NAME,
    }
    return panel, metadata


def regression_metrics(y: np.ndarray, y_hat: np.ndarray) -> tuple[float, float]:
    rmse, _mae, _pearson, spearman = base.regression_metrics(y, y_hat)
    return rmse, spearman


def predictive_diagnostics(
    y: np.ndarray,
    pred: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    rmse, _mae, _pearson, spearman = base.regression_metrics(y, pred)
    fold_regrets: list[float] = []
    for _train_idx, test_idx in folds:
        chosen = int(np.argmin(pred[test_idx]))
        fold_regrets.append(float(y[test_idx][chosen] - np.min(y[test_idx])))
    tail_count = max(5, int(np.ceil(LOWER_TAIL_FRAC * len(y))))
    tail_idx = np.argsort(pred)[:tail_count]
    residual = pred[tail_idx] - y[tail_idx]
    return {
        "rmse": float(rmse),
        "spearman": float(spearman),
        "fold_mean_regret_at_1": float(np.mean(fold_regrets)),
        "lower_tail_optimism": float(np.mean(np.maximum(y[tail_idx] - pred[tail_idx], 0.0))),
        "low_tail_rmse": float(np.sqrt(np.mean(residual * residual))),
    }


def solve_multi_two_phase(
    log_cs: np.ndarray,
    coefficients: np.ndarray,
    *,
    natural: np.ndarray,
    phase_fractions: np.ndarray,
    kl_reg: float,
    repetition_caps: np.ndarray,
) -> tuple[np.ndarray, float, float, str]:
    n_components = len(log_cs)
    n_phases = len(phase_fractions)
    n_domains = len(natural)
    phase_weights = cp.Variable((n_phases, n_domains))
    coeff = coefficients.reshape(n_components, n_phases, n_domains)
    component_logits = [
        cp.sum(cp.multiply(coeff[component_idx], phase_weights)) for component_idx in range(n_components)
    ]
    predicted = cp.sum(cp.exp(log_cs)) / n_components + cp.sum(cp.exp(cp.hstack(component_logits))) / n_components
    aggregate = cp.sum(cp.multiply(phase_fractions[:, None], phase_weights), axis=0)
    kl = cp.sum(cp.rel_entr(aggregate, natural))
    constraints: list[Any] = [
        phase_weights >= 0,
        cp.sum(phase_weights, axis=1) == 1,
        aggregate <= repetition_caps,
    ]
    problem = cp.Problem(cp.Minimize(predicted + float(kl_reg) * kl), constraints)
    status = solve_problem(problem)
    if phase_weights.value is None:
        raise RuntimeError("CVXPY returned no phase weights")
    solved = np.asarray(phase_weights.value, dtype=float)
    solved = np.clip(solved, 0.0, None)
    solved = solved / solved.sum(axis=1, keepdims=True)
    predicted_value = float(np.mean(predict_components(log_cs, coefficients, solved[None, :, :])))
    aggregate_solved = np.einsum("p,pd->d", phase_fractions, solved)
    aggregate_kl = float(
        np.sum(aggregate_solved * (np.log(np.clip(aggregate_solved, 1e-12, 1.0)) - np.log(natural)))
    )
    regularized = predicted_value + kl_reg * aggregate_kl
    return solved, predicted_value, regularized, status


def solve_multi_single(
    log_cs: np.ndarray,
    coefficients: np.ndarray,
    *,
    natural: np.ndarray,
    kl_reg: float,
    repetition_caps: np.ndarray,
) -> tuple[np.ndarray, float, float, str]:
    n_components = len(log_cs)
    n_domains = len(natural)
    weights = cp.Variable(n_domains)
    component_logits = [cp.sum(cp.multiply(coefficients[component_idx], weights)) for component_idx in range(n_components)]
    predicted = cp.sum(cp.exp(log_cs)) / n_components + cp.sum(cp.exp(cp.hstack(component_logits))) / n_components
    kl = cp.sum(cp.rel_entr(weights, natural))
    constraints: list[Any] = [weights >= 0, cp.sum(weights) == 1, weights <= repetition_caps]
    problem = cp.Problem(cp.Minimize(predicted + float(kl_reg) * kl), constraints)
    status = solve_problem(problem)
    if weights.value is None:
        raise RuntimeError("CVXPY returned no weights")
    solved = np.asarray(weights.value, dtype=float)
    solved = np.clip(solved, 0.0, None)
    solved = solved / solved.sum()
    predicted_value = float(np.mean(predict_components(log_cs, coefficients, solved[None, :])))
    regularized = predicted_value + kl_reg * float(
        np.sum(solved * (np.log(np.clip(solved, 1e-12, 1.0)) - np.log(natural)))
    )
    return np.stack([solved, solved]), predicted_value, regularized, status


def solve_problem(problem: cp.Problem) -> str:
    errors: list[str] = []
    for solver in ("CLARABEL", "ECOS", "SCS"):
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
        except Exception as exc:
            errors.append(f"{solver}: {exc}")
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return str(problem.status)
        errors.append(f"{solver}: status={problem.status}")
    raise RuntimeError(f"CVXPY solve failed: {errors}")


def predict_components(log_cs: np.ndarray, coefficients: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [base.predict(float(log_c), coefficients[idx], weights) for idx, log_c in enumerate(log_cs)]
    )


def feature_tensor(panel: pd.DataFrame, columns: list[str], domains: list[str], variant: str) -> np.ndarray:
    weights = panel[columns].astype(float).to_numpy().reshape(len(panel), 2, len(domains))
    if variant == "single_tied":
        return np.einsum("p,npd->nd", PHASE_FRACTIONS, weights)
    if variant == "two_phase_adapted":
        return weights
    raise ValueError(f"Unknown variant: {variant}")


def fit_variant(
    *,
    variant: str,
    huber_delta: float,
    panel: pd.DataFrame,
    metadata: dict[str, Any],
    columns: list[str],
    domains: list[str],
    natural: np.ndarray,
    token_counts: np.ndarray,
    target_budget: int,
    repetition_caps: np.ndarray,
    output_dir: Path,
    fit_n_starts: int,
) -> tuple[VariantSummary, pd.DataFrame, pd.DataFrame]:
    components = list(metadata["components"])
    features = feature_tensor(panel, columns, domains, variant)
    targets = panel[components].astype(float).to_numpy()
    folds = base.kfold_indices(len(panel), n_splits=N_SPLITS, seed=CV_SEED)
    log_cs: list[float] = []
    coefficients: list[np.ndarray] = []
    train_predictions = np.zeros_like(targets, dtype=float)
    oof_predictions = np.zeros_like(targets, dtype=float)
    component_rows: list[ComponentFit] = []
    print(f"Fitting {variant} delta={huber_delta:g} over {len(components)} components", flush=True)
    for component_idx, component in enumerate(components, start=1):
        y = targets[:, component_idx - 1]
        log_c, coef, loss = base.fit_olmix_loglinear(
            features,
            y,
            delta=huber_delta,
            seed=FIT_SEED + component_idx,
            n_starts=fit_n_starts,
            verbose=False,
        )
        log_cs.append(log_c)
        coefficients.append(coef)
        train_predictions[:, component_idx - 1] = base.predict(log_c, coef, features)
        for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
            fold_log_c, fold_coef, _ = base.fit_olmix_loglinear(
                features[train_idx],
                y[train_idx],
                delta=huber_delta,
                seed=FIT_SEED + component_idx * 100 + fold_idx,
                n_starts=fit_n_starts,
                verbose=False,
            )
            oof_predictions[test_idx, component_idx - 1] = base.predict(fold_log_c, fold_coef, features[test_idx])
        train_rmse, train_spearman = regression_metrics(y, train_predictions[:, component_idx - 1])
        oof_rmse, oof_spearman = regression_metrics(y, oof_predictions[:, component_idx - 1])
        component_rows.append(
            ComponentFit(
                variant=variant,
                huber_delta=float(huber_delta),
                component=component,
                n_rows=int(len(panel)),
                fit_log_c=float(log_c),
                fit_huber_loss=float(loss),
                train_rmse=float(train_rmse),
                train_spearman=float(train_spearman),
                oof_rmse=float(oof_rmse),
                oof_spearman=float(oof_spearman),
            )
        )
        if component_idx % 5 == 0 or component_idx == len(components):
            print(f"  component {component_idx}/{len(components)}", flush=True)

    log_cs_array = np.asarray(log_cs, dtype=float)
    coefficients_array = np.vstack(coefficients)
    if variant == "single_tied":
        optimum_weights, predicted_objective, regularized_objective, cvxpy_status = solve_multi_single(
            log_cs_array,
            coefficients_array,
            natural=natural,
            kl_reg=KL_REG,
            repetition_caps=repetition_caps,
        )
    else:
        optimum_weights, predicted_objective, regularized_objective, cvxpy_status = solve_multi_two_phase(
            log_cs_array,
            coefficients_array,
            natural=natural,
            phase_fractions=PHASE_FRACTIONS,
            kl_reg=KL_REG,
            repetition_caps=repetition_caps,
        )

    observed_macro = targets.mean(axis=1)
    train_macro_pred = train_predictions.mean(axis=1)
    oof_macro_pred = oof_predictions.mean(axis=1)
    train_rmse, _mae, _pearson, train_spearman = base.regression_metrics(observed_macro, train_macro_pred)
    oof = predictive_diagnostics(observed_macro, oof_macro_pred, folds)
    phase_weights = panel[columns].astype(float).to_numpy().reshape(len(panel), 2, len(domains))
    distances = base.mean_phase_tv(phase_weights, optimum_weights)
    nearest_idx = int(np.argmin(distances))
    best_idx = int(np.argmin(observed_macro))
    reference = np.stack([natural, natural])
    ratios = optimum_weights / np.clip(reference, 1e-12, None)
    sim_epochs = base.simulated_epochs(optimum_weights, token_counts, target_budget=target_budget)
    prop_rows = panel["run_name"].eq("baseline_proportional")
    if int(prop_rows.sum()) != 1:
        raise ValueError("Expected one baseline_proportional fit row")
    prop_actual = float(panel.loc[prop_rows, "table9_macro_bpb"].iloc[0])
    if variant == "single_tied":
        prop_feature = natural[None, :]
    else:
        prop_feature = reference[None, :, :]
    prop_pred = float(np.mean(predict_components(log_cs_array, coefficients_array, prop_feature)))
    summary = VariantSummary(
        variant=variant,
        huber_delta=float(huber_delta),
        n_components=len(components),
        n_rows=len(panel),
        train_macro_rmse=float(train_rmse),
        train_macro_spearman=float(train_spearman),
        oof_macro_rmse=float(oof["rmse"]),
        oof_macro_spearman=float(oof["spearman"]),
        fold_mean_regret_at_1=float(oof["fold_mean_regret_at_1"]),
        lower_tail_optimism=float(oof["lower_tail_optimism"]),
        low_tail_rmse=float(oof["low_tail_rmse"]),
        cvxpy_status=cvxpy_status,
        kl_reg=KL_REG,
        repetition_factor=REPETITION_FACTOR,
        predicted_macro_bpb=float(predicted_objective),
        regularized_objective=float(regularized_objective),
        proportional_macro_bpb=prop_actual,
        proportional_predicted_macro_bpb=prop_pred,
        best_observed_run_name=str(panel.iloc[best_idx]["run_name"]),
        best_observed_macro_bpb=float(observed_macro[best_idx]),
        nearest_observed_run_name=str(panel.iloc[nearest_idx]["run_name"]),
        nearest_observed_macro_bpb=float(observed_macro[nearest_idx]),
        nearest_observed_mean_phase_tv=float(distances[nearest_idx]),
        mean_phase_tv_to_proportional=float(0.5 * np.abs(optimum_weights - reference).sum(axis=1).mean()),
        max_epoch_multiplier=float(np.max(ratios)),
        q95_epoch_multiplier=float(np.quantile(ratios, 0.95)),
        max_simulated_epoch=float(np.max(sim_epochs)),
        q95_simulated_epoch=float(np.quantile(sim_epochs, 0.95)),
        max_repetition_cap_violation=float(np.max(sim_epochs - REPETITION_FACTOR)),
    )

    predictions = panel[["run_name", "source_experiment", "panel_source", "table9_macro_bpb"]].copy()
    predictions["train_pred_macro_bpb"] = train_macro_pred
    predictions["oof_pred_macro_bpb"] = oof_macro_pred
    predictions["train_residual"] = train_macro_pred - observed_macro
    predictions["oof_residual"] = oof_macro_pred - observed_macro

    weights_out = pd.DataFrame(
        {
            "domain": domains,
            "proportional": natural,
            "phase_0_weight": optimum_weights[0],
            "phase_1_weight": optimum_weights[1],
            "aggregate_weight": base.aggregate_phase_weights(optimum_weights),
            "available_tokens": token_counts,
            "simulated_epochs": sim_epochs,
            "phase_0_epoch_multiplier": ratios[0],
            "phase_1_epoch_multiplier": ratios[1],
            "phase_0_delta": optimum_weights[0] - natural,
            "phase_1_delta": optimum_weights[1] - natural,
        }
    )
    weights_out["max_abs_delta"] = weights_out[["phase_0_delta", "phase_1_delta"]].abs().max(axis=1)

    variant_dir = output_dir / f"{variant}_delta_{str(huber_delta).replace('.', 'p')}"
    variant_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(row) for row in component_rows]).to_csv(variant_dir / "component_fit_summary.csv", index=False)
    predictions.to_csv(variant_dir / "macro_fit_predictions.csv", index=False)
    weights_out.to_csv(variant_dir / "proposed_mixture_weights.csv", index=False)
    with (variant_dir / "summary.json").open("w") as f:
        json.dump({"summary": asdict(summary), "metadata": metadata}, f, indent=2, sort_keys=True)
    write_variant_plots(variant_dir, variant, huber_delta, predictions, weights_out, summary)
    return summary, pd.DataFrame([asdict(row) for row in component_rows]), predictions


def write_variant_plots(
    variant_dir: Path,
    variant: str,
    huber_delta: float,
    predictions: pd.DataFrame,
    weights: pd.DataFrame,
    summary: VariantSummary,
) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=predictions["table9_macro_bpb"],
            y=predictions["oof_pred_macro_bpb"],
            mode="markers",
            marker={
                "size": 8,
                "color": predictions["panel_source"].map({"qsplit_signal": 0, "domain_deletion": 1}),
                "colorscale": "RdYlGn_r",
            },
            text=predictions["run_name"],
            name="fit rows",
        )
    )
    lo = float(min(predictions["table9_macro_bpb"].min(), predictions["oof_pred_macro_bpb"].min()))
    hi = float(max(predictions["table9_macro_bpb"].max(), predictions["oof_pred_macro_bpb"].max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"dash": "dash", "color": "#555"}, name="y=x"))
    fig.update_layout(
        title=f"{variant} delta={huber_delta:g}: OOF predicted vs observed Table-9 macro BPB",
        xaxis_title="Observed mean component BPB",
        yaxis_title="OOF predicted mean component BPB",
        template="plotly_white",
        width=900,
        height=720,
    )
    fig.write_html(variant_dir / "macro_oof_scatter.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    top = weights.sort_values("max_abs_delta", ascending=False).head(30).iloc[::-1]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(y=top["domain"], x=top["phase_0_epoch_multiplier"], orientation="h", name="phase 0"))
    fig2.add_trace(go.Bar(y=top["domain"], x=top["phase_1_epoch_multiplier"], orientation="h", name="phase 1"))
    fig2.add_vline(x=1.0, line_dash="dash", line_color="#444")
    fig2.add_vline(x=REPETITION_FACTOR, line_dash="dot", line_color="#666", annotation_text="epoch cap=4")
    fig2.update_layout(
        title=(
            f"{variant} OLMix proposal: KL={summary.kl_reg:g}, cap={summary.repetition_factor:g}, "
            f"delta={huber_delta:g}"
        ),
        xaxis_title="Epoch multiplier relative to proportional",
        yaxis_title="Domain",
        template="plotly_white",
        width=1150,
        height=900,
        barmode="group",
    )
    fig2.write_html(variant_dir / "proposed_mixture_epoch_multipliers.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    heat = weights.sort_values("domain")
    z = np.log2(
        np.clip(heat[["phase_0_epoch_multiplier", "phase_1_epoch_multiplier"]].to_numpy(dtype=float), 1e-9, None)
    )
    text = np.vectorize(lambda value: f"{value:.1f}x")(
        heat[["phase_0_epoch_multiplier", "phase_1_epoch_multiplier"]].to_numpy(dtype=float)
    )
    fig3 = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["phase 0", "phase 1"],
            y=heat["domain"],
            colorscale="RdYlGn_r",
            zmid=0.0,
            colorbar={"title": "log2 epoch multiplier"},
            text=text,
            texttemplate="%{text}",
            hovertemplate="domain=%{y}<br>phase=%{x}<br>log2 multiplier=%{z:.2f}<extra></extra>",
        )
    )
    fig3.update_layout(
        title=f"{variant} OLMix proposal heatmap",
        xaxis_title="Phase",
        yaxis_title="Domain",
        template="plotly_white",
        width=760,
        height=1250,
    )
    fig3.write_html(variant_dir / "proposed_mixture_heatmap.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_summary_plots(output_dir: Path, summaries: pd.DataFrame, components: pd.DataFrame) -> None:
    fig = go.Figure()
    for variant, group in summaries.groupby("variant"):
        fig.add_trace(
            go.Scatter(
                x=group["huber_delta"],
                y=group["oof_macro_spearman"],
                mode="lines+markers",
                name=f"{variant} OOF Spearman",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=group["huber_delta"],
                y=group["predicted_macro_bpb"],
                mode="lines+markers",
                name=f"{variant} proposed pred BPB",
                yaxis="y2",
            )
        )
    fig.update_layout(
        title="Paper-faithful OLMix sweep: component-fit average objective",
        xaxis_title="Huber delta",
        yaxis_title="OOF Spearman on Table-9 macro BPB",
        yaxis2={"title": "Predicted proposed macro BPB", "overlaying": "y", "side": "right"},
        template="plotly_white",
        width=1000,
        height=650,
    )
    fig.write_html(output_dir / "paper_faithful_olmix_sweep_summary.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    best_key = summaries.sort_values(["oof_macro_spearman", "oof_macro_rmse"], ascending=[False, True]).iloc[0]
    best_components = components[
        components["variant"].eq(best_key["variant"]) & components["huber_delta"].eq(best_key["huber_delta"])
    ].sort_values("oof_spearman")
    fig2 = go.Figure(
        go.Bar(
            x=best_components["oof_spearman"],
            y=best_components["component"].str.replace("olmo_base_eval/easy_bpb/", "", regex=False).str.replace(
                "/bpb", "", regex=False
            ),
            orientation="h",
        )
    )
    fig2.update_layout(
        title=f"Component OOF Spearman for best macro fit: {best_key['variant']} delta={best_key['huber_delta']}",
        xaxis_title="OOF Spearman",
        yaxis_title="Table-9 component",
        template="plotly_white",
        width=1000,
        height=1300,
    )
    fig2.write_html(output_dir / "best_component_oof_spearman.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(output_dir: Path, summaries: pd.DataFrame, metadata: dict[str, Any]) -> None:
    rows = summaries.sort_values(["oof_macro_spearman", "oof_macro_rmse"], ascending=[False, True])
    lines = [
        "# Paper-faithful OLMix baselines for OLMoBaseEval Easy",
        "",
        "This benchmark fits one OLMix log-linear model per Table-9 BPB component, then optimizes the unweighted mean predicted component BPB.",
        "",
        "- `single_tied`: phase-weighted exposure average is used as the single-simplex feature; the solved mixture is deployed in both phases.",
        "- `two_phase_adapted`: concatenated phase weights are modeled directly; this is a Marin extension, not the paper's single-simplex setting.",
        "- Both variants use KL and epoch caps on aggregate exposure, so the two-phase extension is not penalized more heavily than the single-simplex baseline.",
        f"- KL regularization: `{KL_REG}`.",
        f"- Aggregate simulated epoch cap: `{REPETITION_FACTOR}`.",
        f"- Fit rows: `241` ex-ante qsplit rows plus `39` domain deletions; the baseline proportional target is the mean of `{metadata['n_proportional_reference_rows']}` proportional observations.",
        "",
        "| variant | Huber delta | OOF Spearman | OOF RMSE | regret@1 | lower-tail optimism | predicted proposed BPB | proportional BPB | max sim epoch | nearest observed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            f"| `{row.variant}` | {row.huber_delta:g} | {row.oof_macro_spearman:.4f} | "
            f"{row.oof_macro_rmse:.6f} | {row.fold_mean_regret_at_1:.6f} | "
            f"{row.lower_tail_optimism:.6f} | {row.predicted_macro_bpb:.6f} | "
            f"{row.proportional_macro_bpb:.6f} | {row.max_simulated_epoch:.2f} | "
            f"`{row.nearest_observed_run_name}` |"
        )
    lines.extend(
        [
            "",
            "Component set:",
            f"- `{len(metadata['components'])}` components.",
            "- Minerva MATH, ARC, MultiPL-E/MT-MBPP, and Basic Skills use their subtasks as standalone components.",
            "- MMLU leaves are replaced by the four OLMix category averages.",
            "- OLMoBaseEval aggregate rows such as `olmobase_easy_qa`, `olmobase_easy_code`, and `olmobase_easy_math` are not included in the paper-faithful objective.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _signal, columns, domains, natural = base.load_raw_signal_panel()
    target_budget = base.load_target_budget()
    token_counts = base.load_domain_token_counts(domains)
    repetition_caps = base.repetition_weight_caps(
        token_counts,
        target_budget=target_budget,
        repetition_factor=REPETITION_FACTOR,
    )
    if np.any(natural - repetition_caps > 1e-12):
        raise ValueError("Proportional baseline violates the requested repetition cap")

    panel, metadata = build_fit_panel(columns)
    panel.to_csv(args.output_dir / "fit_panel_table9_macro.csv", index=False)
    with (args.output_dir / "component_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    summaries: list[VariantSummary] = []
    component_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for huber_delta in args.huber_deltas:
        for variant in args.variants:
            summary, component_frame, prediction_frame = fit_variant(
                variant=variant,
                huber_delta=float(huber_delta),
                panel=panel,
                metadata=metadata,
                columns=columns,
                domains=domains,
                natural=natural,
                token_counts=token_counts,
                target_budget=target_budget,
                repetition_caps=repetition_caps,
                output_dir=args.output_dir,
                fit_n_starts=int(args.fit_n_starts),
            )
            summaries.append(summary)
            component_frames.append(component_frame)
            prediction_frames.append(
                prediction_frame.assign(variant=variant, huber_delta=float(huber_delta))
            )

    summary_frame = pd.DataFrame([asdict(row) for row in summaries])
    component_frame = pd.concat(component_frames, ignore_index=True)
    prediction_frame = pd.concat(prediction_frames, ignore_index=True)
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    component_frame.to_csv(args.output_dir / "component_fit_summary.csv", index=False)
    prediction_frame.to_csv(args.output_dir / "macro_fit_predictions.csv", index=False)
    with (args.output_dir / "summary.json").open("w") as f:
        json.dump([asdict(row) for row in summaries], f, indent=2, sort_keys=True)
    write_summary_plots(args.output_dir, summary_frame, component_frame)
    write_report(args.output_dir, summary_frame, metadata)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
