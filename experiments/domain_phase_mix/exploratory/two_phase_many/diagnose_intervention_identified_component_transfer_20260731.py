# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy", "tabulate"]
# ///
"""Falsify intervention-identified bucket-to-benchmark transfer at 300M.

The 78 central antithetic log tilts identify a local simplex-tangent gradient.
The 39 full domain deletions are a different intervention class and are never
used to estimate that gradient. Passing this diagnostic licenses evaluation of
a bounded benchmark-production aggregate spine; it does not promote one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_proportional_controllability_log_tilts as pctrl,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    fit_olmo_base_easy_paper_faithful_olmix_300m as table9,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
MANIFEST_PATH = REFERENCE_OUTPUTS / "proportional_controllability_300m_20260520" / "training_manifest.csv"
UNCHEATABLE_INTERVENTIONS_PATH = (
    REFERENCE_OUTPUTS / "pctrl_training_eval_wandb_collect_20260623" / "pctrl_final_metric_matrix_with_training_eval.csv"
)
UNCHEATABLE_CONTROLS_PATH = (
    SCRIPT_DIR / "metric_registry" / "raw_metric_matrix_300m" / "raw_metric_matrix_300m_with_proportional_noise.csv"
)
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "intervention_identified_component_transfer_20260731"

CANDIDATE_ID = "WSD80-SUR-064"
BOOTSTRAP_DRAWS = 4_000
BOOTSTRAP_SEED = 20260731
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

UNCHEATABLE_COMPONENT_NAMES = (
    "ao3_english",
    "arxiv_computer_science",
    "arxiv_physics",
    "bbc_news",
    "github_cpp",
    "github_python",
    "wikipedia_english",
)

GATES = {
    "aggregate_gradient_p_max": 0.01,
    "aggregate_deletion_spearman_min": 0.35,
    "aggregate_deletion_spearman_ci_low_min": 0.0,
    "aggregate_deletion_sign_accuracy_min": 0.60,
    "aggregate_deletion_rmse_improvement_min": 0.05,
    "aggregate_deletion_rmse_improvement_ci_low_min": 0.0,
    "component_gradient_bh_fraction_min": 0.50,
    "component_positive_deletion_spearman_fraction_min": 0.60,
    "component_median_deletion_spearman_min": 0.20,
}


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def protocol() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "candidate_id": CANDIDATE_ID,
        "purpose": "Cheapest falsification of an intervention-identified benchmark-production spine",
        "training_interventions": {
            "kind": "central antithetic log tilts",
            "rows": 78,
            "pairs": 39,
            "alpha": pctrl.ALPHA,
            "estimand": "local BPB gradient in the 38-dimensional simplex tangent space",
        },
        "falsification_interventions": {
            "kind": "full domain deletion with proportional redistribution",
            "rows": 39,
            "used_for_fitting": False,
        },
        "anchor": {
            "kind": "physically tied proportional controls",
            "rows": 11,
            "use": "estimate deletion effects and observational noise",
        },
        "targets": {
            "uncheatable": "fixed weighted aggregation of seven component BPBs",
            "table9": "exact unweighted mean of 51 component BPBs",
        },
        "gradient_estimator": (
            "For each component, d_i=(L_i(+alpha)-L_i(-alpha))/(2 alpha); "
            "solve d=A r by Moore-Penrose inverse, project r onto the simplex tangent, "
            "then q=r/sqrt(p)."
        ),
        "deletion_prediction": "delta_L_hat = q dot (w_deletion - w_proportional)",
        "important_non_evidence": (
            "The 39 tilt directions span only 38 tangent dimensions, so in-sample direction fit is "
            "nearly saturated and is never used as evidence. Only held-out intervention-class transfer counts."
        ),
        "bootstrap": {
            "unit": "deleted domain",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "interval": [0.025, 0.975],
        },
        "gates": GATES,
        "decision": (
            "Both aggregate targets and both component-support summaries must pass. "
            "Passing licenses a bounded aggregate-production model; failure blocks this identification route."
        ),
        "data_paths": {
            "manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
            "uncheatable_interventions": str(UNCHEATABLE_INTERVENTIONS_PATH.relative_to(REPO_ROOT)),
            "uncheatable_controls": str(UNCHEATABLE_CONTROLS_PATH.relative_to(REPO_ROOT)),
            "table9_wide": str(table9.OLMO_FULL_WIDE.relative_to(REPO_ROOT)),
        },
        "source_sha256": source_sha256(),
    }
    payload["protocol_sha256"] = canonical_hash(payload)
    return payload


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def intervention_columns(manifest: pd.DataFrame, phase: int) -> list[str]:
    prefix = f"phase_{phase}_"
    return [column for column in manifest.columns if column.startswith(prefix)]


def load_manifest() -> tuple[pd.DataFrame, list[str]]:
    manifest = read_csv(MANIFEST_PATH)
    counts = manifest["intervention_type"].value_counts().to_dict()
    if counts != {"central_log_tilt": 78, "domain_deletion": 39}:
        raise ValueError(f"Unexpected intervention counts: {counts}")
    phase_0_columns = intervention_columns(manifest, 0)
    phase_1_columns = intervention_columns(manifest, 1)
    if len(phase_0_columns) != 39 or len(phase_1_columns) != 39:
        raise ValueError("Expected 39 bucket columns in each phase")
    domains = [column.removeprefix("phase_0_") for column in phase_0_columns]
    expected_phase_1 = [f"phase_1_{domain}" for domain in domains]
    if phase_1_columns != expected_phase_1:
        raise ValueError("Phase columns do not share one ordered domain list")
    phase_gap = np.max(
        np.abs(manifest[phase_0_columns].to_numpy(dtype=float) - manifest[phase_1_columns].to_numpy(dtype=float))
    )
    if phase_gap > 1e-12:
        raise ValueError(f"Intervention panel is not physically tied: max phase gap {phase_gap}")
    row_sums = manifest[phase_0_columns].sum(axis=1).to_numpy(dtype=float)
    if np.max(np.abs(row_sums - 1.0)) > 1e-9:
        raise ValueError("Intervention weights do not sum to one")
    return manifest, domains


def table9_family(component: str) -> str:
    if component in {table9.metric_key(task) for task in table9.MINERVA_SUBTASKS}:
        return "minerva_math"
    if component in {
        table9.metric_key("codex_humaneval"),
        table9.metric_key("mbpp"),
        *[table9.metric_key(task) for task in table9.MT_MBPP_SUBTASKS],
    }:
        return "code"
    if component in {table9.metric_key("arc_easy"), table9.metric_key("arc_challenge")}:
        return "arc"
    if component.startswith("mmlu_"):
        return "mmlu"
    if component in {table9.metric_key(task) for task in ("csqa", "hellaswag", "winogrande", "socialiqa", "piqa")}:
        return "commonsense"
    if component in {table9.metric_key(task) for task in ("coqa", "drop", "jeopardy", "naturalqs", "squad", "sciq")}:
        return "reading_qa"
    if component in {table9.metric_key(task) for task in table9.BASIC_SKILLS_SUBTASKS}:
        return "basic_skills"
    if component == table9.metric_key("lambada"):
        return "language"
    if component == table9.metric_key("medmcqa"):
        return "medical"
    raise ValueError(f"Unclassified Table-9 component: {component}")


def uncheatable_family(component: str) -> str:
    name = component.split("/")[-2]
    if name.startswith("github_"):
        return "code"
    if name.startswith("arxiv_"):
        return "research"
    return {
        "ao3_english": "narrative",
        "bbc_news": "news",
        "wikipedia_english": "encyclopedic",
    }[name]


def constrained_aggregation_weights(component_values: np.ndarray, aggregate: np.ndarray) -> np.ndarray:
    count = component_values.shape[1]
    gram = component_values.T @ component_values
    rhs = component_values.T @ aggregate
    system = np.block(
        [
            [gram, np.ones((count, 1))],
            [np.ones((1, count)), np.zeros((1, 1))],
        ]
    )
    solution = np.linalg.solve(system, np.concatenate([rhs, np.ones(1)]))[:count]
    if np.min(solution) < -1e-6:
        raise ValueError(f"Recovered negative aggregate weight: {solution}")
    solution = np.clip(solution, 0.0, None)
    return solution / solution.sum()


def load_uncheatable(
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray, list[str], dict[str, float]]:
    components = [f"eval/uncheatable_eval/{name}/bpb" for name in UNCHEATABLE_COMPONENT_NAMES]
    aggregate = "eval/uncheatable_eval/bpb"
    interventions = read_csv(UNCHEATABLE_INTERVENTIONS_PATH)
    needed = ["run_name", *components, aggregate]
    joined = manifest.merge(interventions[needed], on="run_name", how="left", validate="one_to_one")
    if joined[needed[1:]].isna().any().any():
        raise ValueError("Uncheatable intervention outcomes are incomplete")

    controls = read_csv(UNCHEATABLE_CONTROLS_PATH)
    controls = controls[
        controls["run_name"].eq("baseline_proportional") | controls["row_kind"].eq("noise_variable_subset_proportional")
    ][needed].copy()
    if len(controls) != 11 or controls[needed[1:]].isna().any().any():
        raise ValueError("Expected 11 complete Uncheatable proportional controls")

    training_rows = pd.concat(
        [
            joined[joined["intervention_type"].eq("central_log_tilt")][needed],
            controls[needed],
        ],
        ignore_index=True,
    )
    weights = constrained_aggregation_weights(
        training_rows[components].to_numpy(dtype=float), training_rows[aggregate].to_numpy(dtype=float)
    )
    all_reconstruction = joined[components].to_numpy(dtype=float) @ weights
    diagnostics = {
        "aggregation_train_max_abs_error": float(
            np.max(
                np.abs(
                    training_rows[components].to_numpy(dtype=float) @ weights
                    - training_rows[aggregate].to_numpy(dtype=float)
                )
            )
        ),
        "aggregation_deletion_max_abs_error": float(
            np.max(
                np.abs(
                    all_reconstruction[joined["intervention_type"].eq("domain_deletion").to_numpy()]
                    - joined.loc[joined["intervention_type"].eq("domain_deletion"), aggregate].to_numpy(dtype=float)
                )
            )
        ),
    }
    if max(diagnostics.values()) > 2e-6:
        raise ValueError(f"Uncheatable component aggregation is not exact enough: {diagnostics}")
    return joined, controls, components, weights, [uncheatable_family(c) for c in components], diagnostics


def load_table9(
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray, list[str], dict[str, float]]:
    components = table9.table9_component_order()
    wide = table9.load_olmo_wide_with_table9_components()
    joined = manifest.merge(wide, on="run_name", how="left", validate="one_to_one")
    if joined[components].isna().any().any():
        raise ValueError("Table-9 intervention outcomes are incomplete")
    controls = wide[wide["run_name"].eq("baseline_proportional") | wide["panel"].eq("proportional_noise")].copy()
    if len(controls) != 11 or controls[components].isna().any().any():
        raise ValueError("Expected 11 complete Table-9 proportional controls")
    weights = np.full(len(components), 1.0 / len(components), dtype=float)
    joined = joined.copy()
    controls = controls.copy()
    joined["table9_macro_bpb"] = joined[components].to_numpy(dtype=float) @ weights
    controls["table9_macro_bpb"] = controls[components].to_numpy(dtype=float) @ weights
    return (
        joined,
        controls,
        components,
        weights,
        [table9_family(c) for c in components],
        {"aggregation_max_abs_error": 0.0},
    )


def preflight_payload() -> dict[str, Any]:
    manifest, manifest_domains = load_manifest()
    geometry = pctrl.load_geometry()
    if set(manifest_domains) != set(geometry.domains):
        raise ValueError("Manifest and analytic intervention geometry disagree")
    domains = geometry.domains
    uncheatable = load_uncheatable(manifest)
    table9_bundle = load_table9(manifest)
    return {
        "rows": len(manifest),
        "domains": len(domains),
        "intervention_counts": manifest["intervention_type"].value_counts().to_dict(),
        "physically_tied": True,
        "geometry_rank": int(np.linalg.matrix_rank(geometry.v * geometry.sqrt_p[None, :])),
        "uncheatable_components": len(uncheatable[2]),
        "table9_components": len(table9_bundle[2]),
        "proportional_controls": len(uncheatable[1]),
        "uncheatable_aggregation": uncheatable[5],
        "table9_aggregation": table9_bundle[5],
    }


def freeze_protocol(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = protocol()
    path = output_dir / "protocol.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise ValueError(f"Frozen protocol differs from current source: {path}")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def verify_protocol(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "protocol.json"
    if not path.exists():
        raise FileNotFoundError(f"Freeze the protocol before evaluation: {path}")
    frozen = json.loads(path.read_text())
    current = protocol()
    if frozen != current:
        raise ValueError("Current source or protocol differs from the frozen protocol")
    return frozen


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def estimate_gradients(
    frame: pd.DataFrame,
    controls: pd.DataFrame,
    components: list[str],
    weights: np.ndarray,
    domains: list[str],
    geometry: pctrl.Geometry,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, float]]:
    tilts = frame[frame["intervention_type"].eq("central_log_tilt")]
    plus = tilts[tilts["tilt_sign"].eq("plus")].set_index("target_domain").loc[domains]
    minus = tilts[tilts["tilt_sign"].eq("minus")].set_index("target_domain").loc[domains]
    derivative = (plus[components].to_numpy(dtype=float) - minus[components].to_numpy(dtype=float)) / (2.0 * pctrl.ALPHA)
    design = geometry.v * geometry.sqrt_p[None, :]
    pinv = np.linalg.pinv(design, rcond=1e-10)
    tangent = np.eye(len(domains)) - np.outer(geometry.sqrt_p, geometry.sqrt_p)
    r = tangent @ (pinv @ derivative)
    q = (r / geometry.sqrt_p[:, None]).T
    q = q - (q @ geometry.p)[:, None]
    derivative_hat = design @ (q.T * geometry.sqrt_p[:, None])

    control_sd = controls[components].std(axis=0, ddof=1).to_numpy(dtype=float)
    derivative_noise_sd = control_sd / (pctrl.ALPHA * math.sqrt(2.0))
    projected_energy = np.sum(derivative_hat * derivative_hat, axis=0)
    chi_square = projected_energy / np.maximum(derivative_noise_sd * derivative_noise_sd, 1e-30)
    gradient_p = stats.chi2.sf(chi_square, df=np.linalg.matrix_rank(design))
    gradient_q = benjamini_hochberg(gradient_p)
    gradient_norm = np.sqrt(np.sum(geometry.p[None, :] * q * q, axis=1))

    rows = []
    for index, component in enumerate(components):
        rows.append(
            {
                "component": component,
                "gradient_norm": gradient_norm[index],
                "alpha_gradient_over_control_sd": (
                    pctrl.ALPHA * gradient_norm[index] / control_sd[index] if control_sd[index] > 0.0 else math.nan
                ),
                "gradient_chi_square": chi_square[index],
                "gradient_p": gradient_p[index],
                "gradient_bh_q": gradient_q[index],
                "direction_fit_rmse_non_evidence": float(
                    np.sqrt(np.mean((derivative[:, index] - derivative_hat[:, index]) ** 2))
                ),
            }
        )

    aggregate_q = weights @ q
    aggregate_derivative = derivative @ weights
    aggregate_derivative_hat = derivative_hat @ weights
    aggregate_control = controls[components].to_numpy(dtype=float) @ weights
    aggregate_sd = float(np.std(aggregate_control, ddof=1))
    aggregate_noise_sd = aggregate_sd / (pctrl.ALPHA * math.sqrt(2.0))
    aggregate_chi_square = float(
        np.sum(aggregate_derivative_hat * aggregate_derivative_hat) / max(aggregate_noise_sd**2, 1e-30)
    )
    aggregate = {
        "gradient_norm": float(np.sqrt(np.sum(geometry.p * aggregate_q * aggregate_q))),
        "gradient_chi_square": aggregate_chi_square,
        "gradient_p": float(stats.chi2.sf(aggregate_chi_square, df=np.linalg.matrix_rank(design))),
        "direction_fit_rmse_non_evidence": float(
            np.sqrt(np.mean((aggregate_derivative - aggregate_derivative_hat) ** 2))
        ),
    }
    return q, aggregate_q, pd.DataFrame(rows), aggregate


def finite_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    rmse = float(np.sqrt(np.mean(residual * residual)))
    null_rmse = float(np.sqrt(np.mean(observed * observed)))
    spearman = float(stats.spearmanr(predicted, observed).statistic)
    sign_accuracy = float(np.mean(np.sign(predicted) == np.sign(observed)))
    denominator = float(np.dot(predicted, predicted))
    slope_zero = float(np.dot(predicted, observed) / denominator) if denominator > 0.0 else math.nan
    if float(np.std(predicted)) > 0.0:
        slope, intercept = np.polyfit(predicted, observed, 1)
    else:
        slope, intercept = math.nan, math.nan
    return {
        "rmse": rmse,
        "null_rmse": null_rmse,
        "rmse_improvement": 1.0 - rmse / null_rmse if null_rmse > 0.0 else math.nan,
        "spearman": spearman,
        "sign_accuracy": sign_accuracy,
        "observed_on_predicted_slope_zero": slope_zero,
        "observed_on_predicted_slope": float(slope),
        "observed_on_predicted_intercept": float(intercept),
        "bias_predicted_minus_observed": float(np.mean(residual)),
        "worst_abs_error": float(np.max(np.abs(residual))),
    }


def bootstrap_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    count = len(observed)
    values: dict[str, list[float]] = {
        "rmse_improvement": [],
        "spearman": [],
        "sign_accuracy": [],
        "observed_on_predicted_slope_zero": [],
    }
    for _ in range(BOOTSTRAP_DRAWS):
        indices = rng.integers(0, count, size=count)
        metrics = finite_metrics(observed[indices], predicted[indices])
        for key in values:
            values[key].append(metrics[key])
    output = {}
    for key, samples in values.items():
        finite = np.asarray(samples, dtype=float)
        finite = finite[np.isfinite(finite)]
        output[f"{key}_ci_low"] = float(np.quantile(finite, 0.025))
        output[f"{key}_ci_high"] = float(np.quantile(finite, 0.975))
    return output


def evaluate_target(
    *,
    target: str,
    frame: pd.DataFrame,
    controls: pd.DataFrame,
    components: list[str],
    weights: np.ndarray,
    component_families: list[str],
    domains: list[str],
    geometry: pctrl.Geometry,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    q, aggregate_q, gradient_rows, aggregate_gradient = estimate_gradients(
        frame, controls, components, weights, domains, geometry
    )
    gradient_rows.insert(0, "target", target)
    gradient_rows["component_family"] = component_families

    deletions = frame[frame["intervention_type"].eq("domain_deletion")].set_index("target_domain").loc[domains]
    weight_columns = [f"phase_0_{domain}" for domain in domains]
    deletion_weights = deletions[weight_columns].to_numpy(dtype=float)
    delta_weights = deletion_weights - geometry.p[None, :]
    predicted_component = delta_weights @ q.T
    anchor_component = controls[components].mean(axis=0).to_numpy(dtype=float)
    observed_component = deletions[components].to_numpy(dtype=float) - anchor_component[None, :]

    prediction_rows = []
    component_metric_rows = []
    for component_index, component in enumerate(components):
        metrics = finite_metrics(observed_component[:, component_index], predicted_component[:, component_index])
        component_metric_rows.append(
            {
                "target": target,
                "component": component,
                "component_family": component_families[component_index],
                **gradient_rows.iloc[component_index].drop(labels=["target", "component", "component_family"]).to_dict(),
                **metrics,
            }
        )
        for domain_index, domain in enumerate(domains):
            prediction_rows.append(
                {
                    "target": target,
                    "component": component,
                    "component_family": component_families[component_index],
                    "deleted_domain": domain,
                    "observed_delta_bpb": observed_component[domain_index, component_index],
                    "predicted_delta_bpb": predicted_component[domain_index, component_index],
                }
            )

    observed_aggregate = observed_component @ weights
    predicted_aggregate = delta_weights @ aggregate_q
    aggregate_metrics = {
        "target": target,
        **{f"gradient_{key}": value for key, value in aggregate_gradient.items()},
        **finite_metrics(observed_aggregate, predicted_aggregate),
        **bootstrap_metrics(observed_aggregate, predicted_aggregate),
    }
    component_metrics = pd.DataFrame(component_metric_rows)
    positive_spearman = component_metrics["spearman"].gt(0.0)
    component_summary = {
        "component_count": len(component_metrics),
        "gradient_bh_q_lt_0p1_fraction": float(component_metrics["gradient_bh_q"].lt(0.1).mean()),
        "positive_deletion_spearman_fraction": float(positive_spearman.mean()),
        "median_deletion_spearman": float(component_metrics["spearman"].median()),
        "median_deletion_rmse_improvement": float(component_metrics["rmse_improvement"].median()),
    }
    aggregate_metrics.update(component_summary)

    aggregate_prediction_rows = pd.DataFrame(
        {
            "target": target,
            "component": "__aggregate__",
            "component_family": "aggregate",
            "deleted_domain": domains,
            "observed_delta_bpb": observed_aggregate,
            "predicted_delta_bpb": predicted_aggregate,
        }
    )
    predictions = pd.concat([pd.DataFrame(prediction_rows), aggregate_prediction_rows], ignore_index=True)
    q_rows = pd.DataFrame(q, index=components, columns=domains).rename_axis("component").reset_index()
    q_rows.insert(0, "target", target)
    return predictions, component_metrics, q_rows, aggregate_metrics


def gate_target(row: pd.Series) -> tuple[bool, list[str]]:
    checks = {
        "aggregate gradient": row["gradient_gradient_p"] <= GATES["aggregate_gradient_p_max"],
        "deletion Spearman": row["spearman"] >= GATES["aggregate_deletion_spearman_min"],
        "deletion Spearman CI": row["spearman_ci_low"] > GATES["aggregate_deletion_spearman_ci_low_min"],
        "deletion sign accuracy": row["sign_accuracy"] >= GATES["aggregate_deletion_sign_accuracy_min"],
        "deletion RMSE improvement": row["rmse_improvement"] >= GATES["aggregate_deletion_rmse_improvement_min"],
        "deletion RMSE improvement CI": (
            row["rmse_improvement_ci_low"] > GATES["aggregate_deletion_rmse_improvement_ci_low_min"]
        ),
        "component local-gradient support": (
            row["gradient_bh_q_lt_0p1_fraction"] >= GATES["component_gradient_bh_fraction_min"]
        ),
        "component deletion-rank sign": (
            row["positive_deletion_spearman_fraction"] >= GATES["component_positive_deletion_spearman_fraction_min"]
        ),
        "component median deletion rank": (
            row["median_deletion_spearman"] >= GATES["component_median_deletion_spearman_min"]
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return not failures, failures


def write_plots(
    predictions: pd.DataFrame, component_metrics: pd.DataFrame, q_rows: pd.DataFrame, output_dir: Path
) -> None:
    aggregate = predictions[predictions["component"].eq("__aggregate__")]
    fig = px.scatter(
        aggregate,
        x="observed_delta_bpb",
        y="predicted_delta_bpb",
        color="target",
        hover_name="deleted_domain",
        facet_col="target",
        title="Antithetic local gradients predicting held-out full-domain deletion effects",
    )
    for target, rows in aggregate.groupby("target"):
        lo = float(min(rows["observed_delta_bpb"].min(), rows["predicted_delta_bpb"].min()))
        hi = float(max(rows["observed_delta_bpb"].max(), rows["predicted_delta_bpb"].max()))
        fig.add_trace(
            go.Scatter(
                x=[lo, hi], y=[lo, hi], mode="lines", line={"dash": "dash", "color": "#60717c"}, showlegend=False
            ),
            row=1,
            col=1 if target == aggregate["target"].unique()[0] else 2,
        )
    fig.update_layout(height=700, width=1450)
    fig.write_html(output_dir / "aggregate_deletion_transfer.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    metric_fig = px.scatter(
        component_metrics,
        x="alpha_gradient_over_control_sd",
        y="spearman",
        color="target",
        symbol="component_family",
        hover_name="component",
        title="Local gradient strength versus finite-deletion transfer by benchmark component",
    )
    metric_fig.add_hline(y=0.0, line_dash="dash", line_color="#60717c")
    metric_fig.update_layout(height=800, width=1450)
    metric_fig.write_html(
        output_dir / "component_identification_vs_transfer.html", include_plotlyjs="cdn", config=PLOT_CONFIG
    )

    for target, rows in q_rows.groupby("target"):
        matrix = rows.set_index("component").drop(columns="target")
        zmax = float(np.nanpercentile(np.abs(matrix.to_numpy(dtype=float)), 98))
        heatmap = px.imshow(
            matrix,
            color_continuous_scale="RdYlGn_r",
            zmin=-zmax,
            zmax=zmax,
            aspect="auto",
            title=f"{target}: intervention-identified local bucket-transfer map",
        )
        heatmap.update_layout(height=max(700, 18 * len(matrix)), width=1700, margin={"l": 400, "b": 300})
        heatmap.write_html(
            output_dir / f"{target}_component_transfer_map.html", include_plotlyjs="cdn", config=PLOT_CONFIG
        )


def write_report(
    aggregate_metrics: pd.DataFrame, component_metrics: pd.DataFrame, decision: str, output_dir: Path
) -> None:
    lines = [
        "# Intervention-Identified Benchmark Transfer Diagnostic",
        "",
        f"Candidate: `{CANDIDATE_ID}`",
        "",
        f"Decision: **{decision}**",
        "",
        "The 78 central antithetic log tilts identify a local simplex-tangent transfer map. The 39 full "
        "domain deletions are a different intervention class and are not used to estimate or tune that map.",
        "",
        "The in-sample directional fit is nearly saturated by construction (39 directions for a rank-38 "
        "tangent space) and is explicitly excluded from the decision.",
        "",
        "## Aggregate Gates",
        "",
        aggregate_metrics.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Component Summary",
        "",
        component_metrics.groupby("target")
        .agg(
            components=("component", "count"),
            median_gradient_snr=("alpha_gradient_over_control_sd", "median"),
            significant_gradient_fraction=("gradient_bh_q", lambda values: float(np.mean(values < 0.1))),
            median_deletion_spearman=("spearman", "median"),
            positive_deletion_spearman_fraction=("spearman", lambda values: float(np.mean(values > 0.0))),
            median_rmse_improvement=("rmse_improvement", "median"),
        )
        .reset_index()
        .to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Interpretation Boundary",
        "",
        "A pass would license fitting a bounded aggregate-production response with the transfer map frozen. "
        "It would not establish a temporal mechanism or promote a full surrogate. A failure blocks this route "
        "before any new nonlinear response is fit.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def evaluate(output_dir: Path) -> None:
    frozen = verify_protocol(output_dir)
    manifest, manifest_domains = load_manifest()
    geometry = pctrl.load_geometry()
    if set(manifest_domains) != set(geometry.domains):
        raise ValueError("Manifest and analytic geometry domain order disagree")
    domains = geometry.domains

    bundles = {
        "uncheatable": load_uncheatable(manifest),
        "table9": load_table9(manifest),
    }
    all_predictions = []
    all_component_metrics = []
    all_q = []
    aggregate_rows = []
    for target, bundle in bundles.items():
        frame, controls, components, weights, families, _ = bundle
        predictions, component_metrics, q_rows, aggregate_metrics = evaluate_target(
            target=target,
            frame=frame,
            controls=controls,
            components=components,
            weights=weights,
            component_families=families,
            domains=domains,
            geometry=geometry,
        )
        all_predictions.append(predictions)
        all_component_metrics.append(component_metrics)
        all_q.append(q_rows)
        aggregate_rows.append(aggregate_metrics)

    predictions = pd.concat(all_predictions, ignore_index=True)
    component_metrics = pd.concat(all_component_metrics, ignore_index=True)
    q_rows = pd.concat(all_q, ignore_index=True)
    aggregate_metrics = pd.DataFrame(aggregate_rows)
    passes = []
    failures = []
    for _index, row in aggregate_metrics.iterrows():
        passed, target_failures = gate_target(row)
        passes.append(passed)
        failures.append("; ".join(target_failures))
    aggregate_metrics["passes_frozen_gate"] = passes
    aggregate_metrics["gate_failures"] = failures
    decision = (
        "PASS: bounded aggregate-production model licensed" if all(passes) else "FAIL: identification route blocked"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "deletion_predictions.csv", index=False)
    component_metrics.to_csv(output_dir / "component_metrics.csv", index=False)
    q_rows.to_csv(output_dir / "component_transfer_map.csv", index=False)
    aggregate_metrics.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    preflight = preflight_payload()
    (output_dir / "coverage.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    decision_payload = {
        "candidate_id": CANDIDATE_ID,
        "protocol_sha256": frozen["protocol_sha256"],
        "decision": decision,
        "targets_passed": {row["target"]: bool(row["passes_frozen_gate"]) for _, row in aggregate_metrics.iterrows()},
        "gate_failures": {row["target"]: row["gate_failures"] for _, row in aggregate_metrics.iterrows()},
    }
    (output_dir / "decision.json").write_text(json.dumps(decision_payload, indent=2, sort_keys=True) + "\n")
    write_plots(predictions, component_metrics, q_rows, output_dir)
    write_report(aggregate_metrics, component_metrics, decision, output_dir)
    print(json.dumps(decision_payload, indent=2, sort_keys=True))
    print(aggregate_metrics.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "freeze", "evaluate"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "preflight":
        print(json.dumps(preflight_payload(), indent=2, sort_keys=True))
        return
    if args.mode == "freeze":
        freeze_protocol(args.output_dir)
        return
    evaluate(args.output_dir)


if __name__ == "__main__":
    main()
