# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Materialize a frozen adversarial 3e18 surrogate stress panel.

The panel compares two model specifications frozen before observing any new
outcomes: the inverse-deficit log-link baseline and the early-family asymmetric
deficit candidate. It samples outside the 280-row fit support, retains policies
that at least one frozen model predicts at or beyond the fit-swarm frontier,
and balances consensus with model-specific challenges. Historical 3e18 rows
are used only for coordinate deduplication; their targets never enter proposal
generation, scoring, or selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import plotly.express as px

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_deficit_output_link_20260716 as output_link,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as coverage,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_deficit_response_20260716 as deficit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/delphi_3e18_adversarial_stress_panel_20260716"
DEFAULT_GCS_OUTPUT_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_3e18_adversarial_stress_panel_20260716"
)
DEFICIT_SOURCE_METRICS = SCRIPT_DIR / "reference_outputs/hierarchical_deficit_response_20260716/metrics.csv"
TARGETS = ("uncheatable", "table9")
BASELINE_MODEL_ID = "inverse_deficit_log_link"
CHALLENGER_MODEL_ID = "early_family_asymmetric"
MODELS = (BASELINE_MODEL_ID, CHALLENGER_MODEL_ID)
SELECTION_STRATA = ("baseline_ranked", "challenger_ranked", "high_disagreement")
POLICY_QUOTAS = {"single_phase_tied": 6, "two_phase": 54}
BASE_TIED_SAMPLES = 8_000
BASE_TWO_PHASE_SAMPLES = 24_000
FRONTIER_NEIGHBOR_SAMPLES = 4_000
ELITES_PER_MODEL_POLICY = 20
MUTATIONS_PER_ELITE = 16
MAX_SIMULATED_EPOCH = 64.0
MIN_FIT_POLICY_TV = 0.02
MIN_HELDOUT_POLICY_TV = 0.005
MIN_SELECTED_POLICY_TV = 0.004
PREDICTION_BATCH = 1024
RANDOM_SEED = 20260716
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class ModelBundle:
    model_id: str
    model: Any


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def phase_fractions(dataset: Any) -> tuple[float, float]:
    alpha0, alpha1 = observatory.phase_fractions(dataset)
    return float(alpha0), float(alpha1)


def fit_models(
    dataset: Any,
    target: str,
) -> list[ModelBundle]:
    dataset_id = (
        coverage.DatasetId.DELPHI_3E18_UNCHEATABLE if target == "uncheatable" else coverage.DatasetId.DELPHI_3E18_TABLE9
    )
    family_dataset = coverage.load_dataset(dataset_id)
    if tuple(dataset.domain_names) != tuple(family_dataset.domains):
        raise ValueError(f"Domain ordering differs for {target}")
    if not np.allclose(dataset.weights, family_dataset.weights, atol=1e-12, rtol=0.0):
        raise ValueError(f"Fit weights differ for {target}")

    source_metrics = pd.read_csv(DEFICIT_SOURCE_METRICS)
    indices = np.arange(dataset.n)
    baseline_deficit = output_link.selected_deficit_config(
        dataset_id,
        deficit.Variant.POWER_DEFICIT_HYBRID_REPLAY,
        source_metrics,
    )
    challenger_deficit = output_link.selected_deficit_config(
        dataset_id,
        deficit.Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_ASYMMETRIC,
        source_metrics,
    )
    if target == "uncheatable":
        baseline_link = output_link.LinkConfig(output_link.Link.LOG_EXCESS, 0.9, 0.001)
        challenger_link = output_link.LinkConfig(output_link.Link.IDENTITY, 0.0, 0.001)
    else:
        baseline_link = output_link.LinkConfig(output_link.Link.LOG_EXCESS, 0.75, 0.001)
        challenger_link = output_link.LinkConfig(output_link.Link.LOG_EXCESS, 0.75, 0.01)

    bundles = [
        ModelBundle(
            model_id=BASELINE_MODEL_ID,
            model=output_link.fit_model(family_dataset, baseline_deficit, baseline_link, indices),
        ),
        ModelBundle(
            model_id=CHALLENGER_MODEL_ID,
            model=output_link.fit_model(family_dataset, challenger_deficit, challenger_link, indices),
        ),
    ]
    for bundle in bundles:
        print(f"Fitted frozen {target}/{bundle.model_id}", flush=True)
    return bundles


def load_existing_heldout_weights(reference: Any) -> tuple[int, np.ndarray]:
    """Load completed heldout policy coordinates without reading their targets."""
    columns = [
        "training_state",
        "checkpoint_declared_complete",
        "policy_class",
        "phase_0_fraction",
        "phase_0_weights_json",
        "phase_1_weights_json",
    ]
    frame = pd.read_csv(observatory.DELPHI_3E18_HELDOUTS, usecols=columns)
    complete = (frame["training_state"] == "finished") & (frame["checkpoint_declared_complete"] == 1)
    frame = frame.loc[complete].reset_index(drop=True)
    domains = list(reference.domain_names)

    def parse_weights(value: str) -> list[float]:
        weights = json.loads(value)
        return [float(weights[domain]) for domain in domains]

    phase0 = np.asarray([parse_weights(value) for value in frame["phase_0_weights_json"]], dtype=float)
    phase1 = np.asarray([parse_weights(value) for value in frame["phase_1_weights_json"]], dtype=float)
    weights = np.stack([phase0, phase1], axis=1)
    if len(frame) != 364 or not np.allclose(weights.sum(axis=2), 1.0):
        raise ValueError(f"Expected 364 completed normalized 3e18 heldout coordinates, found {len(frame)}")
    heldout_alpha0 = frame["phase_0_fraction"].to_numpy(dtype=float)
    fit_alpha0, _fit_alpha1 = phase_fractions(reference)
    mismatched_split = ~np.isclose(heldout_alpha0, fit_alpha0, atol=1e-12)
    if (frame.loc[mismatched_split, "policy_class"] != "single_phase_tied").any():
        raise ValueError("A phase-varying heldout uses a different phase split from the fit swarm")
    return len(frame), weights


def predict_in_batches(bundle: ModelBundle, weights: np.ndarray) -> np.ndarray:
    parts = []
    for start in range(0, len(weights), PREDICTION_BATCH):
        batch = weights[start : start + PREDICTION_BATCH]
        parts.append(bundle.model.predict(batch))
    prediction = np.concatenate(parts)
    if prediction.shape != (len(weights),) or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid predictions for {bundle.model_id}")
    return prediction


def structured_base_pool(dataset: Any, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    natural = observatory.natural_weights(dataset, phase_fractions(dataset)[0])
    log_natural = np.log(np.clip(natural, 1e-12, 1.0))
    m = dataset.m

    tied_sigma = np.exp(rng.uniform(np.log(0.08), np.log(2.5), size=BASE_TIED_SAMPLES))
    tied_logits = log_natural + rng.normal(size=(BASE_TIED_SAMPLES, m)) * tied_sigma[:, None]
    tied = softmax_rows(tied_logits)
    tied_weights = np.stack([tied, tied], axis=1)

    alpha0, alpha1 = phase_fractions(dataset)
    aggregate_sigma = np.exp(rng.uniform(np.log(0.08), np.log(2.5), size=BASE_TWO_PHASE_SAMPLES))
    phase_sigma = np.exp(rng.uniform(np.log(0.02), np.log(1.5), size=BASE_TWO_PHASE_SAMPLES))
    shared = rng.normal(size=(BASE_TWO_PHASE_SAMPLES, m)) * aggregate_sigma[:, None]
    contrast = rng.normal(size=(BASE_TWO_PHASE_SAMPLES, m)) * phase_sigma[:, None]
    phase0 = softmax_rows(log_natural + shared + alpha1 * contrast)
    phase1 = softmax_rows(log_natural + shared - alpha0 * contrast)
    two_phase = np.stack([phase0, phase1], axis=1)

    top_fit = np.argsort(dataset.y)[:20]
    neighbor_weights = []
    for index in np.resize(top_fit, FRONTIER_NEIGHBOR_SAMPLES):
        base = np.clip(dataset.weights[index], 1e-12, 1.0)
        sigma = float(np.exp(rng.uniform(np.log(0.02), np.log(0.5))))
        perturbed = softmax_rows(np.log(base) + rng.normal(size=base.shape) * sigma)
        neighbor_weights.append(perturbed)
    neighbors = np.asarray(neighbor_weights, dtype=float)
    origins = np.concatenate(
        [
            np.repeat("structured_tied", len(tied_weights)),
            np.repeat("structured_two_phase", len(two_phase)),
            np.repeat("fit_frontier_neighbor", len(neighbors)),
        ]
    )
    weights = np.concatenate([tied_weights, two_phase, neighbors])
    return weights, origins


def policy_types(weights: np.ndarray) -> np.ndarray:
    tied = np.max(np.abs(weights[:, 0] - weights[:, 1]), axis=1) <= 1e-10
    return np.where(tied, "single_phase_tied", "two_phase")


def max_epochs(dataset: Any, weights: np.ndarray) -> np.ndarray:
    exposure = weights[:, 0] * dataset.c0[None, :] + weights[:, 1] * dataset.c1[None, :]
    return exposure.max(axis=1)


def elite_mutations(
    base_weights: np.ndarray,
    base_predictions: np.ndarray,
    model_ids: tuple[str, ...],
    frontier: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    types = policy_types(base_weights)
    mutations: list[np.ndarray] = []
    origins: list[str] = []
    sigmas = np.asarray([0.02, 0.05, 0.1, 0.2], dtype=float)
    for model_index, model_id in enumerate(model_ids):
        for policy_type in POLICY_QUOTAS:
            eligible = np.flatnonzero(types == policy_type)
            ordering = eligible[np.argsort(base_predictions[eligible, model_index])]
            frontier_ordering = ordering[base_predictions[ordering, model_index] <= frontier]
            elites = frontier_ordering[:ELITES_PER_MODEL_POLICY]
            if len(elites) < ELITES_PER_MODEL_POLICY:
                elites = ordering[:ELITES_PER_MODEL_POLICY]
            for elite_index in elites:
                base = np.clip(base_weights[elite_index], 1e-12, 1.0)
                for mutation_index in range(MUTATIONS_PER_ELITE):
                    sigma = sigmas[mutation_index % len(sigmas)]
                    if policy_type == "single_phase_tied":
                        logits = np.log(base[0]) + rng.normal(size=base.shape[1]) * sigma
                        tied = softmax_rows(logits[None, :])[0]
                        candidate = np.stack([tied, tied])
                    else:
                        candidate = softmax_rows(np.log(base) + rng.normal(size=base.shape) * sigma)
                    mutations.append(candidate)
                    origins.append(f"elite_mutation:{model_id}")
    return np.asarray(mutations, dtype=float), np.asarray(origins, dtype=object)


def all_predictions(bundles: list[ModelBundle], weights: np.ndarray) -> np.ndarray:
    return np.column_stack([predict_in_batches(bundle, weights) for bundle in bundles])


def weighted_policy_tv(left: np.ndarray, right: np.ndarray, alpha0: float, alpha1: float) -> np.ndarray:
    return 0.5 * (
        alpha0 * np.abs(left[..., 0, :] - right[..., 0, :]).sum(axis=-1)
        + alpha1 * np.abs(left[..., 1, :] - right[..., 1, :]).sum(axis=-1)
    )


def minimum_reference_distance(
    candidates: np.ndarray,
    reference: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> np.ndarray:
    output = np.full(len(candidates), np.inf, dtype=float)
    for start in range(0, len(candidates), 128):
        batch = candidates[start : start + 128]
        distance = weighted_policy_tv(batch[:, None], reference[None, :], alpha0, alpha1)
        output[start : start + len(batch)] = distance.min(axis=1)
    return output


def categorical_kl(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    values = np.clip(left, 1e-12, 1.0)
    reference = np.clip(right, 1e-12, 1.0)
    return np.sum(values * (np.log(values) - np.log(reference)), axis=-1)


def normalized_rank(values: np.ndarray, *, larger_is_better: bool) -> np.ndarray:
    order = np.argsort(values if larger_is_better else -values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=True)
    return ranks


def proposal_rows(
    *,
    target: str,
    dataset: Any,
    bundles: list[ModelBundle],
    weights: np.ndarray,
    origins: np.ndarray,
    predictions: np.ndarray,
    heldout_weights: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    model_ids = tuple(bundle.model_id for bundle in bundles)
    frontier = float(np.min(dataset.y))
    types = policy_types(weights)
    proposal_models: list[set[str]] = [set() for _ in range(len(weights))]
    for model_index, model_id in enumerate(model_ids):
        for policy_type in POLICY_QUOTAS:
            eligible = np.flatnonzero((types == policy_type) & (predictions[:, model_index] <= frontier))
            for candidate_index in eligible:
                proposal_models[candidate_index].add(model_id)
    proposed = np.asarray([bool(models) for models in proposal_models])
    weights = weights[proposed]
    origins = origins[proposed]
    predictions = predictions[proposed]
    types = types[proposed]
    proposal_models = [models for models, keep in zip(proposal_models, proposed, strict=True) if keep]

    alpha0, alpha1 = phase_fractions(dataset)
    fit_distance = minimum_reference_distance(weights, dataset.weights, alpha0, alpha1)
    heldout_distance = minimum_reference_distance(weights, heldout_weights, alpha0, alpha1)
    epochs = max_epochs(dataset, weights)
    eligible = (
        (fit_distance >= MIN_FIT_POLICY_TV)
        & (heldout_distance >= MIN_HELDOUT_POLICY_TV)
        & (epochs <= MAX_SIMULATED_EPOCH)
    )
    weights = weights[eligible]
    origins = origins[eligible]
    predictions = predictions[eligible]
    types = types[eligible]
    fit_distance = fit_distance[eligible]
    heldout_distance = heldout_distance[eligible]
    epochs = epochs[eligible]
    proposal_models = [models for models, keep in zip(proposal_models, eligible, strict=True) if keep]

    natural = observatory.natural_weights(dataset, alpha0)
    aggregate = alpha0 * weights[:, 0] + alpha1 * weights[:, 1]
    aggregate_kl = categorical_kl(aggregate, natural)
    phase_information = alpha0 * categorical_kl(weights[:, 0], aggregate) + alpha1 * categorical_kl(
        weights[:, 1], aggregate
    )
    phase_tv = 0.5 * np.abs(weights[:, 0] - weights[:, 1]).sum(axis=1)
    prediction_min = predictions.min(axis=1)
    prediction_max = predictions.max(axis=1)
    prediction_median = np.median(predictions, axis=1)
    prediction_std = predictions.std(axis=1)
    consensus = (predictions <= frontier).sum(axis=1)
    baseline_frontier = predictions[:, model_ids.index(BASELINE_MODEL_ID)] <= frontier
    challenger_frontier = predictions[:, model_ids.index(CHALLENGER_MODEL_ID)] <= frontier
    frontier_relation = np.select(
        [baseline_frontier & challenger_frontier, baseline_frontier, challenger_frontier],
        ["consensus_frontier", "baseline_only_frontier", "challenger_only_frontier"],
        default="not_frontier",
    )
    if np.any(frontier_relation == "not_frontier"):
        raise RuntimeError("Proposal filtering retained a candidate with no frontier proposer")
    proposer_prediction = np.asarray(
        [
            min(predictions[index, model_ids.index(model)] for model in models)
            for index, models in enumerate(proposal_models)
        ]
    )
    rows = pd.DataFrame(
        {
            "target": target,
            "policy_class": types,
            "origin": origins,
            "proposal_models": [",".join(sorted(models)) for models in proposal_models],
            "frontier_bpb": frontier,
            "proposer_prediction": proposer_prediction,
            "proposer_margin_below_frontier": frontier - proposer_prediction,
            "prediction_min": prediction_min,
            "prediction_median": prediction_median,
            "prediction_max": prediction_max,
            "prediction_std": prediction_std,
            "prediction_range": prediction_max - prediction_min,
            "frontier_consensus_count": consensus,
            "frontier_relation": frontier_relation,
            "min_fit_policy_tv": fit_distance,
            "min_existing_heldout_policy_tv": heldout_distance,
            "aggregate_kl_to_proportional": aggregate_kl,
            "phase_information_kl": phase_information,
            "phase_tv": phase_tv,
            "max_simulated_epoch": epochs,
            "max_weight": weights.max(axis=(1, 2)),
            "near_zero_weight_count": (weights < 1e-5).sum(axis=(1, 2)),
        }
    )
    for model_index, model_id in enumerate(model_ids):
        rows[f"predicted_{model_id}"] = predictions[:, model_index]
    rows["selection_utility"] = (
        0.30 * normalized_rank(rows["proposer_margin_below_frontier"].to_numpy(), larger_is_better=True)
        + 0.30 * normalized_rank(rows["prediction_range"].to_numpy(), larger_is_better=True)
        + 0.20 * normalized_rank(rows["min_fit_policy_tv"].to_numpy(), larger_is_better=True)
        + 0.20 * normalized_rank(rows["frontier_consensus_count"].to_numpy(), larger_is_better=True)
    )
    return rows, weights


def candidate_distance(candidate: np.ndarray, selected: list[np.ndarray], alpha0: float, alpha1: float) -> float:
    if not selected:
        return np.inf
    references = np.asarray(selected, dtype=float)
    return float(weighted_policy_tv(candidate[None, None], references[None, :], alpha0, alpha1).min())


def select_balanced_panel(
    rows: pd.DataFrame,
    weights: np.ndarray,
    dataset: Any,
    excluded_weights: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    alpha0, alpha1 = phase_fractions(dataset)
    selected_indices: list[int] = []
    distance_references = [weight for weight in excluded_weights]
    selected_strata: dict[int, str] = {}
    for policy_class, quota in POLICY_QUOTAS.items():
        policy_indices = rows.index[rows["policy_class"].eq(policy_class)].to_numpy(dtype=int)
        per_stratum = quota // len(SELECTION_STRATA)
        if per_stratum * len(SELECTION_STRATA) != quota:
            raise ValueError(f"Policy quota {quota} is not divisible across selection strata")
        for _ in range(per_stratum):
            for stratum in SELECTION_STRATA:
                candidates = [index for index in policy_indices if index not in selected_indices]
                eligible = [
                    index
                    for index in candidates
                    if candidate_distance(weights[index], distance_references, alpha0, alpha1) >= MIN_SELECTED_POLICY_TV
                ]
                if not eligible:
                    raise RuntimeError(
                        f"Could not fill {policy_class}/{stratum}; selected "
                        f"{sum(value == stratum for value in selected_strata.values())}/{per_stratum}"
                    )
                if stratum == "baseline_ranked":
                    criterion = rows.loc[eligible, f"predicted_{BASELINE_MODEL_ID}"].to_numpy(dtype=float)
                    criterion_score = normalized_rank(criterion, larger_is_better=False)
                elif stratum == "challenger_ranked":
                    criterion = rows.loc[eligible, f"predicted_{CHALLENGER_MODEL_ID}"].to_numpy(dtype=float)
                    criterion_score = normalized_rank(criterion, larger_is_better=False)
                else:
                    criterion = rows.loc[eligible, "prediction_range"].to_numpy(dtype=float)
                    criterion_score = normalized_rank(criterion, larger_is_better=True)
                distances = np.asarray(
                    [candidate_distance(weights[index], distance_references, alpha0, alpha1) for index in eligible]
                )
                distance_score = normalized_rank(distances, larger_is_better=True)
                utility_score = rows.loc[eligible, "selection_utility"].to_numpy(dtype=float)
                combined_score = 0.70 * criterion_score + 0.15 * utility_score + 0.15 * distance_score
                best_index = eligible[int(np.argmax(combined_score))]
                selected_indices.append(best_index)
                selected_strata[best_index] = stratum
                distance_references.append(weights[best_index])

    selected = rows.loc[selected_indices].copy().reset_index(drop=True)
    selected["selection_stratum"] = [selected_strata[index] for index in selected_indices]
    selected_weights_array = weights[np.asarray(selected_indices, dtype=int)]
    selected["selected_order"] = np.arange(len(selected))
    policy_tags = np.where(selected["policy_class"].eq("single_phase_tied"), "1p", "2p")
    selected["candidate_id"] = [
        f"adv3e18c_{selected.at[index, 'target']}_{policy_tag}_{index:03d}"
        for index, policy_tag in enumerate(policy_tags)
    ]
    return selected, selected_weights_array


def write_mixtures(output_dir: Path, manifest: pd.DataFrame, weights: np.ndarray, dataset: Any) -> None:
    mixtures_dir = output_dir / "mixtures"
    mixtures_dir.mkdir(parents=True, exist_ok=True)
    alpha0, _alpha1 = phase_fractions(dataset)
    natural = observatory.natural_weights(dataset, alpha0)
    for row, candidate_weights in zip(manifest.to_dict(orient="records"), weights, strict=True):
        frame = pd.DataFrame(
            {
                "domain": dataset.domain_names,
                "natural_weight": natural,
                "phase_0_weight": candidate_weights[0],
                "phase_1_weight": candidate_weights[1],
                "simulated_epochs": candidate_weights[0] * dataset.c0 + candidate_weights[1] * dataset.c1,
            }
        )
        frame.to_csv(mixtures_dir / f"{row['candidate_id']}.csv", index=False)


def write_launcher_source_panel(
    output_dir: Path,
    manifest: pd.DataFrame,
    weights: np.ndarray,
    dataset: Any,
) -> tuple[Path, str]:
    rows = []
    for row, candidate_weights in zip(manifest.to_dict(orient="records"), weights, strict=True):
        source_row = {
            "candidate_id": row["candidate_id"],
            "target": row["target"],
            "policy_class": row["policy_class"],
            "selection_stratum": row["selection_stratum"],
            "proposal_models": row["proposal_models"],
            "frontier_bpb": row["frontier_bpb"],
            f"predicted_{BASELINE_MODEL_ID}": row[f"predicted_{BASELINE_MODEL_ID}"],
            f"predicted_{CHALLENGER_MODEL_ID}": row[f"predicted_{CHALLENGER_MODEL_ID}"],
            "min_fit_policy_tv": row["min_fit_policy_tv"],
            "min_existing_heldout_policy_tv": row["min_existing_heldout_policy_tv"],
            "max_simulated_epoch": row["max_simulated_epoch"],
        }
        for phase_index in (0, 1):
            for domain, weight in zip(dataset.domain_names, candidate_weights[phase_index], strict=True):
                source_row[f"phase_{phase_index}_{domain}"] = float(weight)
        rows.append(source_row)
    source_path = output_dir / "launcher_source_panel.csv"
    pd.DataFrame(rows).to_csv(source_path, index=False)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return source_path, source_sha256


def upload_artifact(local_path: Path, remote_path: str) -> None:
    with local_path.open("rb") as source, fsspec.open(remote_path, "wb") as destination:
        destination.write(source.read())


def render_diagnostics(manifest: pd.DataFrame, output_dir: Path) -> None:
    figure = px.scatter(
        manifest,
        x="min_fit_policy_tv",
        y="proposer_margin_below_frontier",
        color="frontier_consensus_count",
        symbol="policy_class",
        size="max_simulated_epoch",
        facet_col="target",
        hover_name="candidate_id",
        hover_data=[
            "proposal_models",
            "selection_stratum",
            "prediction_range",
            "phase_information_kl",
            "aggregate_kl_to_proportional",
            "min_existing_heldout_policy_tv",
        ],
        color_continuous_scale="RdYlGn_r",
        title="Frozen Delphi 3e18 adversarial surrogate stress panel",
    )
    figure.update_xaxes(title_text="minimum policy TV from 280-row fit support")
    figure.update_yaxes(title_text="proposer-predicted improvement over fit frontier (BPB)")
    figure.update_layout(width=1500, height=760, margin={"l": 70, "r": 30, "t": 100, "b": 70})
    figure.write_html(output_dir / "adversarial_panel_diagnostics.html", include_plotlyjs=True, config=PLOT_CONFIG)


def write_report(
    pool_rows: pd.DataFrame,
    manifest: pd.DataFrame,
    output_dir: Path,
) -> None:
    composition = (
        manifest.groupby(["target", "policy_class", "selection_stratum"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    lines = [
        "# Frozen Delphi 3e18 adversarial surrogate stress panel",
        "",
        "## Purpose",
        "",
        (
            "Compare two frozen surrogate specifications on fresh out-of-support policies that at least one model "
            "predicts at or beyond the 280-row fit-swarm frontier. The panel is a confirmatory stress test for the "
            "current model pair; it becomes development evidence if its outcomes are later used to change a model."
        ),
        "",
        "## Selection protocol",
        "",
        ("- Fit both frozen models on the complete two-phase Delphi 3e18 280-row swarm only."),
        (
            f"- `{BASELINE_MODEL_ID}` uses inverse-power bucket/family deficits, hybrid literal replay, and the "
            "fit-OOF-selected log-reducible BPB link."
        ),
        (
            f"- `{CHALLENGER_MODEL_ID}` adds three phase-0 semantic-family deficit channels and bounded asymmetric "
            "surplus credit. Its Uncheatable identity link and Table-9 log-reducible link were frozen using the old "
            "3e18 development archive before this panel was generated."
        ),
        (
            "- Linear and OLMix loglinear are excluded from proposal generation because their known weak frontier "
            "calibration would spend stress-test budget on uninformative candidates."
        ),
        (
            "- Generate phase-tied and phase-varying structured log-tilts, plus perturbations around observed "
            "fit-frontier rows."
        ),
        "- Refine each model's best frontier-level samples with deterministic local mutations.",
        (
            "- Reject policies within 0.02 policy TV of the fit panel, within 0.005 of any existing 3e18 heldout, "
            "or above 64 simulated epochs."
        ),
        (
            "- Select 60 rows per target: 6 phase-tied and 54 phase-varying. Within each policy class, allocate "
            "one third each to baseline-ranked, challenger-ranked, and maximum-disagreement candidates."
        ),
        (
            "- Rank, model-disagreement, and geometric-diversity terms are rank-normalized before combination so "
            "the named stratum remains the dominant selection criterion."
        ),
        (
            "- The 1:9 tied-to-two-phase allocation reflects the narrow credible tied frontier while retaining "
            "controls for whether errors require phase variation."
        ),
        "- Existing heldout target values are never used for candidate scoring or selection.",
        (
            "- Candidates proposed for the two objectives are also deduplicated because every checkpoint yields "
            "both metrics."
        ),
        "",
        "## Composition",
        "",
        composition.to_markdown(index=False),
        "",
        "## Geometry and prediction ranges",
        "",
        manifest[
            [
                "target",
                "policy_class",
                "proposer_margin_below_frontier",
                "prediction_range",
                "min_fit_policy_tv",
                "min_existing_heldout_policy_tv",
                "max_simulated_epoch",
            ]
        ]
        .groupby(["target", "policy_class"])
        .agg(["min", "median", "max"])
        .to_markdown(floatfmt=".5f"),
        "",
        "## Leakage boundary",
        "",
        (
            "Historical 3e18 targets are never read by the selector. Historical coordinates are used only to enforce "
            "a minimum policy-TV distance. The old archive informed the already-frozen model/link choice, so this new "
            "panel can compare those frozen choices once; after inspection it becomes development evidence."
        ),
        "",
        f"Generated pool rows surviving frontier proposal before geometric filters: {len(pool_rows)}.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gcs-output-dir", default=DEFAULT_GCS_OUTPUT_DIR)
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    all_manifests = []
    all_pool_rows = []
    all_weights = []
    heldout_counts: dict[str, int] = {}
    cross_target_weights: np.ndarray | None = None
    for target in TARGETS:
        print(f"Building {target} proposals", flush=True)
        dataset = observatory.load_delphi_3e18_fit_dataset(target)
        if cross_target_weights is None:
            cross_target_weights = np.empty((0, 2, len(dataset.domain_names)), dtype=float)
        heldout_count, heldout_weights = load_existing_heldout_weights(dataset)
        heldout_counts[target] = heldout_count
        bundles = fit_models(dataset, target)
        base_weights, base_origins = structured_base_pool(dataset, rng)
        base_epoch_mask = max_epochs(dataset, base_weights) <= MAX_SIMULATED_EPOCH
        base_weights = base_weights[base_epoch_mask]
        base_origins = base_origins[base_epoch_mask]
        base_predictions = all_predictions(bundles, base_weights)
        mutations, mutation_origins = elite_mutations(
            base_weights,
            base_predictions,
            tuple(bundle.model_id for bundle in bundles),
            float(np.min(dataset.y)),
            rng,
        )
        combined_weights = np.concatenate([base_weights, mutations])
        combined_origins = np.concatenate([base_origins, mutation_origins])
        combined_predictions = all_predictions(bundles, combined_weights)
        pool_rows, pool_weights = proposal_rows(
            target=target,
            dataset=dataset,
            bundles=bundles,
            weights=combined_weights,
            origins=combined_origins,
            predictions=combined_predictions,
            heldout_weights=heldout_weights,
        )
        print(
            pool_rows.groupby(["policy_class", "frontier_relation"]).size().rename("rows").to_string(),
            flush=True,
        )
        manifest, selected_weights = select_balanced_panel(
            pool_rows,
            pool_weights,
            dataset,
            cross_target_weights,
        )
        cross_target_weights = np.concatenate([cross_target_weights, selected_weights])
        write_mixtures(args.output_dir, manifest, selected_weights, dataset)
        all_pool_rows.append(pool_rows)
        all_manifests.append(manifest)
        all_weights.append(selected_weights)

    pool = pd.concat(all_pool_rows, ignore_index=True)
    manifest = pd.concat(all_manifests, ignore_index=True)
    weights = np.concatenate(all_weights)
    manifest["selected_order"] = np.arange(len(manifest))
    expected_rows = len(TARGETS) * sum(POLICY_QUOTAS.values())
    if len(manifest) != expected_rows or manifest["candidate_id"].duplicated().any():
        raise ValueError(f"Expected {expected_rows} unique candidates, found {len(manifest)}")
    alpha0, alpha1 = phase_fractions(dataset)
    uncheatable_weights = weights[manifest["target"].eq("uncheatable").to_numpy()]
    table9_weights = weights[manifest["target"].eq("table9").to_numpy()]
    cross_target_min_tv = float(
        weighted_policy_tv(
            uncheatable_weights[:, None],
            table9_weights[None, :],
            alpha0,
            alpha1,
        ).min()
    )
    if cross_target_min_tv < MIN_SELECTED_POLICY_TV:
        raise ValueError(f"Cross-target candidates are only {cross_target_min_tv:.6f} policy TV apart")
    manifest_path = args.output_dir / "candidate_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    pool.to_csv(args.output_dir / "eligible_candidate_pool.csv", index=False)
    phase_rows = []
    domains = observatory.load_delphi_3e18_fit_dataset("uncheatable").domain_names
    for row, candidate_weights in zip(manifest.to_dict(orient="records"), weights, strict=True):
        for phase_index in (0, 1):
            for domain, weight in zip(domains, candidate_weights[phase_index], strict=True):
                phase_rows.append(
                    {
                        "candidate_id": row["candidate_id"],
                        "target": row["target"],
                        "phase": phase_index,
                        "domain": domain,
                        "weight": float(weight),
                    }
                )
    phase_frame = pd.DataFrame(phase_rows)
    phase_frame.to_csv(args.output_dir / "phase_weights.csv", index=False)
    source_panel_path, source_panel_sha256 = write_launcher_source_panel(args.output_dir, manifest, weights, dataset)
    gcs_source_panel = f"{args.gcs_output_dir}/source/launcher_source_panel-{source_panel_sha256[:16]}.csv"
    gcs_candidate_manifest = f"{args.gcs_output_dir}/source/candidate_manifest-{manifest_sha256[:16]}.csv"
    summary = {
        "panel_rows": len(manifest),
        "rows_per_target": manifest.groupby("target").size().to_dict(),
        "rows_per_policy": manifest.groupby("policy_class").size().to_dict(),
        "rows_per_stratum": manifest.groupby("selection_stratum").size().to_dict(),
        "fit_rows": 280,
        "existing_heldouts_deduplicated_against": heldout_counts,
        "models": list(MODELS),
        "random_seed": RANDOM_SEED,
        "selection_uses_existing_heldout_targets": False,
        "cross_target_min_policy_tv": cross_target_min_tv,
        "candidate_manifest_sha256": manifest_sha256,
        "launcher_source_panel": str(source_panel_path),
        "launcher_source_panel_sha256": source_panel_sha256,
        "gcs_launcher_source_panel": gcs_source_panel,
        "gcs_candidate_manifest": gcs_candidate_manifest,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    render_diagnostics(manifest, args.output_dir)
    write_report(pool, manifest, args.output_dir)
    if args.upload:
        upload_artifact(source_panel_path, gcs_source_panel)
        upload_artifact(manifest_path, gcs_candidate_manifest)
        upload_artifact(summary_path, f"{args.gcs_output_dir}/source/summary.json")
        upload_artifact(args.output_dir / "report.md", f"{args.gcs_output_dir}/source/report.md")
        upload_artifact(args.output_dir / "phase_weights.csv", f"{args.gcs_output_dir}/source/phase_weights.csv")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
