# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Select a reserve-aware Delphi prefix using only the canonical 300M panels.

The aggregate is the established one-phase OLMix Table-9 proposal. HPR then
changes only phase order at that aggregate, using the exact 280-row 300M
two-phase panel. Delphi outcomes are deliberately absent from every input path.

Every emitted mixture is on Levanter's 2,048-example mixture-block lattice.
This matters here: passing nominal continuous weights would silently train a
different coordinate after per-block integer truncation.

This script selects a conservative prefix, not a claimed endpoint optimum.
The selected contrast moves the 80%-of-training prefix by about 0.5% total
variation while reserving a larger compensating dose for phase 1. A subsequent
shared-prefix continuation search must validate whether that reserve is useful.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_hierarchical_phase_replay_validation_panel_3e18 as hpr_panel,
)

REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "reserve_aware_prefix_300m_20260820"
ANCHOR_PATH = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_one_phase_model_sweeps_300m_20260628"
    / "olmix_one_phase_cap4_delta0p01_kl0p075"
    / "proposed_mixture_weights.csv"
)
ANCHOR_SUMMARY_PATH = REFERENCE_OUTPUTS / "olmo_base_easy_one_phase_model_sweeps_300m_20260628" / "summary.json"
CANONICAL_TWO_PHASE_PATH = (
    REFERENCE_OUTPUTS
    / "two_phase_surrogate_collaborator_packet_20260721"
    / "data"
    / "canonical"
    / "300m_two_phase_fit.csv"
)

TARGET = "table9"
FIT_ROWS = 280
SELECTED_PHASE_INFORMATION_BUDGET = 5e-4
PHASE_INFORMATION_BUDGETS = (1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2)
PREFIX_TV_LIMIT = 0.006
FOLD_SEED = 0
MIXTURE_BLOCK_SIZE = 2048
EXPECTED_OLMIX_ROW_COUNTS = {"n_rows": 279, "n_signal_rows": 240, "n_deletion_rows": 39}
EXPECTED_OLMIX_PROPORTIONAL_REFERENCES = 11


@dataclass(frozen=True)
class PredictionAdapter:
    """Expose a uniform prediction method for established Observatory models."""

    function: Callable[[np.ndarray], np.ndarray]

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return self.function(weights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-fold-stability", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_counts(weights: np.ndarray) -> np.ndarray:
    """Mirror MixtureDataset's deterministic counts for one mixture block."""
    counts = np.asarray(weights * MIXTURE_BLOCK_SIZE, dtype=np.int64)
    largest = int(np.argmax(counts))
    counts[largest] += MIXTURE_BLOCK_SIZE - int(counts.sum())
    if int(counts.sum()) != MIXTURE_BLOCK_SIZE or int(counts.min()) < 0:
        raise ValueError("Invalid runtime mixture-block counts")
    return counts


def runtime_weights(weights: np.ndarray) -> np.ndarray:
    return runtime_counts(weights) / MIXTURE_BLOCK_SIZE


def project_fixed_aggregate(weights: np.ndarray, aggregate: np.ndarray, alpha0: float, alpha1: float) -> np.ndarray:
    """Project an 80/20 fixed-aggregate policy onto exact mixture-block counts."""
    if not np.isclose(alpha0, 0.8) or not np.isclose(alpha1, 0.2):
        raise ValueError(f"Integer fixed-aggregate projection requires 80/20 phases, found {alpha0}/{alpha1}")
    aggregate_counts = runtime_counts(aggregate)
    if not np.array_equal(aggregate_counts / MIXTURE_BLOCK_SIZE, aggregate):
        raise ValueError("Aggregate is not on the runtime mixture-block lattice")

    target_shift = aggregate_counts - MIXTURE_BLOCK_SIZE * weights[0]
    lower = np.ceil(-aggregate_counts / 4).astype(np.int64)
    upper = aggregate_counts.copy()
    shift = np.clip(np.rint(target_shift).astype(np.int64), lower, upper)

    while int(shift.sum()) != 0:
        if int(shift.sum()) > 0:
            valid = np.flatnonzero(shift > lower)
            candidates = shift[valid] - 1
        else:
            valid = np.flatnonzero(shift < upper)
            candidates = shift[valid] + 1
        if len(valid) == 0:
            raise ValueError("Could not balance projected fixed-aggregate shift")
        cost = (candidates - target_shift[valid]) ** 2 - (shift[valid] - target_shift[valid]) ** 2
        chosen = int(valid[int(np.argmin(cost))])
        shift[chosen] += -1 if int(shift.sum()) > 0 else 1

    phase0_counts = aggregate_counts - shift
    phase1_counts = aggregate_counts + 4 * shift
    projected = np.stack([phase0_counts, phase1_counts]) / MIXTURE_BLOCK_SIZE
    if not np.array_equal(runtime_weights(projected[0]), projected[0]):
        raise ValueError("Projected phase 0 is not runtime-exact")
    if not np.array_equal(runtime_weights(projected[1]), projected[1]):
        raise ValueError("Projected phase 1 is not runtime-exact")
    realized_aggregate = alpha0 * projected[0] + alpha1 * projected[1]
    if not np.allclose(realized_aggregate, aggregate, atol=1e-15):
        raise ValueError("Integer projection failed to preserve the aggregate")
    return projected


def load_anchor(dataset: Any) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    frame = pd.read_csv(ANCHOR_PATH)
    summary = json.loads(ANCHOR_SUMMARY_PATH.read_text())
    if summary["row_counts"] != EXPECTED_OLMIX_ROW_COUNTS:
        raise ValueError(f"Unexpected OLMix row counts: {summary['row_counts']}")
    if summary["proportional_reference_n"] != EXPECTED_OLMIX_PROPORTIONAL_REFERENCES:
        raise ValueError(f"Unexpected OLMix proportional reference count: {summary['proportional_reference_n']}")
    indexed = frame.set_index("domain")
    missing = set(dataset.domain_names) - set(indexed.index)
    if missing:
        raise ValueError(f"OLMix anchor is missing buckets: {sorted(missing)}")
    source_aggregate = np.asarray([indexed.loc[name, "aggregate_weight"] for name in dataset.domain_names], dtype=float)
    if not np.isclose(source_aggregate.sum(), 1.0, atol=1e-10):
        raise ValueError(f"OLMix aggregate sums to {source_aggregate.sum():.12f}, not one")
    aggregate = runtime_weights(source_aggregate)
    return frame, source_aggregate, aggregate, summary


def load_canonical_panel(reference: Any, target: str) -> Any:
    canonical = pd.read_csv(CANONICAL_TWO_PHASE_PATH)
    phase0 = canonical[[f"phase_0_weight::{name}" for name in reference.domain_names]].to_numpy(dtype=float)
    phase1 = canonical[[f"phase_1_weight::{name}" for name in reference.domain_names]].to_numpy(dtype=float)
    weights = np.stack([phase0, phase1], axis=1)
    if weights.shape != reference.weights.shape or not np.allclose(weights, reference.weights, atol=1e-12):
        raise ValueError("In-memory HPR coordinates do not match the hashed canonical-280 CSV")
    run_names = reference.frame["run_name"].astype(str).to_numpy()
    if not np.array_equal(run_names, canonical["row_id"].astype(str).to_numpy()):
        raise ValueError("Canonical packet rows do not align with the HPR panel metadata")
    frame = reference.frame.copy()
    frame[hpr_panel.TARGET_COLUMNS[target]] = canonical[hpr_panel.TARGET_COLUMNS[target]].to_numpy(dtype=float)
    return hpr_panel.make_raw_dataset(
        reference,
        frame,
        weights,
        target,
        f"300m_{target}",
    )


def fit_hpr(
    dataset: Any,
    target: str,
    indices: np.ndarray,
    config: Any,
    sweep: pd.DataFrame,
) -> hpr_panel.FittedPolicy:
    model = observatory.hierarchical_phase_replay_fit(dataset, indices, config)
    return hpr_panel.FittedPolicy(target, observatory.TWO_PHASE, dataset, config, model, sweep)


def candidate_metrics(
    weights: np.ndarray,
    aggregate: np.ndarray,
    dataset: Any,
    natural: np.ndarray,
) -> dict[str, float]:
    alpha0, alpha1 = observatory.phase_fractions(dataset)
    geometry = hpr_panel.candidate_geometry(weights, dataset, natural, dataset.weights)
    realized_aggregate = alpha0 * weights[0] + alpha1 * weights[1]
    return {
        "phase_information_kl": float(geometry["phase_information_kl"]),
        "phase_total_variation": float(geometry["phase_total_variation"]),
        "prefix_tv_to_anchor": float(0.5 * np.abs(weights[0] - aggregate).sum()),
        "continuation_tv_to_anchor": float(0.5 * np.abs(weights[1] - aggregate).sum()),
        "aggregate_tv_error": float(0.5 * np.abs(realized_aggregate - aggregate).sum()),
        "max_bucket_weight": float(geometry["max_bucket_weight"]),
        "max_simulated_epoch": float(geometry["max_simulated_epoch"]),
        "min_fit_policy_tv": float(geometry["min_fit_policy_tv"]),
    }


def established_models(dataset: Any, hpr_model: Any) -> dict[str, Any]:
    indices = np.arange(dataset.n)
    separate_l2, _sweep = observatory.select_separate_l2(dataset, observatory.TWO_PHASE)
    separate = observatory.separate_fit(dataset, indices, separate_l2, observatory.TWO_PHASE)
    compact_l2, _sweep = observatory.select_compact_l2(dataset, observatory.TWO_PHASE)
    compact = observatory.compact_fit(dataset, indices, compact_l2, observatory.TWO_PHASE)
    return {
        "hierarchical_phase_bucket_replay": hpr_model,
        "separate_heads": PredictionAdapter(
            lambda weights: observatory.separate_predict(
                separate,
                dataset,
                weights,
                observatory.TWO_PHASE,
            )
        ),
        "compact_retained_state": compact,
    }


def fold_stability(
    dataset: Any,
    aggregate: np.ndarray,
    config: Any,
    sweep: pd.DataFrame,
    reference_weights: np.ndarray,
) -> pd.DataFrame:
    alpha0, alpha1 = observatory.phase_fractions(dataset)
    reference_contrast = reference_weights[1] - reference_weights[0]
    rows: list[dict[str, float | int]] = []
    for fold, (train, _test) in enumerate(observatory.folds(dataset, FOLD_SEED)):
        fitted = fit_hpr(dataset, TARGET, train, config, sweep)
        tied = np.stack([aggregate, aggregate])
        tied_prediction = hpr_panel.scalar_prediction(fitted.model, tied)
        result = hpr_panel.optimize_fixed_aggregate(
            fitted,
            aggregate,
            SELECTED_PHASE_INFORMATION_BUDGET,
            alpha0,
            alpha1,
        )
        projected = project_fixed_aggregate(result.weights, aggregate, alpha0, alpha1)
        contrast = projected[1] - projected[0]
        contrast_norm = np.linalg.norm(contrast)
        reference_norm = np.linalg.norm(reference_contrast)
        cosine = float(np.dot(contrast, reference_contrast) / (contrast_norm * reference_norm))
        rows.append(
            {
                "fold": fold,
                "fit_rows": len(train),
                "predicted_gain_bpb": tied_prediction - hpr_panel.scalar_prediction(fitted.model, projected),
                "phase_total_variation": 0.5 * np.abs(contrast).sum(),
                "prefix_tv_to_anchor": 0.5 * np.abs(projected[0] - aggregate).sum(),
                "cosine_to_full_contrast": cosine,
                "contrast_tv_to_full": 0.5 * np.abs(contrast - reference_contrast).sum(),
            }
        )
    return pd.DataFrame(rows)


def bucket_frame(
    anchor_frame: pd.DataFrame,
    dataset: Any,
    source_aggregate: np.ndarray,
    aggregate: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    indexed = anchor_frame.set_index("domain")
    frame = pd.DataFrame(
        {
            "domain": dataset.domain_names,
            "proportional": [indexed.loc[name, "proportional"] for name in dataset.domain_names],
            "source_aggregate_weight": source_aggregate,
            "aggregate_weight": aggregate,
            "phase_0_weight": weights[0],
            "phase_1_weight": weights[1],
            "available_tokens": [indexed.loc[name, "available_tokens"] for name in dataset.domain_names],
            "aggregate_simulated_epochs": [indexed.loc[name, "simulated_epochs"] for name in dataset.domain_names],
        }
    )
    frame["phase_0_minus_aggregate"] = frame["phase_0_weight"] - frame["aggregate_weight"]
    frame["phase_1_minus_aggregate"] = frame["phase_1_weight"] - frame["aggregate_weight"]
    frame["phase_1_minus_phase_0"] = frame["phase_1_weight"] - frame["phase_0_weight"]
    frame["phase_0_materialized_epochs"] = weights[0] * np.asarray(dataset.c0, dtype=float)
    frame["phase_1_materialized_epochs"] = weights[1] * np.asarray(dataset.c1, dtype=float)
    return frame


def write_report(
    output_dir: Path,
    path_frame: pd.DataFrame,
    model_frame: pd.DataFrame,
    direction_frame: pd.DataFrame,
    stability: pd.DataFrame,
    buckets: pd.DataFrame,
    config: Any,
    uncheatable_guardrail: float,
    aggregate_snap_tv: float,
    olmix_summary: dict[str, Any],
) -> None:
    selected = path_frame.loc[path_frame["selected"]].iloc[0]
    late = buckets.nlargest(8, "phase_1_minus_phase_0")
    early = buckets.nsmallest(8, "phase_1_minus_phase_0")
    lines = [
        "# Canonical-300M reserve-aware prefix",
        "",
        "## Decision",
        "",
        (
            f"Use the OLMix Table-9 cap-4, KL-0.075 aggregate and the HPR fixed-aggregate contrast at "
            f"phase-information budget `{SELECTED_PHASE_INFORMATION_BUDGET:g}`. The resulting phase-0 prefix is "
            f"{selected['prefix_tv_to_anchor']:.6f} TV from the OLMix aggregate; phase 1 is "
            f"{selected['continuation_tv_to_anchor']:.6f} TV away. The aggregate is preserved to "
            f"{selected['aggregate_tv_error']:.2e} TV."
        ),
        "",
        "This is a prefix-selection rule, not a claim that the HPR endpoint gain is calibrated. All three established "
        "models agree on the sign and independently optimize toward similar contrasts, but they share exposure-based "
        "inductive biases and are not independent statistical evidence. The candidate remains far from empirical "
        "two-phase fit support. The shared-prefix experiment must measure the continuation response directly at 3e18.",
        "",
        "## Data boundary",
        "",
        f"- Aggregate source: `{ANCHOR_PATH.relative_to(REPO_ROOT)}`.",
        (
            f"  OLMix used {olmix_summary['row_counts']['n_rows']} coordinate rows "
            f"({olmix_summary['row_counts']['n_signal_rows']} signal and "
            f"{olmix_summary['row_counts']['n_deletion_rows']} deletion) plus "
            f"{olmix_summary['proportional_reference_n']} proportional repeats."
        ),
        (
            f"  Runtime block-count realization moves the nominal OLMix aggregate by {aggregate_snap_tv:.6f} TV; "
            "the reserve optimization and all emitted controls use that exactly realizable aggregate."
        ),
        f"- Phase-model source: `{CANONICAL_TWO_PHASE_PATH.relative_to(REPO_ROOT)}`, exactly 280 300M rows.",
        "- No Delphi outcome, heldout score, or 3e18 model fit is read by this script.",
        "- Table-9 selects the prefix. The 300M Uncheatable fit is a non-selecting harm guardrail.",
        "",
        "## Why this budget",
        "",
        f"The selected rung is the largest candidate-ladder reserve rung whose 80%-duration prefix remains "
        f"within {PREFIX_TV_LIMIT:.3f} TV of the OLMix aggregate. This limits damage to the empirically strong "
        "one-phase recipe "
        "while creating a four-times-larger compensating continuation displacement under the 80/20 split.",
        "The canonical panel has a local design gap: its median phase TV is about 0.50, while the selected contrast is "
        "0.026. The panel identifies a stable large-scale ordering direction, not the correct local reserve magnitude.",
        f"All phase weights are exact multiples of 1/{MIXTURE_BLOCK_SIZE}, matching the runtime mixture block.",
        "",
        "## Stability",
        "",
    ]
    if stability.empty:
        lines.append("Fold stability was skipped.")
    else:
        lines.extend(
            [
                (
                    f"Across five delete-fold refits, contrast cosine to the full fit ranges from "
                    f"{stability['cosine_to_full_contrast'].min():.4f} to "
                    f"{stability['cosine_to_full_contrast'].max():.4f}; prefix TV ranges from "
                    f"{stability['prefix_tv_to_anchor'].min():.6f} to "
                    f"{stability['prefix_tv_to_anchor'].max():.6f}."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "The established-model prediction table is diagnostic only:",
            "",
            model_frame.to_markdown(index=False, floatfmt=".6f"),
            "",
            "Independent optimization at the same phase-information budget:",
            "",
            direction_frame.to_markdown(index=False, floatfmt=".6f"),
            "",
            f"The 300M Uncheatable HPR guardrail predicts selected-minus-tied = {uncheatable_guardrail:+.6f} BPB "
            "(negative is better).",
            "",
            "## Reserve anatomy",
            "",
            "Largest late shifts:",
            "",
            late[["domain", "phase_0_weight", "phase_1_weight", "phase_1_minus_phase_0"]].to_markdown(
                index=False,
                floatfmt=".6f",
            ),
            "",
            "Largest early shifts:",
            "",
            early[["domain", "phase_0_weight", "phase_1_weight", "phase_1_minus_phase_0"]].to_markdown(
                index=False,
                floatfmt=".6f",
            ),
            "",
            "## HPR shape",
            "",
            "```json",
            json.dumps(
                {
                    "variant": config.variant.value,
                    "shape_index": config.shape_index,
                    **asdict(config.shape),
                },
                indent=2,
            ),
            "```",
            "",
            "## Required next gate",
            "",
            "Use this phase-0 mixture as one shared prefix. Search phase-1 continuations broadly, retain the exact "
            "boundary checkpoint and boundary evaluation, include tied/OLMix and nominal HPR continuations, then "
            "fresh-seed confirm the best complete policy against a separately trained one-phase comparator. Do not "
            "claim joint optimality: a "
            "second round must re-optimize phase 0 after learning the continuation response.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not ANCHOR_PATH.exists() or not CANONICAL_TWO_PHASE_PATH.exists():
        raise FileNotFoundError("Canonical-280 source artifacts are missing")

    reference = hpr_panel.policy_datasets("300m", TARGET, None)[observatory.TWO_PHASE]
    dataset = load_canonical_panel(reference, TARGET)
    if dataset.n != FIT_ROWS or dataset.m != 39:
        raise ValueError(f"Expected a 280x39 fit panel, found {dataset.n}x{dataset.m}")
    anchor_frame, source_aggregate, aggregate, olmix_summary = load_anchor(dataset)
    aggregate_snap_tv = float(0.5 * np.abs(aggregate - source_aggregate).sum())
    alpha0, alpha1 = observatory.phase_fractions(dataset)
    natural = observatory.natural_weights(dataset, alpha0)

    config, selection = observatory.select_hierarchical_phase_replay_config(dataset, observatory.TWO_PHASE)
    sweep = pd.DataFrame(selection["candidateSweep"])
    fitted = fit_hpr(dataset, TARGET, np.arange(dataset.n), config, sweep)
    tied = np.stack([aggregate, aggregate])
    tied_prediction = hpr_panel.scalar_prediction(fitted.model, tied)

    candidates: dict[float, np.ndarray] = {}
    path_rows: list[dict[str, float | bool]] = []
    for budget in PHASE_INFORMATION_BUDGETS:
        result = hpr_panel.optimize_fixed_aggregate(fitted, aggregate, budget, alpha0, alpha1)
        projected = project_fixed_aggregate(result.weights, aggregate, alpha0, alpha1)
        prediction = hpr_panel.scalar_prediction(fitted.model, projected)
        metrics = candidate_metrics(projected, aggregate, dataset, natural)
        selected = np.isclose(budget, SELECTED_PHASE_INFORMATION_BUDGET)
        candidates[budget] = projected
        path_rows.append(
            {
                "phase_information_budget": budget,
                "selected": selected,
                "hpr_predicted_bpb": prediction,
                "hpr_predicted_gain_bpb": tied_prediction - prediction,
                "successful_optimizer_starts": result.successful_starts,
                **metrics,
            }
        )
    path_frame = pd.DataFrame(path_rows)
    selected_row = path_frame.loc[path_frame["selected"]].iloc[0]
    if selected_row["prefix_tv_to_anchor"] > PREFIX_TV_LIMIT + 1e-9:
        raise ValueError("Selected reserve prefix exceeds its frozen TV limit")
    selected_weights = candidates[SELECTED_PHASE_INFORMATION_BUDGET]

    models = established_models(dataset, fitted.model)
    model_rows = []
    direction_rows = []
    selected_contrast = selected_weights[1] - selected_weights[0]
    for name, model in models.items():
        tied_value = float(model.predict(tied[None])[0])
        selected_value = float(model.predict(selected_weights[None])[0])
        model_rows.append(
            {
                "model": name,
                "tied_prediction": tied_value,
                "selected_prediction": selected_value,
                "predicted_gain_bpb": tied_value - selected_value,
            }
        )
        model_fit = hpr_panel.FittedPolicy(TARGET, observatory.TWO_PHASE, dataset, config, model, sweep)
        optimum = hpr_panel.optimize_fixed_aggregate(
            model_fit,
            aggregate,
            SELECTED_PHASE_INFORMATION_BUDGET,
            alpha0,
            alpha1,
        )
        projected = project_fixed_aggregate(optimum.weights, aggregate, alpha0, alpha1)
        contrast = projected[1] - projected[0]
        contrast_norm = np.linalg.norm(contrast)
        selected_norm = np.linalg.norm(selected_contrast)
        direction_rows.append(
            {
                "model": name,
                "phase_total_variation": 0.5 * np.abs(contrast).sum(),
                "prefix_tv_to_anchor": 0.5 * np.abs(projected[0] - aggregate).sum(),
                "cosine_to_hpr_contrast": float(np.dot(contrast, selected_contrast) / (contrast_norm * selected_norm)),
                "contrast_tv_to_hpr": 0.5 * np.abs(contrast - selected_contrast).sum(),
            }
        )
    model_frame = pd.DataFrame(model_rows)
    direction_frame = pd.DataFrame(direction_rows)
    if not bool((model_frame["predicted_gain_bpb"] > 0).all()):
        raise ValueError("An established phase model predicts harm from the selected reserve direction")

    stability = pd.DataFrame()
    if not args.skip_fold_stability:
        stability = fold_stability(dataset, aggregate, config, sweep, selected_weights)
        if float(stability["cosine_to_full_contrast"].min()) < 0.9:
            raise ValueError("Reserve direction is unstable across fit folds")

    uncheatable_reference = hpr_panel.policy_datasets("300m", "uncheatable", None)[observatory.TWO_PHASE]
    uncheatable = load_canonical_panel(uncheatable_reference, "uncheatable")
    uncheatable_config, uncheatable_selection = observatory.select_hierarchical_phase_replay_config(
        uncheatable,
        observatory.TWO_PHASE,
    )
    uncheatable_fit = fit_hpr(
        uncheatable,
        "uncheatable",
        np.arange(uncheatable.n),
        uncheatable_config,
        pd.DataFrame(uncheatable_selection["candidateSweep"]),
    )
    uncheatable_guardrail = hpr_panel.scalar_prediction(
        uncheatable_fit.model,
        selected_weights,
    ) - hpr_panel.scalar_prediction(uncheatable_fit.model, tied)
    if uncheatable_guardrail > 0.002:
        message = f"Selected Table-9 reserve direction violates the Uncheatable guardrail: {uncheatable_guardrail}"
        raise ValueError(message)

    buckets = bucket_frame(anchor_frame, dataset, source_aggregate, aggregate, selected_weights)
    path_frame.to_csv(args.output_dir / "candidate_path.csv", index=False)
    model_frame.to_csv(args.output_dir / "established_model_predictions.csv", index=False)
    direction_frame.to_csv(args.output_dir / "established_model_directions.csv", index=False)
    stability.to_csv(args.output_dir / "fold_stability.csv", index=False)
    buckets.to_csv(args.output_dir / "selected_prefix_and_nominal_continuation.csv", index=False)
    buckets[["domain", "phase_0_weight"]].rename(columns={"phase_0_weight": "weight"}).to_csv(
        args.output_dir / "selected_phase_0_prefix.csv",
        index=False,
    )
    manifest = {
        "selection_target": TARGET,
        "fit_rows": dataset.n,
        "bucket_count": dataset.m,
        "phase_fractions": [alpha0, alpha1],
        "aggregate_source": str(ANCHOR_PATH.relative_to(REPO_ROOT)),
        "aggregate_source_sha256": file_sha256(ANCHOR_PATH),
        "aggregate_summary_source": str(ANCHOR_SUMMARY_PATH.relative_to(REPO_ROOT)),
        "aggregate_summary_source_sha256": file_sha256(ANCHOR_SUMMARY_PATH),
        "aggregate_source_row_counts": olmix_summary["row_counts"],
        "aggregate_source_proportional_reference_n": olmix_summary["proportional_reference_n"],
        "runtime_mixture_block_size": MIXTURE_BLOCK_SIZE,
        "aggregate_runtime_tv_to_source": aggregate_snap_tv,
        "canonical_two_phase_source": str(CANONICAL_TWO_PHASE_PATH.relative_to(REPO_ROOT)),
        "canonical_two_phase_source_sha256": file_sha256(CANONICAL_TWO_PHASE_PATH),
        "selected_phase_information_budget": SELECTED_PHASE_INFORMATION_BUDGET,
        "prefix_tv_limit": PREFIX_TV_LIMIT,
        "delphi_outcomes_used": False,
    }
    (args.output_dir / "design_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_report(
        args.output_dir,
        path_frame,
        model_frame,
        direction_frame,
        stability,
        buckets,
        config,
        uncheatable_guardrail,
        aggregate_snap_tv,
        olmix_summary,
    )
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
