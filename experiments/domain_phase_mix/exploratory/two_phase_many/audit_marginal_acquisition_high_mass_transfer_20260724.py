# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Test phase-order transfer to unseen high-mass bucket swaps.

The aggregate fit and phase-treatment budget match the exact 280-checkpoint
audit. Phase-head fitting excludes every high-mass pair direction. The frozen
test asks whether a marginal-value phase rule learned from domain-vs-rest and
balanced-partition interventions composes to bucket-to-bucket swaps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_fixed_budget_aggregate_comparators_20260724 as comparators,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_fixed_budget_marginal_acquisition_joint_20260724 as joint,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_fixed_budget_pooled_acquisition_protocol_20260724 as strict_protocol,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_frontier_control_aggregate_identification_20260724 as aggregate_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_marginal_acquisition_phase_potential_20260724 as phase_potential,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "marginal_acquisition_high_mass_transfer_20260724"
DEFAULT_SEEDS = joint.DEFAULT_SEEDS
BOOTSTRAP_DRAWS = 20_000
HIGH_MASS_FAMILY = "high_mass_pair"
CANDIDATE_NAMES = (
    "marginal_global_phase_potential",
    "marginal_family_phase_potential",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    return parser.parse_args()


def candidate_by_name(name: str) -> phase_potential.Candidate:
    return next(candidate for candidate in phase_potential.CANDIDATES if candidate.name == name)


def transfer_selection(
    dataset: phase_potential.PairDataset,
    model: Any,
    treatment_count: int,
    seed: int,
) -> np.ndarray:
    """Select complete pairs without consulting outcomes or high-mass swaps."""

    order = joint.interleaved_pair_order(dataset, model, seed)
    eligible = order[~dataset.frame.iloc[order]["contrast_family"].eq(HIGH_MASS_FAMILY).to_numpy()]
    anchors = sorted(dataset.frame["anchor_id"].unique())
    queues = {
        anchor: eligible[dataset.frame.iloc[eligible]["anchor_id"].eq(anchor).to_numpy()].tolist() for anchor in anchors
    }
    selected_list: list[int] = []
    while len(selected_list) < treatment_count // 2:
        for anchor in anchors:
            if len(selected_list) >= treatment_count // 2:
                break
            if not queues[anchor]:
                raise ValueError(f"Insufficient non-high-mass pairs for anchor {anchor}")
            selected_list.append(queues[anchor].pop(0))
    selected = np.asarray(selected_list, dtype=int)
    if len(selected) != treatment_count // 2:
        raise ValueError("Insufficient non-high-mass pairs for the charged treatment budget")
    anchor_counts = dataset.frame.iloc[selected].groupby("anchor_id", sort=True).size()
    if int(anchor_counts.max() - anchor_counts.min()) > 1:
        raise ValueError("Transfer training pairs are not balanced by anchor")
    return selected


def fit_candidate(
    dataset: phase_potential.PairDataset,
    model: Any,
    candidate: phase_potential.Candidate,
    selected: np.ndarray,
) -> tuple[phase_potential.FitResult, np.ndarray]:
    design, family_group = phase_potential.candidate_design(dataset, model, candidate)
    ridge = 0.0 if candidate.level == "marginal_global" else joint.ODD_RIDGE[dataset.target]
    fitted = phase_potential.fit_nonnegative_head(
        design[selected],
        dataset.odd[selected],
        dataset.noise[selected],
        family_group,
        ridge,
        0.0,
    )
    return fitted, design


def cluster_bootstrap_ratio(
    observed: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap RMSE ratios over complete swap-direction clusters."""

    unique_groups = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    ratios = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled_groups = rng.choice(unique_groups, len(unique_groups), replace=True)
        sampled = np.concatenate([indices[group] for group in sampled_groups])
        candidate_rmse = np.sqrt(np.mean((predicted[sampled] - observed[sampled]) ** 2))
        zero_rmse = np.sqrt(np.mean(observed[sampled] ** 2))
        ratios[draw] = candidate_rmse / max(zero_rmse, 1e-12)
    return {
        "rmse_ratio_ci_low": float(np.quantile(ratios, 0.025)),
        "rmse_ratio_ci_high": float(np.quantile(ratios, 0.975)),
        "probability_better_than_zero": float(np.mean(ratios < 1.0)),
    }


def anchor_marginal_cosine(
    dataset: phase_potential.PairDataset,
    model: Any,
) -> float:
    aggregate, _contrast = phase_potential.aligned_pair_arrays(dataset)
    anchors = dataset.frame["anchor_id"].drop_duplicates().tolist()
    if len(anchors) != 2:
        raise ValueError(f"Expected two anchors, found {anchors}")
    marginals = []
    for anchor in anchors:
        index = int(np.flatnonzero(dataset.frame["anchor_id"].eq(anchor).to_numpy())[0])
        marginals.append(phase_potential.marginal_bucket_value(model, aggregate[[index]])[0])
    denominator = float(np.linalg.norm(marginals[0]) * np.linalg.norm(marginals[1]))
    return float(marginals[0] @ marginals[1] / max(denominator, 1e-12))


def run_target(
    target: str,
    seeds: tuple[int, ...],
    draws: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    (
        _reference,
        _heldout_frame,
        _heldout_weights,
        single,
        controls,
        _evaluation_frame,
        _evaluation_weights,
        _observed,
        _clusters,
    ) = comparators.target_data(target)
    dataset = phase_potential.pair_datasets()[target]
    test = np.flatnonzero(dataset.frame["contrast_family"].eq(HIGH_MASS_FAMILY).to_numpy())
    if len(test) != 18:
        raise ValueError(f"Expected 18 high-mass pair rows, found {len(test)}")

    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    coefficients: list[dict[str, Any]] = []
    for arm in joint.BUDGET_ARMS:
        if arm.treatment_count == 0:
            continue
        for seed_index, seed in enumerate(seeds):
            training = strict_protocol.aggregate_training_dataset(
                target,
                single,
                controls,
                arm,
                seed,
            )
            fold = strict_protocol.grouped_stratified_folds(training, seed)
            aggregate_fit = aggregate_audit.frozen_pooled_fit(training, fold)
            model = aggregate_fit.model
            selected = transfer_selection(dataset, model, arm.treatment_count, seed)
            selected_families = dataset.frame.iloc[selected]["contrast_family"].value_counts().sort_index().to_dict()
            for candidate_index, candidate_name in enumerate(CANDIDATE_NAMES):
                candidate = candidate_by_name(candidate_name)
                fitted, design = fit_candidate(dataset, model, candidate, selected)
                predicted = fitted.predict(design[test])
                observed = dataset.odd[test]
                row = {
                    "target": target,
                    "arm": arm.name,
                    "seed": seed,
                    "candidate": candidate_name,
                    "parameter_count": len(fitted.coefficients),
                    "treatment_count": arm.treatment_count,
                    "selected_pair_count": len(selected),
                    "selected_families": json.dumps(selected_families, sort_keys=True),
                    "anchor_marginal_cosine": anchor_marginal_cosine(dataset, model),
                    **phase_potential.metric_row(observed, predicted),
                }
                row.update(
                    cluster_bootstrap_ratio(
                        observed,
                        predicted,
                        dataset.frame.iloc[test]["direction_id"].to_numpy(dtype=object),
                        draws,
                        20260724 + 10_000 * seed_index + 100 * candidate_index,
                    )
                )
                metrics.append(row)
                local = dataset.frame.iloc[test].copy()
                local["target"] = target
                local["arm"] = arm.name
                local["seed"] = seed
                local["candidate"] = candidate_name
                local["observed_odd"] = observed
                local["predicted_odd"] = predicted
                local["residual"] = predicted - observed
                predictions.append(local)
                for index, coefficient in enumerate(fitted.coefficients):
                    coefficients.append(
                        {
                            "target": target,
                            "arm": arm.name,
                            "seed": seed,
                            "candidate": candidate_name,
                            "coefficient_index": index,
                            "coefficient": float(coefficient),
                        }
                    )
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True), pd.DataFrame(coefficients)


def git_metadata() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def write_report(output_dir: Path, metrics: pd.DataFrame) -> None:
    columns = [
        "target",
        "arm",
        "seed",
        "candidate",
        "parameter_count",
        "selected_pair_count",
        "anchor_marginal_cosine",
        "rmse",
        "rmse_ratio",
        "rmse_ratio_ci_low",
        "rmse_ratio_ci_high",
        "probability_better_than_zero",
        "spearman",
        "calibration_slope",
        "resolved_sign_accuracy",
    ]
    lines = [
        "# Frozen high-mass phase-order transfer",
        "",
        "Every high-mass bucket-swap direction is excluded from fitting and hyperparameter decisions.",
        "The phase head sees only charged domain-vs-rest and balanced-partition antithetic pairs.",
        "",
        metrics[columns].sort_values(["target", "arm", "candidate", "seed"]).to_markdown(index=False, floatfmt=".6f"),
        "",
        "The sealed targeted pairwise panel was not accessed.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    results = [run_target(target, seeds, int(args.bootstrap_draws)) for target in ("uncheatable", "table9")]
    metrics = pd.concat([result[0] for result in results], ignore_index=True)
    predictions = pd.concat([result[1] for result in results], ignore_index=True)
    coefficients = pd.concat([result[2] for result in results], ignore_index=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    coefficients.to_csv(args.output_dir / "coefficients.csv", index=False)
    write_report(args.output_dir, metrics)
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "total_checkpoint_budget": strict_protocol.TOTAL_BUDGET,
                "arms": [asdict(arm) for arm in joint.BUDGET_ARMS],
                "seeds": seeds,
                "high_mass_pair_rows_used_for_fit": 0,
                "high_mass_pair_rows_used_for_hyperparameters": 0,
                "candidate_names": CANDIDATE_NAMES,
                "bootstrap_unit": "bucket-swap direction across both anchors",
                "sealed_targeted_pairwise_panel_accessed": False,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "git": git_metadata(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
