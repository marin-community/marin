# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "gcsfs>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "pyarrow>=20.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///

"""Freeze sub-280-row Compact raw optima for Delphi 3e18 validation.

The source learning curve used redundant full-logit policy coordinates. SciPy
often exhausted its iteration budget there despite reaching stable finite
endpoints. This materializer independently refits every frozen subset and
reoptimizes it in an orthonormal phase-contrast basis with analytic gradients.
The recorded endpoint remains an explicit candidate, so the numerical audit
can only preserve or lower the fitted objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.optimize import minimize
from scipy.special import softmax

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_compact_retained_state_sample_efficiency_3e18_20260721 as sample_eff,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_delphi_phase_policy_sample_efficiency_20260721 as common,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_compact_policy_optimizer_3e18_20260721 as optimizer_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_LEARNING_CURVE = (
    REFERENCE_OUTPUTS / "compact_retained_state_sample_efficiency_3e18_20260721" / "learning_curve_runs.csv"
)
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_compact_sub280_optimum_validation_panel_20260721"
DEFAULT_FIT_SWARM = REFERENCE_OUTPUTS / "delphi_augmented_swarm_3e18_20260714/delphi_augmented_swarm_3e18_wide.csv"
DEFAULT_HELDOUTS = REFERENCE_OUTPUTS / "delphi_3e18_append_only_heldouts_20260714/heldout_current.csv"

MODEL_ID = "compact_retained_state"
POLICY_CLASS = observatory.TWO_PHASE
SUB_280_SAMPLE_SIZES = (48, 64, 80, 112, 144, 184, 232)
EXPECTED_SOURCE_ROWS = 140
OPTIMIZER_STARTS = 24
OPTIMIZER_MAX_ITERATIONS = 2_500
POLICY_DEDUP_TV = 1e-6
EXACT_COORDINATE_TOLERANCE = 1e-10
RECORDED_OBJECTIVE_TOLERANCE = 1e-6
TARGET_ORDER = {"uncheatable": 0, "table9": 1}
TARGET_TAGS = {"uncheatable": "u", "table9": "t"}
DESIGN_TAGS = {"panel_stratified": "ps", "intervention_core": "ic"}


@dataclass(frozen=True)
class PanelSummary:
    source_fit_optima: int
    unique_training_coordinates: int
    deduplicated_aliases: int
    uncheatable_proposals: int
    table9_proposals: int
    exact_fit_overlaps: int
    exact_heldout_overlaps: int
    optimizer_finite_endpoints: int
    optimizer_successful_endpoints: int
    source_panel_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-curve", type=Path, default=DEFAULT_LEARNING_CURVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fit-swarm", type=Path, default=DEFAULT_FIT_SWARM)
    parser.add_argument("--heldouts", type=Path, default=DEFAULT_HELDOUTS)
    parser.add_argument("--optimizer-starts", type=int, default=OPTIMIZER_STARTS)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def policy_sha256(domains: list[str], weights: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update("\0".join(domains).encode())
    hasher.update(np.asarray(weights, dtype="<f8").tobytes())
    return hasher.hexdigest()


def recorded_weights(row: pd.Series) -> np.ndarray:
    return np.stack(
        [
            np.asarray(json.loads(str(row["raw_phase_0_weights_json"])), dtype=float),
            np.asarray(json.loads(str(row["raw_phase_1_weights_json"])), dtype=float),
        ]
    )


def weighted_policy_tv(left: np.ndarray, right: np.ndarray, alpha0: float, alpha1: float) -> float:
    return common.weighted_policy_tv(left, right, alpha0, alpha1)


def analytic_optimum(
    dataset: common.pooled.Dataset,
    model: Any,
    spec: common.FrozenSpec,
    recorded: np.ndarray,
    seed: int,
    count: int,
) -> tuple[common.RawOptimum, dict[str, int]]:
    """Optimize one fitted Compact surface without redundant logits."""
    basis = helmert(dataset.m, full=False).T

    def weights_to_contrasts(weights: np.ndarray) -> np.ndarray:
        return (np.log(np.maximum(weights, 1e-12)) @ basis).ravel()

    def contrasts_to_weights(contrasts: np.ndarray) -> np.ndarray:
        logits = np.asarray(contrasts, dtype=float).reshape(2, dataset.m - 1) @ basis.T
        return softmax(logits, axis=1)

    def objective_and_gradient(contrasts: np.ndarray) -> tuple[float, np.ndarray]:
        weights = contrasts_to_weights(contrasts)
        prediction, weight_gradient = optimizer_audit.compact_prediction_and_weight_gradient(model, weights)
        logit_gradient = weights * (weight_gradient - np.sum(weight_gradient * weights, axis=1, keepdims=True))
        return prediction, (logit_gradient @ basis).ravel()

    recorded_prediction = float(model.predict(recorded[None, :, :])[0])
    candidates: list[tuple[float, np.ndarray, bool]] = [(recorded_prediction, recorded.copy(), False)]
    messages: dict[str, int] = {"recorded_endpoint": 1}
    starts = common.optimum_starts(dataset, model, spec, seed, count, previous=recorded)
    for start in starts:
        start_weights = common.logits_to_weights(start, dataset.m)
        result = minimize(
            objective_and_gradient,
            weights_to_contrasts(start_weights),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": OPTIMIZER_MAX_ITERATIONS,
                "maxfun": 250_000,
                "ftol": 1e-12,
                "gtol": 1e-8,
                "maxls": 60,
            },
        )
        message = str(result.message)
        messages[message] = messages.get(message, 0) + 1
        if np.isfinite(result.fun) and np.isfinite(result.x).all():
            candidates.append(
                (
                    float(result.fun),
                    contrasts_to_weights(np.asarray(result.x, dtype=float)),
                    bool(result.success),
                )
            )
    if len(candidates) == 1:
        raise RuntimeError(f"No finite analytic-gradient endpoint for {dataset.name}")
    best = min(candidates, key=lambda candidate: candidate[0])
    return (
        common.RawOptimum(
            weights=best[1],
            predicted_bpb=best[0],
            optimizer_converged=best[2],
            successful_starts=sum(candidate[2] for candidate in candidates[1:]),
            finite_starts=len(candidates) - 1,
        ),
        messages,
    )


def reconstruct_optima(source: pd.DataFrame, optimizer_starts: int) -> tuple[pd.DataFrame, list[np.ndarray], list[str]]:
    rows: list[dict[str, Any]] = []
    policies: list[np.ndarray] = []
    domains: list[str] | None = None
    source = source.loc[source["sample_size"].isin(SUB_280_SAMPLE_SIZES)].copy()
    source = source.sort_values(["target", "sampling_design", "sample_size", "seed"]).reset_index(drop=True)
    if len(source) != EXPECTED_SOURCE_ROWS:
        raise ValueError(f"Expected {EXPECTED_SOURCE_ROWS} sub-280 fit optima, found {len(source)}")

    for source_row in source.itertuples(index=False):
        target = str(source_row.target)
        design = str(source_row.sampling_design)
        sample_size = int(source_row.sample_size)
        subset_seed = int(source_row.seed)
        reference = observatory.load_delphi_3e18_fit_dataset(target)
        if domains is None:
            domains = list(reference.domain_names)
        elif domains != list(reference.domain_names):
            raise ValueError("Target datasets use different domain orderings")

        selected = sample_eff.nested_subsets(reference.frame, (sample_size,), design, subset_seed)[sample_size]
        subset_sha256 = sample_eff.row_digest(selected)
        if subset_sha256 != str(source_row.subset_sha256):
            raise ValueError(f"Subset drift for {target}/{design}/n={sample_size}/seed={subset_seed}")
        train = sample_eff.subset_dataset(
            reference,
            selected,
            target,
            f"compact_sub280_validation_{target}_{design}_{sample_size}_{subset_seed}",
        )
        model = observatory.compact_fit(
            train,
            np.arange(train.n),
            float(source_row.selected_l2),
            POLICY_CLASS,
        )
        spec = common.frozen_spec(target, POLICY_CLASS, MODEL_ID)
        recorded = recorded_weights(pd.Series(source_row._asdict()))
        recorded_prediction = float(model.predict(recorded[None, :, :])[0])
        if abs(recorded_prediction - float(source_row.raw_predicted_bpb)) > RECORDED_OBJECTIVE_TOLERANCE:
            raise ValueError(
                f"Recorded objective drift for {target}/{design}/n={sample_size}/seed={subset_seed}: "
                f"{recorded_prediction} vs {source_row.raw_predicted_bpb}"
            )
        optimum, messages = analytic_optimum(
            train,
            model,
            spec,
            recorded,
            seed=20_260_721 + 100 * sample_size + subset_seed,
            count=optimizer_starts,
        )
        alpha0, alpha1 = observatory.phase_fractions(train)
        aggregate = alpha0 * optimum.weights[0] + alpha1 * optimum.weights[1]
        proportional = observatory.natural_weights(train, alpha0)
        exposure = optimum.weights[0] * train.c0 + optimum.weights[1] * train.c1
        row = {
            "target": target,
            "sampling_design": design,
            "sample_size": sample_size,
            "subset_seed": subset_seed,
            "subset_sha256": subset_sha256,
            "selected_l2": float(source_row.selected_l2),
            "recorded_predicted_bpb": recorded_prediction,
            "robust_predicted_bpb": optimum.predicted_bpb,
            "objective_improvement": recorded_prediction - optimum.predicted_bpb,
            "policy_tv_from_recorded": weighted_policy_tv(recorded, optimum.weights, alpha0, alpha1),
            "optimizer_converged": optimum.optimizer_converged,
            "optimizer_successful_starts": optimum.successful_starts,
            "optimizer_finite_starts": optimum.finite_starts,
            "optimizer_messages_json": json.dumps(messages, sort_keys=True),
            "max_bucket_weight": float(optimum.weights.max()),
            "max_simulated_epochs": float(exposure.max()),
            "phase_total_variation": float(0.5 * np.abs(optimum.weights[0] - optimum.weights[1]).sum()),
            "aggregate_tv_to_proportional": float(0.5 * np.abs(aggregate - proportional).sum()),
            "standardized_fit_support_distance": common.standardized_support_distance(train, optimum.weights),
            "source_key": f"{target}:{design}:n{sample_size}:s{subset_seed}",
        }
        rows.append(row)
        policies.append(optimum.weights)
    assert domains is not None
    return pd.DataFrame(rows), policies, domains


def deduplicate_optima(
    audit: pd.DataFrame,
    policies: list[np.ndarray],
    domains: list[str],
) -> tuple[pd.DataFrame, list[np.ndarray], pd.DataFrame]:
    representatives: list[np.ndarray] = []
    representative_rows: list[dict[str, Any]] = []
    aliases: list[list[dict[str, Any]]] = []
    alpha0, alpha1 = 0.8, 0.2

    for row, policy in zip(audit.to_dict(orient="records"), policies, strict=True):
        match = None
        for index, representative in enumerate(representatives):
            if weighted_policy_tv(policy, representative, alpha0, alpha1) <= POLICY_DEDUP_TV:
                match = index
                break
        if match is None:
            representatives.append(policy)
            representative_rows.append(row)
            aliases.append([row])
        else:
            aliases[match].append(row)

    panel_rows: list[dict[str, Any]] = []
    alias_rows: list[dict[str, Any]] = []
    for index, (row, policy, group) in enumerate(zip(representative_rows, representatives, aliases, strict=True)):
        target = str(row["target"])
        design = str(row["sampling_design"])
        candidate_id = (
            f"crslow_{TARGET_TAGS[target]}_{DESIGN_TAGS[design]}_"
            f"n{int(row['sample_size']):03d}_s{int(row['subset_seed'])}"
        )
        target_aliases = sorted({str(item["target"]) for item in group})
        row_aliases = [str(item["source_key"]) for item in group]
        panel_row: dict[str, Any] = {
            "_policy_index": index,
            "candidate_id": candidate_id,
            "target": target,
            "policy_class": "two_phase",
            "candidate_kind": "compact_raw_optimum_sub280_learning_curve",
            "fit_source": "delphi_3e18",
            "aggregate_kl_coefficient": "",
            "phase_information_budget": "",
            "model": MODEL_ID,
            "sampling_design": design,
            "fit_rows": int(row["sample_size"]),
            "subset_seed": int(row["subset_seed"]),
            "selected_l2": float(row["selected_l2"]),
            "proposal_predicted_bpb": float(row["robust_predicted_bpb"]),
            "max_bucket_weight": float(row["max_bucket_weight"]),
            "max_simulated_epochs": float(row["max_simulated_epochs"]),
            "phase_total_variation": float(row["phase_total_variation"]),
            "aggregate_tv_to_proportional": float(row["aggregate_tv_to_proportional"]),
            "standardized_fit_support_distance": float(row["standardized_fit_support_distance"]),
            "source_alias_count": len(group),
            "source_aliases_json": json.dumps(row_aliases, separators=(",", ":")),
            "proposal_target_aliases_json": json.dumps(target_aliases, separators=(",", ":")),
            "policy_sha256": policy_sha256(domains, policy),
        }
        for phase_index in range(2):
            for domain_index, domain in enumerate(domains):
                panel_row[f"phase_{phase_index}_{domain}"] = float(policy[phase_index, domain_index])
        panel_rows.append(panel_row)
        for alias in group:
            alias_rows.append(
                {
                    "candidate_id": candidate_id,
                    "representative_source_key": row["source_key"],
                    "alias_source_key": alias["source_key"],
                    "alias_policy_tv": weighted_policy_tv(
                        policies[audit.index[audit["source_key"].eq(alias["source_key"])][0]],
                        policy,
                        alpha0,
                        alpha1,
                    ),
                }
            )
    panel = pd.DataFrame(panel_rows)
    panel["target_order"] = panel["target"].map(TARGET_ORDER)
    panel = panel.sort_values(["target_order", "sampling_design", "fit_rows", "subset_seed"])
    ordered_policies = [representatives[int(index)] for index in panel["_policy_index"]]
    panel = panel.drop(columns=["target_order", "_policy_index"])
    if panel["candidate_id"].duplicated().any() or panel["policy_sha256"].duplicated().any():
        raise ValueError("Deduplicated panel still contains duplicate IDs or exact policies")
    return panel.reset_index(drop=True), ordered_policies, pd.DataFrame(alias_rows)


def load_fit_weights(path: Path, domains: list[str]) -> np.ndarray:
    frame = pd.read_csv(path)
    return np.stack(
        [
            frame[[f"phase_0_{domain}" for domain in domains]].to_numpy(dtype=float),
            frame[[f"phase_1_{domain}" for domain in domains]].to_numpy(dtype=float),
        ],
        axis=1,
    )


def load_heldout_weights(path: Path, domains: list[str]) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(path)
    weights: list[np.ndarray] = []
    identities: list[str] = []
    for row in frame.itertuples(index=False):
        phase0 = json.loads(str(row.phase_0_weights_json))
        phase1 = json.loads(str(row.phase_1_weights_json))
        weights.append(
            np.asarray(
                [
                    [float(phase0[domain]) for domain in domains],
                    [float(phase1[domain]) for domain in domains],
                ]
            )
        )
        identities.append(str(row.heldout_id))
    return np.stack(weights), identities


def overlap_audit(
    panel: pd.DataFrame,
    policies: list[np.ndarray],
    domains: list[str],
    fit_swarm: Path,
    heldouts: Path,
) -> pd.DataFrame:
    fit_weights = load_fit_weights(fit_swarm, domains)
    heldout_weights, heldout_ids = load_heldout_weights(heldouts, domains)
    rows = []
    for panel_row, policy in zip(panel.to_dict(orient="records"), policies, strict=True):
        fit_max = np.max(np.abs(fit_weights - policy), axis=(1, 2))
        heldout_max = np.max(np.abs(heldout_weights - policy), axis=(1, 2))
        fit_tv = np.asarray([weighted_policy_tv(item, policy, 0.8, 0.2) for item in fit_weights])
        heldout_tv = np.asarray([weighted_policy_tv(item, policy, 0.8, 0.2) for item in heldout_weights])
        nearest_heldout = int(np.argmin(heldout_tv))
        rows.append(
            {
                "candidate_id": panel_row["candidate_id"],
                "policy_sha256": panel_row["policy_sha256"],
                "exact_fit_overlap_count": int(np.sum(fit_max <= EXACT_COORDINATE_TOLERANCE)),
                "exact_heldout_overlap_count": int(np.sum(heldout_max <= EXACT_COORDINATE_TOLERANCE)),
                "nearest_fit_weighted_tv": float(fit_tv.min()),
                "nearest_heldout_weighted_tv": float(heldout_tv[nearest_heldout]),
                "nearest_heldout_id": heldout_ids[nearest_heldout],
            }
        )
    return pd.DataFrame(rows)


def write_mixtures(panel: pd.DataFrame, domains: list[str], output_dir: Path) -> None:
    mixture_dir = output_dir / "mixtures"
    mixture_dir.mkdir(parents=True, exist_ok=True)
    for _, row in panel.iterrows():
        mixture = pd.DataFrame(
            {
                "domain": domains,
                "phase_0_weight": [float(row[f"phase_0_{domain}"]) for domain in domains],
                "phase_1_weight": [float(row[f"phase_1_{domain}"]) for domain in domains],
            }
        )
        mixture.to_csv(mixture_dir / f"{row['candidate_id']}.csv", index=False, float_format="%.17g")


def write_report(
    panel: pd.DataFrame,
    numerical_audit: pd.DataFrame,
    overlap: pd.DataFrame,
    summary: PanelSummary,
    output_dir: Path,
) -> None:
    composition = panel.groupby(["target", "sampling_design", "fit_rows"]).size().rename("coordinates").reset_index()
    optimizer_summary = (
        numerical_audit.groupby(["target", "sampling_design", "sample_size"])
        .agg(
            fit_instances=("source_key", "size"),
            mean_objective_improvement=("objective_improvement", "mean"),
            max_policy_tv_from_recorded=("policy_tv_from_recorded", "max"),
            max_weight=("max_bucket_weight", "max"),
            max_epoch=("max_simulated_epochs", "max"),
            max_support_distance=("standardized_fit_support_distance", "max"),
        )
        .reset_index()
    )
    lines = [
        "# Compact retained-state sub-280 raw-optimum validation panel",
        "",
        "This panel extends the already validated 280-row-and-larger Compact raw-optimum path. It freezes every ",
        "sub-280 optimum from two nested sampling designs, five subset seeds, seven fit-row budgets, and both targets.",
        "",
        f"- Source fit optima: {summary.source_fit_optima}.",
        f"- Unique training coordinates after policy-TV <= {POLICY_DEDUP_TV:g} deduplication: "
        f"{summary.unique_training_coordinates}.",
        f"- Deduplicated aliases: {summary.deduplicated_aliases}.",
        f"- Source-panel SHA-256: `{summary.source_panel_sha256}`.",
        "- Every checkpoint receives both Uncheatable and Marin-native Table-9 evaluation.",
        "- Candidate selection uses only the fit subset and frozen inner-CV L2; no heldout target outcome is used.",
        "- Numerical gate: every recorded endpoint is retained as a candidate and challenged by 24 analytic-gradient ",
        "  orthonormal-contrast starts. Therefore reoptimization cannot worsen the frozen fitted objective.",
        f"- Exact fit-panel overlaps: {summary.exact_fit_overlaps}; exact existing-heldout overlaps: "
        f"{summary.exact_heldout_overlaps}.",
        "",
        "The very small fits deliberately produce unsupported and sometimes single-bucket raw optima. These are not ",
        "deployment recommendations; their observed transfer is the response variable needed for the sample-efficiency ",
        "learning curve.",
        "",
        "## Composition",
        "",
        composition.to_markdown(index=False),
        "",
        "## Numerical and geometry audit",
        "",
        optimizer_summary.to_markdown(index=False),
        "",
        f"Nearest existing-heldout weighted policy-TV spans {overlap['nearest_heldout_weighted_tv'].min():.6f} to "
        f"{overlap['nearest_heldout_weighted_tv'].max():.6f}.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.optimizer_starts < 8:
        raise ValueError("At least eight optimizer starts are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.learning_curve)
    numerical_audit, source_policies, domains = reconstruct_optima(source, args.optimizer_starts)
    numerical_audit.to_csv(args.output_dir / "numerical_optimum_audit.csv", index=False, float_format="%.17g")
    panel, unique_policies, aliases = deduplicate_optima(numerical_audit, source_policies, domains)
    aliases.to_csv(args.output_dir / "policy_aliases.csv", index=False, float_format="%.17g")
    overlap = overlap_audit(panel, unique_policies, domains, args.fit_swarm, args.heldouts)
    overlap.to_csv(args.output_dir / "coordinate_overlap_audit.csv", index=False, float_format="%.17g")
    exact_fit = int(overlap["exact_fit_overlap_count"].sum())
    exact_heldout = int(overlap["exact_heldout_overlap_count"].sum())
    if exact_fit or exact_heldout:
        raise ValueError(f"Panel has {exact_fit} fit and {exact_heldout} existing-heldout coordinate overlaps")
    source_panel = args.output_dir / "launcher_source_panel.csv"
    panel.to_csv(source_panel, index=False, float_format="%.17g")
    write_mixtures(panel, domains, args.output_dir)
    source_sha256 = file_sha256(source_panel)
    summary = PanelSummary(
        source_fit_optima=len(numerical_audit),
        unique_training_coordinates=len(panel),
        deduplicated_aliases=len(numerical_audit) - len(panel),
        uncheatable_proposals=int(panel["target"].eq("uncheatable").sum()),
        table9_proposals=int(panel["target"].eq("table9").sum()),
        exact_fit_overlaps=exact_fit,
        exact_heldout_overlaps=exact_heldout,
        optimizer_finite_endpoints=int(numerical_audit["optimizer_finite_starts"].gt(0).sum()),
        optimizer_successful_endpoints=int(numerical_audit["optimizer_converged"].sum()),
        source_panel_sha256=source_sha256,
    )
    (args.output_dir / "selection_summary.json").write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n")
    write_report(panel, numerical_audit, overlap, summary, args.output_dir)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
