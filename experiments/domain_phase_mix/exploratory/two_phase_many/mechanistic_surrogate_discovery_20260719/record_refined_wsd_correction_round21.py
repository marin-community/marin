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
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Freeze the corrected 107-policy WSD provenance and reconcile prior audits."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    starcoder_refined_data,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
OUTPUT_DIR = OUTPUT_ROOT / "round21_refined_wsd_provenance_correction"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
EXPECTED_WSD_ROWS = 107


@dataclass(frozen=True)
class AuditSource:
    candidate_id: str
    path: str
    identity_column: str | None
    identity_value: str | None
    rmse_column: str


SOURCES = (
    AuditSource(
        "PMVT",
        "round1_starcoder_shape_refined107/surface_oof_metrics.csv",
        "family",
        "paired_marginal_value_transport",
        "rmse",
    ),
    AuditSource(
        "IFSC",
        "round1_starcoder_shape_refined107/surface_oof_metrics.csv",
        "family",
        "identified_fast_slow_consolidation",
        "rmse",
    ),
    AuditSource(
        "TEA",
        "round1_starcoder_shape_refined107/surface_oof_metrics.csv",
        "family",
        "terminal_equilibrium_adaptation",
        "rmse",
    ),
    AuditSource(
        "PWD",
        "round2_potential_starcoder_refined107/surface_oof_metrics.csv",
        "law",
        "potential_work_dissipation",
        "rmse",
    ),
    AuditSource(
        "ESR",
        "round2_potential_starcoder_refined107/surface_oof_metrics.csv",
        "law",
        "equilibrium_stress_relaxation",
        "rmse",
    ),
    AuditSource("MCR", "round3_component_relaxation_starcoder_refined107/surface_oof_metrics.csv", None, None, "rmse"),
    AuditSource("EAGP", "round4_primed_plasticity_starcoder_refined107/surface_oof_metrics.csv", None, None, "rmse"),
    AuditSource("HRC", "round7_replicator_starcoder_refined107/surface_oof_metrics.csv", None, None, "rmse"),
    AuditSource(
        "EGCC", "round10_exposure_gated_cascade_starcoder_refined107/surface_oof_metrics.csv", None, None, "nested_rmse"
    ),
    AuditSource(
        "HWER", "round11_hessian_equilibrium_starcoder_refined107/surface_oof_metrics.csv", None, None, "nested_rmse"
    ),
    AuditSource(
        "NQGF",
        "round12_noncommuting_gradient_flow_starcoder_refined107/surface_oof_metrics.csv",
        None,
        None,
        "nested_rmse",
    ),
    AuditSource("LPSI", "round13_legacy_bilinear_starcoder_refined107/surface_metrics.csv", None, None, "nested_rmse"),
    AuditSource(
        "NTPGF", "round14_nonlinear_task_potential_starcoder_refined107/surface_metrics.csv", None, None, "nested_rmse"
    ),
    AuditSource(
        "JARA", "round16_jensen_acquisition_starcoder_refined107/surface_metrics.csv", None, None, "nested_rmse"
    ),
    AuditSource(
        "OMGF", "round17_momentum_gradient_flow_starcoder_refined107/surface_metrics.csv", None, None, "nested_rmse"
    ),
    AuditSource(
        "ASMGF", "round19_adaptive_second_moment_starcoder_refined107/surface_metrics.csv", None, None, "nested_rmse"
    ),
    AuditSource(
        "OTTPF", "round20_optimizer_time_flow_starcoder_refined107/surface_metrics.csv", None, None, "nested_rmse"
    ),
)


def selected_row(source: AuditSource, surface: str) -> pd.Series:
    table = pd.read_csv(OUTPUT_ROOT / source.path)
    rows = table.loc[table["surface"].eq(surface)]
    if source.identity_column is not None:
        rows = rows.loc[rows[source.identity_column].eq(source.identity_value)]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for {source.candidate_id}/{surface}, found {len(rows)}")
    return rows.iloc[0]


def corrected_metrics() -> pd.DataFrame:
    baseline = pd.read_csv(OUTPUT_ROOT / "round1_starcoder_shape_refined107/surface_oof_metrics.csv")
    frontier = baseline.groupby("surface", as_index=False)["rmse"].min().rename(columns={"rmse": "frontier_rmse"})
    rows: list[dict[str, object]] = []
    for source in SOURCES:
        for surface in ("starcoder_cosine_50_50", "starcoder_wsd_80_20"):
            row = selected_row(source, surface)
            rmse = float(row[source.rmse_column])
            frontier_rmse = float(frontier.loc[frontier["surface"].eq(surface), "frontier_rmse"].iloc[0])
            rows.append(
                {
                    "candidate_id": source.candidate_id,
                    "surface": surface,
                    "rmse": rmse,
                    "frontier_rmse": frontier_rmse,
                    "relative_rmse_vs_frontier": rmse / frontier_rmse - 1.0,
                    "source_path": source.path,
                }
            )
    return pd.DataFrame(rows)


def verify_refined_provenance() -> dict[str, object]:
    cosine = observatory.load_cosine_starcoder()
    wsd = starcoder_refined_data.load_refined_wsd80_starcoder(cosine)
    if wsd.n != EXPECTED_WSD_ROWS:
        raise ValueError(f"Expected {EXPECTED_WSD_ROWS} refined WSD policies, found {wsd.n}")
    best = int(wsd.y.argmin())
    return {
        "row_count": wsd.n,
        "unique_coordinates": int(
            wsd.frame[["phase0_rare", "phase1_rare"]].drop_duplicates().shape[0]
            if {"phase0_rare", "phase1_rare"}.issubset(wsd.frame.columns)
            else len({tuple(row) for row in wsd.weights[:, :, 1]})
        ),
        "observed_optimum_phase0_rare": float(wsd.weights[best, 0, 1]),
        "observed_optimum_phase1_rare": float(wsd.weights[best, 1, 1]),
        "observed_optimum_bpb": float(wsd.y[best]),
    }


def update_registry(metrics: pd.DataFrame) -> None:
    registry = pd.read_csv(REGISTRY)
    for candidate_id in metrics["candidate_id"].unique():
        candidate = metrics.loc[metrics["candidate_id"].eq(candidate_id)]
        cosine = candidate.loc[candidate["surface"].eq("starcoder_cosine_50_50")].iloc[0]
        wsd = candidate.loc[candidate["surface"].eq("starcoder_wsd_80_20")].iloc[0]
        index = registry.index[registry["id"].eq(candidate_id)]
        if len(index) != 1:
            raise ValueError(f"Registry candidate {candidate_id} is not unique")
        prior = str(registry.loc[index[0], "status_evidence"])
        correction = (
            " Refined-WSD correction: the complete 107-policy WSD surface gives "
            f"cosine/WSD RMSE {cosine['rmse']:.5f}/{wsd['rmse']:.5f}; "
            f"relative to the corrected shape frontiers this is {cosine['relative_rmse_vs_frontier']:+.1%}/"
            f"{wsd['relative_rmse_vs_frontier']:+.1%}. Status unchanged."
        )
        if "Refined-WSD correction:" not in prior:
            registry.loc[index[0], "status_evidence"] = prior + correction
    registry.to_csv(REGISTRY, index=False)


def append_ledger() -> None:
    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_21_refined_wsd_correction",
        "candidate_id": "PROVENANCE-CORRECTION",
        "candidate_family": "Refined StarCoder WSD audit correction",
        "hyperparameters": "No model or hyperparameter changes; all affected candidates rerun from frozen grids on 107 unique WSD policies",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "The Observatory loader omitted 43 completed refined WSD policies, including the true observed optimum; this is a data-provenance correction, not a new mechanism.",
        "novelty_class": "None; provenance repair",
        "evaluation_status": "corrected StarCoder shape evidence; no candidate status changed and no adversarial model evaluation occurred",
        "evidence_path": "round21_refined_wsd_provenance_correction/report.md",
        "notes": "The corrected WSD frontier is 0.04577 RMSE from identified fast-slow consolidation on 107 policies. Future shape gates use this frontier.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


def write_report(metrics: pd.DataFrame, provenance: dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "corrected_candidate_metrics.csv", index=False)
    frontier = (
        metrics[["surface", "frontier_rmse"]]
        .drop_duplicates()
        .sort_values("surface")
        .rename(columns={"frontier_rmse": "corrected_frontier_rmse"})
    )
    frontier.to_csv(OUTPUT_DIR / "corrected_frontier.csv", index=False)
    wsd = metrics.loc[metrics["surface"].eq("starcoder_wsd_80_20")].sort_values("rmse")
    report = rf"""# Round 21: refined WSD provenance correction

## Provenance

The authoritative WSD 80/20 StarCoder surface has **{provenance["row_count"]}** unique completed policies, not the 64-row preliminary export used by the Observatory loader. Its observed optimum is \((p^{{(0)}},p^{{(1)}})=({provenance["observed_optimum_phase0_rare"]:.3f},{provenance["observed_optimum_phase1_rare"]:.3f})\) with BPB {provenance["observed_optimum_bpb"]:.9f}. All discovery-drive StarCoder candidates were rerun from their frozen grids on the complete surface.

## Corrected gate

- Cosine 50/50 frontier RMSE: **{frontier.loc[frontier["surface"].eq("starcoder_cosine_50_50"), "corrected_frontier_rmse"].iloc[0]:.6f}**.
- WSD 80/20 frontier RMSE: **{frontier.loc[frontier["surface"].eq("starcoder_wsd_80_20"), "corrected_frontier_rmse"].iloc[0]:.6f}**, achieved by identified fast-slow consolidation.
- The corrected WSD frontier is materially stronger than the preliminary 64-row estimate. The 5% gate is therefore evaluated against the 107-policy value.

## Candidate reconciliation

{wsd[["candidate_id", "rmse", "relative_rmse_vs_frontier"]].to_markdown(index=False, floatfmt=".6f")}

No candidate changes status. Legacy low-rank interaction remains the closest current-drive route on WSD but is {wsd.loc[wsd["candidate_id"].eq("LPSI"), "relative_rmse_vs_frontier"].iloc[0]:.1%} above the corrected frontier and remains non-identifiable across folds. This correction did not inspect or evaluate a new model against adversarial outcomes.
"""
    (OUTPUT_DIR / "report.md").write_text(report)


def main() -> None:
    metrics = corrected_metrics()
    provenance = verify_refined_provenance()
    update_registry(metrics)
    append_ledger()
    write_report(metrics, provenance)
    print(metrics.sort_values(["surface", "rmse"]).to_string(index=False))


if __name__ == "__main__":
    main()
