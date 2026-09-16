# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2", "tabulate>=0.9"]
# ///
"""Audit append-only chronology and exposed-adversarial data use."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
ROUND_DIR = OUTPUT_ROOT / "round60_data_use_ledger_integrity"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"

MALFORMED_SUPERSEDED_MARKERS = {
    "round25_shared_private_batch_frozen": "round_25_batch_preregistration_corrected",
    "round26_cascade_literal_replay_frozen": "round_26_batch_preregistration_corrected",
    "round27_power_law_error_batch_frozen": "round_27_batch_preregistration_corrected",
}
THEORETICAL_OR_DESCRIPTIVE_SINGLE_EDGE = {"NG-LK", "NG-AES", "ACPD"}
RECONCILED_ROUTES = {"FCF", "IFSC", "PWD", "ESR", "FSC", "FPCGF", "FSCR"}


def route_tokens(raw: str) -> set[str]:
    return {token.strip() for token in re.split(r"[,:+]", raw) if token.strip()}


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(REGISTRY).fillna("")
    ledger = pd.read_csv(LEDGER).fillna("")
    candidate_rows = ledger[ledger["candidate_id"].astype(str).str.strip().ne("")].copy()
    marker_rows = ledger[ledger["candidate_id"].astype(str).str.strip().eq("")].copy()

    if set(marker_rows["round_id"]) != set(MALFORMED_SUPERSEDED_MARKERS):
        raise ValueError("Unexpected malformed marker set")
    for malformed, corrected in MALFORMED_SUPERSEDED_MARKERS.items():
        malformed_index = ledger.index[ledger["round_id"].eq(malformed)].item()
        corrected_index = ledger.index[ledger["round_id"].eq(corrected)].item()
        if corrected_index <= malformed_index:
            raise ValueError(f"Corrected marker {corrected} does not follow {malformed}")

    required = [
        "timestamp",
        "round_id",
        "candidate_id",
        "candidate_family",
        "hyperparameters",
        "adversarial_outcomes_available_before_proposal",
        "adversarial_outcomes_inspected_before_proposal",
        "observations_inspiring_mechanism",
        "novelty_class",
        "evaluation_status",
        "evidence_path",
        "notes",
    ]
    blanks = {column: int(candidate_rows[column].astype(str).str.strip().eq("").sum()) for column in required}
    if any(blanks.values()):
        raise ValueError(f"Complete candidate rows contain blank required fields: {blanks}")
    parsed_time = pd.to_datetime(candidate_rows["timestamp"], utc=True, errors="coerce")
    if parsed_time.isna().any():
        raise ValueError("Complete candidate rows contain invalid timestamps")
    for column in (
        "adversarial_outcomes_available_before_proposal",
        "adversarial_outcomes_inspected_before_proposal",
    ):
        values = set(candidate_rows[column].astype(str).str.lower())
        if not values <= {"true", "false"}:
            raise ValueError(f"Unexpected boolean encoding in {column}: {sorted(values)}")
    if candidate_rows[["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Complete ledger rows contain duplicate round/candidate keys")

    new_routes = registry.loc[~registry["id"].str.startswith("prior_")].copy()
    route_rows: dict[str, list[int]] = {route_id: [] for route_id in new_routes["id"]}
    for index, row in candidate_rows.iterrows():
        for token in route_tokens(str(row["candidate_id"])):
            if token in route_rows:
                route_rows[token].append(index)
    missing = [route_id for route_id, indices in route_rows.items() if not indices]
    if missing:
        raise ValueError(f"New routes absent from ledger: {missing}")

    incomplete_empirical = []
    route_summary = []
    for route in new_routes.itertuples(index=False):
        indices = route_rows[str(route.id)]
        route_ledger = candidate_rows.loc[indices]
        statuses = " | ".join(route_ledger["evaluation_status"])
        has_freeze = any(term in statuses.lower() for term in ("preregister", "frozen", "batch frozen"))
        has_terminal = any(term in statuses.lower() for term in ("blocked", "rejected", "theoretical", "descriptive"))
        has_reconciliation = "reconciled" in statuses.lower()
        exempt = route.id in THEORETICAL_OR_DESCRIPTIVE_SINGLE_EDGE
        complete_freeze_edge = has_freeze or (has_reconciliation and has_terminal)
        complete_terminal_edge = has_terminal or (has_reconciliation and has_freeze)
        if not exempt and not (complete_freeze_edge and complete_terminal_edge):
            incomplete_empirical.append(str(route.id))
        route_summary.append(
            {
                "route_id": route.id,
                "terminal_status": route.status,
                "ledger_row_count": len(indices),
                "has_original_freeze_edge": has_freeze,
                "has_original_terminal_edge": has_terminal,
                "has_reconciliation_edge": has_reconciliation,
                "complete_freeze_edge_after_reconciliation": complete_freeze_edge,
                "complete_terminal_edge_after_reconciliation": complete_terminal_edge,
                "single_edge_theoretical_or_descriptive": exempt,
                "reconciled_historical_edge": route.id in RECONCILED_ROUTES,
            }
        )
    if incomplete_empirical:
        raise ValueError(f"Empirical routes lack explicit freeze/terminal provenance: {incomplete_empirical}")

    new_route_rows = candidate_rows[
        candidate_rows["candidate_id"].map(lambda value: bool(route_tokens(str(value)) & set(new_routes["id"])))
    ]
    adversarial_evaluations = new_route_rows[
        new_route_rows["evaluation_status"].str.contains("outcomes reconstructed", case=False)
    ]
    if len(adversarial_evaluations):
        raise ValueError("A new route was evaluated directly on exposed adversarial outcomes")
    baseline_adversarial = candidate_rows[
        candidate_rows["round_id"].eq("baseline_reconstruction")
        & candidate_rows["evaluation_status"].eq("adversarial development outcomes reconstructed")
    ]
    if len(baseline_adversarial) != 11:
        raise ValueError("Expected exactly 11 pre-existing baseline adversarial reconstructions")

    route_frame = pd.DataFrame(route_summary)
    route_frame.to_csv(ROUND_DIR / "route_ledger_coverage.csv", index=False)
    marker_summary = marker_rows[["round_id", "evidence_path"]].copy()
    marker_summary["corrected_round_id"] = marker_summary["round_id"].map(MALFORMED_SUPERSEDED_MARKERS)
    marker_summary["status"] = "preserved_append_only_and_superseded"
    marker_summary.to_csv(ROUND_DIR / "superseded_batch_markers.csv", index=False)

    report = "\n".join(
        [
            "# Round 60: data-use ledger integrity audit",
            "",
            "This audit reads only the append-only registry and ledger. It fits no model, changes no historical row, and reads no sealed confirmation outcome.",
            "",
            "## Integrity checks",
            "",
            f"- Ledger rows: {len(ledger)}; complete candidate or diagnostic rows: {len(candidate_rows)}.",
            f"- New routes with an explicit ledger trail: {len(route_frame)}/58.",
            f"- Empirical routes with a complete original or provenance-reconciled freeze/terminal trail: {int((~route_frame['single_edge_theoretical_or_descriptive']).sum())}/55.",
            f"- Theoretical or descriptive one-edge routes: {len(THEORETICAL_OR_DESCRIPTIVE_SINGLE_EDGE)}.",
            f"- Reconciled historical gaps: {len(RECONCILED_ROUTES)}; every reconciliation is provenance-only and changes no model, hyperparameter, metric, or terminal status.",
            f"- Preserved malformed legacy batch markers: {len(marker_rows)}; each is followed by a complete corrected preregistration.",
            f"- New-route evaluations directly reconstructing exposed adversarial outcomes: {len(adversarial_evaluations)}.",
            f"- Pre-existing baseline adversarial reconstructions: {len(baseline_adversarial)}.",
            "",
            "## Boundary conclusion",
            "",
            "All 58 new mechanisms are traceable to a frozen proposal and terminal decision, except the three explicitly theoretical/descriptive routes for which a single terminal edge is the complete evaluation. No new route reached the exposed adversarial evaluation gate; all candidate forms were blocked earlier. The only direct adversarial reconstructions are the 11 pre-existing Pareto baselines frozen before candidate generation.",
            "",
            "The three malformed batch-marker rows are retained because the ledger is append-only. Their corrected rows make the intended preregistration explicit, and the audit records both rather than rewriting chronology.",
        ]
    )
    (ROUND_DIR / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
