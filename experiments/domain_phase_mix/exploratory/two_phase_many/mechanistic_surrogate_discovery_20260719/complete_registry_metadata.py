# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# ///

"""Complete inherited registry metadata without changing scientific decisions."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

OUTPUT_ROOT = Path(__file__).parents[1] / "reference_outputs" / "mechanistic_surrogate_discovery_20260719"
REGISTRY_PATH = OUTPUT_ROOT / "approach_registry.csv"
LEDGER_PATH = OUTPUT_ROOT / "data_use_ledger.csv"
SUMMARY_PATH = OUTPUT_ROOT / "registry_summary.json"


LATENT_STATE_BACKFILLS = {
    "prior_A": (
        "Per-bucket retained useful capability \\(s_i\\in[0,1]\\); zero is no retained capability "
        "and one is saturated capability."
    ),
    "prior_F": (
        "Per-family learned capability \\(s_f\\in[0,1]\\), coupled through a rank-one or "
        "rank-two family-competition embedding."
    ),
    "prior_J": (
        "Per-family retained capability state, acquired from compatible in-family exposure and "
        "decayed by out-of-family updates."
    ),
    "prior_M": (
        "Per-bucket unresolved error mass \\(z_i\\ge 0\\), with acquisition removing mass and "
        "family absence restoring mass through forgetting."
    ),
    "prior_N": (
        "The unresolved error masses \\(z_i\\ge 0\\) from model M, driven by competition-adjusted effective evidence."
    ),
    "prior_R": ("Foundation competence \\(s_F(t)\\in[0,1]\\) and the specialist effective-exposure integrals it gates."),
    "prior_U": (
        "Per-bucket retained unique-coverage fraction \\(u_i\\in[0,1]\\); its complement is unseen or forgotten mass."
    ),
    "prior_W": (
        "Per-bucket cumulative physical exposure \\(E_i\\) and retained competence "
        "\\(s_i\\in[0,1]\\), coupled through the current duplicate-token rate."
    ),
    "prior_Z": (
        "Per-bucket collision-limited evidence dose \\(x_i=e_i/(1+cC_i)\\), where \\(C_i\\) is "
        "the fixed finite-corpus collision invariant."
    ),
    "prior_AO": (
        "Bounded group competence \\(z_g\\in[0,1]\\) and proportional-mass-weighted family "
        "competence \\(z_f\\), which gates cross-family interference."
    ),
    "prior_AM": (
        "Per-group bounded competence under a rank-one displacement field induced by excess family concentration."
    ),
    "prior_AN": (
        "Per-phase acquisition efficiency \\(h_t\\in[0,1]\\) and the resulting gated cumulative "
        "bucket exposure \\(x_i=\\sum_t h_te_i^{(t)}\\)."
    ),
}


STATE_TRANSITION_BACKFILLS = {
    "prior_B": (
        "Static finite-population map from cumulative physical exposure: "
        "\\(u_i=1-e^{-e_i}\\) and \\(r_i=e_i-u_i\\); no additional recurrent state."
    ),
    "prior_C": (
        "Static aggregation of bucket unique coverage into family coverage "
        "\\(c_f=\\sum_{i\\in f}p_iu_i/\\sum_{i\\in f}p_i\\); no recurrent transition."
    ),
    "prior_D": ("For each phase, \\(z_k^{(t+1)}=z_k^{(t)}\\exp[-\\sum_iq_{ki}e_i^{(t)}]\\exp[g_k\\Delta_t]\\)."),
    "prior_E": (
        "Static dose-response map \\(s_i=1-\\exp[-(e_i/\\kappa_f)^{k_f}]\\) with literal "
        "duplicate mass \\(r_i=e_i-(1-e^{-e_i})\\); no recurrent transition."
    ),
    "prior_G": (
        "Static conversion of family coverage into nonnegative deficit "
        "\\(d_f=(c_f+\\delta)^{-\\alpha}-(1+\\delta)^{-\\alpha}\\)."
    ),
    "prior_H": (
        "Each phase independently maps exposure to unique coverage and duplicate mass: "
        "\\(u_i^{(t)}=1-e^{-e_i^{(t)}}\\), \\(r_i^{(t)}=e_i^{(t)}-u_i^{(t)}\\)."
    ),
    "prior_I": (
        "Exposure is accumulated through the normalized recency kernel "
        "\\(x_i=\\int_0^1k_\\lambda(t)w_i(t)D\\,dt\\); this closed-form convolution is the "
        "state update."
    ),
    "prior_K": (
        "Bucket coverage is pooled hierarchically into structural groups and families; the "
        "overload ablation attenuates acquired state by \\(\\exp(-\\zeta r)\\)."
    ),
    "prior_L": (
        "The frozen retained-exposure transition is unchanged; terminal ratios "
        "\\(r_i=x_i/x_i^{\\mathrm{prop}}\\) deterministically define the support-divergence state."
    ),
    "prior_O": (
        "The frozen inverse-deficit retained-state transition is unchanged; only the terminal "
        "monotone response link is replaced."
    ),
    "prior_P": (
        "The existing GRP retained-exposure transition is unchanged; terminal bucket and "
        "family coverage ratios gate the response."
    ),
    "prior_Q": (
        "The chosen retained-exposure or inverse-deficit transition is shared across all 51 "
        "component heads; only their response amplitudes differ."
    ),
    "prior_S": (
        "Evidence is either summed physically across phases or accumulated by the exact "
        "normalized exponential-memory integral before adding equivalent prior exposure."
    ),
    "prior_T": (
        "No learned recurrent state: physical epochs deterministically produce collision load "
        "\\(c_i=\\max(e_i-1,0)^2\\) or duplicate mass \\(r_i=e_i-(1-e^{-e_i})\\)."
    ),
    "prior_V": (
        "Group evidence follows either physical accumulation or model U's bounded retained-coverage "
        "transition, then enters a CES family production function."
    ),
    "prior_X": (
        "Distinct target-relevant coverage accumulates under a finite-population sampling law; "
        "an optional global survival factor discounts phase-0 coverage before phase 1."
    ),
    "prior_Y": (
        "No recurrent state: realized token allocations and fixed bucket sizes deterministically "
        "define Kish effective sample size and collision load."
    ),
    "prior_AA": (
        "No recurrent state: smoothed policy-to-proportional density ratios determine an "
        "importance-weight effective-sample-size fraction that scales the base evidence state."
    ),
    "prior_AB": (
        "No recurrent state: the phase-boundary distributions deterministically define directed "
        "KL or Jensen-Shannon adaptation debt, which vanishes for phase-tied policies."
    ),
    "prior_AC": (
        "Per-bucket evidence accumulates additively as "
        "\\(x_i=\\sum_te_i^{(t)}\\bar\\eta_t^q\\) under the fixed learning-rate schedule."
    ),
    "prior_AD": (
        "Each phase's exposure is first scaled by "
        "\\((1+k\\sum_iw_i^2)^{-1}\\), then passed through the unchanged bounded acquisition "
        "transition."
    ),
}


def _load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), list(reader)


def main() -> None:
    fieldnames, rows = _load_csv(REGISTRY_PATH)
    ids = {row["id"] for row in rows}
    assert set(LATENT_STATE_BACKFILLS) <= ids
    assert set(STATE_TRANSITION_BACKFILLS) <= ids

    changed = 0
    for row in rows:
        route_id = row["id"]
        if not row["latent_state"].strip() and route_id in LATENT_STATE_BACKFILLS:
            row["latent_state"] = LATENT_STATE_BACKFILLS[route_id]
            changed += 1
        if not row["state_transition"].strip() and route_id in STATE_TRANSITION_BACKFILLS:
            row["state_transition"] = STATE_TRANSITION_BACKFILLS[route_id]
            changed += 1

    missing = {
        field: [row["id"] for row in rows if not row[field].strip()]
        for field in fieldnames
        if any(not row[field].strip() for row in rows)
    }
    assert not missing, missing
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["family"] for row in rows}) == len(rows)

    with REGISTRY_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _, ledger_rows = _load_csv(LEDGER_PATH)
    statuses = Counter(row["status"] for row in rows)
    summary = {
        "registry": REGISTRY_PATH.relative_to(OUTPUT_ROOT.parents[1]).as_posix(),
        "ledger": LEDGER_PATH.relative_to(OUTPUT_ROOT.parents[1]).as_posix(),
        "route_count": len(rows),
        "prior_routes": sum(row["id"].startswith("prior_") for row in rows),
        "active_routes": sum(row["status"] in {"active", "promoted"} for row in rows),
        "status_counts": dict(sorted(statuses.items())),
        "ledger_rows": len(ledger_rows),
        "metadata_cells_backfilled": changed,
        "required_fields_complete": True,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
