# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy","numpy","pandas","scipy","scikit-learn","plotly","fsspec","gcsfs"]
# ///
"""Table-9 controlled-tilt panel: does a two-phase mixture beat the eff-exp one-phase optimum
(dsp_onephase_effexp_table9_kl0p1 = 1.070728)? Two-phase is premature for Table-9 per the surrogate, and
the naive eff-exp two-phase argmin over-tilts (gamma=17.8, hoards phase-1). So test the CONTROLLED analog
of the uncheatable tilt sweep: hold the 1.0707 aggregate FIXED and add a MODERATE late tilt of its
high-value (overweighted) domains, via phase-1 share s_i = f1 + k*(log(a_i/nat_i) - gbar) (preserves the
phase-1 budget for any k). k=0 = the exact one-phase ablation (should reproduce 1.0707); k>0 = two-phase.
If some k beats k=0 -> two-phase genuinely helps Table-9; else premature is confirmed."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_per_component_dsp_kl_sweep_300m as per_component,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "table9_controlled_tilt_panel_20260705"
BASE_CSV = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_one_phase_table9_validation_mixtures_20260628/mixtures/dsp_onephase_effexp_table9_kl0p1.csv"
)
# tilt strengths: 0 = one-phase ablation; moderate levels chosen to give high-value p1/p0 ~1.5 and ~2.5.
TILTS = {"tilt_k0_onephase": 0.0, "tilt_kmod": 0.10, "tilt_khigh": 0.20, "tilt_kmax": 0.32}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    f0, f1 = base.PHASE_FRACTIONS

    with fsspec.open(BASE_CSV, "r") as fh:
        b = pd.read_csv(fh)
    domains = b["domain"].tolist()
    natural = b["proportional"].to_numpy(float)
    tc = b["available_tokens"].to_numpy(float)
    agg = b["aggregate_weight"].to_numpy(float)
    agg = agg / agg.sum()
    tb = base.load_target_budget()
    g = np.log(np.clip(agg / np.clip(natural, 1e-12, None), 1e-6, None))
    gbar = float(np.sum(g * agg))  # weighted mean so sum(s_i*a_i)=f1 for any k

    rows = []
    for role, k in TILTS.items():
        s = np.clip(f1 + k * (g - gbar), 0.0, 1.0)
        p1 = s * agg / f1
        p0 = (1.0 - s) * agg / f0
        w = np.stack([p0 / p0.sum(), p1 / p1.sum()])
        frame = per_component.mixture_frame(
            domains=domains, natural=natural, weights=w, token_counts=tc, target_budget=tb
        )
        name = f"t9tilt_{role}"
        d = args.output_dir / name
        d.mkdir(parents=True, exist_ok=True)
        frame.to_csv(d / "proposed_mixture_weights.csv", index=False)
        new_agg = base.aggregate_phase_weights(w)
        sim = base.simulated_epochs(w, tc, target_budget=tb)
        # tilt on the top-value quartile of domains
        top = np.argsort(agg / natural)[::-1][: max(1, len(domains) // 4)]
        p1p0_top = float(np.median(w[1][top] / np.clip(w[0][top], 1e-9, None)))
        rows.append(
            dict(
                candidate=name,
                role=role,
                k=k,
                agg_tv_to_base=float(0.5 * np.abs(new_agg - agg).sum()),
                top_value_p1_over_p0=p1p0_top,
                max_weight=float(w.max()),
                max_sim_epoch=float(sim.max()),
                q95_sim_epoch=float(np.quantile(sim, 0.95)),
            )
        )
        print(
            f"  {name:26s} k={k:<4g} agg_tv_to_base={rows[-1]['agg_tv_to_base']:.4f} "
            f"top_p1/p0={p1p0_top:.2f} maxep={sim.max():.1f} q95ep={rows[-1]['q95_sim_epoch']:.2f}",
            flush=True,
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    print(f"\nWrote {len(rows)} candidates. k=0 must reproduce the 1.0707 base (agg_tv_to_base~0).")


if __name__ == "__main__":
    main()
