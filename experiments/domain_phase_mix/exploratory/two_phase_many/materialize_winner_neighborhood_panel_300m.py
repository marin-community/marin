# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///
# ruff: noqa: E501
"""Winner-neighborhood panel for 3e18 validation (uncheatable primary + Table-9).

The proven frontier (uncheatable 0.985974 = dsp_uncheatable_exposure_all_deficits) AGGRESSIVELY
OVERWEIGHTS eval-relevant domains in AGGREGATE (arxiv 1.88x, code 1.64x, wiki 1.55x, math 1.70x the
one-phase level) with only a MODEST late tilt (p1/p0 ~1.3-1.9x). The 300M model's phase-1 tilt
preference is anti-transferable, so we do NOT trust a model-optimized tilt; instead we perturb the
winner EMPIRICALLY and let 3e18 decide which direction beats it.

Two orthogonal sweeps around the winner, each preserving simplex + phase-budget constraints exactly:
  - TILT sweep: phase-1 share s_i of each domain's aggregate; s_i_new = f1 + k*(s_i - f1). k=0 -> one-phase
    (no tilt), k=1 -> winner, k>1 -> more late tilt. sum(s_i*a_i)=f1 is preserved for any k.
  - OVERWEIGHT sweep: a_i_new = onephase_a_i + lambda*(winner_a_i - onephase_a_i), renormalized; then
    split with the winner's tilt pattern. lambda<1 pulls the eval-relevant overweight back, lambda>1 pushes it.

Reads winner + one-phase aggregates, writes per_component mixture_frame CSVs for launch.
"""

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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "winner_neighborhood_panel_20260705"
REPAIR_GCS = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_dsp_exposure_repair_validation_mixtures_20260702/mixtures"
)
PANEL1_DIR = SCRIPT_DIR / "reference_outputs" / "sufficiency_floored_panel_20260705"


def load_winner(objective):
    key = f"dsp_{objective}_exposure_all_deficits"
    with fsspec.open(f"{REPAIR_GCS}/{key}.csv", "r") as fh:
        df = pd.read_csv(fh)
    return df.set_index("domain")


def load_reference(objective):
    """Panel-1 one-phase-control frame: carries proportional, available_tokens, and one-phase aggregate."""
    name = "uncheatable_one_phase_control" if objective == "uncheatable" else "table9_t9_one_phase_control"
    return pd.read_csv(PANEL1_DIR / name / "proposed_mixture_weights.csv").set_index("domain")


def split_from_share(agg, s, f0, f1):
    """Reconstruct (p0, p1) simplices from aggregate a_i and phase-1 share s_i = f1*p1_i/a_i."""
    p1 = s * agg / f1
    p0 = (1.0 - s) * agg / f0
    return np.stack([p0 / p0.sum(), p1 / p1.sum()])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    f0, f1 = base.PHASE_FRACTIONS

    # (objective, mode, param, role)
    panel = [
        ("uncheatable", "winner", 1.0, "winner_exact"),
        ("uncheatable", "tilt", 0.0, "tilt_k0_onephase"),
        ("uncheatable", "tilt", 0.5, "tilt_k0p5_less"),
        ("uncheatable", "tilt", 1.5, "tilt_k1p5_more"),
        ("uncheatable", "overweight", 0.7, "overweight_0p7"),
        ("uncheatable", "overweight", 1.3, "overweight_1p3"),
        ("table9", "winner", 1.0, "t9_winner_exact"),
        ("table9", "tilt", 1.5, "t9_tilt_k1p5_more"),
        ("table9", "overweight", 1.3, "t9_overweight_1p3"),
    ]
    cache = {}
    rows = []
    for objective, mode, param, role in panel:
        if objective not in cache:
            ref = load_reference(
                objective
            )  # panel-1 one-phase control: proportional, available_tokens, one-phase aggregate
            domains = list(ref.index)
            win = load_winner(objective).reindex(domains)  # align winner phase weights to reference domain order
            natural = ref["proportional"].to_numpy(float)
            tc = ref["available_tokens"].to_numpy(float)
            agg_w = win["aggregate_weight"].to_numpy(float)
            p0 = win["phase_0_weight"].to_numpy(float)
            p1 = win["phase_1_weight"].to_numpy(float)
            s = (f1 * p1) / np.clip(agg_w, 1e-12, None)  # phase-1 share of each domain's aggregate
            onephase_agg = ref["aggregate_weight"].to_numpy(float)
            tb = base.load_target_budget()
            cache[objective] = dict(
                domains=domains, natural=natural, tc=tc, agg=agg_w, p0=p0, p1=p1, s=s, one=onephase_agg, tb=tb
            )
        c = cache[objective]
        agg, s = c["agg"], c["s"]
        if mode == "winner":
            w = np.stack([c["p0"] / c["p0"].sum(), c["p1"] / c["p1"].sum()])
        elif mode == "tilt":
            s_new = np.clip(f1 + param * (s - f1), 0.0, 1.0)
            w = split_from_share(agg, s_new, f0, f1)
        else:  # overweight aggregate toward/beyond winner's eval-relevant excess, keep winner tilt pattern
            a_new = c["one"] + param * (agg - c["one"])
            a_new = np.clip(a_new, 1e-9, None)
            a_new = a_new / a_new.sum()
            w = split_from_share(a_new, s, f0, f1)

        frame = per_component.mixture_frame(
            domains=c["domains"], natural=c["natural"], weights=w, token_counts=c["tc"], target_budget=c["tb"]
        )
        name = f"{objective}_{role}"
        d = args.output_dir / name
        d.mkdir(parents=True, exist_ok=True)
        frame.to_csv(d / "proposed_mixture_weights.csv", index=False)
        new_agg = base.aggregate_phase_weights(w)
        sim = base.simulated_epochs(w, c["tc"], target_budget=c["tb"])
        tv1p = float(0.5 * np.abs(new_agg - c["one"]).sum())
        p1p0_med = float(np.median(w[1] / np.clip(w[0], 1e-9, None)))
        rows.append(
            dict(
                candidate=name,
                objective=objective,
                mode=mode,
                param=param,
                role=role,
                tv_agg_to_onephase=tv1p,
                median_p1_over_p0=p1p0_med,
                max_weight=float(w.max()),
                max_simulated_epoch=float(sim.max()),
                q95_simulated_epoch=float(np.quantile(sim, 0.95)),
            )
        )
        print(
            f"  {name:34s} tv_agg->1p={tv1p:.3f} med_p1/p0={p1p0_med:.2f} maxw={w.max():.3f} q95ep={rows[-1]['q95_simulated_epoch']:.2f}",
            flush=True,
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    print(f"\nWrote {len(rows)} candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
