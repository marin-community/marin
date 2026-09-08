# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy","numpy","pandas","scipy","scikit-learn","plotly","fsspec","gcsfs"]
# ///
# ruff: noqa: E402, E501
"""Table-9 FRESH-ANNEAL two-phase panel. The high-value-late tilt HURT Table-9 by re-epoching the
already-saturated synthetic/reasoning domains (synth_math 16.5 epochs). The dolmino/OLMo anneal does the
OPPOSITE: put FRESH, UNDER-EXPOSED high-quality data late. The eff-exp 1phase (1.0707) under-exposes exactly
those HQ domains (common_crawl_hq, synth_qa, olmocr_pdfs at <1 epoch, plenty of room). So hold the 1.0707
aggregate fixed and tilt the FRESH-HQ (HQ AND low-epoch) domains late, keeping saturated domains early:
s_i = f1 + k*(fresh_hq_i - mean). k=0 = one-phase ablation (reproduces 1.0707); k>0 = fresh anneal.
If a k>0 beats k=0 -> a well-motivated two-phase beats one-phase for Table-9 (avoids the over-epoch failure)."""

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
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    fit_olmix_reference_deletion_augmented_300m as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    fit_olmo_base_easy_per_component_dsp_kl_sweep_300m as per_component,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "table9_fresh_anneal_panel_20260705"
BASE_CSV = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_one_phase_table9_validation_mixtures_20260628/mixtures/dsp_onephase_effexp_table9_kl0p1.csv"
)
# fresh-HQ = high-quality / instruction / QA domains that are UNDER-exposed in the 1phase (room to anneal).
HQ_SUBSTR = ["common_crawl_hq", "synth_qa", "olmocr", "pes2o", "stackexchange", "wiki", "flan", "instruct"]
EPOCH_ROOM = 3.0  # only anneal HQ domains with < this many epochs (avoid re-epoching saturated ones)
TILTS = {"anneal_k0_onephase": 0.0, "anneal_kmod": 0.15, "anneal_khigh": 0.35, "anneal_kmax": 0.6}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    f0, f1 = base.PHASE_FRACTIONS
    with fsspec.open(BASE_CSV, "r") as fh:
        b = pd.read_csv(fh)
    domains = b["domain"].tolist()
    natural = b["proportional"].to_numpy(float)
    tc = b["available_tokens"].to_numpy(float)
    agg = b["aggregate_weight"].to_numpy(float)
    agg /= agg.sum()
    ep = b["simulated_epochs"].to_numpy(float)
    tb = base.load_target_budget()
    fresh = np.array(
        [1.0 if (any(s in d for s in HQ_SUBSTR) and ep[i] < EPOCH_ROOM) else 0.0 for i, d in enumerate(domains)]
    )
    hqbar = float(np.sum(fresh * agg))
    print(f"fresh-HQ anneal domains ({int(fresh.sum())}): " + ", ".join(d for i, d in enumerate(domains) if fresh[i]))
    rows = []
    for role, k in TILTS.items():
        s = np.clip(f1 + k * (fresh - hqbar), 0.0, 1.0)
        p1 = s * agg / f1
        p0 = (1.0 - s) * agg / f0
        w = np.stack([p0 / p0.sum(), p1 / p1.sum()])
        frame = per_component.mixture_frame(
            domains=domains, natural=natural, weights=w, token_counts=tc, target_budget=tb
        )
        name = f"t9anneal_{role}"
        d = args.output_dir / name
        d.mkdir(parents=True, exist_ok=True)
        frame.to_csv(d / "proposed_mixture_weights.csv", index=False)
        na = base.aggregate_phase_weights(w)
        sim = base.simulated_epochs(w, tc, target_budget=tb)
        fi = [i for i in range(len(domains)) if fresh[i]]
        p1p0 = float(np.median(w[1][fi] / np.clip(w[0][fi], 1e-9, None))) if fi else 0.0
        rows.append(
            dict(
                candidate=name,
                role=role,
                k=k,
                agg_tv_to_base=float(0.5 * np.abs(na - agg).sum()),
                fresh_hq_p1_over_p0=p1p0,
                max_sim_epoch=float(sim.max()),
                q95_sim_epoch=float(np.quantile(sim, 0.95)),
            )
        )
        print(
            f"  {name:26s} k={k:<4g} agg_tv={rows[-1]['agg_tv_to_base']:.4f} freshHQ_p1/p0={p1p0:.2f} maxep={sim.max():.1f} q95ep={rows[-1]['q95_sim_epoch']:.2f}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    print(
        f"\nWrote {len(rows)}. k=0 must reproduce 1.0707; anneals FRESH HQ late (avoids over-epoch of saturated domains)."
    )


if __name__ == "__main__":
    main()
