# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy","numpy","pandas","scipy","scikit-learn","plotly","fsspec","gcsfs"]
# ///
# ruff: noqa: E402, E501
"""Table-9 fresh-anneal v2: strengthen the (validated) moderate fresh-anneal win. v1 kmod (3-domain anneal,
k=0.15) beat one-phase by 0.0024 (~0.5 sigma). v2 EXPANDS the anneal set to ALL under-exposed high-quality
domains (dolmino HQ + dolma3_cc/*_high quality subsets, epochs<2) for a bigger effect, sweeps strength finely
around the peak, and repeats k0 for the noise floor. Aggregate held fixed; saturated domains stay early."""

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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "table9_fresh_anneal_v2_panel_20260705"
BASE_CSV = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_one_phase_table9_validation_mixtures_20260628/mixtures/dsp_onephase_effexp_table9_kl0p1.csv"
)
HQ_SUBSTR = ["common_crawl_hq", "synth_qa", "olmocr", "pes2o", "_high"]  # HQ + quality-filtered CC subsets
EPOCH_ROOM = 2.0
# (role, anneal_set expanded?, k)
PANEL = [
    ("k0_onephase", True, 0.0),
    ("k0_repeat", True, 0.0),
    ("expanded_k0p10", True, 0.10),
    ("expanded_k0p18", True, 0.18),
    ("expanded_k0p28", True, 0.28),
    ("narrow_k0p15", False, 0.15),
]
NARROW = ["dolmino_common_crawl_hq", "dolmino_synth_qa", "dolmino_olmocr_pdfs_hq"]  # the v1 3-domain set (best v1)


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
    exp_fresh = np.array(
        [1.0 if (any(s in d for s in HQ_SUBSTR) and ep[i] < EPOCH_ROOM) else 0.0 for i, d in enumerate(domains)]
    )
    nar_fresh = np.array([1.0 if d in NARROW else 0.0 for i, d in enumerate(domains)])
    print(
        f"expanded anneal set ({int(exp_fresh.sum())} domains, agg={float(np.sum(exp_fresh * agg)):.3f}); narrow set ({int(nar_fresh.sum())})"
    )
    rows = []
    for role, expanded, k in PANEL:
        fresh = exp_fresh if expanded else nar_fresh
        hqbar = float(np.sum(fresh * agg))
        s = np.clip(f1 + k * (fresh - hqbar), 0.0, 1.0)
        p1 = s * agg / f1
        p0 = (1.0 - s) * agg / f0
        w = np.stack([p0 / p0.sum(), p1 / p1.sum()])
        frame = per_component.mixture_frame(
            domains=domains, natural=natural, weights=w, token_counts=tc, target_budget=tb
        )
        name = f"t9an2_{role}"
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
                expanded=expanded,
                k=k,
                agg_tv=float(0.5 * np.abs(na - agg).sum()),
                fresh_p1_over_p0=p1p0,
                max_sim_epoch=float(sim.max()),
                q95_sim_epoch=float(np.quantile(sim, 0.95)),
            )
        )
        print(
            f"  {name:22s} exp={expanded} k={k:<4g} agg_tv={rows[-1]['agg_tv']:.4f} fresh_p1/p0={p1p0:.2f} maxep={sim.max():.1f}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    print(
        f"\nWrote {len(rows)}. k0+k0_repeat = noise floor; expanded set = bigger anneal; narrow_k0p15 reproduces v1 kmod (1.0696)."
    )


if __name__ == "__main__":
    main()
