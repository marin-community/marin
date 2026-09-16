# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy","numpy","pandas","scipy","scikit-learn","plotly","fsspec","gcsfs"]
# ///
# ruff: noqa: RUF059
"""Panel 3: does the eval-relevance-augmented surrogate's OPTIMIZED per-domain aggregate profile beat
the manual winner (0.985974)? Panel 2 showed the winner is a local optimum under RESCALING (overweight
and tilt sweeps peak at it), so beating it needs a different PROFILE. The augmented model
(L = bowl_gamma1 - lam_rel*sum rel_i*log1p(z_agg_i)) proposes one: more arxiv/wiki/math, less code/stack.

Candidates: augmented aggregate at the winner's TOTAL overweight (evalrel_ow ~ 4.25) with the winner's
near-optimal tilt s*, its one-phase control (re-confirm two-phase helps on the new profile), and a
50/50 winner(+)augmented aggregate blend (hedge). Aggregate axis is the transferable one (deployment-fit
inversion), so the profile is a principled test, not a 300M artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    diagnose_dsp_uncheatable_eta_heldout as eta_diag,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_paper_faithful_olmix_300m as paper_olmix,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_per_component_dsp_kl_sweep_300m as per_component,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_top_level_dsp_300m as top_level_dsp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "augmented_profile_panel_20260705"
REPAIR_GCS = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_dsp_exposure_repair_validation_mixtures_20260702/mixtures"
)
LINEAR_REG = 0.01
LTF = 0.15
KL = 0.1
EVAL_RELEVANT = [
    "dolma3_arxiv",
    "dolma3_stack_edu",
    "dolmino_stack_edu_fim",
    "dolmino_synth_code",
    "dolma3_wikipedia",
    "dolmino_synth_math",
    "dolma3_finemath_3plus",
]


def fit_head(design, y, l2):
    dm = design.mean(0, keepdims=True)
    ym = float(y.mean())
    cd, ct = design - dm, y - ym
    if l2 > 0:
        cd = np.vstack([cd, np.sqrt(l2) * np.eye(cd.shape[1])])
        ct = np.concatenate([ct, np.zeros(cd.shape[1])])
    coef, _ = nnls(cd, ct)
    return ym - float((dm @ coef).item()), coef


def bowl_design(z, mu):
    d = np.log1p(z) - mu[None, :]
    return np.hstack([np.minimum(d, 0.0) ** 2, np.maximum(d, 0.0) ** 2])


def fit_bowl_gamma1(packet):
    e0 = packet.w[:, 0, :] * packet.c0[None, :]
    e1 = packet.w[:, 1, :] * packet.c1[None, :]
    z = e0 + e1
    m = packet.m
    bm = np.clip(np.median(np.log1p(np.where(z > 1e-8, z, np.nan)), axis=0), -2, 8)
    bm = np.where(np.isfinite(bm), bm, 2.0)

    def prof(mu):
        de = bowl_design(z, mu)
        b0, c = fit_head(de, packet.y, LINEAR_REG)
        p = b0 + de @ c
        idx = np.argsort(p)[: max(5, int(np.ceil(LTF * len(packet.y))))]
        return float(np.sqrt(np.mean((p - packet.y) ** 2))) + 0.5 * float(
            np.mean(np.maximum(packet.y[idx] - p[idx], 0.0))
        )

    best = None
    for sh in (-1.5, -0.5, 0.5):
        r = minimize(
            prof,
            np.clip(bm + sh, -2, 8),
            method="L-BFGS-B",
            bounds=[(-2.0, 8.0)] * m,
            options={"maxiter": 80, "ftol": 1e-8},
        )
        if best is None or float(r.fun) < float(best.fun):
            best = r
    mu = np.asarray(best.x, float)
    b0, c = fit_head(bowl_design(z, mu), packet.y, LINEAR_REG)
    return mu, b0, c


def optimize_aggregate(packet, natural, rel, lam_rel):
    m = packet.m
    cagg = packet.c0 + packet.c1
    mu, b0, c = optimize_aggregate.cache

    def to_a(lg):
        e = np.exp(lg - lg.max())
        return e / e.sum()

    def obj(lg):
        a = to_a(lg)
        zg = a * cagg
        v = float((b0 + bowl_design(zg[None, :], mu) @ c)[0])
        v += KL * float(base.weighted_multiclass_kl(np.stack([a, a]), natural, base.PHASE_FRACTIONS))
        v -= lam_rel * float(np.sum(rel * np.log1p(zg)))
        return v

    bv, ba = np.inf, None
    for s in [np.zeros(m), np.log(np.clip(natural, 1e-9, 1))]:
        r = minimize(obj, s, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-10})
        if float(r.fun) < bv:
            bv, ba = float(r.fun), to_a(r.x)
    return ba


def lam_for_overweight(packet, natural, rel, ridx, target_ow):
    lo, hi = 0.0, 0.02
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        a = optimize_aggregate(packet, natural, rel, mid)
        if float(np.mean(a[ridx] / natural[ridx])) < target_ow:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def load_uncheatable():
    packet, panel, domains, natural, tc, tb = eta_diag.load_packet()
    return packet, list(domains), np.asarray(natural, float), np.asarray(tc, float), int(tb)


def load_table9():
    _s, columns, domains, natural = base.load_raw_signal_panel()
    tc = base.load_domain_token_counts(domains)
    fitpanel, _m = paper_olmix.build_fit_panel(columns)
    tb = base.load_target_budget()
    packet = top_level_dsp.build_dsp_packet(fitpanel, columns, domains, tc, "table9_macro_bpb")
    return packet, list(domains), np.asarray(natural, float), np.asarray(tc, float), int(tb)


def split_from_share(agg, s, f0, f1):
    p1 = s * agg / f1
    p0 = (1.0 - s) * agg / f0
    return np.stack([p0 / p0.sum(), p1 / p1.sum()])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    f0, f1 = base.PHASE_FRACTIONS
    rows = []

    for objective, loader in [("uncheatable", load_uncheatable), ("table9", load_table9)]:
        packet, domains, natural, tc, tb = loader()
        idx = {d: i for i, d in enumerate(domains)}
        rel = np.array([1.0 if d in EVAL_RELEVANT else 0.0 for d in domains])
        ridx = [idx[d] for d in EVAL_RELEVANT if d in idx]
        optimize_aggregate.cache = fit_bowl_gamma1(packet)
        with fsspec.open(f"{REPAIR_GCS}/dsp_{objective}_exposure_all_deficits.csv", "r") as fh:
            win = pd.read_csv(fh).set_index("domain").reindex(domains)
        win_agg = win["aggregate_weight"].to_numpy(float)
        win_s = (f1 * win["phase_1_weight"].to_numpy(float)) / np.clip(win_agg, 1e-12, None)
        win_ow = float(np.mean(win_agg[ridx] / natural[ridx]))
        lam = lam_for_overweight(packet, natural, rel, ridx, win_ow)
        aug_agg = optimize_aggregate(packet, natural, rel, lam)

        cands = {
            "aug_profile": (aug_agg, win_s, "two"),
            "aug_profile_1phase": (aug_agg, np.full_like(win_s, f1), "one"),
            "blend_winner_aug": (0.5 * aug_agg + 0.5 * win_agg, win_s, "two"),
        }
        if objective == "table9":
            cands = {"aug_profile": cands["aug_profile"]}
        for role, (agg, s, phase) in cands.items():
            agg = agg / agg.sum()
            w = split_from_share(agg, s, f0, f1)
            frame = per_component.mixture_frame(
                domains=domains, natural=natural, weights=w, token_counts=tc, target_budget=tb
            )
            name = f"{objective}_{role}"
            d = args.output_dir / name
            d.mkdir(parents=True, exist_ok=True)
            frame.to_csv(d / "proposed_mixture_weights.csv", index=False)
            sim = base.simulated_epochs(w, tc, target_budget=tb)
            rows.append(
                dict(
                    candidate=name,
                    objective=objective,
                    role=role,
                    phase=phase,
                    lam_rel=lam,
                    evalrel_ow=float(np.mean(agg[ridx] / natural[ridx])),
                    tv_to_winner=float(0.5 * np.abs(agg - win_agg).sum()),
                    max_weight=float(w.max()),
                    max_sim_epoch=float(sim.max()),
                    q95_sim_epoch=float(np.quantile(sim, 0.95)),
                )
            )
            print(
                f"  {name:30s} ow={rows[-1]['evalrel_ow']:.2f} tv_win={rows[-1]['tv_to_winner']:.3f} "
                f"maxw={w.max():.3f} maxep={sim.max():.1f} q95ep={rows[-1]['q95_sim_epoch']:.2f}",
                flush=True,
            )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    print(f"\nWrote {len(rows)} candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
