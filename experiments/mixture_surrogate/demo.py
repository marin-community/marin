# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Runnable quickstart for the mixture surrogate (humaneval bpb).

    python experiments/mixture_surrogate/demo.py

Loads the frozen model, re-verifies it reproduces the research pipeline's out-of-fold Spearman,
scores the anchor mixture, proposes the best sweeps, and shows the always-on epoch penalty --
dormant on normal in-support mixtures, firing on a deliberate over-repetition mixture.
"""

import numpy as np
from scipy.stats import spearmanr
from surrogate import MixtureSurrogate, content, sq_hellinger

BUDGET = 1.0372343704053e13  # the swarm's effective (simulated) training budget the mixtures encode


def _selfcheck(m: MixtureSurrogate) -> float:
    """Out-of-fold Spearman of the content kernel on the frozen train set -- must match the pipeline."""
    h = content(m._train_w, m.v)
    d2 = sq_hellinger(h, h)
    y = m._y
    order = np.random.default_rng(0).permutation(len(y))  # 5-fold split, numpy only
    pred = np.zeros(len(y))
    for fold in np.array_split(order, 5):
        te = fold
        tr = np.setdiff1d(order, te)
        med = np.median(d2[np.ix_(tr, tr)][~np.eye(len(tr), dtype=bool)])
        g = m.meta["gamma_factor"] / med
        k = np.exp(-g * d2[np.ix_(tr, tr)])
        k[np.diag_indices_from(k)] += m.alpha
        a = np.linalg.solve(k, y[tr] - y[tr].mean())
        pred[te] = np.exp(-g * d2[np.ix_(te, tr)]) @ a + y[tr].mean()
    return float(spearmanr(pred, y).statistic)


def main() -> None:
    m = MixtureSurrogate(target="humaneval_bpb", budget_tokens=BUDGET)
    print(f"humaneval_bpb | {len(m.buckets)} buckets, {len(m._y)} train runs, budget {BUDGET:.0e}")
    print(f"self-check OOF Spearman = {_selfcheck(m):.3f}  (content kernel; epoch harm is additive on top)")

    p = m.predict(m.anchor())
    print(
        f"\nanchor (token-proportional): mean={p['mean']:.4f} = content {p['content_mean']:.4f} + "
        f"harm {p['epoch_harm']:.4f}   sd={p['sd']:.4f} nn={p['nn_distance']:.3f} in_support={p['in_support']}"
    )

    prop = m.propose(n=20000, top_k=3, seed=0)
    print(f"top-3 proposed (best first), in-support {prop['n_in_support']}/{prop['n_sampled']}:")
    for i in range(3):
        top = np.argsort(prop["weights"][i, 0])[::-1][:3]
        doms = ", ".join(f"{m.buckets[b]}={prop['weights'][i, 0, b]:.2f}" for b in top)
        print(
            f"  #{i+1} mean={prop['mean'][i]:.4f} (content {prop['content_mean'][i]:.4f} + "
            f"harm {prop['epoch_harm'][i]:.4f})  Δanchor={p['mean']-prop['mean'][i]:+.4f}  phase0-top: {doms}"
        )

    # guardrail demo: dump ~35% of phase-0 weight onto the smallest code bucket -> real over-repetition
    small_code = int(np.where(m._mask_code)[0][np.argmin(m._tj[m._mask_code])])
    w = m.anchor().copy()
    w[0] *= 0.65
    w[0, small_code] += 0.35
    over = m.predict(w)
    e0 = w[0, small_code] * m._f0 * BUDGET / m._tj[small_code]
    print(
        f"\nepoch guardrail: 35% phase-0 on {m.buckets[small_code]} "
        f"({m._tj[small_code]:.2e} tokens -> {e0:.0f} epochs at this budget)"
    )
    print(
        f"  mean={over['mean']:.4f} = content {over['content_mean']:.4f} + harm {over['epoch_harm']:.4f} bpb "
        f"({over['epoch_harm'] / m.meta['target']['seed_sigma_bpb']:.0f} sigma penalty)"
    )
    print("  (harm is 0 for the same mixture at a budget too small to force repetition -- e.g. 10B)")


if __name__ == "__main__":
    main()
