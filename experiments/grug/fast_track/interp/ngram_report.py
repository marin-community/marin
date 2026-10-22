# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report data for the n-gram neuron section, from local copies of ``ngram_score.npz``, ``ngram_validate.npz``,
``ngram_tokens.npz``, ``tokens.npz``, ``stats.npz`` and ``vocab.json`` (one directory per study tag).

- Data counts over both token samples: per bigram (a, b) with at least ``MIN_COUNT`` occurrences, H(next | b),
  H(next | a, b), the modal c and p(c | a, b).
- Per validated candidate: necessity (mean drop of log p(c) under mean ablation, first mentions and repeats),
  prefix-swap activation ratio and p(c), z-swap ratio, routing share of b, screen metrics, data entropies.
- Census: clear units (necessity >= ``CLEAR_NECESSITY`` and prefix swap keeps <= ``CLEAR_SWAP`` of the activation),
  counted directly in the top tier and extrapolated from the random tier, per layer.
- Hypothesis test: share of bigrams with an eligible unit by information gain and token kind, by H(next | b) x
  H(next | a, b), and clear units against matched bigrams without any candidate unit.

Usage: ``python -m experiments.grug.fast_track.interp.ngram_report <data root> <out.json>``.
"""

import itertools
import json
import sys

import numpy as np

from experiments.grug.fast_track.interp import ngram, ngram_validate

V = ngram.V
MIN_COUNT = 20
MIN_COUNT_RATE = 50
CLEAR_NECESSITY = 0.5
CLEAR_SWAP = 0.25
IG_BINS = (0, 1, 2, 3, 4, 5, 8)
HB_BINS = (0, 2, 4, 6, 12)
HAB_BINS = (0, 1, 2, 3, 12)
KINDS = ("word piece", "whole word", "symbol/other")
NUM_CONTROLS = 5
TAGS = ("tn-d512-ckpt1-rno", "lc1-l171-d512")


def token_kind(s: str) -> str:
    if len(s) >= 2 and s[0] == " " and s[1].isalpha():
        return "whole word"
    if s[:1].isalpha():
        return "word piece"
    return "symbol/other"


def bigram_counts(samples: list[dict]) -> dict[str, np.ndarray]:
    keys = []
    for t in samples:
        prev, _, nxt = ngram.context_arrays(t["tokens"], t["segment_ids"])
        ok = (prev >= 0) & (nxt >= 0) & (t["valid"] > 0)
        keys.append(((prev * V + t["tokens"].astype(np.int64)) * V + nxt)[ok])
    tri, n = np.unique(np.concatenate(keys), return_counts=True)
    ab, c = tri // V, tri % V
    bc = (ab % V) * V + c
    ubc, inv = np.unique(bc, return_inverse=True)
    nbc = np.bincount(inv, weights=n)
    ub, binv = np.unique(ubc // V, return_inverse=True)
    nb = np.bincount(binv, weights=nbc)
    h_b = np.bincount(binv, weights=-nbc * np.log(nbc / nb[binv])) / nb
    uab, ainv = np.unique(ab, return_inverse=True)
    nab = np.bincount(ainv, weights=n)
    h_ab = np.bincount(ainv, weights=-n * np.log(n / nab[ainv])) / nab
    order = np.lexsort((-n, ainv))
    first = np.r_[True, ainv[order][1:] != ainv[order][:-1]]
    keep = nab >= MIN_COUNT
    bidx = np.searchsorted(ub, uab % V)
    return dict(
        ab=uab[keep],
        n_ab=nab[keep],
        H_ab=h_ab[keep],
        H_b=h_b[bidx][keep],
        c=c[order][first][keep],
        p_c_ab=(n[order][first] / nab)[keep],
    )


def _mean(x, mask):
    return np.where(mask.sum(1) > 0, np.nansum(np.where(mask, x, 0), 1) / np.maximum(mask.sum(1), 1), np.nan)


def validated(r: dict) -> dict[str, np.ndarray]:
    ok, is_c, first = r["valid"], r["is_c"], r["first"]
    act = _mean(r["base/act"], ok)
    swap_act = np.stack([_mean(r[f"swap{k}/act"], ok) for k in range(ngram_validate.NUM_SWAPS)], 1)
    swap_p = np.stack([_mean(np.exp(r[f"swap{k}/logp_c"]), ok & is_c) for k in range(ngram_validate.NUM_SWAPS)], 1)
    swap_sel = np.stack([_mean(r[f"swap{k}/selected"].astype(float), ok) for k in range(ngram_validate.NUM_SWAPS)], 1)
    drop = r["base/logp_c"] - r["ablate/logp_c"]
    hz = r["has_z"]
    zratio = _mean(r["zswap/act"], ok & hz) / np.maximum(_mean(r["base/act"], ok & hz), 1e-6)
    ctx = np.nanmean(swap_act, 1) / np.maximum(act, 1e-6)
    nec = _mean(drop, ok & is_c & first)
    return dict(
        nec=nec,
        nec_rep=_mean(drop, ok & is_c & ~first),
        act=act,
        ctx=ctx,
        p_c=_mean(np.exp(r["base/logp_c"]), ok & is_c),
        p_swap=np.nanmean(swap_p, 1),
        sel_swap=np.nanmean(swap_sel, 1),
        zratio=zratio,
        clear=(nec >= CLEAR_NECESSITY) & (ctx <= CLEAR_SWAP),
    )


def census(r: dict, v: dict, n_eligible: int) -> dict:
    tiers, layer = r["tiers"], r["candidates"][:, 0]
    top, rnd = tiers == 0, tiers == 1
    weight = (n_eligible - top.sum()) / rnd.sum()
    masks = {"clear": v["clear"], **{f"nec>={t}": v["nec"] >= t for t in (0.1, 0.5, 1)}}
    est = {
        k: [float((m & top & (layer == li)).sum() + weight * (m & rnd & (layer == li)).sum()) for li in range(6)]
        for k, m in masks.items()
    }
    return dict(est=est, n_elig=n_eligible, w=float(weight))


def unit_rates(m: dict, cnt: dict, kinds: np.ndarray) -> dict:
    ok = ngram_validate.eligible(m)
    with_unit = set((m["cand_a"][ok] * V + m["cand_b"][ok]).tolist())
    has = np.asarray([x in with_unit for x in cnt["ab"]])
    kb = kinds[cnt["ab"] % V]
    gain = cnt["H_b"] - cnt["H_ab"]
    frequent = cnt["n_ab"] >= MIN_COUNT_RATE
    by_kind = {}
    for k in KINDS:
        rates = []
        for lo, hi in itertools.pairwise(IG_BINS):
            s = (kb == k) & (gain >= lo) & (gain < hi) & frequent
            rates.append(float(has[s].mean()) if s.any() else None)
        by_kind[k] = dict(rates=rates, overall=float(has[(kb == k) & frequent].mean()))
    grid = []
    for i in range(len(HB_BINS) - 1):
        row = []
        for j in range(len(HAB_BINS) - 1):
            s = (
                (cnt["H_b"] >= HB_BINS[i])
                & (cnt["H_b"] < HB_BINS[i + 1])
                & (cnt["H_ab"] >= HAB_BINS[j])
                & (cnt["H_ab"] < HAB_BINS[j + 1])
                & frequent
            )
            row.append([float(has[s].mean()) if s.any() else None, int(s.sum())])
        grid.append(row)
    return dict(
        ig_bins=list(IG_BINS),
        by_kind=by_kind,
        grid=grid,
        hb=list(HB_BINS),
        hab=list(HAB_BINS),
        n_elig=int(ok.sum()),
        elig_by_layer=np.bincount(m["unit_index"][ok, 0], minlength=6).tolist(),
    )


def matched_controls(rows: list[dict], m: dict, cnt: dict, kinds: np.ndarray) -> dict:
    """Each clear unit's bigram against bigrams with no candidate unit at all (any unit whose modal context is that
    bigram with selectivity >= 3), matched on frequency (within 2x) and p(c | a, b) (within 0.1)."""
    any_unit = (m["cand_a"] >= 0) & (m["n_ab"] >= ngram_validate.MIN_OCCURRENCES) & (m["selectivity"] >= 3)
    used = set((m["cand_a"][any_unit] * V + m["cand_b"][any_unit]).tolist())
    free = np.asarray([x not in used for x in cnt["ab"]])
    rng = np.random.default_rng(0)
    units, controls = [], []
    for x in rows:
        if not x["clear"] or x["n_ab"] is None:
            continue
        s = (
            free
            & (cnt["n_ab"] >= x["n_ab"] / 2)
            & (cnt["n_ab"] <= x["n_ab"] * 2)
            & (np.abs(cnt["p_c_ab"] - x["p_c_ab"]) <= 0.1)
        )
        pool = np.flatnonzero(s)
        if not len(pool):
            continue
        units.append(x)
        controls += list(rng.choice(pool, size=min(NUM_CONTROLS, len(pool)), replace=False))
    controls = np.asarray(controls)

    def summary(k, hb, hab):
        k, hb, hab = np.asarray(k), np.asarray(hb), np.asarray(hab)
        return dict(
            frag=float((k == "word piece").mean()),
            word=float((k == "whole word").mean()),
            other=float((k == "symbol/other").mean()),
            H_b=float(np.median(hb)),
            H_ab=float(np.median(hab)),
            IG=float(np.median(hb - hab)),
        )

    return dict(
        units=summary([x["kind"] for x in units], [x["H_b"] for x in units], [x["H_ab"] for x in units]),
        controls=summary(kinds[cnt["ab"][controls] % V], cnt["H_b"][controls], cnt["H_ab"][controls]),
        n_units=len(units),
        n_controls=len(controls),
    )


def _rounded(x, d=3):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def rows_for(r: dict, v: dict, m: dict, cnt: dict, stats: dict, vocab: list[str], kinds: np.ndarray) -> list[dict]:
    index = {tuple(u): j for j, u in enumerate(m["unit_index"])}
    routed_ids, token_counts = stats["expert_token_ids"], stats["token_counts"]
    out = []
    for j, (li, e, i, a, b, c, z) in enumerate(r["candidates"]):
        mj = index[(li, e, i)]
        k = np.searchsorted(cnt["ab"], a * V + b)
        found = k < len(cnt["ab"]) and cnt["ab"][k] == a * V + b

        out.append(
            dict(
                unit=f"L{li}·{'S' if e < 0 else f'E{e}'}·n{i}",
                layer=int(li),
                tier=int(r["tiers"][j]),
                a=vocab[a],
                b=vocab[b],
                c=vocab[c],
                z=vocab[z] if z >= 0 else None,
                kind=str(kinds[b]),
                route=_rounded(routed_ids[li, e, b] / max(token_counts[b], 1)) if e >= 0 else 1.0,
                conc=_rounded(m["concentration"][mj]),
                selv=_rounded(m["selectivity"][mj]),
                r2g=_rounded(m["r2_gain"][mj]),
                crank=int(m["ot_c_rank"][mj]),
                n_ab=_rounded(cnt["n_ab"][k]) if found else None,
                H_b=_rounded(cnt["H_b"][k]) if found else None,
                H_ab=_rounded(cnt["H_ab"][k]) if found else None,
                p_c_ab=_rounded(cnt["p_c_ab"][k]) if found else None,
                clear=bool(v["clear"][j]),
                **{
                    key: _rounded(v[key][j])
                    for key in ("nec", "nec_rep", "act", "ctx", "p_c", "p_swap", "sel_swap", "zratio")
                },
            )
        )
    return out


def load(path: str) -> dict:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def tag_report(root: str, tag: str) -> dict:
    d = f"{root}/{tag}"
    with open(f"{d}/vocab.json") as f:
        vocab = json.load(f)
    kinds = np.asarray([token_kind(s) for s in vocab])
    cnt = bigram_counts([load(f"{d}/tokens.npz"), load(f"{d}/ngram_tokens.npz")])
    m = ngram_validate.score_metrics(load(f"{d}/ngram_score.npz"))
    r = load(f"{d}/ngram_validate.npz")
    v = validated(r)
    rows = rows_for(r, v, m, cnt, load(f"{d}/stats.npz"), vocab, kinds)
    n_eligible = int(ngram_validate.eligible(m).sum())
    pop = dict(
        thresholds=[None, *ngram_validate.POPULATION_THRESHOLDS],
        loss=[float(x) for x in r["pop_loss"]],
        members=[0, *[int((r["necessity"] >= t).sum()) for t in ngram_validate.POPULATION_THRESHOLDS]],
    )
    return dict(
        rows=rows,
        census=census(r, v, n_eligible),
        rates=unit_rates(m, cnt, kinds),
        matched=matched_controls(rows, m, cnt, kinds),
        pop=pop,
    )


def main() -> None:
    root, out = sys.argv[1], sys.argv[2]
    tn, l171 = (tag_report(root, t) for t in TAGS)
    report = dict(
        rows=tn["rows"],
        census=tn["census"],
        pop=tn["pop"],
        hyp=dict(tn=tn["rates"], l171=l171["rates"], matched=tn["matched"]),
        l171=dict(census=l171["census"], pop=l171["pop"], matched=l171["matched"]),
    )
    with open(out, "w") as f:
        json.dump(report, f)


if __name__ == "__main__":
    main()
