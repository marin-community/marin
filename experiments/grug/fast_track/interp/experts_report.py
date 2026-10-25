# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-side analysis of the ``experts`` passes (numpy only): role experts from the class census, layer-wide
exclusion on the census samples against the shuffle and matched-covariance Gaussian nulls, and the per-expert deep
dive (sub-role selectivity, clustering, sub-role exclusion, ablation concentration, steering)."""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from experiments.grug.fast_track.interp import experts

SEQ_LEN = 4096
LIFT_THRESHOLDS = (0.5, 0.2, 0.1)
MIN_EXPECTED = 20.0
MIN_EXPECTED_STRONG = 5.0  # strong events are rare: the nulls at the same row count calibrate the noise


# ---------------------------------------------------------------------------------------------------------------
# Role experts


def role_table(census: dict, min_tokens: int = 200) -> dict:
    """Per class and layer the expert with the largest lift-weighted share: ``class_share`` (the share of the
    expert's routed tokens in the class), ``share_of_class`` (the share of the class's tokens routed to it; uniform
    is ``K / E``), ``lift`` (class share over the class base rate) and ``score = class_share * share_of_class``."""
    counts = census["counts"]  # [L, E, C]
    totals = census["class_tokens"]  # [C]
    classes = [str(c) for c in census["classes"]]
    base = totals / totals[0]
    out = {}
    for ci, c in enumerate(classes):
        if ci == 0 or totals[ci] < min_tokens:
            continue
        cs = counts[:, :, ci] / np.maximum(counts[:, :, 0], 1)
        soc = counts[:, :, ci] / totals[ci]
        score = cs * soc
        rows = []
        for li in range(counts.shape[0]):
            order = np.argsort(-score[li])[:3]
            rows.append(
                [
                    dict(
                        layer=li,
                        expert=int(e),
                        class_share=float(cs[li, e]),
                        share_of_class=float(soc[li, e]),
                        lift=float(cs[li, e] / max(base[ci], 1e-12)),
                        score=float(score[li, e]),
                    )
                    for e in order
                ]
            )
        out[c] = dict(base_rate=float(base[ci]), tokens=float(totals[ci]), by_layer=rows)
    return out


def specialization_summary(census: dict) -> dict:
    """Per layer: the routing entropy of each class over experts (bits; uniform log2(E)) and the share of the class
    held by its top-10 experts (the top-k slots hold 10 of the 10 * N tokens' assignments)."""
    counts = census["counts"]
    totals = census["class_tokens"]
    classes = [str(c) for c in census["classes"]]
    res = {}
    for ci, c in enumerate(classes):
        if totals[ci] < 200:
            continue
        p = counts[:, :, ci] / np.maximum(counts[:, :, ci].sum(-1, keepdims=True), 1)
        ent = -np.sum(np.where(p > 0, p * np.log2(np.maximum(p, 1e-30)), 0), axis=-1)
        top10 = np.sort(p, axis=-1)[:, ::-1][:, :10].sum(-1)
        res[c] = dict(entropy_bits=ent.tolist(), top10_share=top10.tolist())
    return res


def hist_quantile(hist: np.ndarray, q: float) -> float:
    c = np.cumsum(hist) / max(hist.sum(), 1e-30)
    return float(np.searchsorted(c, q))


def concentration_summary(census: dict) -> dict:
    """Per layer the median token's units for 50/80/90/99% of ``sum c`` and the median participation ratio."""
    kh = census["k_hist"]  # [L, Q, I + 1]
    out = {
        f"k{int(q * 100)}": [hist_quantile(kh[li, qi], 0.5) for li in range(kh.shape[0])]
        for qi, q in enumerate(census["quants"])
    }
    out["pr"] = [hist_quantile(census["pr_hist"][li], 0.5) for li in range(kh.shape[0])]
    act = census["active_hist"]
    out["active"] = [
        float(np.sum(np.arange(act.shape[1]) * act[li]) / max(act[li].sum(), 1)) for li in range(act.shape[0])
    ]
    return out


# ---------------------------------------------------------------------------------------------------------------
# Exclusion


def gaussian_null(pre: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` rows from a Gaussian with the data's mean and covariance of the pre-activations (the matched-covariance
    "orthant" null: a pair's co-firing is then whatever its linear correlation implies)."""
    mu = pre.mean(0)
    cov = np.cov(pre, rowvar=False) + 1e-6 * np.eye(pre.shape[1])
    return rng.multivariate_normal(mu, cov, size=n, method="cholesky").astype(np.float32)


def pair_census(bits: np.ndarray, min_expected: float = MIN_EXPECTED) -> dict:
    """On/off (or strong-event) co-occurrence census of ``[n, I]`` bits: per eligible pair (expected co-count at
    least ``min_expected``), lift and the hypergeometric shuffle-null z. Returns counts below the lift thresholds and
    the lift / z arrays of eligible pairs."""
    b = bits.astype(np.float32)
    n = b.shape[0]
    cnt = b.T @ b
    ni = np.diag(cnt)
    mu = np.outer(ni, ni) / n
    iu = np.triu_indices(len(ni), k=1)
    elig = mu[iu] >= min_expected
    obs = cnt[iu][elig]
    m = mu[iu][elig]
    lift = obs / m
    a, c = ni[iu[0]][elig], ni[iu[1]][elig]
    var = a * c * (n - a) * (n - c) / (n * n * max(n - 1, 1))
    z = (obs - m) / np.sqrt(np.maximum(var, 1e-6))
    return dict(
        pairs=int(elig.sum()),
        below={str(t): int((lift < t).sum()) for t in LIFT_THRESHOLDS},
        z_lt_minus5=float((z < -5).mean()) if len(z) else float("nan"),
        lift_p01=float(np.percentile(lift, 1)) if len(lift) else float("nan"),
        lift_p99=float(np.percentile(lift, 99)) if len(lift) else float("nan"),
        lift_min=float(lift.min()) if len(lift) else float("nan"),
        lift=lift,
        z=z,
        idx=(iu[0][elig], iu[1][elig]),
    )


def strong_bits(pre: np.ndarray, norms: np.ndarray, kind: str, q: float = 0.9) -> np.ndarray:
    act = np.square(np.maximum(pre.astype(np.float32), 0))
    if kind == "on":
        return act > 0
    if kind == "q90":
        thr = np.quantile(act, q, axis=0)
        return (act > thr[None, :]) & (act > 0)
    if kind == "c5":
        c = act * norms[None, :]
        return c >= 0.05 * np.maximum(c.sum(-1, keepdims=True), 1e-30)
    raise ValueError(kind)


def exclusion_vs_nulls(
    pre: np.ndarray, norms: np.ndarray, rng: np.random.Generator, null_rows: int | None = None
) -> dict:
    """Observed vs shuffle vs matched-Gaussian exclusion for on/off, q90 and c5 events of one expert."""
    n = pre.shape[0]
    g = gaussian_null(pre.astype(np.float64), null_rows or n, rng)
    shuf = np.stack([rng.permutation(pre[:, i]) for i in range(pre.shape[1])], axis=1)
    out = {}
    for kind in ("on", "q90", "c5"):
        res = {}
        for name, x in (("obs", pre), ("shuffle", shuf), ("gauss", g)):
            r = pair_census(strong_bits(x, norms, kind), MIN_EXPECTED if kind == "on" else MIN_EXPECTED_STRONG)
            res[name] = {k: v for k, v in r.items() if k not in ("lift", "z", "idx")}
            if name == "obs":
                res["obs_lift"] = r["lift"]
                res["obs_idx"] = r["idx"]
        out[kind] = res
    return out


def merge_census(rows: list[dict]) -> dict:
    """Sum pair counts over experts; median of the percentiles."""
    out = dict(pairs=sum(r["pairs"] for r in rows))
    out["below"] = {t: sum(r["below"][t] for r in rows) for t in rows[0]["below"]}
    out["below_frac"] = {t: out["below"][t] / max(out["pairs"], 1) for t in out["below"]}
    for k in ("z_lt_minus5", "lift_p01", "lift_p99", "lift_min"):
        vals = [r[k] for r in rows if np.isfinite(r[k])]
        out[k] = float(np.median(vals)) if vals else float("nan")
    out["lift_min"] = float(min([r["lift_min"] for r in rows if np.isfinite(r["lift_min"])], default=np.nan))
    return out


def layer_exclusion(census: dict, rng: np.random.Generator, experts_per_layer: int = 8) -> list[dict]:
    """Layer-wide exclusion on the census samples (``experts_per_layer`` of the sampled experts per layer)."""
    res = []
    for li in range(census["sample_pre"].shape[0]):
        per = {k: {n: [] for n in ("obs", "shuffle", "gauss")} for k in ("on", "q90", "c5")}
        for j in range(experts_per_layer):
            n = int(census["sample_filled"][li, j])
            if n < 512:
                continue
            e = int(census["sample_experts"][li, j])
            pre = census["sample_pre"][li, j, :n].astype(np.float32)
            r = exclusion_vs_nulls(pre, census["norms"][li, e], rng, null_rows=n)
            for k in per:
                for name in per[k]:
                    per[k][name].append(r[k][name])
        res.append({k: {name: merge_census(v) for name, v in d.items() if v} for k, d in per.items()})
    return res


# ---------------------------------------------------------------------------------------------------------------
# Deep dive


def row_features(deep: dict, feats, vocab) -> dict:
    """Per deep row: current / previous / next token, digit features (first digit of the current and next token,
    position in the digit run, run length, run digits) and the class labels."""
    tok_flat = deep["tok"]
    seq, pos = tok_flat // SEQ_LEN, tok_flat % SEQ_LEN
    tokens = deep["tokens"].astype(np.int64)
    segs = deep["segment_ids"]
    cur = tokens[seq, pos]
    nxt = np.where(pos + 1 < SEQ_LEN, tokens[seq, np.minimum(pos + 1, SEQ_LEN - 1)], -1)
    nxt_same = (pos + 1 < SEQ_LEN) & (segs[seq, np.minimum(pos + 1, SEQ_LEN - 1)] == segs[seq, pos])
    nxt = np.where(nxt_same, nxt, -1)
    prv = np.where(pos > 0, tokens[seq, np.maximum(pos - 1, 0)], -1)
    cls = np.zeros((len(cur), experts.NC), bool)
    run_pos = np.full(len(cur), -1)
    run_len = np.full(len(cur), -1)
    run_digits = np.full(len(cur), -1)
    for s in np.unique(seq):
        m = seq == s
        t = tokens[s : s + 1]
        sg = segs[s : s + 1]
        c = experts.class_masks(t, sg, np.ones_like(t), str(deep["sources"][s]), feats, vocab)[0]
        runs = experts.digit_runs(t, sg, feats)
        cls[m] = c[pos[m]] > 0
        run_pos[m] = runs["run_pos"][0, pos[m]]
        run_len[m] = runs["run_len"][0, pos[m]]
        run_digits[m] = runs["run_digits"][0, pos[m]]
    fd = feats["first_digit"]
    return dict(
        cur=cur,
        nxt=nxt,
        prv=prv,
        cur_digit=fd[cur],
        next_digit=np.where(nxt >= 0, fd[np.maximum(nxt, 0)], -1),
        next_is_digit=(nxt >= 0) & feats["digit"][np.maximum(nxt, 0)],
        run_pos=run_pos,
        run_len=run_len,
        run_digits=run_digits,
        cls=cls,
        source=np.asarray([str(deep["sources"][s]) for s in seq]),
    )


def selectivity(act: np.ndarray, groups: np.ndarray, values: list[int]) -> dict:
    """Per unit: eta^2 of the activation over the groups (rows with a group in ``values``), the preferred group and
    its selectivity index ``(m_best - m_rest) / (m_best + m_rest)``."""
    keep = np.isin(groups, values)
    a, g = act[keep], groups[keep]
    means = np.stack([a[g == v].mean(0) if (g == v).any() else np.zeros(a.shape[1]) for v in values])
    counts = np.asarray([(g == v).sum() for v in values])
    grand = a.mean(0)
    between = np.sum(counts[:, None] * np.square(means - grand[None]), axis=0)
    total = np.sum(np.square(a - grand[None]), axis=0)
    eta2 = between / np.maximum(total, 1e-30)
    best = np.argmax(means, axis=0)
    m_best = means[best, np.arange(a.shape[1])]
    rest = (np.sum(means * counts[:, None], 0) - m_best * counts[best]) / np.maximum(counts.sum() - counts[best], 1)
    si = (m_best - rest) / np.maximum(m_best + rest, 1e-30)
    return dict(eta2=eta2, best=np.asarray(values)[best], si=si, means=means, counts=counts)


def cluster_units(act: np.ndarray, k: int, min_rate: float = 0.01) -> dict:
    """Average-linkage clusters of the units (firing at least ``min_rate`` on the rows) on ``1 - corr`` of their
    activations."""
    live = np.nonzero((act > 0).mean(0) >= min_rate)[0]
    x = act[:, live]
    x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-12)
    corr = np.clip(x.T @ x / x.shape[0], -1, 1)
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(z, k, criterion="maxclust")
    return dict(units=live, labels=labels, corr=corr)
