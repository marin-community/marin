# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# H3 Phase 2: content surrogate proposes a mixture over 39 old + 1 new bucket (dolma_starcoder).
# Surrogate = KME_RIDGE + KERNEL_HELLINGER (per-phase semantic), ensemble = rank-average.
# Reuses retrodiction.py feature code paths (imported), extended to 40 domains.
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import KFold

from experiments.datakit.mixture_features import featurize


def _rj(path):
    with open(path) as f:
        return json.load(f)


# --- verbatim copies from experiments/datakit/mixture_features/retrodiction.py ---
SEED = 0
N_INNER_FOLDS = 5
RIDGE_ALPHAS = np.logspace(-3, 3, 25)
KR_GAMMA_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
KR_ALPHAS = np.logspace(-3, 2, 6)


def _ridge_solve(x_tr, y_tr, x_te, alphas):
    mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    a, b = (x_tr - mu) / sd, (x_te - mu) / sd
    ym = y_tr.mean()
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    uty = u.T @ (y_tr - ym)
    bv = b @ vt.T
    out = np.empty((x_te.shape[0], len(alphas)))
    for i, al in enumerate(alphas):
        out[:, i] = bv @ (uty * s / (s**2 + al)) + ym
    return out


def ridge_cv_predict(x_tr, y_tr, x_te):
    kf = KFold(N_INNER_FOLDS, shuffle=True, random_state=SEED)
    sse = np.zeros(len(RIDGE_ALPHAS))
    for tr, va in kf.split(x_tr):
        p = _ridge_solve(x_tr[tr], y_tr[tr], x_tr[va], RIDGE_ALPHAS)
        sse += ((p - y_tr[va][:, None]) ** 2).sum(axis=0)
    best = int(np.argmin(sse))
    return _ridge_solve(x_tr, y_tr, x_te, RIDGE_ALPHAS[best : best + 1])[:, 0]


def _sq_hellinger(h_phases):
    n = h_phases.shape[0]
    d = np.zeros((n, n))
    for p in range(h_phases.shape[1]):
        s = np.sqrt(np.clip(h_phases[:, p, :], 0.0, None))
        d += np.clip(1.0 - s @ s.T, 0.0, None)
    return d / h_phases.shape[1]


def _arm_features(w, v40, v1000, v5000, rff, centroids):
    n = w.shape[0]
    h40 = np.stack([w[:, p, :] @ v40.T for p in range(2)], axis=1)
    h1000 = np.stack([w[:, p, :] @ v1000.T for p in range(2)], axis=1)
    h5000 = np.stack([w[:, p, :] @ v5000.T for p in range(2)], axis=1)
    return {
        "h40": h40.reshape(n, -1),
        "h1000": h1000.reshape(n, -1),
        "kme": (h5000 @ centroids).reshape(n, -1),
        "rff": np.concatenate([w[:, p, :] @ rff for p in range(2)], axis=1),
        "d2_hell": _sq_hellinger(h1000),
        "h40_phases": h40,
    }


# --- end verbatim copies ---

SCRATCH = Path("scratch/mixture_features")
HIST = SCRATCH / "domain_histograms"
BASIS = SCRATCH / "basis"
CAND = SCRATCH / "h3" / "candidate_histograms"
OUT = SCRATCH / "h3"
NEW = "dolma_starcoder"
TARGET = "eval/uncheatable_eval/bpb"
NEW_CAP = 0.20  # max new-bucket share per phase
N_CAND = 100_000
TOP_K = 64
rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Load 39 domain histograms + candidate starcoder histogram -> 40-domain basis
# ---------------------------------------------------------------------------
def load_hist(hist_dir, meta, domain, dmeta, rff_by_domain, basis):
    df = pd.read_parquet(hist_dir / dmeta["parquet"])
    counts = {
        (int(c), int(q)): int(t)
        for c, q, t in zip(df["cluster_id"], df["quality_bucket"], df["token_count"], strict=True)
    }
    bs = dmeta["bucket_stats"]
    return featurize.DomainHistogram(
        domain=domain,
        basis=basis,
        sample_size=dmeta["sample_size"],
        token_count=dmeta["token_count"],
        seed=dmeta["seed"],
        counts=counts,
        rff_mean=tuple(np.asarray(rff_by_domain[domain], dtype=np.float64).tolist()),
        stats=featurize.BucketStats(
            total_tokens_available=bs["total_tokens_available"],
            mean_doc_tokens=bs["mean_doc_tokens"],
            duplicate_frac=bs["duplicate_frac"],
            loss_masked_frac=bs["loss_masked_frac"],
        ),
    )


meta = json.loads((HIST / "_meta.json").read_text())
b = meta["basis"]
basis = featurize.MixtureBasis(
    embedder=b["embedder"],
    tokenizer=b["tokenizer"],
    centroids_path=b["centroids_path"],
    centroids_sha256=b["centroids_sha256"],
    k=b["k"],
    view_paths={int(k): v for k, v in b["view_paths"].items()},
    view_sha256={int(k): v for k, v in b["view_sha256"].items()},
    quality_scorer=b["quality_scorer"],
    quality_scorer_sha256=b["quality_scorer_sha256"],
    rff_dim=b["rff_dim"],
    rff_seed=b["rff_seed"],
    rff_bandwidth=b["rff_bandwidth"],
)
rff39 = np.load(HIST / meta["rff_means_file"])
rff_by = dict(zip(rff39["domains"].tolist(), rff39["rff_means"], strict=True))
hists = [load_hist(HIST, meta, d, dm, rff_by, basis) for d, dm in meta["domains"].items()]

# candidate starcoder
cmeta = json.loads((CAND / "_meta.json").read_text())
crff = np.load(CAND / "rff_means.npz")
crff_by = dict(zip(crff["domains"].tolist(), crff["rff_means"], strict=True))
hists.append(load_hist(CAND, cmeta, NEW, cmeta["domains"][NEW], crff_by, basis))
hists.sort(key=lambda h: h.domain)

views = {k: np.load(BASIS / f"lookup_5000_to_{k}.npy") for k in (40, 1000)}
centroids = np.load(BASIS / "centroids_5000.npy").astype(np.float64)
V40, order = featurize.composition_matrix(hists, 40, views)
V1000, _ = featurize.composition_matrix(hists, 1000, views)
V5000, _ = featurize.composition_matrix(hists, 5000, views)  # (5000,40)
rff_mat = np.stack([np.asarray(dict((h.domain, h.rff_mean) for h in hists)[d]) for d in order], axis=0)  # (40,2048)
DOMAINS = order  # sorted 40
NEW_J = DOMAINS.index(NEW)
D = len(DOMAINS)
print(f"40 domains, new bucket '{NEW}' at sorted index {NEW_J}")

# ---------------------------------------------------------------------------
# Training runs: 238 300M rule-passers -> w (238,2,40) [starcoder col = 0], y = bpb
# ---------------------------------------------------------------------------
runs = pd.read_parquet(SCRATCH / "runs.parquet")
sub = runs[runs["scale"] == "300m_6b"].reset_index(drop=True)
assert len(sub) == 238
y_tr = sub[TARGET].to_numpy(dtype=np.float64)
w_tr = np.zeros((len(sub), 2, D))
for j, d in enumerate(DOMAINS):
    if d == NEW:
        continue
    for p in (0, 1):
        w_tr[:, p, j] = sub[f"phase_{p}_{d}"].to_numpy(dtype=np.float64)
assert np.allclose(w_tr.sum(axis=2), 1.0, atol=1e-6)

# token prior over 40 buckets
tok = _rj("experiments/datakit/mixture_features/data/domain_token_counts.json")["available_tokens"]
tok[NEW] = _rj(str(OUT / "candidate_token_avail.json"))[NEW]
tok_vec = np.array([tok[d] for d in DOMAINS], dtype=np.float64)
tok_prior = tok_vec / tok_vec.sum()


# ---------------------------------------------------------------------------
# Feature builders on 40-domain basis (semantic arm only)
# ---------------------------------------------------------------------------
def kme_feats(w):
    return _arm_features(w, V40, V1000, V5000, rff_mat, centroids)["kme"]  # (n,384)


def h1000_phases(w):  # (n,2,1000)
    return np.stack([w[:, p, :] @ V1000.T for p in range(2)], axis=1)


kme_tr = kme_feats(w_tr)
h1000_tr = h1000_phases(w_tr)


# ---------------------------------------------------------------------------
# KERNEL_HELLINGER: replicate kernel_cv_predict, train=238, predict arbitrary candidate block
# (avoids materializing a 100k x 100k matrix -- only tr-tr and cand-tr blocks)
# ---------------------------------------------------------------------------
def sqrth(h):  # (n,2,1000) -> list of sqrt per phase
    return [np.sqrt(np.clip(h[:, p, :], 0.0, None)) for p in range(2)]


s_tr = sqrth(h1000_tr)


def d2_hell_block(hA_sqrt, hB_sqrt):  # mean over phases of (1 - sA@sB.T)
    n = 2
    d = np.zeros((hA_sqrt[0].shape[0], hB_sqrt[0].shape[0]))
    for p in range(n):
        d += np.clip(1.0 - hA_sqrt[p] @ hB_sqrt[p].T, 0.0, None)
    return d / n


d_trtr = d2_hell_block(s_tr, s_tr)


def _kr_fit(k_tr, yv, k_te_tr, alpha):
    ym = yv.mean()
    dual = np.linalg.solve(k_tr + alpha * np.eye(len(yv)), yv - ym)
    return k_te_tr @ dual + ym


# select (gamma, alpha) by inner 5-fold CV on train (exact copy of kernel_cv_predict logic)
med = float(np.median(d_trtr[~np.eye(len(y_tr), dtype=bool)]))
gammas = np.asarray(KR_GAMMA_FACTORS) / max(med, 1e-12)
kf = KFold(N_INNER_FOLDS, shuffle=True, random_state=SEED)
folds = list(kf.split(np.arange(len(y_tr))))
best, best_sse = None, np.inf
for g in gammas:
    kfull = np.exp(-g * d_trtr)
    for al in KR_ALPHAS:
        sse = 0.0
        for itr, iva in folds:
            p = _kr_fit(kfull[np.ix_(itr, itr)], y_tr[itr], kfull[np.ix_(iva, itr)], al)
            sse += ((p - y_tr[iva]) ** 2).sum()
        if sse < best_sse:
            best, best_sse = (g, al), sse
KH_G, KH_AL = best
K_trtr = np.exp(-KH_G * d_trtr)
KH_dual = np.linalg.solve(K_trtr + KH_AL * np.eye(len(y_tr)), y_tr - y_tr.mean())
print(f"KERNEL_HELLINGER: gamma={KH_G:.4g} alpha={KH_AL:.4g} (median d2={med:.4g})")


def kh_predict(w_block):  # (m,2,40) -> predicted bpb
    s_b = sqrth(h1000_phases(w_block))
    d_bt = d2_hell_block(s_b, s_tr)  # (m,238)
    return np.exp(-KH_G * d_bt) @ KH_dual + y_tr.mean()


def kme_predict(w_block):
    return ridge_cv_predict(kme_tr, y_tr, kme_feats(w_block))


# ---------------------------------------------------------------------------
# RegMix-style candidate generation (two-phase, 40 buckets)
# ---------------------------------------------------------------------------
def cap_new(w2):  # w2 (m,2,D): cap new bucket share <= NEW_CAP per phase, renormalize rest
    over = w2[:, :, NEW_J] > NEW_CAP
    for p in range(2):
        m = over[:, p]
        if m.any():
            rest = w2[m, p, :].copy()
            rest[:, NEW_J] = 0.0
            rest = rest / rest.sum(axis=1, keepdims=True) * (1 - NEW_CAP)
            rest[:, NEW_J] = NEW_CAP
            w2[m, p, :] = rest
    return w2


# 70% Dirichlet(alpha = token_prior * temp), temp in {0.2,0.5,1.0}
n_dir = int(0.70 * N_CAND)
temps = [0.2, 0.5, 1.0]
dir_blocks = []
per = n_dir // len(temps)
for t in temps:
    alpha = np.maximum(tok_prior * t * D, 1e-3)  # scale so mean concentration ~ temp
    for _p in range(2):
        pass
    w0 = rng.dirichlet(alpha, size=per)
    w1 = rng.dirichlet(alpha, size=per)
    dir_blocks.append(np.stack([w0, w1], axis=1))
w_dir = np.concatenate(dir_blocks, axis=0)

# 30% perturbations of top-10 historical mixtures with new bucket injected at share in [0.01,0.20]
n_pert = N_CAND - w_dir.shape[0]
top10_idx = np.argsort(y_tr)[:10]
pert = []
per_mix = n_pert // 10 + 1
for idx in top10_idx:
    base = w_tr[idx].copy()  # (2,40), starcoder=0
    for _ in range(per_mix):
        s = rng.uniform(0.01, 0.20)
        m = np.zeros((2, D))
        for p in (0, 1):
            b39 = base[p].copy()
            b39[NEW_J] = 0.0
            b39 = b39 / b39.sum() * (1 - s)
            b39[NEW_J] = s
            # Dirichlet perturbation: mix with dirichlet noise (concentration 200)
            noise = rng.dirichlet(np.maximum(b39 * 200.0, 1e-3))
            m[p] = 0.7 * b39 + 0.3 * noise
            m[p] /= m[p].sum()
        pert.append(m)
w_pert = np.array(pert[:n_pert])
w_cand = np.concatenate([w_dir, w_pert], axis=0)[:N_CAND]
w_cand = cap_new(w_cand)
w_cand = w_cand / w_cand.sum(axis=2, keepdims=True)
print(f"generated {w_cand.shape[0]} candidates ({n_dir} dirichlet + {w_pert.shape[0]} perturb)")


# ---------------------------------------------------------------------------
# Predict, ensemble (rank-average), pick top-64
# ---------------------------------------------------------------------------
def batched(fn, w, bs=5000):
    out = []
    for i in range(0, w.shape[0], bs):
        out.append(fn(w[i : i + bs]))
    return np.concatenate(out)


kme_pred = batched(kme_predict, w_cand)
kh_pred = batched(kh_predict, w_cand)

rank_ens = (rankdata(kme_pred) + rankdata(kh_pred)) / 2.0  # lower bpb -> lower rank -> better
top = np.argsort(rank_ens)[:TOP_K]
PROPOSAL = w_cand[top].mean(axis=0)
PROPOSAL = PROPOSAL / PROPOSAL.sum(axis=1, keepdims=True)
print(f"top-64 ensemble mean predicted bpb: kme={kme_pred[top].mean():.4f} kh={kh_pred[top].mean():.4f}")

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
best_j = int(np.argmin(y_tr))
anchor = w_tr[best_j].copy()  # (2,40), starcoder=0 -- best historical mixture
anchor_realized = float(y_tr[best_j])

# OLMIX_REUSE: shrink best historical by token-proportional new share, both phases
s_tp = tok[NEW] / tok_vec.sum()
olmix = np.zeros((2, D))
for p in (0, 1):
    a = anchor[p].copy()
    a[NEW_J] = 0.0
    a = a / a.sum() * (1 - s_tp)
    a[NEW_J] = s_tp
    olmix[p] = a

# TOKEN_PROPORTIONAL over 40, both phases
tokprop = np.stack([tok_prior, tok_prior], axis=0)


def predict_one(w2):  # (2,40)
    wb = w2[None]
    return float(kme_predict(wb)[0]), float(kh_predict(wb)[0])


mixes = {"PROPOSAL": PROPOSAL, "OLMIX_REUSE": olmix, "TOKEN_PROPORTIONAL": tokprop, "ANCHOR": anchor}
pred = {}
for name, w2 in mixes.items():
    assert np.allclose(w2.sum(axis=1), 1.0, atol=1e-6), (name, w2.sum(axis=1))
    km, kh = predict_one(w2)
    pred[name] = {
        "kme_pred_bpb": km,
        "kh_pred_bpb": kh,
        "ensemble_mean_pred_bpb": (km + kh) / 2.0,
        "new_bucket_share_phase": [float(w2[0, NEW_J]), float(w2[1, NEW_J])],
    }

for name in mixes:
    pred[name]["pred_regret_vs_anchor"] = pred[name]["ensemble_mean_pred_bpb"] - pred["ANCHOR"]["ensemble_mean_pred_bpb"]
pred["ANCHOR"]["realized_bpb_historical"] = anchor_realized
pred["ANCHOR"]["realized_run_name"] = sub.loc[best_j, "run_name"]


# top-10 weights per mix (by phase-averaged weight)
def top10(w2):
    avg = w2.mean(axis=0)
    idx = np.argsort(avg)[::-1][:10]
    return {DOMAINS[i]: {"phase_0": float(w2[0, i]), "phase_1": float(w2[1, i])} for i in idx}


result = {
    "new_bucket": NEW,
    "new_bucket_sorted_index": NEW_J,
    "domains_sorted": DOMAINS,
    "new_cap_per_phase": NEW_CAP,
    "token_proportional_new_share": s_tp,
    "kernel_hellinger": {"gamma": KH_G, "alpha": KH_AL, "median_d2": med},
    "anchor_realized_bpb": anchor_realized,
    "anchor_run_name": sub.loc[best_j, "run_name"],
    "predictions": pred,
    "top10_weights": {name: top10(w2) for name, w2 in mixes.items()},
    "full_weights": {
        name: {
            "phase_0": {DOMAINS[i]: float(w2[0, i]) for i in range(D)},
            "phase_1": {DOMAINS[i]: float(w2[1, i]) for i in range(D)},
        }
        for name, w2 in mixes.items()
    },
}
with open(OUT / "surrogate_result.json", "w") as f:
    json.dump(result, f, indent=2)
np.savez(OUT / "surrogate_mixes.npz", **{k: v for k, v in mixes.items()})
print("\n=== PREDICTED bpb (lower=better) ===")
for name in ["PROPOSAL", "OLMIX_REUSE", "TOKEN_PROPORTIONAL", "ANCHOR"]:
    p = pred[name]
    print(
        f"{name:20s} kme={p['kme_pred_bpb']:.4f} kh={p['kh_pred_bpb']:.4f} "
        f"ens={p['ensemble_mean_pred_bpb']:.4f} reg_vs_anchor={p['pred_regret_vs_anchor']:+.4f} "
        f"new_share={p['new_bucket_share_phase'][0]:.3f}/{p['new_bucket_share_phase'][1]:.3f}"
    )
print(f"\nANCHOR realized (historical) bpb = {anchor_realized:.4f} ({sub.loc[best_j,'run_name']})")
print("wrote surrogate_result.json + surrogate_mixes.npz")
