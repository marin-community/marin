# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Do activations sit on a small manifold? Linear and intrinsic dimension of the residual stream and of each routed
expert's post-activations, low-rank patching of the experts, and the geometry of specialized experts.

Phase ``fit`` (one process per layer shard, ``--layers``), on unseen training sequences:

- pass A: the residual stream entering each layer's MLP gate (``mlp_gate_in``) -- its full covariance and a random
  row sample -- and per-expert routed-token counts by current-token class (``CLASSES``), which pick each layer's
  specialized experts (the highest-purity expert of the digit, punctuation and whitespace classes) plus two random
  experts;
- pass B: every expert's post-activation mean and covariance over its routed tokens (capacity ``CAP`` rows per expert
  per batch, the overflow counted), and the selected experts' raw rows with their token context.

On the host: PCA spectra (PCs for 50/90/99% of variance, participation ratio) of the stream and of every expert, and
of the selected experts on their own class; the TwoNN and Levina-Bickel MLE intrinsic dimension of the stream and the
selected experts at several subsample sizes against two nulls (a matched-covariance Gaussian and a column shuffle);
the geometry of each selected expert's class rows (top PCs, linear and ring probes of digit value, position in the
number, number length). Writes ``manifold_fit_L<i>.json`` and the top-``RMAX`` PCA bases ``manifold_basis_L<i>.npz``.

Phase ``eval`` (all layers' bases), on held-out sequences with the routing frozen at the clean selection: the loss
change when each expert's post-activations are replaced by their projection on its top-``r`` PCs (affine: mean plus
the centred projection), per layer and in every layer at once, against random rank-``r`` subspaces and the oracle
top-``k`` units per token by magnitude (``act * ||W_down row||``, the exclusion study's lens); and for the selected
experts alone, with the all-token or the class basis, the loss change on their routed class and non-class positions.
Writes ``manifold_eval.json``.
"""

import argparse
import io
import logging
import time

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh

from experiments.grug.fast_track.interp import categories, common, forward

logger = logging.getLogger(__name__)

BATCH = 4
CAP = 2048  # expert rows per batch in the covariance pass (mean load ~430)
SEL_ROWS = 20000  # raw rows kept per selected expert
RESID_PER_BATCH = 512
RANKS = (4, 8, 16, 32, 64)
RMAX = max(RANKS)
TOPK = (16, 32, 64, 128, 256)
CLASS_RANKS = (4, 16, 64)
CLASSES = ("digit", "punct", "space", "function_word", "word_start", "continuation", "other")
STRUCT_CLASSES = ("digit", "punct", "space")
SUBSAMPLES = (500, 1000, 2000, 4000, 8000)
NULL_N = 4000
MLE_K = 10
PROJ_POINTS = 2000
SPECTRUM_KEEP = 256


def _is_digits(s: str) -> bool:
    return s != "" and s.isascii() and s.isdigit()


def token_classes(vocab: list[str]) -> np.ndarray:
    """Each token id's current-token class (index into ``CLASSES``)."""
    feats = categories.token_features(vocab)
    out = np.full(len(vocab), CLASSES.index("other"), np.int32)
    for name, key in (
        ("continuation", "alpha_cont"),
        ("word_start", "word_start"),
        ("function_word", "function_word"),
        ("space", "whitespace"),
        ("punct", "punctuation"),
    ):
        out[feats[key]] = CLASSES.index(name)
    pure_digit = np.array([_is_digits(s.strip()) for s in vocab])
    out[pure_digit] = CLASSES.index("digit")
    return out


# ---------------------------------------------------------------------------------------------------------------
# Spectra and intrinsic dimension


def spectrum_stats(eig: np.ndarray) -> dict:
    """PCs for 50/90/99% of variance and the participation ratio of a (descending, non-negative) spectrum."""
    eig = np.clip(np.asarray(eig, np.float64), 0, None)
    total = eig.sum()
    if total <= 0:
        return dict(k50=0, k90=0, k99=0, pr=0.0, total=0.0)
    c = np.cumsum(eig) / total
    return dict(
        k50=int(np.searchsorted(c, 0.5) + 1),
        k90=int(np.searchsorted(c, 0.9) + 1),
        k99=int(np.searchsorted(c, 0.99) + 1),
        pr=float(total**2 / np.sum(eig**2)),
        total=float(total),
    )


def sample_spectrum(x: np.ndarray) -> np.ndarray:
    xc = x - x.mean(0, keepdims=True)
    s = np.linalg.svd(xc.astype(np.float64), compute_uv=False)
    return s**2 / max(len(x) - 1, 1)


CANDIDATES = 4 * (MLE_K + 1)
GEOM_N = 4000
PERMUTATIONS = 200


@jax.jit
def _knn_refined(x: jax.Array) -> jax.Array:
    """Each row's ``MLE_K + 1`` nearest other rows' distances (ascending): candidates by the Gram-matrix squared
    distance, then exact differences of the (centred, unit-RMS-norm) rows."""
    sq = jnp.sum(x * x, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * x @ x.T
    d2 = d2 + jnp.diag(jnp.full((x.shape[0],), jnp.inf))
    cand = jax.lax.top_k(-d2, CANDIDATES)[1]
    d = jnp.sqrt(jnp.sum(jnp.square(x[cand] - x[:, None, :]), axis=-1))
    return jnp.sort(d, axis=1)[:, : MLE_K + 1]


def knn_distances(x: np.ndarray) -> np.ndarray:
    """``[n, MLE_K + 1]`` ascending distances to the nearest other rows."""
    xc = (x - x.mean(0)).astype(np.float64)
    xc /= max(np.sqrt(np.mean(np.sum(xc**2, axis=1))), 1e-30)
    return np.asarray(_knn_refined(jnp.asarray(xc, jnp.float32)), np.float64)


def intrinsic_dims(x: np.ndarray) -> dict:
    """TwoNN (Facco et al. 2017: the slope through the origin of ``-log(1 - F(mu))`` on ``log mu``, ``mu = r2 / r1``,
    over the 90% smallest ratios) and the Levina-Bickel MLE with ``MLE_K`` neighbours (MacKay-Ghahramani pooling), on
    rows without exact duplicates."""
    x = np.unique(np.asarray(x, np.float32), axis=0)
    if len(x) < 50:
        return dict(n=len(x), twonn=float("nan"), mle=float("nan"))
    r = knn_distances(x)
    r = r[r[:, 0] > 0]
    mu = np.sort(r[:, 1] / r[:, 0])
    n = len(mu)
    keep = int(0.9 * n)
    lx = np.log(mu[:keep])
    ly = -np.log(1 - np.arange(1, keep + 1) / n)
    twonn = float(np.sum(lx * ly) / np.sum(lx * lx))
    inv = np.mean(np.log(r[:, MLE_K - 1 : MLE_K] / r[:, : MLE_K - 1]), axis=1)
    mle = 1.0 / np.mean(inv[inv > 0]) if np.any(inv > 0) else float("nan")
    return dict(n=int(n), twonn=twonn, mle=float(mle))


def id_report(x: np.ndarray, rng: np.random.Generator) -> dict:
    """Linear and intrinsic dimension at several subsample sizes, and both nulls at ``NULL_N`` rows."""
    out: dict = {"n_total": len(x), "by_n": {}}
    for n in (*[s for s in SUBSAMPLES if s < len(x)], len(x)):
        idx = rng.choice(len(x), size=min(n, len(x)), replace=False)
        sub = x[idx]
        row = spectrum_stats(sample_spectrum(sub))
        if n <= max(SUBSAMPLES):
            row.update(intrinsic_dims(sub))
        out["by_n"][str(n)] = row
    n = min(NULL_N, len(x))
    sub = x[rng.choice(len(x), size=n, replace=False)].astype(np.float64)
    mean, cov = sub.mean(0), np.cov(sub, rowvar=False)
    gauss = rng.multivariate_normal(mean, cov, size=n, method="eigh").astype(np.float32)
    shuffled = np.stack([rng.permutation(sub[:, j]) for j in range(sub.shape[1])], axis=1).astype(np.float32)
    out["null_gauss"] = {**intrinsic_dims(gauss), **spectrum_stats(sample_spectrum(gauss))}
    out["null_shuffle"] = {**intrinsic_dims(shuffled), **spectrum_stats(sample_spectrum(shuffled))}
    return out


# ---------------------------------------------------------------------------------------------------------------
# Token context of selected rows


def digit_context(tokens: np.ndarray, segs: np.ndarray, vocab: list[str], cls: np.ndarray) -> dict[str, np.ndarray]:
    """Per position of ``[B, S]`` tokens: last / first digit of the token (-1 if not a pure-digit token), the token's
    digit count, its index in its run of consecutive pure-digit tokens, the run's total digits, the next token's first
    digit (-1 if not a digit token)."""
    digit = cls == CLASSES.index("digit")
    strings = np.asarray([s.strip() for s in vocab], dtype=object)
    last = np.asarray([int(s[-1]) if _is_digits(s) else -1 for s in strings])
    first = np.asarray([int(s[0]) if _is_digits(s) else -1 for s in strings])
    ndig = np.asarray([len(s) if _is_digits(s) else 0 for s in strings])
    b, s = tokens.shape
    run_pos = np.zeros((b, s), np.int32)
    run_digits = np.zeros((b, s), np.int32)
    is_d = digit[tokens]
    for i in range(b):
        t = 0
        while t < s:
            if not is_d[i, t]:
                t += 1
                continue
            u = t
            while u + 1 < s and is_d[i, u + 1] and segs[i, u + 1] == segs[i, t]:
                u += 1
            run_pos[i, t : u + 1] = np.arange(u - t + 1)
            run_digits[i, t : u + 1] = ndig[tokens[i, t : u + 1]].sum()
            t = u + 1
    nxt = np.pad(tokens[:, 1:], ((0, 0), (0, 1)))
    return dict(
        last_digit=last[tokens],
        first_digit=first[tokens],
        token_digits=ndig[tokens],
        run_pos=run_pos,
        run_digits=run_digits,
        next_first_digit=np.where(np.pad(segs[:, 1:] == segs[:, :-1], ((0, 0), (0, 1))), first[nxt], -1),
    )


def ridge_r2(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    """Held-out (30%) R^2 of a ridge regression ``x -> y`` (``y`` 1-D or 2-D; mean over columns)."""
    y = y.reshape(len(y), -1).astype(np.float64)
    n = len(x)
    if n < 40:
        return float("nan")
    perm = rng.permutation(n)
    tr, te = perm[: int(0.7 * n)], perm[int(0.7 * n) :]
    xm = x[tr].mean(0)
    ym = y[tr].mean(0)
    a = x[tr] - xm
    lam = 1e-3 * np.trace(a.T @ a) / a.shape[1]
    w = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ (y[tr] - ym))
    pred = (x[te] - xm) @ w + ym
    ss = np.sum((y[te] - y[te].mean(0)) ** 2, axis=0)
    r2 = 1 - np.sum((y[te] - pred) ** 2, axis=0) / np.maximum(ss, 1e-12)
    return float(np.mean(r2))


def structure_report(rows: np.ndarray, ctx: dict[str, np.ndarray], tok: np.ndarray, vocab, cls_name: str, rng) -> dict:
    """Top PCs of an expert's class rows (class-centred), probes of the context variables, and chart points."""
    x = rows.astype(np.float64)
    mean = x.mean(0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    pcs = (x - mean) @ vt[:32].T  # [n, 32]
    out: dict = {"n": len(x), "pc_var": (s[:16] ** 2 / np.sum(s**2)).tolist()}
    pick = rng.choice(len(x), size=min(PROJ_POINTS, len(x)), replace=False)
    points: dict = {"pc1": pcs[pick, 0], "pc2": pcs[pick, 1], "pc3": pcs[pick, 2]}
    # Each top PC's best-correlated context variable.
    variables = {k: v.astype(np.float64) for k, v in ctx.items()}
    variables["token_rank"] = tok.astype(np.float64)
    corr = {}
    for k, v in variables.items():
        if np.std(v) == 0:
            continue
        corr[k] = [float(np.corrcoef(pcs[:, j], v)[0, 1]) for j in range(6)]
    out["pc_corr"] = corr
    if cls_name == "digit":
        d = ctx["last_digit"].astype(np.float64)
        ang = 2 * np.pi * d / 10
        ring = np.stack([np.cos(ang), np.sin(ang)], 1)
        probes = {
            "last_digit_linear": ridge_r2(pcs, d, rng),
            "last_digit_ring": ridge_r2(pcs, ring, rng),
            "last_digit_onehot": ridge_r2(pcs, np.eye(10)[d.astype(int)], rng),
            "run_pos": ridge_r2(pcs, ctx["run_pos"], rng),
            "run_digits": ridge_r2(pcs, ctx["run_digits"], rng),
            "token_digits": ridge_r2(pcs, ctx["token_digits"], rng),
        }
        nd = ctx["next_first_digit"]
        has = nd >= 0
        if has.sum() >= 40:
            na = 2 * np.pi * nd[has] / 10
            probes["next_digit_linear"] = ridge_r2(pcs[has], nd[has].astype(np.float64), rng)
            probes["next_digit_ring"] = ridge_r2(pcs[has], np.stack([np.cos(na), np.sin(na)], 1), rng)
        out["probes"] = probes
        # The ring plane: the 2-D span of the cos/sin probe on the full (centred) rows; per-digit centroids there.
        a = x - mean
        lam = 1e-3 * np.trace(a.T @ a) / a.shape[1]
        w = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ (ring - ring.mean(0)))
        q, _ = np.linalg.qr(w)
        plane = a @ q
        cent = np.stack([plane[d == k].mean(0) if np.any(d == k) else np.full(2, np.nan) for k in range(10)])
        angles = np.arctan2(cent[:, 1] - np.nanmean(cent[:, 1]), cent[:, 0] - np.nanmean(cent[:, 0]))
        order = np.argsort(angles)
        steps = np.diff(np.concatenate([order, order[:1]]))
        out["ring"] = dict(
            centroids=cent.tolist(),
            centroid_angles=angles.tolist(),
            angular_order=order.tolist(),
            adjacent_frac=float(np.mean(np.isin(np.mod(steps, 10), (1, 9)))),
        )
        points["ring1"], points["ring2"] = plane[pick, 0], plane[pick, 1]
    for k, v in ctx.items():
        points[k] = v[pick]
    tops = np.unique(tok, return_counts=True)
    top_ids = tops[0][np.argsort(-tops[1])][:8]
    out["top_tokens"] = [vocab[i] for i in top_ids]
    points["token"] = np.asarray([int(np.where(top_ids == t)[0][0]) if t in top_ids else 8 for t in tok[pick]])
    out["points"] = {k: np.round(np.asarray(v, np.float64), 4).tolist() for k, v in points.items()}
    return out


# ---------------------------------------------------------------------------------------------------------------
# Fit phase


def make_pass_a(layers: tuple[int, ...]):
    @jax.jit
    def run(model, tokens, seg, cls_ids, pos):
        tr = forward.replicate(forward.trace_forward(model, tokens, seg, forward.no_ablation()))
        cls_flat = cls_ids[tokens.reshape(-1)]
        out = []
        for i in layers:
            lay = tr["layers"][i]
            x = lay["mlp_gate_in"].astype(jnp.float32).reshape(-1, forward.HIDDEN)
            onehot = jax.nn.one_hot(cls_flat, len(CLASSES), dtype=jnp.float32)  # [T, C]
            sel = jax.nn.one_hot(lay["selected"], forward.NUM_EXPERTS, dtype=jnp.float32).sum(1)  # [T, E]
            out.append(dict(s1=x.sum(0), s2=x.T @ x, rows=x[pos], counts=sel.T @ onehot))
        return out

    return run


def make_pass_b(layers: tuple[int, ...], sel_experts: dict[int, np.ndarray]):
    sel = {i: jnp.asarray(sel_experts[i]) for i in layers}

    @jax.jit
    def run(model, tokens, seg):
        tr = forward.replicate(forward.trace_forward(model, tokens, seg, forward.no_ablation()))
        out = []
        for i in layers:
            lay = tr["layers"][i]
            act = lay["expert_act"]
            starts, counts = lay["expert_starts"], lay["expert_counts"]
            idx = starts[:, None] + jnp.arange(CAP)[None, :]
            valid = jnp.arange(CAP)[None, :] < counts[:, None]
            xs = jnp.where(valid[..., None], act[jnp.clip(idx, 0, act.shape[0] - 1)], 0.0).astype(jnp.bfloat16)
            s1 = jnp.sum(xs.astype(jnp.float32), axis=1)
            s2 = jnp.einsum("eci,ecj->eij", xs, xs, preferred_element_type=jnp.float32)
            n = jnp.minimum(counts, CAP)
            e = sel[i]
            sidx = starts[e][:, None] + jnp.arange(CAP)[None, :]
            svalid = jnp.arange(CAP)[None, :] < counts[e][:, None]
            sidx_c = jnp.clip(sidx, 0, act.shape[0] - 1)
            out.append(
                dict(
                    s1=s1,
                    s2=s2,
                    n=n,
                    drops=jnp.maximum(counts - CAP, 0),
                    sel_rows=act[sidx_c].astype(jnp.bfloat16),
                    sel_tok=jnp.where(svalid, lay["token_sorted"][sidx_c], -1),
                )
            )
        return out

    return run


@jax.jit
def _eigh_desc(cov: jax.Array) -> tuple[jax.Array, jax.Array]:
    w, v = jnp.linalg.eigh(cov)
    return w[..., ::-1], v[..., ::-1]


def select_experts(counts: np.ndarray, rng: np.random.Generator) -> dict[str, int]:
    """Per class in ``STRUCT_CLASSES`` the highest-purity expert with >= 500 class tokens, plus two random ones."""
    total = counts.sum(1)
    out = {}
    for name in STRUCT_CLASSES:
        c = CLASSES.index(name)
        purity = counts[:, c] / np.maximum(total, 1)
        purity[counts[:, c] < 500] = -1
        out[name] = int(np.argmax(purity))
    others = [e for e in range(counts.shape[0]) if e not in out.values()]
    for j, e in enumerate(rng.choice(others, size=2, replace=False)):
        out[f"random{j}"] = int(e)
    return out


def run_fit(args, model, mesh, vocab, cls_ids: np.ndarray) -> None:
    layers = tuple(int(x) for x in args.layers.split(","))
    rng = np.random.default_rng(100 + layers[0])
    batches = list(common.train_batches(BATCH, args.fit_sequences, seed=31))
    t0 = time.time()
    pass_a = make_pass_a(layers)
    nt = BATCH * common.SEQ_LEN
    acc_a = [dict(s1=0.0, s2=0.0, counts=0.0, rows=[]) for _ in layers]
    with set_mesh(mesh):
        cls_dev = jnp.asarray(cls_ids)
        for batch in batches:
            tokens, seg, _ = common.device_batch(batch, mesh)
            pos = jnp.asarray(np.sort(rng.choice(nt, size=RESID_PER_BATCH, replace=False)))
            res = jax.device_get(pass_a(model, tokens, seg, cls_dev, pos))
            for a, r in zip(acc_a, res, strict=True):
                a["s1"] = a["s1"] + np.asarray(r["s1"], np.float64)
                a["s2"] = a["s2"] + np.asarray(r["s2"], np.float64)
                a["counts"] = a["counts"] + np.asarray(r["counts"], np.float64)
                a["rows"].append(np.asarray(r["rows"], np.float32))
    logger.info("pass A done (%.0fs)", time.time() - t0)
    n_tok = len(batches) * nt
    selected = {i: select_experts(a["counts"], rng) for i, a in zip(layers, acc_a, strict=True)}
    pass_b = make_pass_b(layers, {i: np.asarray(list(selected[i].values())) for i in layers})
    acc_b = [dict(s1=0.0, s2=0.0, n=0.0, drops=0.0, rows=[], ctx=[], tok=[]) for _ in layers]
    with set_mesh(mesh):
        for batch in batches:
            tokens, seg, _ = common.device_batch(batch, mesh)
            res = pass_b(model, tokens, seg)
            ctx = digit_context(batch.tokens, batch.segment_ids, vocab, cls_ids)
            flat_tok = batch.tokens.reshape(-1)
            for a, r in zip(acc_b, res, strict=True):
                # The covariance sums stay on the device (1.4 GB per layer at d1280).
                for key in ("s1", "s2", "n", "drops"):
                    a[key] = a[key] + r[key].astype(jnp.float32)
                st = np.asarray(r["sel_tok"])
                a["rows"].append(np.asarray(r["sel_rows"], np.float32))
                a["tok"].append(st)
                a["ctx"].append(
                    {k: v.reshape(-1)[np.clip(st, 0, None)] for k, v in ctx.items()}
                    | {"id": flat_tok[np.clip(st, 0, None)]}
                )
    logger.info("pass B done (%.0fs)", time.time() - t0)
    for i, a, b in zip(layers, acc_a, acc_b, strict=True):
        report: dict = {"layer": i, "tokens": n_tok, "selected": selected[i]}
        # Residual stream.
        mean = a["s1"] / n_tok
        cov = a["s2"] / n_tok - np.outer(mean, mean)
        eig = np.linalg.eigvalsh(cov)[::-1]
        resid_rows = np.concatenate(a["rows"])
        report["resid"] = {
            "full": spectrum_stats(eig),
            "spectrum": (eig[:SPECTRUM_KEEP] / eig.sum()).tolist(),
            "id": id_report(resid_rows, rng),
        }
        # Every expert.
        n_dev = jnp.maximum(b["n"], 1.0)
        emean_dev = b["s1"] / n_dev[:, None]
        ecov_dev = b["s2"] / n_dev[:, None, None] - jnp.einsum("ei,ej->eij", emean_dev, emean_dev)
        b["n"], b["drops"], emean = np.asarray(b["n"]), np.asarray(b["drops"]), np.asarray(emean_dev)
        eigs, vecs = [], []
        for c0 in range(0, ecov_dev.shape[0], 64):
            w, v = jax.device_get(_eigh_desc(ecov_dev[c0 : c0 + 64]))
            eigs.append(np.asarray(w))
            vecs.append(np.asarray(v[..., :RMAX]))
        eigs, vecs = np.concatenate(eigs), np.concatenate(vecs)
        del ecov_dev
        b["s2"] = None
        per = [spectrum_stats(e) for e in eigs]
        report["experts"] = {k: [p[k] for p in per] for k in ("k50", "k90", "k99", "pr", "total")} | {
            "rows": b["n"].tolist(),
            "drops": float(b["drops"].sum()),
        }
        frac = eigs / np.maximum(eigs.sum(1, keepdims=True), 1e-30)
        report["experts"]["mean_spectrum"] = frac[:, :SPECTRUM_KEEP].mean(0).tolist()
        report["experts"]["captured_by_rank"] = {str(r): float(np.median(frac[:, :r].sum(1))) for r in RANKS}
        # Selected experts: raw rows, class rows.
        rows = np.concatenate(b["rows"], axis=1)  # [nsel, rows, I]
        toks = np.concatenate(b["tok"], axis=1)
        ctxs = {k: np.concatenate([c[k] for c in b["ctx"]], axis=1) for k in b["ctx"][0]}
        class_means, class_bases = [], []
        report["selected_detail"] = {}
        for j, (role, e) in enumerate(selected[i].items()):
            ok = toks[j] >= 0
            x = rows[j][ok][:SEL_ROWS]
            ids = ctxs["id"][j][ok][:SEL_ROWS]
            ctx = {k: v[j][ok][:SEL_ROWS] for k, v in ctxs.items() if k != "id"}
            ccls = cls_ids[ids]
            total = a["counts"][e].sum()
            detail: dict = {"expert": e, "rows": len(x), "class_share": (a["counts"][e] / max(total, 1)).tolist()}
            detail["class_coverage"] = (a["counts"][e] / np.maximum(a["counts"].sum(0), 1)).tolist()
            detail["all"] = spectrum_stats(sample_spectrum(x)) | {"id": id_report(x, rng)}
            cname = role if role in STRUCT_CLASSES else None
            if cname is not None:
                m = ccls == CLASSES.index(cname)
                xc = x[m]
                detail["class"] = spectrum_stats(sample_spectrum(xc)) | {"id": id_report(xc, rng), "n": int(m.sum())}
                detail["structure"] = structure_report(xc, {k: v[m] for k, v in ctx.items()}, ids[m], vocab, cname, rng)
                cm = xc.mean(0)
                _, _, vt = np.linalg.svd(xc - cm, full_matrices=False)
                class_means.append(cm)
                class_bases.append(vt[:RMAX].T)
            report["selected_detail"][role] = detail
        common.save_json(args.tag, f"manifold_fit_L{i}", report)
        buf = dict(
            mean=emean.astype(np.float32),
            basis=vecs.astype(np.float32),
            eig=eigs.astype(np.float32),
            sel_roles=np.asarray([r for r in selected[i] if r in STRUCT_CLASSES]),
            sel_experts=np.asarray([e for r, e in selected[i].items() if r in STRUCT_CLASSES]),
            class_mean=np.stack(class_means).astype(np.float32),
            class_basis=np.stack(class_bases).astype(np.float32),
        )
        common.save_npz(args.tag, f"manifold_basis_L{i}", **buf)
        logger.info(
            "layer %d done (%.0fs): resid k90 %d, expert median k90 %.0f",
            i,
            time.time() - t0,
            report["resid"]["full"]["k90"],
            np.median(report["experts"]["k90"]),
        )


# ---------------------------------------------------------------------------------------------------------------
# Geometry phase: identity-controlled intrinsic dimension and the unsupervised digit-centroid geometry


def dedup_by_token(x: np.ndarray, ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One random row per distinct token id."""
    order = rng.permutation(len(ids))
    _, first = np.unique(ids[order], return_index=True)
    return x[order[first]]


def residualize_by_token(x: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Rows minus their token id's mean row (ids with one row dropped)."""
    uniq, inv, cnt = np.unique(ids, return_inverse=True, return_counts=True)
    sums = np.zeros((len(uniq), x.shape[1]))
    np.add.at(sums, inv, x)
    keep = cnt[inv] > 1
    return (x - sums[inv] / cnt[inv][:, None])[keep]


def id_controls(x: np.ndarray, ids: np.ndarray, rng: np.random.Generator) -> dict:
    """Intrinsic dimension of raw rows, one row per token id, and token-mean-residualized rows (``GEOM_N`` each)."""
    out = {}
    for name, rows in (
        ("raw", x),
        ("dedup_token", dedup_by_token(x, ids, rng)),
        ("residual_token", residualize_by_token(x, ids)),
    ):
        sub = rows[rng.choice(len(rows), size=min(GEOM_N, len(rows)), replace=False)] if len(rows) else rows
        out[name] = intrinsic_dims(sub) | spectrum_stats(sample_spectrum(sub)) if len(sub) >= 50 else {"n": len(sub)}
    return out


def _adjacent_frac(order: np.ndarray, labels: np.ndarray) -> float:
    """Share of angularly consecutive centroids whose labels are cyclically adjacent (mod 10)."""
    lab = labels[order]
    steps = np.mod(np.diff(np.concatenate([lab, lab[:1]])), 10)
    return float(np.mean(np.isin(steps, (1, 9))))


def centroid_geometry(x: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> dict:
    """PCA of the per-label (0-9) centroids: variance shares, angular order in the PC1-PC2 plane and its cyclic
    adjacency against label permutations, and whether centroid distances follow the linear ``|a - b|`` or the circular
    ``min(|a - b|, 10 - |a - b|)`` label distance (Spearman)."""
    present = np.asarray([k for k in range(10) if np.sum(labels == k) >= 20])
    if len(present) < 6:
        return {"labels": present.tolist()}
    cent = np.stack([x[labels == k].mean(0) for k in present])
    cc = cent - cent.mean(0)
    _, sv, vt = np.linalg.svd(cc, full_matrices=False)
    plane = cc @ vt[:2].T
    ang = np.arctan2(plane[:, 1], plane[:, 0])
    order = np.argsort(ang)
    obs = _adjacent_frac(order, present)
    null = np.asarray([_adjacent_frac(order, rng.permutation(present)) for _ in range(PERMUTATIONS)])
    dist = np.linalg.norm(cent[:, None] - cent[None], axis=-1)
    iu = np.triu_indices(len(present), 1)
    da = np.abs(present[:, None] - present[None])[iu]
    lin, circ = da, np.minimum(da, 10 - da)

    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    radius = np.linalg.norm(plane, axis=1)
    return dict(
        labels=present.tolist(),
        var_share=(sv**2 / np.sum(sv**2))[:5].tolist(),
        plane=plane.tolist(),
        angular_order=present[order].tolist(),
        adjacent_frac=obs,
        adjacent_null_mean=float(null.mean()),
        adjacent_p=float(np.mean(null >= obs)),
        radius_cv=float(np.std(radius) / max(np.mean(radius), 1e-30)),
        spearman_linear=spearman(dist[iu], lin),
        spearman_circular=spearman(dist[iu], circ),
        pc1_label_spearman=spearman(plane[:, 0], present.astype(float)),
    )


def probe_null(pcs: np.ndarray, d: np.ndarray, rng: np.random.Generator) -> dict:
    """Ring (cos/sin of 2 pi d / 10) and linear probe R^2 with the true digit labelling vs random relabellings."""

    def ring(lab):
        a = 2 * np.pi * lab / 10
        return np.stack([np.cos(a), np.sin(a)], 1)

    true_ring, true_lin = ridge_r2(pcs, ring(d), rng), ridge_r2(pcs, d.astype(float), rng)
    perm_ring, perm_lin = [], []
    for _ in range(20):
        p = rng.permutation(10)
        perm_ring.append(ridge_r2(pcs, ring(p[d]), rng))
        perm_lin.append(ridge_r2(pcs, p[d].astype(float), rng))
    return dict(
        ring=true_ring,
        ring_perm_mean=float(np.mean(perm_ring)),
        ring_perm_max=float(np.max(perm_ring)),
        linear=true_lin,
        linear_perm_mean=float(np.mean(perm_lin)),
        linear_perm_max=float(np.max(perm_lin)),
    )


def run_geom(args, model, mesh, vocab, cls_ids: np.ndarray) -> None:
    layers = tuple(int(x) for x in args.layers.split(","))
    rng = np.random.default_rng(200 + layers[0])
    fits = {}
    for i in layers:
        with fsspec.open(f"{common.OUTPUT_ROOT}/{args.tag}/manifold_basis_L{i}.npz", "rb") as f:
            z = np.load(io.BytesIO(f.read()))
            fits[i] = dict(roles=[str(r) for r in z["sel_roles"]], experts=z["sel_experts"])
    batches = list(common.train_batches(BATCH, args.fit_sequences, seed=31))
    nt = BATCH * common.SEQ_LEN
    pass_a = make_pass_a(layers)
    pass_b = make_pass_b(layers, {i: fits[i]["experts"] for i in layers})
    t0 = time.time()
    resid = {i: ([], []) for i in layers}
    sel = {i: dict(rows=[], tok=[], ctx=[]) for i in layers}
    with set_mesh(mesh):
        cls_dev = jnp.asarray(cls_ids)
        for batch in batches:
            tokens, seg, _ = common.device_batch(batch, mesh)
            pos_np = np.sort(rng.choice(nt, size=RESID_PER_BATCH, replace=False))
            ra = jax.device_get(pass_a(model, tokens, seg, cls_dev, jnp.asarray(pos_np)))
            rb = pass_b(model, tokens, seg)
            flat = batch.tokens.reshape(-1)
            ctx = digit_context(batch.tokens, batch.segment_ids, vocab, cls_ids)
            for i, a, b in zip(layers, ra, rb, strict=True):
                resid[i][0].append(np.asarray(a["rows"], np.float32))
                resid[i][1].append(flat[pos_np])
                st = np.asarray(b["sel_tok"])
                sel[i]["rows"].append(np.asarray(b["sel_rows"], np.float32))
                sel[i]["tok"].append(st)
                sel[i]["ctx"].append({k: v.reshape(-1)[np.clip(st, 0, None)] for k, v in ctx.items()})
    logger.info("geometry pass done (%.0fs)", time.time() - t0)
    for i in layers:
        x, ids = np.concatenate(resid[i][0]), np.concatenate(resid[i][1])
        report: dict = {"layer": i, "resid": id_controls(x, ids, rng), "selected": {}}
        rows = np.concatenate(sel[i]["rows"], axis=1)
        toks = np.concatenate(sel[i]["tok"], axis=1)
        ctxs = {k: np.concatenate([c[k] for c in sel[i]["ctx"]], axis=1) for k in sel[i]["ctx"][0]}
        for j, (role, e) in enumerate(zip(fits[i]["roles"], fits[i]["experts"], strict=True)):
            ok = toks[j] >= 0
            # sel_tok indexes the batch's flat tokens; map to ids per batch.
            ids_j = np.concatenate(
                [
                    np.where(t[j] >= 0, b.tokens.reshape(-1)[np.clip(t[j], 0, None)], -1)
                    for t, b in zip(sel[i]["tok"], batches, strict=True)
                ]
            )[ok]
            xj = rows[j][ok][:SEL_ROWS]
            ids_j = ids_j[:SEL_ROWS]
            ctx = {k: v[j][ok][:SEL_ROWS] for k, v in ctxs.items()}
            m = cls_ids[ids_j] == CLASSES.index(role)
            det: dict = {"expert": int(e), "class_n": int(m.sum()), "id": id_controls(xj[m], ids_j[m], rng)}
            if role == "digit" and m.sum() >= 200:
                xc = xj[m].astype(np.float64)
                cm = {k: v[m] for k, v in ctx.items()}
                single = cm["token_digits"] == 1
                pcs = (xc - xc.mean(0)) @ np.linalg.svd(xc - xc.mean(0), full_matrices=False)[2][:32].T
                det["last_digit_centroids"] = centroid_geometry(xc, cm["last_digit"], rng)
                det["single_digit_centroids"] = centroid_geometry(xc[single], cm["last_digit"][single], rng)
                nd = cm["next_first_digit"]
                det["next_digit_centroids"] = centroid_geometry(xc[nd >= 0], nd[nd >= 0], rng)
                det["run_pos_centroids"] = centroid_geometry(xc, np.minimum(cm["run_pos"], 9), rng)
                det["probe_null_last"] = probe_null(pcs, cm["last_digit"], rng)
                if np.sum(nd >= 0) >= 200:
                    det["probe_null_next"] = probe_null(pcs[nd >= 0], nd[nd >= 0], rng)
                # Unsupervised ring view: the rows projected on the last-digit centroids' PC1-PC2 plane.
                lab = cm["last_digit"]
                cent = np.stack([xc[lab == k].mean(0) if np.any(lab == k) else xc.mean(0) for k in range(10)])
                vt = np.linalg.svd(cent - cent.mean(0), full_matrices=False)[2][:2]
                proj = (xc - cent.mean(0)) @ vt.T
                pick = rng.choice(len(xc), size=min(PROJ_POINTS, len(xc)), replace=False)
                det["centroid_plane_points"] = [
                    [round(float(proj[k, 0]), 4), round(float(proj[k, 1]), 4), int(lab[k])] for k in pick
                ]
                det["centroid_plane_centroids"] = ((cent - cent.mean(0)) @ vt.T).round(4).tolist()
            report["selected"][role] = det
        common.save_json(args.tag, f"manifold_geom_L{i}", report)
        logger.info("geom layer %d done (%.0fs)", i, time.time() - t0)


# ---------------------------------------------------------------------------------------------------------------
# Eval phase

MODE_OFF, MODE_LOWRANK, MODE_TOPK = 0, 1, 2


def make_eval(model):
    w_down_norm = jnp.stack(
        [jnp.linalg.norm(layer.mlp.expert_mlp.w_down.astype(jnp.float32), axis=-1) for layer in model.layers()]
    )  # [L, E, I]

    @jax.jit
    def run(model, tokens, seg, sel_override, means, bases, mode, rank, k, expert_mask):
        def act_map(layer, es, counts, act):
            def lowrank(act):
                mu = means[layer][es]
                c = (act - mu).astype(jnp.bfloat16)
                u = bases[layer]  # [E, I, R]
                coef = forward._local_ragged_dot(c, u.astype(jnp.bfloat16), counts).astype(jnp.float32)
                coef = coef * (jnp.arange(RMAX) < rank[layer])[None, :]
                ut = jnp.swapaxes(u, 1, 2).astype(jnp.bfloat16)
                rec = forward._local_ragged_dot(coef.astype(jnp.bfloat16), ut, counts).astype(jnp.float32)
                return mu + rec

            def topk(act):
                mag = act * w_down_norm[layer][es]
                vals, _ = jax.lax.top_k(mag, max(TOPK))
                kth = jnp.take(vals, jnp.clip(k[layer] - 1, 0, max(TOPK) - 1), axis=1)
                return jnp.where(mag >= kth[:, None], act, 0.0)

            new = jax.lax.switch(mode[layer], [lambda a: a, lowrank, topk], act)
            on = expert_mask[layer][es][:, None]
            return jnp.where(on, new, act)

        t = tokens.shape[0] * tokens.shape[1]
        routing = forward.no_override(t)._replace(
            sel_mask=jnp.ones((forward.NUM_LAYERS, t), bool), sel_override=sel_override
        )
        tr = forward.trace_forward(model, tokens, seg, forward.no_ablation(), routing=routing, act_map=act_map)
        return forward.replicate(tr["loss"])

    @jax.jit
    def clean(model, tokens, seg):
        tr = forward.replicate(forward.trace_forward(model, tokens, seg, forward.no_ablation()))
        return tr["loss"], jnp.stack([lay["selected"] for lay in tr["layers"]])

    return run, clean


def load_bases(tag: str) -> list[dict]:
    out = []
    for i in range(forward.NUM_LAYERS):
        with fsspec.open(f"{common.OUTPUT_ROOT}/{tag}/manifold_basis_L{i}.npz", "rb") as f:
            z = np.load(io.BytesIO(f.read()))
            out.append({k: z[k] for k in z.files})
    return out


def run_eval(args, model, mesh, vocab, cls_ids: np.ndarray) -> None:
    L, E = forward.NUM_LAYERS, forward.NUM_EXPERTS
    rng = np.random.default_rng(7)
    fits = load_bases(args.tag)
    means = jnp.asarray(np.stack([f["mean"] for f in fits]))
    bases_np = np.stack([f["basis"] for f in fits])
    bases = jnp.asarray(bases_np)
    rand_np = np.linalg.qr(rng.standard_normal((L, E, forward.EXPERT_WIDTH, RMAX)))[0].astype(np.float32)
    rand_bases = jnp.asarray(rand_np)
    batches = list(common.train_batches(BATCH, args.eval_sequences, seed=32))
    with set_mesh(mesh):
        run, clean = make_eval(model)
    t0 = time.time()
    data = []
    with set_mesh(mesh):
        for b in batches:
            tokens, seg, _ = common.device_batch(b, mesh)
            loss, sel = jax.device_get(clean(model, tokens, seg))
            scored = (b.loss_weight > 0) & np.pad(b.segment_ids[:, 1:] == b.segment_ids[:, :-1], ((0, 0), (0, 1)))
            data.append(
                dict(
                    tokens=tokens,
                    seg=seg,
                    sel=jnp.asarray(sel),
                    loss=np.asarray(loss),
                    scored=scored,
                    sel_np=np.asarray(sel),
                    tok=b.tokens,
                )
            )
    logger.info("clean pass done (%.0fs)", time.time() - t0)
    ones = np.ones((L, E), bool)

    def evaluate(mode, rank, k, mask, basis=None, masks_fn=None, mean_arr=None):
        mode_a, rank_a, k_a = (jnp.asarray(np.asarray(x, np.int32)) for x in (mode, rank, k))
        m = jnp.asarray(mask)
        sums: dict[str, list[float]] = {}
        with set_mesh(mesh):
            for d in data:
                loss = np.asarray(
                    run(
                        model,
                        d["tokens"],
                        d["seg"],
                        d["sel"],
                        means if mean_arr is None else mean_arr,
                        bases if basis is None else basis,
                        mode_a,
                        rank_a,
                        k_a,
                        m,
                    )
                )
                diff = loss - d["loss"]
                groups = {"all": d["scored"]} if masks_fn is None else masks_fn(d)
                for g, gm in groups.items():
                    s = sums.setdefault(g, [0.0, 0.0])
                    s[0] += float(np.sum(diff * gm))
                    s[1] += float(np.sum(gm))
        return {g: dict(dloss=s[0] / max(s[1], 1), n=s[1]) for g, s in sums.items()}

    zeros = np.zeros(L, np.int32)
    results: dict = {
        "base_loss": float(sum(np.sum(d["loss"] * d["scored"]) for d in data) / sum(d["scored"].sum() for d in data))
    }
    results["lowrank_all"] = {r: evaluate(np.full(L, MODE_LOWRANK), np.full(L, r), zeros, ones)["all"] for r in RANKS}
    logger.info(
        "lowrank all-layers %s (%.0fs)",
        {r: round(v["dloss"], 4) for r, v in results["lowrank_all"].items()},
        time.time() - t0,
    )
    results["random_all"] = {
        r: evaluate(np.full(L, MODE_LOWRANK), np.full(L, r), zeros, ones, basis=rand_bases)["all"] for r in (16, 64)
    }
    results["topk_all"] = {k: evaluate(np.full(L, MODE_TOPK), zeros, np.full(L, k), ones)["all"] for k in TOPK}
    logger.info("topk %s (%.0fs)", {k: round(v["dloss"], 4) for k, v in results["topk_all"].items()}, time.time() - t0)
    per_layer = {}
    for i in range(L):
        mode = np.where(np.arange(L) == i, MODE_LOWRANK, MODE_OFF)
        per_layer[i] = {r: evaluate(mode, np.full(L, r), zeros, ones)["all"]["dloss"] for r in RANKS}
        per_layer[i]["topk64"] = evaluate(np.where(np.arange(L) == i, MODE_TOPK, MODE_OFF), zeros, np.full(L, 64), ones)[
            "all"
        ]["dloss"]
    results["lowrank_layer"] = per_layer
    logger.info("per-layer done (%.0fs)", time.time() - t0)
    # Selected experts alone: all-token vs class basis, loss change on their routed class / non-class positions.
    cls_results = {}
    for i in range(L):
        f = fits[i]
        for j, (role, e) in enumerate(zip(f["sel_roles"], f["sel_experts"], strict=True)):
            role, e = str(role), int(e)
            c = CLASSES.index(role)
            mask = np.zeros((L, E), bool)
            mask[i, e] = True
            cb = bases_np.copy()
            cb[i, e] = f["class_basis"][j]
            cmeans = np.stack([g["mean"] for g in fits])
            cmeans[i, e] = f["class_mean"][j]
            cb_dev, cmeans_dev = jnp.asarray(cb), jnp.asarray(cmeans)

            def masks_fn(d, i=i, e=e, c=c):
                routed = np.any(d["sel_np"][i] == e, axis=-1).reshape(d["tok"].shape)
                is_c = cls_ids[d["tok"]] == c
                return {"class": routed & is_c & d["scored"], "other": routed & ~is_c & d["scored"]}

            row = {}
            for r in CLASS_RANKS:
                mode = np.where(np.arange(L) == i, MODE_LOWRANK, MODE_OFF)
                row[f"all_r{r}"] = evaluate(mode, np.full(L, r), zeros, mask, masks_fn=masks_fn)
                row[f"class_r{r}"] = evaluate(
                    mode, np.full(L, r), zeros, mask, basis=cb_dev, masks_fn=masks_fn, mean_arr=cmeans_dev
                )
            row["topk64"] = evaluate(
                np.where(np.arange(L) == i, MODE_TOPK, MODE_OFF), zeros, np.full(L, 64), mask, masks_fn=masks_fn
            )
            cls_results[f"L{i}.{role}.E{e}"] = row
        logger.info("class patches layer %d (%.0fs)", i, time.time() - t0)
    results["class_patch"] = cls_results
    common.save_json(args.tag, "manifold_eval", results)
    logger.info("eval done (%.0fs)", time.time() - t0)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("fit", "geom", "eval"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--layers", default="", help="fit: comma-separated layers of this shard")
    parser.add_argument("--fit-sequences", type=int, default=128)
    parser.add_argument("--eval-sequences", type=int, default=32)
    args = parser.parse_args()
    cfg = common.model_config(args.tag)
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    with set_mesh(mesh):
        model = forward.replicate(params)
    vocab = common.vocab_strings()
    cls_ids = token_classes(vocab)
    if args.phase == "fit":
        run_fit(args, model, mesh, vocab, cls_ids)
    elif args.phase == "geom":
        run_geom(args, model, mesh, vocab, cls_ids)
    else:
        run_eval(args, model, mesh, vocab, cls_ids)


if __name__ == "__main__":
    main()
