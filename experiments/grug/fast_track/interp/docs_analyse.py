# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-token analysis of whole documents from the ``docs_index`` corpus: how each next-token prediction is made and
how many units it uses.

For each scored position ``t`` (target ``y = tokens[t + 1]``), with the units that ran at ``t`` as candidates (the 10
selected experts' 384 units in each layer plus the shared units, ``NU`` = 25,344):

- attribution patching: one exact backward of ``log p(y)`` per position (rows of a batch are copies of the document,
  each differentiating a different position), with zero taps at every site (``forward.UnitEdit``): unit attribution
  ``act x grad`` (zero baseline) and ``(act - mean) x grad`` (the unit's mean activation when its expert is
  selected); each head's contribution x grad, the MLA value-embedding share, the embedding; each earlier position's
  value x grad per head (what each head moved) and embedding x grad; the unit attribution by position (``t`` and the
  three SConv lags vs older);
- direct logit attribution of the centred target logit (final gate weights and RMS scale fixed): every unit at ``t``
  through the MLP SConv's lag-0 tap, heads, VE, embedding, and each layer's MLP total (the rest is the SConv lags);
- the attribution-ranked sets: per baseline, the share of the positive attribution mass in the top ``CURVE_K`` and
  the smallest ``k`` holding 50/80/90% of it (``knet``: of the net total, undefined when that is not positive);
- interventions at ``t`` only (routing frozen at the clean selection): sufficiency (keep the top ``k`` units, ablate
  every other candidate, zero or mean) and necessity (ablate the top ``k``), ``log p(y)`` at each ``k``;
- the local-gradient attribution (``docs_index``'s) at every position, for its top-64 overlap with the exact one.

Per document: the units of the tokens' 80% sets (zero baseline) weighted by their attribution. Writes
``docs_<name>.npz``.
"""

import argparse
import logging
import time

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh

from experiments.grug.fast_track.interp import common, docs_index, forward

logger = logging.getLogger(__name__)

L, E, I, K, H = forward.NUM_LAYERS, forward.NUM_EXPERTS, forward.EXPERT_WIDTH, forward.TOP_K, forward.NUM_HEADS
SW, D, HD = forward.SHARED_WIDTH, forward.HIDDEN, forward.HEAD_DIM
NR = L * K * I
NU = NR + L * SW
MAX_DOC = 1023  # bucket(n + 1) must fit the largest bucket
BUCKETS = (256, 512, 1024)
G = 16  # attribution rows (positions) per backward
R = 32  # intervention rows per forward
CURVE_K = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
SUFF_K = (0, 1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096)
NEC_K = (4, 64, 1024)
FRACTIONS = (0.5, 0.8, 0.9)
TOP_POS, TOP_NEG = 40, 8
SRC_TOP, EMB_TOP, LOCAL_TOP = 3, 6, 64
KEY_UNITS = 512
# Intervention modes: (baseline of the ranking and of the ablation, necessity).
MODES = (("zero", False), ("mean", False), ("zero", True), ("mean", True))


def bucket(n: int) -> int:
    return next(b for b in BUCKETS if b >= n)


def candidate_unit_ids(selected: np.ndarray) -> np.ndarray:
    """``[NU]`` global unit ids of one position's candidates from its ``[L, K]`` selected experts."""
    li = np.arange(L)[:, None, None]
    routed = ((li * E + selected[:, :, None]) * I + np.arange(I)[None, None, :]).reshape(-1)
    shared = docs_index.NUM_ROUTED + (np.arange(L)[:, None] * SW + np.arange(SW)[None, :]).reshape(-1)
    return np.concatenate([routed, shared])


def _row_gather(x: jax.Array, rows: jax.Array, targets: jax.Array) -> jax.Array:
    return x[rows, targets]


def _first_reaching(cum: jax.Array, total: jax.Array, frac: float) -> jax.Array:
    """1 + the first index where ``cum`` reaches ``frac * total`` (-1 when the total is not positive)."""
    hit = cum >= frac * total[:, None]
    k = jnp.argmax(hit, axis=1) + 1
    return jnp.where((total > 0) & jnp.any(hit, axis=1), k, -1)


@jax.jit
def attribute(model, tokens, seg, targets, unit_mean, shared_mean):
    """Exact per-position attribution: row ``g`` differentiates ``log p`` of its target at ``targets[g]``."""
    m = forward.replicate(model)
    g, s = tokens.shape
    t = g * s
    rows = jnp.arange(g)
    q = rows * s + targets
    z = jnp.zeros
    taps = dict(
        routed=z((L, t * K, I)),
        shared=z((L, t, SW)),
        head=z((L, g, s, H, D)),
        v=z((L, g, s, H, HD)),
        ve=z((L, g, s, H, HD)),
        embed=z((g, s, D)),
    )

    def objective(taps):
        edit = forward.UnitEdit(
            routed_tap=taps["routed"],
            shared_tap=taps["shared"],
            head_tap=taps["head"],
            v_tap=taps["v"],
            ve_tap=taps["ve"],
            embed_tap=taps["embed"],
        )
        tr = forward.trace_forward(m, tokens, seg, forward.no_ablation(), edit=edit)
        return -jnp.sum(forward.replicate(tr["loss"])[rows, targets]), tr

    grads, tr = jax.grad(objective, has_aux=True)(taps)
    tr = forward.replicate(tr)
    out = {}
    labels = tr["labels"][rows, targets]
    out["logp"] = -tr["loss"][rows, targets]
    out["top_logp"] = tr["top_logp"][rows, targets]
    out["top_ids"] = tr["top_ids"][rows, targets]
    out["entropy"] = tr["entropy"][rows, targets]

    unembed = forward.unembed_direction(m)
    u_y = unembed.T[labels] - jnp.mean(unembed, axis=1)[None]  # [G, D]
    inv = tr["inv_rms"][rows, targets]
    wf = tr["w_final"][:, rows, targets]  # [L + 1, G]
    direct = [u_y * (wf[src] * inv)[:, None] for src in range(L + 1)]  # [G, D] per source
    hidden = _row_gather(tr["hidden"], rows, targets).astype(jnp.float32)
    out["dla_raw"] = jnp.sum(hidden * u_y / forward.final_gain(m), axis=-1)
    emb = _row_gather(tr["embed"], rows, targets).astype(jnp.float32)
    out["dla_embed"] = jnp.sum(emb * direct[0], axis=-1)
    out["attr_embed"] = jnp.sum(emb * _row_gather(grads["embed"], rows, targets), axis=-1)
    e_sal = jnp.sum(tr["embed"].astype(jnp.float32) * grads["embed"], axis=-1)  # [G, S]
    causal = jnp.arange(s)[None, :] <= targets[:, None]
    _, out["emb_src_pos"] = jax.lax.top_k(jnp.where(causal, jnp.abs(e_sal), -1.0), EMB_TOP)
    out["emb_src_val"] = jnp.take_along_axis(e_sal, out["emb_src_pos"], axis=1)

    a0, am, act_all, dla_all, sel_all = [], [], [], [], []
    prof = jnp.zeros((g, s))
    keys = ("dla_head", "dla_ve", "attr_head", "attr_ve", "dla_shared", "dla_routed", "dla_mlp", "dla_lag")
    acc = {k: [] for k in ("head_src_pos", "head_src_val", "head_sal_total", "head_sal_self", *keys)}
    for li, (lay, layer) in enumerate(zip(tr["layers"], m.layers(), strict=True)):
        d_l = direct[li + 1]
        flat = lay["token_sorted"] * K + lay["slot_sorted"]
        inv_perm = jnp.zeros((t * K,), jnp.int32).at[flat].set(jnp.arange(t * K, dtype=jnp.int32))
        js = inv_perm[q[:, None] * K + jnp.arange(K)[None, :]]  # [G, K]
        act = lay["expert_act"][js]  # [G, K, I]
        grad = grads["routed"][li][js]
        sel = lay["selected"][q]  # [G, K]
        used = lay["used_combine"][q]
        a0.append(act * grad)
        am.append((act - unit_mean[li][sel]) * grad)
        tap0 = layer.sconv_mlp.weight.astype(jnp.float32)[0]
        gvec = tap0[None] * d_l  # [G, D]
        v_lat = gvec @ layer.mlp.w_latent_up.astype(jnp.float32).T  # [G, R]
        wd = layer.mlp.expert_mlp.w_down[sel].astype(jnp.float32)  # [G, K, I, R]
        unit_dla = act * used[..., None] * jnp.einsum("gkir,gr->gki", wd, v_lat)
        sact = lay["shared_act"][q]  # [G, SW]
        sgrad = grads["shared"][li][q]
        s_dla = sact * (gvec @ layer.shared[0].w_down.astype(jnp.float32).T)
        a0.append(sact * sgrad)
        am.append((sact - shared_mean[li][None]) * sgrad)
        act_all += [act, sact]
        dla_all += [unit_dla, s_dla]
        sel_all.append(sel)
        tok_attr = jax.ops.segment_sum(
            jnp.sum(lay["expert_act"] * grads["routed"][li], axis=-1), lay["token_sorted"], num_segments=t
        )
        prof = prof + tok_attr.reshape(g, s) + jnp.sum(lay["shared_act"] * grads["shared"][li], axis=-1).reshape(g, s)

        hc = _row_gather(lay["head_out"], rows, targets).astype(jnp.float32)  # [G, H, D]
        hg = _row_gather(grads["head"][li], rows, targets)
        vc = _row_gather(lay["ve_head_out"], rows, targets).astype(jnp.float32)
        acc["attr_head"].append(jnp.sum(hc * hg, axis=-1))
        acc["attr_ve"].append(jnp.sum(vc * hg, axis=-1))
        acc["dla_head"].append(jnp.einsum("ghd,gd->gh", hc, d_l))
        acc["dla_ve"].append(jnp.einsum("ghd,gd->gh", vc, d_l))
        acc["dla_shared"].append(jnp.sum(s_dla, axis=-1))
        acc["dla_routed"].append(jnp.sum(unit_dla, axis=(1, 2)))
        mlp_dla = jnp.sum(_row_gather(lay["mlp_out"], rows, targets).astype(jnp.float32) * d_l, axis=-1)
        acc["dla_mlp"].append(mlp_dla)
        acc["dla_lag"].append(mlp_dla - acc["dla_shared"][-1] - acc["dla_routed"][-1])

        if li in forward.KDA_LAYERS:
            sal = jnp.sum(lay["v"].astype(jnp.float32) * grads["v"][li], axis=-1)  # [G, S, H]
        else:
            sal = jnp.sum(lay["v_proj"].astype(jnp.float32) * grads["v"][li], axis=-1) + jnp.sum(
                lay["ve"].astype(jnp.float32) * grads["ve"][li], axis=-1
            )
        sal = jnp.where(causal[..., None], sal, 0.0).transpose(0, 2, 1)  # [G, H, S]
        _, pos = jax.lax.top_k(jnp.abs(sal), SRC_TOP)
        acc["head_src_pos"].append(pos)
        acc["head_src_val"].append(jnp.take_along_axis(sal, pos, axis=-1))
        acc["head_sal_total"].append(jnp.sum(sal, axis=-1))
        acc["head_sal_self"].append(sal[rows, :, targets])
    for k, v in acc.items():
        out[k] = jnp.stack(v, axis=1)  # [G, L, ...]
    lag = targets[:, None] - jnp.arange(s)[None, :]
    out["prof"] = jnp.stack(
        [jnp.sum(jnp.where(lag == j, prof, 0.0), axis=1) for j in range(4)]
        + [jnp.sum(jnp.where((lag > 3) & causal, prof, 0.0), axis=1)],
        axis=1,
    )

    def flat_units(parts):
        routed = jnp.stack([p for p in parts[0::2]], axis=1).reshape(g, NR)
        shared = jnp.stack([p for p in parts[1::2]], axis=1).reshape(g, L * SW)
        return jnp.concatenate([routed, shared], axis=1)

    s0, sm = flat_units(a0), flat_units(am)
    act_f, dla_f = flat_units(act_all), flat_units(dla_all)
    out["selected"] = jnp.stack(sel_all, axis=1)  # [G, L, K]
    out["attr_routed_l"] = jnp.stack([jnp.sum(p, axis=(1, 2)) for p in a0[0::2]], axis=1)
    out["attr_shared_l"] = jnp.stack([jnp.sum(p, axis=1) for p in a0[1::2]], axis=1)
    for name, sc in (("0", s0), ("m", sm)):
        srt = -jnp.sort(-sc, axis=1)
        cum = jnp.cumsum(srt, axis=1)
        total = cum[:, -1]
        out[f"total{name}"] = total
        out[f"knet{name}"] = jnp.stack([_first_reaching(cum, total, f) for f in FRACTIONS], axis=1)
        # The positive pushes cancel against the negative ones (the net total is often <= 0), so the sets are
        # defined on the positive attribution mass.
        pcum = jnp.cumsum(jnp.maximum(srt, 0.0), axis=1)
        ptot = pcum[:, -1]
        out[f"curve{name}"] = pcum[:, jnp.asarray(CURVE_K) - 1] / ptot[:, None]
        out[f"k{name}"] = jnp.stack([_first_reaching(pcum, ptot, f) for f in FRACTIONS], axis=1)
    abs_cum = jnp.cumsum(-jnp.sort(-jnp.abs(s0), axis=1), axis=1)
    out["kabs"] = jnp.stack([_first_reaching(abs_cum, abs_cum[:, -1], f) for f in FRACTIONS], axis=1)
    out["pos0"] = jnp.sum(jnp.maximum(s0, 0.0), axis=1)
    out["neg0"] = jnp.sum(jnp.minimum(s0, 0.0), axis=1)
    out["active"] = jnp.sum(act_f > 0, axis=1)
    _, top = jax.lax.top_k(s0, TOP_POS)
    _, bot = jax.lax.top_k(-s0, TOP_NEG)
    idx = jnp.concatenate([top, bot], axis=1)
    out["top_idx"] = idx
    for name, arr in (("a0", s0), ("am", sm), ("dla", dla_f), ("act", act_f)):
        out[f"top_{name}"] = jnp.take_along_axis(arr, idx, axis=1)
    return out, s0, sm


@jax.jit
def intervene(model, tokens, seg, targets, scores, ks, necessity, unit_mean, shared_mean, selected):
    """``log p`` of each row's target with the candidates at its target position edited: the top ``ks`` by
    ``scores`` kept (the rest ablated to ``unit_mean`` / ``shared_mean``) or, where ``necessity``, ablated."""
    m = forward.replicate(model)
    r, s = tokens.shape
    rows = jnp.arange(r)
    srt = -jnp.sort(-scores, axis=1)
    thr = srt[rows, jnp.clip(ks - 1, 0, NU - 1)]
    top = (scores >= thr[:, None]) & (ks[:, None] > 0)
    keep = jnp.where(necessity[:, None], ~top, top).astype(jnp.bfloat16)
    rk = keep[:, :NR].reshape(r, L, K, I).transpose(1, 0, 2, 3)
    sk = keep[:, NR:].reshape(r, L, SW).transpose(1, 0, 2)
    routed_keep = jnp.ones((L, r, s, K, I), jnp.bfloat16).at[:, rows, targets].set(rk).reshape(L, r * s, K, I)
    shared_keep = jnp.ones((L, r, s, SW), jnp.bfloat16).at[:, rows, targets].set(sk).reshape(L, r * s, SW)
    abl = forward.no_ablation()._replace(neuron_mean=unit_mean, shared_mean=shared_mean)
    routing = forward.no_override(r * s)._replace(
        sel_mask=jnp.ones((L, r * s), bool),
        sel_override=jnp.broadcast_to(selected[:, None, :s], (L, r, s, K)).reshape(L, r * s, K),
    )
    edit = forward.UnitEdit(routed_keep=routed_keep, shared_keep=shared_keep)
    tr = forward.trace_forward(m, tokens, seg, abl, routing=routing, edit=edit)
    return -forward.replicate(tr["loss"])[rows, targets]


@jax.jit
def local_attribution(model, tokens, seg):
    """``docs_index``'s local-gradient unit attribution at every position of row 0: top-``LOCAL_TOP`` candidate
    indices and the totals."""
    m = forward.replicate(model)
    g, s = tokens.shape
    t = g * s
    weight = jnp.ones((g, s), jnp.float32)
    mask = docs_index.target_mask(seg, weight)
    taps = (jnp.zeros((L, t * K, I)), jnp.zeros((L, t, SW)))

    def objective(taps):
        edit = forward.UnitEdit(routed_tap=taps[0], shared_tap=taps[1])
        tr = forward.trace_forward(m, tokens, seg, forward.no_ablation(), edit=edit, local_grad=True)
        return -jnp.sum(tr["loss"] * mask), tr

    (g_r, g_s), tr = jax.grad(objective, has_aux=True)(taps)
    tr = forward.replicate(tr)
    q = jnp.arange(s)
    parts = []
    for li, lay in enumerate(tr["layers"]):
        flat = lay["token_sorted"] * K + lay["slot_sorted"]
        inv_perm = jnp.zeros((t * K,), jnp.int32).at[flat].set(jnp.arange(t * K, dtype=jnp.int32))
        js = inv_perm[q[:, None] * K + jnp.arange(K)[None, :]]
        parts.append((lay["expert_act"][js] * g_r[li][js]).reshape(s, K * I))
    routed = jnp.stack(parts, axis=1).reshape(s, NR)
    shared = jnp.stack([tr["layers"][li]["shared_act"][:s] * g_s[li][:s] for li in range(L)], axis=1).reshape(s, -1)
    sc = jnp.concatenate([routed, shared], axis=1)
    _, top = jax.lax.top_k(sc, LOCAL_TOP)
    return top, jnp.sum(sc, axis=1)


def unit_means(tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Each routed unit's mean activation over the tokens routed to its expert, and each shared unit's mean, from the
    ``atlas`` pass."""
    with fsspec.open(f"{common.OUTPUT_ROOT}/{tag}/atlas.npz", "rb") as f:
        z = np.load(f)
        act_sum, routed, s_act_sum, tokens = z["act_sum"], z["expert_dom"].sum(-1), z["s_act_sum"], z["dom_tokens"]
    return (act_sum / np.maximum(routed, 1)[..., None]).astype(np.float32), (s_act_sum / tokens.sum()).astype(np.float32)


def load_docs(tag: str, index: str, doc_ids: list[int]) -> list[dict]:
    with fsspec.open(f"{common.OUTPUT_ROOT}/{tag}/{index}.npz", "rb") as f:
        z = np.load(f)
        meta = {k: z[f"doc_{k}"] for k in ("source", "seq", "start", "end")}
    with fsspec.open(f"{common.OUTPUT_ROOT}/{tag}/{index}_tokens.npz", "rb") as f:
        tokens = np.load(f)["tokens"]
    docs = []
    for d in doc_ids:
        seq, start, end = int(meta["seq"][d]), int(meta["start"][d]), int(meta["end"][d])
        docs.append(dict(doc=d, source=int(meta["source"][d]), tokens=tokens[seq, start : min(end, start + MAX_DOC)]))
    return docs


def padded(row: np.ndarray, s: int, copies: int) -> tuple[np.ndarray, np.ndarray]:
    tok = np.zeros(s, np.int32)
    seg = np.ones(s, np.int32)
    n = min(len(row), s)
    tok[:n] = row[:n]
    seg[:n] = 0
    return np.tile(tok, (copies, 1)), np.tile(seg, (copies, 1))


def analyse_doc(model, mesh, doc: dict, means, n_intervene: int, rng: np.random.Generator) -> dict:
    row = doc["tokens"].astype(np.int32)
    n = len(row)
    targets = np.arange(n - 1)
    unit_mean, shared_mean = means
    res: dict[str, list] = {}
    s0_all = np.zeros((n - 1, NU), np.float32)
    sm_all = np.zeros((n - 1, NU), np.float32)
    for c in range(0, n - 1, G):
        tg = targets[c : c + G]
        pad_tg = np.concatenate([tg, np.full(G - len(tg), tg[-1])])
        s = bucket(int(pad_tg.max()) + 2)
        tok, seg = padded(row, s, G)
        with set_mesh(mesh):
            out, s0, sm = attribute(
                model, jnp.asarray(tok), jnp.asarray(seg), jnp.asarray(pad_tg, jnp.int32), unit_mean, shared_mean
            )
            out = jax.device_get(out)
            s0_all[c : c + len(tg)] = np.asarray(s0)[: len(tg)]
            sm_all[c : c + len(tg)] = np.asarray(sm)[: len(tg)]
        for k, v in out.items():
            res.setdefault(k, []).append(np.asarray(v)[: len(tg)])
    res = {k: np.concatenate(v) for k, v in res.items()}
    selected = res["selected"]  # [n - 1, L, K]
    res["top_uid"] = np.stack([candidate_unit_ids(selected[j])[res["top_idx"][j]] for j in range(n - 1)]).astype(
        np.int32
    )

    # Interventions on a subset of positions (all when ``n_intervene`` covers the document).
    pick = targets if n_intervene >= n - 1 else np.sort(rng.choice(targets, n_intervene, replace=False))
    sel_doc = np.zeros((L, bucket(n + 1), K), np.int32)
    sel_doc[:, : n - 1] = selected.transpose(1, 0, 2)
    with set_mesh(mesh):
        sel_dev = jnp.asarray(sel_doc)
        for name, nec in MODES:
            ks = NEC_K if nec else SUFF_K
            jobs = [(t, j, k) for t in pick for j, k in enumerate(ks)]
            vals = np.full((n - 1, len(ks)), np.nan, np.float32)
            scores_src = s0_all if name == "zero" else sm_all
            um = jnp.zeros_like(unit_mean) if name == "zero" else unit_mean
            sm_ = jnp.zeros_like(shared_mean) if name == "zero" else shared_mean
            for c in range(0, len(jobs), R):
                chunk = jobs[c : c + R]
                chunk = chunk + [chunk[-1]] * (R - len(chunk))
                tg = np.asarray([j[0] for j in chunk], np.int32)
                s = bucket(int(tg.max()) + 2)
                tok, seg = padded(row, s, R)
                lp = intervene(
                    model,
                    jnp.asarray(tok),
                    jnp.asarray(seg),
                    jnp.asarray(tg),
                    jnp.asarray(scores_src[tg]),
                    jnp.asarray([j[2] for j in chunk], jnp.int32),
                    jnp.full((R,), nec),
                    um,
                    sm_,
                    sel_dev[:, :s],
                )
                lp = np.asarray(jax.device_get(lp))
                for (t, j, _), v in zip(chunk, lp, strict=True):
                    vals[t, j] = v
            res[f"{'nec' if nec else 'suff'}_{name}"] = vals
        tok, seg = padded(row, bucket(n + 1), 2)
        top_local, local_total = jax.device_get(local_attribution(model, jnp.asarray(tok), jnp.asarray(seg)))
    top_local = np.asarray(top_local)[: n - 1]
    exact_top = np.argsort(-s0_all, axis=1)[:, :LOCAL_TOP]
    res["local_overlap"] = np.asarray(
        [len(np.intersect1d(a, b)) / LOCAL_TOP for a, b in zip(exact_top, top_local, strict=True)], np.float32
    )
    res["local_total"] = np.asarray(local_total)[: n - 1].astype(np.float32)

    # The document's key units: each token's 80% set (zero baseline), weighted by attribution.
    key_mass: dict[int, float] = {}
    key_count: dict[int, int] = {}
    union = {f: set() for f in FRACTIONS}
    for j in range(n - 1):
        order = np.argsort(-s0_all[j])
        uids = candidate_unit_ids(selected[j])
        for fi, f in enumerate(FRACTIONS):
            k = int(res["k0"][j, fi])
            if k > 0:
                union[f].update(uids[order[:k]].tolist())
        k80 = int(res["k0"][j, 1])
        for o in order[: max(k80, 0)]:
            u = int(uids[o])
            key_mass[u] = key_mass.get(u, 0.0) + float(s0_all[j, o])
            key_count[u] = key_count.get(u, 0) + 1
    ranked = sorted(key_mass, key=lambda u: -key_mass[u])[:KEY_UNITS]
    res["key_uid"] = np.asarray(ranked, np.int32)
    res["key_mass"] = np.asarray([key_mass[u] for u in ranked], np.float32)
    res["key_ntok"] = np.asarray([key_count[u] for u in ranked], np.int32)
    res["union"] = np.asarray([len(union[f]) for f in FRACTIONS], np.int32)
    res["tokens"] = row
    return res


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="tn-d512-ckpt1-rno")
    parser.add_argument("--index", default="docs_index")
    parser.add_argument("--docs", required=True, help="comma-separated index document ids")
    parser.add_argument("--intervene", default="", help="per document: positions to intervene on (0: all)")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    doc_ids = [int(x) for x in args.docs.split(",")]
    n_int = [int(x) for x in args.intervene.split(",")] if args.intervene else [0] * len(doc_ids)
    cfg = common.model_config(args.tag)
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    docs = load_docs(args.tag, args.index, doc_ids)
    um, sm = unit_means(args.tag)
    rng = np.random.default_rng(0)
    t0 = time.time()
    arrays = {}
    with set_mesh(mesh):
        model = common.compute_model(params)
        means = (jnp.asarray(um), jnp.asarray(sm))
    for di, (doc, ni) in enumerate(zip(docs, n_int, strict=True)):
        res = analyse_doc(model, mesh, doc, means, ni if ni > 0 else MAX_DOC, rng)
        for k, v in res.items():
            arrays[f"d{doc['doc']}/{k}"] = v
        arrays[f"d{doc['doc']}/source"] = np.asarray(doc["source"])
        logger.info(
            "doc %d (%d/%d, %d tokens): median k80 %s, %.0fs",
            doc["doc"],
            di + 1,
            len(docs),
            len(doc["tokens"]),
            np.median(res["k0"][:, 1]),
            time.time() - t0,
        )
        common.save_npz(args.tag, f"docs_{args.name}", **arrays)
    logger.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
