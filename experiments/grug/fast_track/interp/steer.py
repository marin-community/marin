# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-layer routed-unit catalog, each unit's hypothesis tested by steering (one process per group of layers).

1. Selection (pool A, unseen training windows plus the first Paloma sequences): every routed unit's firing count and
   activation mass; per layer the ``N_TOP`` units of largest output mass (activation sum times the norm of the unit's
   output direction ``w_down[e, i] @ w_latent_up``) plus ``N_RAND`` random units that fired at least ``MIN_FIRES``
   times.
2. Hypotheses (pool A profile of the selected units, the activation at every token routed to the unit's expert):
   top-activating contexts; the trigger (activation mass by current token and by token class); previous-token
   dependence (held-out R^2 of the activation from the current token vs from the (previous, current) pair); the
   predicted next tokens ``H_next`` (activation-weighted next tokens at the unit's top firings); the weights-only
   logit lens ``H_lens``; the expert's routed tokens. A label is assembled from these.
3. Steering (pool B, held-out windows; the unit exists only where its expert is routed, so steering is applied only
   at routed positions, or after force-routing the expert, reported separately): at one position per window
   - relevant positions (the unit fires at or above its median positive activation): ``act * m`` for ``MULTS`` and
     ``act := p99`` (its 99th-percentile activation over routed tokens);
   - irrelevant positions (routed, the unit is off, the current token is not among its triggers): ``act := p99``;
   - forced positions (expert not routed): the expert force-routed into the token's last top-k slot, then
     ``act := p99`` (the reference is the force-routed forward without the edit);
   and negative controls on the same windows: a random unit of the same expert moved by the same output norm.
   Measured at the steered position: the full next-token log-probabilities (the change of ``log p(H)``, top-k, KL from
   the reference); over the next ``NEXT`` positions: KL, the change of the actual tokens' loss; per later layer the
   relative change of the block output and of the routing.
4. Greedy generation from a few relevant-position prompts with the unit force-routed and set to p99 at every
   generated position, against the clean and a control continuation.

Writes ``steer_<name>.json`` (catalog, per-row metrics, examples, generations) under ``common.OUTPUT_ROOT``.
"""

import argparse
import functools
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh

from experiments.grug.fast_track.interp import categories, common, forward

logger = logging.getLogger(__name__)

L, E, I, K, V = forward.NUM_LAYERS, forward.NUM_EXPERTS, forward.EXPERT_WIDTH, forward.TOP_K, forward.VOCAB
WIN = 512
WINDOWS_PER_SEQ = common.SEQ_LEN // WIN
BATCH = 16
N_TOP, N_RAND = 24, 16
MIN_FIRES = 100
REL_WINDOWS, IRR_WINDOWS, FORCE_WINDOWS = 12, 8, 4
NEXT = 8
H_SIZE = 5
TOP_FIRINGS = 64
MULTS = (2.0, 5.0, 10.0)
GEN_PROMPT, GEN_NEW, GEN_UNITS = 112, 16, 3  # prompt + new = 128 (the Inkling bias block)
CONTEXTS = 12
CLASSES = ("digit", "punctuation", "whitespace", "function_word", "word_start", "alpha_cont")


# ---------------------------------------------------------------------------------------------------------------
# Data


def _cut(batch: common.TokenBatch, rows: slice) -> tuple[np.ndarray, np.ndarray, list[str]]:
    tok = batch.tokens[rows].reshape(-1, WIN)
    seg = batch.segment_ids[rows].reshape(-1, WIN)
    return tok, seg, [batch.source] * tok.shape[0]


def build_pool(train_sequences: int, seed: int, paloma_lo: int, paloma_hi: int):
    """Windows of ``WIN`` tokens: unseen training sequences (seed ``seed``) and Paloma sequences
    ``[paloma_lo, paloma_hi)`` of every set."""
    toks, segs, srcs = [], [], []
    for b in common.paloma_batches(paloma_hi, max_sequences=paloma_hi):
        live = int((b.loss_weight.sum(axis=1) > 0).sum())
        t, s, src = _cut(b, slice(paloma_lo, min(paloma_hi, live)))
        toks.append(t), segs.append(s), srcs.extend(src)
    for b in common.train_batches(8, train_sequences, seed=seed):
        t, s, src = _cut(b, slice(None))
        toks.append(t), segs.append(s), srcs.extend(src)
    tok, seg = np.concatenate(toks), np.concatenate(segs)
    pad = (-tok.shape[0]) % BATCH
    if pad:
        tok = np.concatenate([tok, tok[:pad]])
        seg = np.concatenate([seg, seg[:pad]])
        srcs += srcs[:pad]
    return tok.astype(np.int32), seg.astype(np.int32), srcs


# ---------------------------------------------------------------------------------------------------------------
# Device passes


@functools.partial(jax.jit, static_argnums=(3,))
def unit_stats(model, tokens, seg, layers):
    tr = forward.replicate(forward.trace_forward(model, tokens, seg, forward.no_ablation()))
    out = []
    for i in layers:
        lay = tr["layers"][i]
        act, es = lay["expert_act"], lay["expert_sorted"]
        fire = jax.ops.segment_sum((act > 0).astype(jnp.float32), es, num_segments=E)
        mass = jax.ops.segment_sum(act, es, num_segments=E)
        out.append((fire, mass))
    return out


@functools.partial(jax.jit, static_argnums=(5,))
def profile(model, tokens, seg, sel_e, sel_u, layers):
    """Per layer ``[U, B * S]``: each selected unit's activation at every token routed to its expert, -1 elsewhere."""
    tr = forward.replicate(forward.trace_forward(model, tokens, seg, forward.no_ablation()))
    t = tokens.size
    outs = []
    for i, e, u in zip(layers, sel_e, sel_u, strict=True):
        lay = tr["layers"][i]
        act, es, ts = lay["expert_act"], lay["expert_sorted"], lay["token_sorted"]
        vals = act[:, u].T  # [U, TK]
        hit = es[None, :] == e[:, None]
        rows = jnp.arange(e.shape[0])[:, None]
        cols = jnp.broadcast_to(ts[None, :], hit.shape)
        outs.append(jnp.full((e.shape[0], t), -1.0).at[rows, cols].max(jnp.where(hit, vals, -1.0)))
    return outs


def _logp(model, hidden):
    logits = jnp.einsum("bnd,dv->bnv", hidden, model.output_proj, preferred_element_type=jnp.float32)
    if model.config.logit_soft_cap is not None:
        logits = forward.soft_cap(logits, model.config.logit_soft_cap)
    return jax.nn.log_softmax(logits, axis=-1)


@jax.jit
def steer_eval(model, tokens, seg, steer: forward.RoutedSteer, pos, ref):
    tr = forward.replicate(
        forward.trace_forward(model, tokens, seg, forward.no_ablation(), edit=forward.UnitEdit(routed_steer=steer))
    )
    b, s = tokens.shape
    tokens = forward.replicate(tokens)
    idx = jnp.clip(pos[:, None] + jnp.arange(NEXT + 1)[None, :], 0, s - 1)  # [B, N]
    hid = jnp.take_along_axis(tr["hidden"], idx[..., None], axis=1)
    logp = _logp(forward.replicate(model), hid)  # [B, N, V]
    ref_logp = logp[ref]
    kl = jnp.sum(jnp.exp(ref_logp) * (ref_logp - logp), axis=-1)  # KL(reference || steered)
    labels = jnp.take_along_axis(tokens, jnp.clip(idx + 1, 0, s - 1), axis=1)
    nll = -jnp.take_along_axis(logp, labels[..., None], axis=-1)[..., 0]
    blocks = jnp.stack([jnp.take_along_axis(x, idx[..., None], axis=1) for x in tr["blocks"][1:]]).astype(jnp.float32)
    diff = jnp.linalg.norm(blocks - blocks[:, ref], axis=-1)
    rel = diff / (jnp.linalg.norm(blocks[:, ref], axis=-1) + 1e-6)  # [L, B, N]
    sel = jnp.stack([lay["selected"].reshape(b, s, K) for lay in tr["layers"]])  # [L, B, S, K]
    sel = jnp.take_along_axis(sel, pos[None, :, None, None], axis=2)[:, :, 0]  # [L, B, K]
    same = jnp.any(sel[..., :, None] == sel[:, ref][..., None, :], axis=-1)
    return dict(
        logp_t=logp[:, 0],
        kl=kl,
        dnll=nll - nll[ref],
        rel=rel[:, :, 0].T,
        rel_next=jnp.mean(rel[:, :, 1:], axis=-1).T,
        route_changed=(K - jnp.sum(same, axis=-1)).T,
    )


@jax.jit
def greedy_step(model, tokens, seg, steer: forward.RoutedSteer, pos):
    tr = forward.replicate(
        forward.trace_forward(model, tokens, seg, forward.no_ablation(), edit=forward.UnitEdit(routed_steer=steer))
    )
    hid = jnp.take_along_axis(tr["hidden"], jnp.broadcast_to(pos, (tokens.shape[0],))[:, None, None], axis=1)
    return jnp.argmax(_logp(forward.replicate(model), hid)[:, 0], axis=-1)


def make_steer(rows: list[dict]) -> forward.RoutedSteer:
    def col(key, dtype, default):
        return jnp.asarray(np.asarray([r.get(key, default) for r in rows], dtype))

    return forward.RoutedSteer(
        layer=col("layer", np.int32, -1),
        expert=col("expert", np.int32, 0),
        unit=col("unit", np.int32, 0),
        pos_lo=col("pos_lo", np.int32, 0),
        pos_hi=col("pos_hi", np.int32, 0),
        mult=col("mult", np.float32, 1.0),
        add=col("add", np.float32, 0.0),
        force=col("force", np.bool_, False),
    )


# ---------------------------------------------------------------------------------------------------------------
# Hypotheses (host)


def _r2_split(y: np.ndarray, keys: np.ndarray, win: np.ndarray) -> float:
    """Held-out R^2 of ``y`` from per-key means fit on even windows, scored on odd windows (unseen keys: the mean)."""
    fit, ev = win % 2 == 0, win % 2 == 1
    if fit.sum() < 10 or ev.sum() < 10:
        return float("nan")
    uniq, inv = np.unique(keys, return_inverse=True)
    sums = np.bincount(inv[fit], weights=y[fit], minlength=len(uniq))
    cnts = np.bincount(inv[fit], minlength=len(uniq))
    mu = y[fit].mean()
    means = np.where(cnts > 0, (sums + 2 * mu) / (cnts + 2), mu)  # shrunk to the mean
    pred = means[inv[ev]]
    sst = np.sum(np.square(y[ev] - y[ev].mean()))
    return float(1.0 - np.sum(np.square(y[ev] - pred)) / max(sst, 1e-12))


def _tok(vocab, i: int) -> str:
    return vocab[int(i)]


def hypothesis(a, tok, seg, vocab, feats, lens_top, lens_bot) -> dict:
    """``a``: ``[N]`` the unit's activation over the flattened pool (-1 where its expert is not routed)."""
    n = tok.size
    flat_tok, flat_seg = tok.reshape(-1), seg.reshape(-1)
    pos_in_win = np.arange(n) % WIN
    win = np.arange(n) // WIN
    routed = a >= 0
    act = np.where(routed, a, 0.0)
    pos_act = act[act > 0]
    p50 = float(np.median(pos_act)) if pos_act.size else 0.0
    p99 = float(np.quantile(act[routed], 0.99)) if routed.any() else 0.0
    mass = act.sum()
    cur_mass = np.bincount(flat_tok, weights=act, minlength=V)
    trig = np.argsort(-cur_mass)[:10]
    trig_share = cur_mass[trig] / max(mass, 1e-12)
    routed_cnt = np.bincount(flat_tok[routed], minlength=V)
    route_top = np.argsort(-routed_cnt)[:5]
    classes = {}
    for c in CLASSES:
        m = feats[c][flat_tok]
        classes[c] = dict(
            mass=float(act[m].sum() / max(mass, 1e-12)), routed=float((m & routed).sum() / max(routed.sum(), 1))
        )
    prev = np.concatenate([[0], flat_tok[:-1]])
    has_prev = (pos_in_win > 0) & (np.concatenate([[False], flat_seg[1:] == flat_seg[:-1]]))
    r = routed & has_prev
    r2_cur = _r2_split(act[r], flat_tok[r], win[r])
    r2_pair = _r2_split(act[r], flat_tok[r].astype(np.int64) * V + prev[r], win[r])
    prev_mass = np.bincount(prev[r], weights=act[r], minlength=V)
    prev_top = np.argsort(-prev_mass)[:5]
    # The unit's strongest firings: contexts and the tokens that follow them.
    order = np.argsort(-act)[:TOP_FIRINGS]
    order = order[act[order] > 0]
    nxt_ok = (pos_in_win[order] < WIN - 1) & (flat_seg[np.minimum(order + 1, n - 1)] == flat_seg[order])
    nxt = flat_tok[np.minimum(order + 1, n - 1)]
    nxt_mass = np.bincount(nxt[nxt_ok], weights=act[order][nxt_ok], minlength=V)
    h_next = [int(x) for x in np.argsort(-nxt_mass)[:H_SIZE] if nxt_mass[x] > 0]
    h_next_share = float(nxt_mass[h_next].sum() / max(nxt_mass.sum(), 1e-12))
    same_as_cur = float(np.mean(nxt[nxt_ok] == flat_tok[order][nxt_ok])) if nxt_ok.any() else 0.0
    contexts = []
    for j in order[:CONTEXTS]:
        lo = max(j - 10, (j // WIN) * WIN)
        while lo < j and flat_seg[lo] != flat_seg[j]:
            lo += 1
        nx = _tok(vocab, flat_tok[j + 1]) if (j % WIN) < WIN - 1 else ""
        contexts.append(
            dict(
                act=float(act[j]),
                before="".join(_tok(vocab, x) for x in flat_tok[lo:j]),
                cur=_tok(vocab, flat_tok[j]),
                next=nx,
            )
        )
    hyp = dict(
        fires=int((act > 0).sum()),
        routed=int(routed.sum()),
        fire_rate=float((act > 0).sum() / max(routed.sum(), 1)),
        p50=p50,
        p99=p99,
        max=float(act.max()),
        trigger=[(int(t), _tok(vocab, t), float(s)) for t, s in zip(trig[:5], trig_share[:5], strict=True)],
        trigger_ids=[int(t) for t in trig],
        trigger_top5_share=float(trig_share[:5].sum()),
        classes=classes,
        r2_cur=r2_cur,
        r2_pair=r2_pair,
        prev_dependence=float(r2_pair - r2_cur) if np.isfinite(r2_cur) and np.isfinite(r2_pair) else float("nan"),
        prev_top=[(int(t), _tok(vocab, t)) for t in prev_top],
        h_next=h_next,
        h_next_str=[_tok(vocab, t) for t in h_next],
        h_next_share=h_next_share,
        next_is_cur=same_as_cur,
        h_lens=[int(t) for t in lens_top[:H_SIZE]],
        h_lens_str=[_tok(vocab, t) for t in lens_top[:H_SIZE]],
        lens_bot_str=[_tok(vocab, t) for t in lens_bot[:H_SIZE]],
        route_top=[(_tok(vocab, t), float(routed_cnt[t] / max(routed.sum(), 1))) for t in route_top],
        contexts=contexts,
    )
    hyp["label"] = label(hyp)
    return hyp


def label(h: dict) -> str:
    trig = h["trigger"]
    if h["trigger_top5_share"] >= 0.4:
        what = "/".join(repr(t[1]) for t in trig[:3])
    else:
        lifts = {c: v["mass"] / max(v["routed"], 1e-6) for c, v in h["classes"].items() if v["mass"] > 0.25}
        what = max(lifts, key=lifts.get).replace("_", " ") + " tokens" if lifts else "mixed tokens"
    s = f"fires on {what}"
    if np.isfinite(h["prev_dependence"]) and h["prev_dependence"] > 0.1:
        s += " after " + "/".join(repr(t[1]) for t in h["prev_top"][:2])
    if h["h_next_str"] and h["h_next_share"] >= 0.3:
        s += "; next " + "/".join(repr(t) for t in h["h_next_str"][:3])
    return s


# ---------------------------------------------------------------------------------------------------------------
# Steering rows


def _positions(a2d: np.ndarray, mask2d: np.ndarray, count: int, rng) -> list[tuple[int, int]]:
    """Up to ``count`` (window, position) pairs in distinct windows where ``mask2d`` (positions 16 .. WIN - NEXT - 2)."""
    m = mask2d.copy()
    m[:, :16] = False
    m[:, WIN - NEXT - 1 :] = False
    wins = np.flatnonzero(m.any(axis=1))
    rng.shuffle(wins)
    out = []
    for w in wins[:count]:
        cand = np.flatnonzero(m[w])
        out.append((int(w), int(rng.choice(cand))))
    return out


def unit_groups(u: dict, a_b: np.ndarray, tok_b: np.ndarray, rng) -> list[list[dict]]:
    """Row groups for one unit, each on one window; ``ref`` is the index (within the group) of the row a row is
    compared with (a reference row is its own)."""
    a2d = a_b.reshape(-1, WIN)
    h = u["hyp"]
    base = dict(layer=u["layer"], expert=u["expert"])
    ctl_unit, ratio, p99 = u["ctl_unit"], u["ctl_ratio"], h["p99"]
    trig = np.isin(tok_b, np.asarray(h["trigger_ids"]))
    groups = []
    for w, t in _positions(a2d, a2d >= max(h["p50"], 1e-9), REL_WINDOWS, rng):
        act = float(a2d[w, t])
        at = dict(w=w, t=t, pos_lo=t, pos_hi=t, ref=0)
        g = [dict(kind="ref", **at)]
        for m in MULTS:
            g.append(dict(kind=f"rel_x{m:g}", **at, **base, unit=u["unit"], mult=m, d=(m - 1) * act))
        g.append(dict(kind="rel_p99", **at, **base, unit=u["unit"], mult=0.0, add=p99, d=p99 - act))
        g.append(dict(kind="ctl_rel_x5", **at, **base, unit=ctl_unit, add=4 * act * ratio, d=4 * act))
        g.append(dict(kind="ctl_rel_p99", **at, **base, unit=ctl_unit, add=(p99 - act) * ratio, d=p99 - act))
        groups.append(g)
    for w, t in _positions(a2d, (a2d == 0) & ~trig, IRR_WINDOWS, rng):
        at = dict(w=w, t=t, pos_lo=t, pos_hi=t, ref=0)
        groups.append(
            [
                dict(kind="ref", **at),
                dict(kind="irr_p99", **at, **base, unit=u["unit"], mult=0.0, add=p99, d=p99),
                dict(kind="ctl_irr_p99", **at, **base, unit=ctl_unit, add=p99 * ratio, d=p99),
            ]
        )
    for w, t in _positions(a2d, a2d < 0, FORCE_WINDOWS, rng):
        at = dict(w=w, t=t, pos_lo=t, pos_hi=t)
        f = dict(**at, **base, force=True)
        groups.append(
            [
                dict(kind="ref", **at, ref=0),
                dict(kind="forced_only", **f, unit=u["unit"], ref=0, d=0.0),
                dict(kind="frc_p99", **f, unit=u["unit"], mult=0.0, add=p99, ref=1, d=p99),
                dict(kind="ctl_frc_p99", **f, unit=ctl_unit, add=p99 * ratio, ref=1, d=p99),
            ]
        )
    return groups


def pack(groups: list[list[dict]]) -> list[list[dict]]:
    """Batches of ``BATCH`` rows; a group never straddles batches (refs become batch indices); padding repeats the
    batch's first row as its own reference."""
    batches, cur = [], []
    for g in groups:
        if len(cur) + len(g) > BATCH:
            batches.append(cur)
            cur = []
        start = len(cur)
        cur.extend({**r, "ref": start + r["ref"]} for r in g)
    if cur:
        batches.append(cur)
    for b in batches:
        while len(b) < BATCH:
            b.append({**b[0], "kind": "pad", "ref": 0, "layer": -1})
    return batches


# ---------------------------------------------------------------------------------------------------------------


def topk(logp: np.ndarray, vocab, k: int = 5) -> list[tuple[str, float]]:
    idx = np.argsort(-logp)[:k]
    return [(vocab[int(i)], float(np.exp(logp[i]))) for i in idx]


def lse(x: np.ndarray, ids: list[int]) -> float:
    if not ids:
        return float("nan")
    v = x[np.asarray(ids)]
    m = v.max()
    return float(m + np.log(np.sum(np.exp(v - m))))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=common.STUDY_TAG)
    parser.add_argument("--layers", required=True, help="comma-separated layers this process handles")
    parser.add_argument("--name", required=True)
    parser.add_argument("--pool-a-train", type=int, default=64, help="unseen 4096-token train sequences, pool A")
    parser.add_argument("--pool-b-train", type=int, default=48, help="pool B (held out)")
    args = parser.parse_args()
    layers = tuple(int(x) for x in args.layers.split(","))
    cfg = common.model_config(args.tag)
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    model = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    vocab = common.vocab_strings()
    feats = categories.token_features(vocab)
    t0 = time.time()
    tok_a, seg_a, _ = build_pool(args.pool_a_train, 41, 0, 2)
    tok_b, seg_b, src_b = build_pool(args.pool_b_train, 42, 2, 4)
    logger.info("pools: A %d windows, B %d windows (%.0fs)", len(tok_a), len(tok_b), time.time() - t0)
    rng = np.random.default_rng(1000 + layers[0])
    with set_mesh(mesh):
        model = forward.replicate(model)
        # Output directions and the logit lens of every unit of these layers (weights only).
        unembed = forward.unembed_direction(model)
        out_norm, lens = {}, {}
        for i in layers:
            lay = model.layers()[i]
            lag0 = lay.sconv_mlp.weight[0].astype(jnp.float32)
            up = lay.mlp.w_latent_up.astype(jnp.float32)
            wd = lay.mlp.expert_mlp.w_down.astype(jnp.float32)
            dirs = jnp.einsum("eir,rd->eid", wd, up) * lag0
            out_norm[i] = np.asarray(jnp.linalg.norm(dirs, axis=-1))
            lens[i] = dirs  # kept on device; projected for the selected units below
        # 1. selection
        fire = {i: np.zeros((E, I)) for i in layers}
        mass = {i: np.zeros((E, I)) for i in layers}
        for s in range(0, len(tok_a), BATCH):
            res = unit_stats(model, jnp.asarray(tok_a[s : s + BATCH]), jnp.asarray(seg_a[s : s + BATCH]), layers)
            for i, (f, m) in zip(layers, res, strict=True):
                fire[i] += np.asarray(f)
                mass[i] += np.asarray(m)
        units = []
        for i in layers:
            impact = mass[i] * out_norm[i]
            flat = np.argsort(-impact.reshape(-1))
            top = flat[:N_TOP]
            pool = np.setdiff1d(np.flatnonzero(fire[i].reshape(-1) >= MIN_FIRES), top)
            rand = rng.choice(pool, size=min(N_RAND, len(pool)), replace=False)
            for kind, ids in (("top", top), ("random", rand)):
                for f in ids:
                    e, unit = divmod(int(f), I)
                    units.append(
                        dict(
                            layer=i,
                            expert=e,
                            unit=unit,
                            kind=kind,
                            impact=float(impact[e, unit]),
                            impact_rank=int(np.flatnonzero(flat == f)[0]),
                            fires_a=float(fire[i][e, unit]),
                        )
                    )
        logger.info("selected %d units (%.0fs)", len(units), time.time() - t0)
        # Lens of the selected units.
        for u in units:
            d = lens[u["layer"]][u["expert"], u["unit"]]
            lg = np.asarray(d @ unembed)
            lg = lg - lg.mean()
            u["lens_top"] = np.argsort(-lg)[:10]
            u["lens_bot"] = np.argsort(lg)[:10]
            others = np.setdiff1d(np.arange(I), [u["unit"]])
            c = int(rng.choice(others))
            u["ctl_unit"] = c
            u["ctl_ratio"] = float(
                out_norm[u["layer"]][u["expert"], u["unit"]] / max(out_norm[u["layer"]][u["expert"], c], 1e-9)
            )
        del lens
        sel_e = tuple(jnp.asarray([u["expert"] for u in units if u["layer"] == i], jnp.int32) for i in layers)
        sel_u = tuple(jnp.asarray([u["unit"] for u in units if u["layer"] == i], jnp.int32) for i in layers)
        layer_units = {i: [u for u in units if u["layer"] == i] for i in layers}

        def run_profile(tok, seg):
            acc = {i: [] for i in layers}
            for s in range(0, len(tok), BATCH):
                res = profile(
                    model, jnp.asarray(tok[s : s + BATCH]), jnp.asarray(seg[s : s + BATCH]), sel_e, sel_u, layers
                )
                for i, r in zip(layers, res, strict=True):
                    acc[i].append(np.asarray(r, np.float32))
            return {i: np.concatenate(v, axis=1) for i, v in acc.items()}

        prof_a = run_profile(tok_a, seg_a)
        logger.info("profile A done (%.0fs)", time.time() - t0)
        for i in layers:
            for j, u in enumerate(layer_units[i]):
                u["hyp"] = hypothesis(prof_a[i][j], tok_a, seg_a, vocab, feats, u["lens_top"], u["lens_bot"])
        del prof_a
        prof_b = run_profile(tok_b, seg_b)
        logger.info("profile B done (%.0fs)", time.time() - t0)
        # 3. steering rows
        all_groups = []
        for i in layers:
            for j, u in enumerate(layer_units[i]):
                u["id"] = f"L{i}.E{u['expert']}.u{u['unit']}"
                for g in unit_groups(u, prof_b[i][j], tok_b, rng):
                    for r in g:
                        r["uid"] = u["id"]
                    all_groups.append(g)
        batches = pack(all_groups)
        logger.info("%d steering batches (%.0fs)", len(batches), time.time() - t0)
        by_id = {u["id"]: u for u in units}
        records = []
        for n, rows in enumerate(batches):
            tok = jnp.asarray(np.stack([tok_b[r["w"]] for r in rows]))
            seg = jnp.asarray(np.stack([seg_b[r["w"]] for r in rows]))
            pos = jnp.asarray(np.asarray([r["t"] for r in rows], np.int32))
            ref = jnp.asarray(np.asarray([r["ref"] for r in rows], np.int32))
            out = jax.device_get(steer_eval(model, tok, seg, make_steer(rows), pos, ref))
            for b, r in enumerate(rows):
                if r["kind"] in ("pad", "ref"):
                    continue
                u = by_id[r["uid"]]
                h = u["hyp"]
                lp, lp_ref = out["logp_t"][b], out["logp_t"][r["ref"]]
                dl = lp - lp_ref
                rec = dict(
                    uid=r["uid"],
                    kind=r["kind"],
                    w=r["w"],
                    t=r["t"],
                    d=float(r.get("d", r.get("add", 0.0))),
                    d_hnext=lse(lp, h["h_next"]) - lse(lp_ref, h["h_next"]),
                    d_hlens=lse(lp, h["h_lens"]) - lse(lp_ref, h["h_lens"]),
                    kl=float(out["kl"][b, 0]),
                    kl_next=float(np.mean(out["kl"][b, 1:])),
                    dnll_t=float(out["dnll"][b, 0]),
                    dnll_next=float(np.mean(out["dnll"][b, 1:])),
                    top1_changed=bool(np.argmax(lp) != np.argmax(lp_ref)),
                    rel=[float(x) for x in out["rel"][b]],
                    rel_next=[float(x) for x in out["rel_next"][b]],
                    route_changed=[int(x) for x in out["route_changed"][b]],
                    up=[(vocab[int(x)], float(dl[x])) for x in np.argsort(-dl)[:5]],
                    up_ids=[int(x) for x in np.argsort(-dl)[:20]],
                )
                if r["kind"] in ("rel_x10", "rel_p99", "irr_p99", "frc_p99", "rel_x5"):
                    rec["before"] = topk(lp_ref, vocab)
                    rec["after"] = topk(lp, vocab)
                    w, t = r["w"], r["t"]
                    rec["prompt"] = "".join(vocab[int(x)] for x in tok_b[w, max(0, t - 30) : t + 1])
                    rec["actual_next"] = vocab[int(tok_b[w, t + 1])]
                    rec["source"] = src_b[w]
                records.append(rec)
            if n % 50 == 0:
                logger.info("batch %d/%d (%.0fs)", n, len(batches), time.time() - t0)
        # 4. greedy generation
        gens = []
        g_rows, g_meta = [], []
        for i in layers:
            for u in [u for u in layer_units[i] if u["kind"] == "top"][:GEN_UNITS]:
                a2d = prof_b[i][layer_units[i].index(u)].reshape(-1, WIN)
                m = a2d >= max(u["hyp"]["p50"], 1e-9)
                m[:, :GEN_PROMPT] = False
                cand = np.argwhere(m)
                if not len(cand):
                    continue
                w, t = (int(x) for x in cand[rng.integers(len(cand))])
                prompt = tok_b[w, t - GEN_PROMPT + 1 : t + 1]
                base = dict(layer=i, expert=u["expert"], pos_lo=GEN_PROMPT - 1, pos_hi=GEN_PROMPT + GEN_NEW, force=True)
                variants = [
                    ("clean", dict()),
                    (
                        "p99_routed_only",
                        {**base, "unit": u["unit"], "mult": 0.0, "add": u["hyp"]["p99"], "force": False},
                    ),
                    ("p99_forced", {**base, "unit": u["unit"], "mult": 0.0, "add": u["hyp"]["p99"]}),
                    ("ctl_forced", {**base, "unit": u["ctl_unit"], "add": u["hyp"]["p99"] * u["ctl_ratio"]}),
                ]
                for name, st in variants:
                    g_rows.append(st)
                    g_meta.append((u["id"], name, prompt))
        for s in range(0, len(g_rows), BATCH):
            rows = g_rows[s : s + BATCH]
            meta = g_meta[s : s + BATCH]
            pad = BATCH - len(rows)
            rows = rows + [rows[0]] * pad
            meta = meta + [meta[0]] * pad
            toks = np.zeros((BATCH, GEN_PROMPT + GEN_NEW), np.int32)
            for b, (_, _, prompt) in enumerate(meta):
                toks[b, :GEN_PROMPT] = prompt
            seg = jnp.zeros(toks.shape, jnp.int32)
            st = make_steer(rows)
            for p in range(GEN_PROMPT - 1, GEN_PROMPT + GEN_NEW - 1):
                nxt = np.asarray(greedy_step(model, jnp.asarray(toks), seg, st, jnp.asarray(p, jnp.int32)))
                toks[:, p + 1] = nxt
            for b in range(BATCH - pad):
                uid, name, prompt = meta[b]
                gens.append(
                    dict(
                        uid=uid,
                        variant=name,
                        prompt="".join(vocab[int(x)] for x in prompt[-24:]),
                        continuation="".join(vocab[int(x)] for x in toks[b, GEN_PROMPT:]),
                    )
                )
        logger.info("generation done (%.0fs)", time.time() - t0)
    catalog = []
    for u in units:
        catalog.append(
            {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in u.items()
                if k not in ("lens_top", "lens_bot")
            }
        )
    common.save_json(
        args.tag, f"steer_{args.name}", dict(layers=list(layers), catalog=catalog, records=records, gens=gens)
    )
    logger.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
