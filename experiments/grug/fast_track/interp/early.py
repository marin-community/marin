# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What the first three layers (KDA layers 0, 1, 2) of the Ckpt 1 d512 model do.

One evaluation set (``PALOMA_ROWS`` sequences of every Paloma subset plus ``TRAIN_ROWS`` unseen training sequences,
rows ``% 3 == 0`` held out for the probes) and four phases, each writing ``early_<phase>.npz``:

- ``ablate``: per-token loss with each L0-L2 component (attention, each KDA head, routed MoE, shared expert, the
  whole MLP, the whole layer) mean-ablated (the ``attrib`` means) and, for attention / routed / shared,
  resample-ablated (its output taken from the next row of the batch at the same position: another document);
- ``probe``: linear probes on the residual mix entering each layer (``in0`` = the embedding, ``in{i+1}`` = after layer
  ``i``, ``final`` = the readout mix) and on each layer's block output, for token identity at offsets 0..-8, the whole
  word at its last piece, the word's first piece, the in-word index, "seen earlier in the document", the source and
  code/prose, and (ridge) log document position, log absolute position and the model's next-token entropy; then the
  same probes refit with each early attention / routed MoE mean-ablated. Also the routed experts of every token;
- ``heads``: each L0-L2 KDA head's exact token-mixing matrix (``heads.kda_matrix``) summarized by distance, self,
  previous token, document start, same token, delimiters, the current word's earlier pieces, weighted by ``|A|`` and
  by ``|A| * |v|``, with example patterns; the induction test (random tokens repeated) under head / layer ablations
  and L3 path patches (query side vs key/value side), with L3's induction attention;
- ``consume``: path patching of the L0-L2 block outputs (mean or resample) into each later AttnRes gate, per side
  (attention query vs keys/values, MoE router vs experts), with per-token loss, routing overlap and, with routing
  frozen, each L3-L5 unit's activation change; the clean gate weights.
"""

import argparse
import functools
import io
import logging
import time

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
import optax
from haliax.partitioning import set_mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

from experiments.grug.fast_track.interp import categories, common, forward, heads

logger = logging.getLogger(__name__)

BATCH = 8
PALOMA_ROWS = 6  # per Paloma set
TRAIN_ROWS = 96
TRAIN_SEED = 7
EARLY = (0, 1, 2)
L = forward.NUM_LAYERS
H = forward.NUM_HEADS
SOURCES = ("train", *common.PALOMA_SETS)
CODE = SOURCES.index(categories.CODE_SOURCE)
WORD_CLASSES = 2000  # multi-piece words probed (the most frequent in the probe-train rows)
OFFSETS = (1, 2, 3, 5, 8)
PROBE_STEPS = 800
PROBE_BATCH = 8192
PROBE_LR = 3e-3
PROBE_WD = 1e-4
PROBE_DTYPE = jnp.bfloat16  # probe matmul inputs (float32 accumulation)
PROBE_MAX_TRAIN = 400_000
RIDGE = 1e-3
EVAL_CHUNK = 8192
HEAD_ROWS = 12
EXCERPT = 128
INDUCTION_BATCHES = 2
CONSUME_BATCHES = 12
SITE_NAMES = (*[f"in{i}" for i in range(L)], "final", *[f"blk{i}" for i in range(L)])


# ---------------------------------------------------------------------------------------------------------------
# Data and labels (host numpy, shared with the report)


def eval_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``tokens, segment_ids, loss_weight [N, S]`` and each row's source index, Paloma rows then train rows."""
    rows = []
    for b in [
        *common.paloma_batches(BATCH, max_sequences=PALOMA_ROWS),
        *common.train_batches(BATCH, TRAIN_ROWS, seed=TRAIN_SEED),
    ]:
        for r in range(b.tokens.shape[0]):
            if b.loss_weight[r].sum() > 0:
                rows.append((b.tokens[r], b.segment_ids[r], b.loss_weight[r], SOURCES.index(b.source)))
    n = len(rows) - len(rows) % BATCH
    rows = rows[:n]
    return (
        np.stack([r[0] for r in rows]),
        np.stack([r[1] for r in rows]),
        np.stack([r[2] for r in rows]),
        np.asarray([r[3] for r in rows], np.int32),
    )


def batches_of(tokens, seg, weight):
    for start in range(0, tokens.shape[0], BATCH):
        sl = slice(start, start + BATCH)
        yield common.TokenBatch("eval", tokens[sl], seg[sl], weight[sl])


def piece_features(vocab: list[str]) -> dict[str, np.ndarray]:
    """Per token id: ``alpha`` (letters after an optional leading space), ``cont`` (letters, no leading space),
    ``ends_alpha`` and ``delim`` (punctuation, whitespace or containing a newline)."""
    n = len(vocab)
    out = {k: np.zeros(n, bool) for k in ("alpha", "cont", "ends_alpha", "delim")}
    feats = categories.token_features(vocab)
    for i, s in enumerate(vocab):
        if not s:
            continue
        stripped = s.strip()
        out["alpha"][i] = stripped.isalpha() and (s[0] == " " or not s[0].isspace())
        out["cont"][i] = s.isalpha()
        out["ends_alpha"][i] = s[-1].isalpha()
        out["delim"][i] = feats["punctuation"][i] or feats["whitespace"][i] or "\n" in s
    return out


def word_index(tokens: np.ndarray, seg: np.ndarray, pf: dict[str, np.ndarray]) -> np.ndarray:
    """``[N, S]`` each token's piece index in its word (0: the word's first piece; -1: not a word piece)."""
    idx = np.full(tokens.shape, -1, np.int32)
    alpha, cont, ends = pf["alpha"][tokens], pf["cont"][tokens], pf["ends_alpha"][tokens]
    for r in range(tokens.shape[0]):
        prev = -1
        for t in range(tokens.shape[1]):
            joined = t > 0 and cont[r, t] and seg[r, t] == seg[r, t - 1] and prev >= 0 and ends[r, t - 1]
            prev = prev + 1 if joined else (0 if alpha[r, t] else -1)
            idx[r, t] = prev
    return idx


def position_in_doc(seg: np.ndarray) -> np.ndarray:
    s = seg.shape[1]
    idx = np.broadcast_to(np.arange(s), seg.shape)
    start = np.concatenate([np.ones_like(seg[:, :1], bool), seg[:, 1:] != seg[:, :-1]], axis=1)
    return idx - np.maximum.accumulate(np.where(start, idx, 0), axis=1)


def seen_before(tokens: np.ndarray, seg: np.ndarray) -> np.ndarray:
    out = np.zeros(tokens.shape, bool)
    for r in range(tokens.shape[0]):
        seen: set[tuple[int, int]] = set()
        for t in range(tokens.shape[1]):
            key = (int(seg[r, t]), int(tokens[r, t]))
            out[r, t] = key in seen
            seen.add(key)
    return out


def probe_labels(tokens, seg, weight, sources, pf, train_rows) -> dict[str, np.ndarray]:
    """``[N, S]`` int labels (-1: not scored) of every classification target."""
    n = tokens.shape[0]
    valid = weight > 0
    pos = position_in_doc(seg)
    out: dict[str, np.ndarray] = {"cur": np.where(valid, tokens, -1)}
    for k in OFFSETS:
        prev = np.full(tokens.shape, -1, np.int32)
        prev[:, k:] = tokens[:, :-k]
        out[f"prev{k}"] = np.where(valid & (pos >= k), prev, -1)
    widx = word_index(tokens, seg, pf)
    nxt = np.pad(widx[:, 1:], ((0, 0), (0, 1)), constant_values=-1)
    last = (widx >= 1) & (nxt <= 0) & valid
    first_piece = np.full(tokens.shape, -1, np.int32)
    words = np.full(tokens.shape, -1, np.int64)
    keys: dict[tuple, int] = {}
    key_of = {}
    for r in range(n):
        for t in np.flatnonzero(widx[r] >= 1):
            first_piece[r, t] = tokens[r, t - widx[r, t]]
        for t in np.flatnonzero(last[r]):
            key = tuple(int(x) for x in tokens[r, t - widx[r, t] : t + 1])
            key_of[(r, t)] = key
            if train_rows[r]:
                keys[key] = keys.get(key, 0) + 1
    top = sorted(keys, key=lambda k: -keys[k])[:WORD_CLASSES]
    cls = {k: i for i, k in enumerate(top)}
    for (r, t), key in key_of.items():
        words[r, t] = cls.get(key, -1)
    out["word"] = words.astype(np.int32)
    out["word_first"] = np.where(valid & (widx >= 1), first_piece, -1)
    out["inword"] = np.where(valid & (widx >= 1), np.minimum(widx, 3) - 1, -1)
    out["seen"] = np.where(valid & (pos >= 1), seen_before(tokens, seg).astype(np.int32), -1)
    out["source"] = np.where(valid, sources[:, None], -1)
    out["code"] = np.where(valid, (sources[:, None] == CODE).astype(np.int32), -1)
    return out


NUM_CLASSES = {
    "cur": forward.VOCAB,
    **{f"prev{k}": forward.VOCAB for k in OFFSETS},
    "word": WORD_CLASSES,
    "word_first": forward.VOCAB,
    "inword": 3,
    "seen": 2,
    "source": len(SOURCES),
    "code": 2,
}
# Targets refit under each early ablation (the vocabulary-sized ones cost most).
ABLATED_TARGETS = ("cur", "prev1", "prev2", "prev3", "word", "word_first", "inword", "seen", "source")


# ---------------------------------------------------------------------------------------------------------------
# Shared forward helpers


def load_means(tag: str) -> tuple[jax.Array, jax.Array]:
    with fsspec.open(f"{common.OUTPUT_ROOT}/{tag}/attrib.npz", "rb") as f:
        z = np.load(io.BytesIO(f.read()))
        return jnp.asarray(z["head_mean"]), jnp.asarray(z["sub_mean"])


def mean_ablation(head_mean, sub_mean, *, subs=(), heads_=()) -> forward.Ablation:
    """Mean-ablate ``subs`` ((layer, SUB_*) pairs) and ``heads_`` ((layer, head) pairs)."""
    base = forward.no_ablation()
    sub_mult, head_mult = base.sub_mult, base.head_mult
    for i, sub in subs:
        sub_mult = sub_mult.at[i, sub].set(0.0)
    for i, h in heads_:
        head_mult = head_mult.at[i, h].set(0.0)
    return base._replace(sub_mult=sub_mult, sub_mean=sub_mean, head_mult=head_mult, head_mean=head_mean)


def roll_rows(x: jax.Array, b: int) -> jax.Array:
    """Resample: each row's value from the next row of the batch (another document), same position."""
    x = reshard(x, P(*([None] * x.ndim)))
    if x.shape[0] == b:
        return jnp.roll(x, 1, axis=0)
    return jnp.roll(x.reshape(b, -1, *x.shape[1:]), 1, axis=0).reshape(x.shape)


# ---------------------------------------------------------------------------------------------------------------
# Phase: ablate


def ablate_configs(head_mean, sub_mean) -> list[tuple[str, forward.Ablation, tuple[int, str] | None]]:
    """``(name, ablation, resample)``: ``resample = (layer, "attn" | "routed" | "shared")`` swaps that output."""
    base = forward.no_ablation()
    m = functools.partial(mean_ablation, head_mean, sub_mean)
    out: list = [("clean", base, None)]
    for i in EARLY:
        out += [
            (f"L{i}.attn:mean", m(subs=[(i, forward.SUB_ATTN)]), None),
            (f"L{i}.attn:resample", base, (i, "attn")),
            (f"L{i}.routed:mean", m(subs=[(i, forward.SUB_ROUTED)]), None),
            (f"L{i}.routed:resample", base, (i, "routed")),
            (f"L{i}.shared:mean", m(subs=[(i, forward.SUB_SHARED)]), None),
            (f"L{i}.shared:resample", base, (i, "shared")),
            (f"L{i}.mlp:mean", m(subs=[(i, forward.SUB_ROUTED), (i, forward.SUB_SHARED)]), None),
            (
                f"L{i}.layer:mean",
                m(subs=[(i, s) for s in (forward.SUB_ATTN, forward.SUB_ROUTED, forward.SUB_SHARED)]),
                None,
            ),
            *[(f"L{i}.h{h}:mean", m(heads_=[(i, h)]), None) for h in range(H)],
        ]
    out += [
        ("L012.attn:mean", m(subs=[(i, forward.SUB_ATTN) for i in EARLY]), None),
        ("L012.mlp:mean", m(subs=[(i, s) for i in EARLY for s in (forward.SUB_ROUTED, forward.SUB_SHARED)]), None),
        ("L3.attn:mean", m(subs=[(3, forward.SUB_ATTN)]), None),
    ]
    return out


def run_ablate(model, mesh, data, head_mean, sub_mean) -> dict[str, np.ndarray]:
    tokens_all, seg_all, w_all, _ = data
    configs = ablate_configs(head_mean, sub_mean)
    b, s = BATCH, common.SEQ_LEN
    t = b * s

    @jax.jit
    def clean(model, tokens, seg):
        tr = forward.trace_forward(model, tokens, seg, forward.no_ablation())
        early = [{k: tr["layers"][i][k] for k in ("head_out", "routed_latent", "shared_act")} for i in EARLY]
        return tr["loss"], tr["entropy"], tr["top_ids"][..., 0], early

    kinds = ("attn", "routed", "shared")

    @jax.jit
    def ablated(model, tokens, seg, abl, early, which):
        """``which [3 kinds, 3 early layers]``: 1 resamples that output (its tap is ``roll - clean``)."""
        early = forward.replicate(early)

        def taps(key, kind):
            out = []
            for i in range(L):
                if i not in EARLY:
                    out.append(jnp.zeros_like(early[0][key]))
                    continue
                c = early[EARLY.index(i)][key].astype(jnp.float32)
                out.append(which[kinds.index(kind), EARLY.index(i)] * (roll_rows(c, b) - c))
            return jnp.stack(out)

        head_tap = taps("head_out", "attn").astype(jnp.bfloat16)
        routing = forward.no_override(t)._replace(latent_delta=taps("routed_latent", "routed"))
        edit = forward.UnitEdit(head_tap=head_tap, shared_tap=taps("shared_act", "shared"))
        return forward.trace_forward(model, tokens, seg, abl, routing=routing, edit=edit)["loss"]

    n = tokens_all.shape[0]
    loss = np.zeros((len(configs), n, s), np.float16)
    entropy = np.zeros((n, s), np.float16)
    top1 = np.zeros((n, s), np.int32)
    t0 = time.time()
    with set_mesh(mesh):
        for bi, batch in enumerate(batches_of(tokens_all, seg_all, w_all)):
            rows = slice(bi * b, (bi + 1) * b)
            tokens, seg, _ = common.device_batch(batch, mesh)
            _, ent, top, early = clean(model, tokens, seg)
            entropy[rows], top1[rows] = np.asarray(ent), np.asarray(top)
            for ci, (_, abl, resample) in enumerate(configs):
                which = np.zeros((len(kinds), len(EARLY)), np.float32)
                if resample is not None:
                    i, kind = resample
                    which[kinds.index(kind), EARLY.index(i)] = 1.0
                # The clean baseline (config 0) runs the same edited path with zero taps.
                loss[ci, rows] = np.asarray(ablated(model, tokens, seg, abl, early, jnp.asarray(which)))
            logger.info("ablate batch %d (%.0fs)", bi, time.time() - t0)
    return dict(names=np.asarray([c[0] for c in configs]), loss=loss, entropy=entropy, top1=top1)


# ---------------------------------------------------------------------------------------------------------------
# Phase: probe


def _xent(logits, y):
    return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits), y[:, None], axis=-1))


def make_probe_fit(num_classes: int):
    opt = optax.adamw(PROBE_LR, weight_decay=PROBE_WD)

    @jax.jit
    def fit(x, y, mu, sd, train_idx, test_idx, test_ok):
        d = x.shape[1]
        params = (jnp.zeros((d, num_classes), jnp.float32), jnp.zeros((num_classes,), jnp.float32))
        state = opt.init(params)

        def feats(idx):
            return ((x[idx].astype(jnp.float32) - mu) / sd).astype(PROBE_DTYPE)

        def step(carry, idx):
            p, st = carry
            xb, yb = feats(idx), y[idx]

            def lossf(p):
                logits = jnp.einsum("bd,dc->bc", xb, p[0].astype(PROBE_DTYPE), preferred_element_type=jnp.float32)
                return _xent(logits + p[1], yb)

            loss, g = jax.value_and_grad(lossf)(p)
            upd, st = opt.update(g, st, p)
            return (optax.apply_updates(p, upd), st), loss

        (params, _), losses = jax.lax.scan(step, (params, state), train_idx)

        def ev(acc, chunk):
            idx, ok = chunk
            logits = jnp.einsum(
                "bd,dc->bc", feats(idx), params[0].astype(PROBE_DTYPE), preferred_element_type=jnp.float32
            )
            logits = logits + params[1]
            yb = y[idx]
            correct = (jnp.argmax(logits, -1) == yb) & ok
            lp = jnp.take_along_axis(jax.nn.log_softmax(logits), yb[:, None], axis=-1)[:, 0]
            return (acc[0] + jnp.sum(correct), acc[1] + jnp.sum(jnp.where(ok, -lp, 0.0)), acc[2] + jnp.sum(ok)), None

        (c, xe, k), _ = jax.lax.scan(ev, (0.0, 0.0, 0.0), (test_idx, test_ok))
        return c / k, xe / k, losses[-50:].mean()

    return fit


def _chunked(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = -(-len(idx) // EVAL_CHUNK) * EVAL_CHUNK
    pad = np.zeros(m, np.int32)
    pad[: len(idx)] = idx
    ok = np.zeros(m, bool)
    ok[: len(idx)] = True
    return pad.reshape(-1, EVAL_CHUNK), ok.reshape(-1, EVAL_CHUNK)


class ProbeSet:
    """The probe targets over the flattened ``[N * S]`` evaluation tokens, split by row."""

    def __init__(self, labels: dict[str, np.ndarray], reg: dict[str, np.ndarray], test_rows: np.ndarray, seed: int = 0):
        self.labels = {k: v.reshape(-1) for k, v in labels.items()}
        self.reg = {k: v.reshape(-1) for k, v in reg.items()}
        s = labels["cur"].shape[1]
        self.test_tok = np.repeat(test_rows, s)
        self.rng = np.random.default_rng(seed)
        self.fits: dict[int, object] = {}

    def split(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ok = y >= 0
        train = np.flatnonzero(ok & ~self.test_tok)
        test = np.flatnonzero(ok & self.test_tok)
        if len(train) > PROBE_MAX_TRAIN:
            train = np.sort(self.rng.choice(train, PROBE_MAX_TRAIN, replace=False))
        return train, test

    def run(self, x_host: np.ndarray, targets) -> dict[str, tuple[float, float]]:
        """``{target: (test accuracy | R^2, test xent | 0)}`` of linear probes on ``x_host [N * S, D]``."""
        x = jnp.asarray(x_host)
        all_train = np.flatnonzero(~self.test_tok)
        sub = self.rng.choice(all_train, min(len(all_train), 200_000), replace=False)
        xs = x_host[sub].astype(np.float32)
        mu, sd = xs.mean(0), xs.std(0) + 1e-3
        out = {}
        for name in targets:
            if name in self.reg:
                out[name] = (self._ridge(x_host, self.reg[name], mu, sd), 0.0)
                continue
            y = self.labels[name]
            train, test = self.split(y)
            c = NUM_CLASSES[name]
            if c not in self.fits:
                self.fits[c] = make_probe_fit(c)
            steps = self.rng.choice(train, (PROBE_STEPS, PROBE_BATCH))
            ti, tok = _chunked(test)
            acc, xe, tr_loss = self.fits[c](
                x,
                jnp.asarray(np.maximum(y, 0)),
                jnp.asarray(mu),
                jnp.asarray(sd),
                jnp.asarray(steps),
                jnp.asarray(ti),
                jnp.asarray(tok),
            )
            out[name] = (float(acc), float(xe))
            logger.info(
                "  probe %s: acc %.4f xent %.3f (train %.3f, n %d/%d)",
                name,
                float(acc),
                float(xe),
                float(tr_loss),
                len(train),
                len(test),
            )
        return out

    def _ridge(self, x_host, y, mu, sd) -> float:
        ok = np.isfinite(y)
        train = np.flatnonzero(ok & ~self.test_tok)
        test = np.flatnonzero(ok & self.test_tok)
        train = self.rng.choice(train, min(len(train), PROBE_MAX_TRAIN), replace=False)
        xt = (x_host[train].astype(np.float64) - mu) / sd
        xt = np.concatenate([xt, np.ones((len(xt), 1))], 1)
        w = np.linalg.solve(xt.T @ xt + RIDGE * len(xt) * np.eye(xt.shape[1]), xt.T @ y[train])
        xe = (x_host[test].astype(np.float64) - mu) / sd
        pred = np.concatenate([xe, np.ones((len(xe), 1))], 1) @ w
        return float(1 - np.mean((pred - y[test]) ** 2) / np.var(y[test]))


def run_probe(model, mesh, data, head_mean, sub_mean, vocab) -> dict[str, np.ndarray]:
    tokens_all, seg_all, w_all, sources = data
    n, s = tokens_all.shape
    test_rows = np.arange(n) % 3 == 0
    pf = piece_features(vocab)
    labels = probe_labels(tokens_all, seg_all, w_all, sources, pf, ~test_rows)

    @jax.jit
    def sites(model, tokens, seg, abl):
        tr = forward.trace_forward(model, tokens, seg, abl)
        mixes = [lay["attn_gate_in"] for lay in tr["layers"]] + [tr["mixed"]]
        blocks = tr["blocks"][1:]
        sel = jnp.stack([lay["selected"] for lay in tr["layers"]])
        return [m.astype(jnp.bfloat16) for m in mixes + blocks], tr["entropy"], sel

    def collect(abl, keep: tuple[str, ...]):
        feats = {k: np.zeros((n * s, forward.HIDDEN), np.float16) for k in keep}
        ent = np.zeros((n, s), np.float32)
        sel = np.zeros((L, n * s, forward.TOP_K), np.uint16)
        with set_mesh(mesh):
            for bi, batch in enumerate(batches_of(tokens_all, seg_all, w_all)):
                tokens, seg, _ = common.device_batch(batch, mesh)
                arrs, e, sl = sites(model, tokens, seg, abl)
                rows = slice(bi * BATCH * s, (bi + 1) * BATCH * s)
                for name, a in zip(SITE_NAMES, arrs, strict=True):
                    if name in feats:
                        feats[name][rows] = np.asarray(a, np.float32).reshape(-1, forward.HIDDEN)
                ent[bi * BATCH : (bi + 1) * BATCH] = np.asarray(e)
                sel[:, rows] = np.asarray(sl)
        return feats, ent, sel

    t0 = time.time()
    feats, ent, sel = collect(forward.no_ablation(), SITE_NAMES)
    pos = position_in_doc(seg_all).astype(np.float64)
    valid = (w_all > 0).reshape(-1)
    reg = {
        "logpos_doc": np.where(valid, np.log1p(pos.reshape(-1)), np.nan),
        "logpos_abs": np.where(valid, np.log1p(np.broadcast_to(np.arange(s), (n, s)).reshape(-1)), np.nan),
        "entropy": np.where(valid, ent.reshape(-1), np.nan),
    }
    probes = ProbeSet(labels, reg, test_rows)
    targets = (*NUM_CLASSES, *reg)
    results: dict[str, np.ndarray] = {}
    for site in SITE_NAMES:
        with set_mesh(mesh):
            r = probes.run(feats[site], targets)
        for k, (acc, xe) in r.items():
            results[f"clean/{site}/{k}"] = np.asarray([acc, xe])
        logger.info("probe site %s done (%.0fs)", site, time.time() - t0)
    del feats
    ablations = [
        (f"L{i}.{name}", mean_ablation(head_mean, sub_mean, subs=[(i, sub)]), i)
        for i in EARLY
        for name, sub in (("attn", forward.SUB_ATTN), ("routed", forward.SUB_ROUTED))
    ]
    ablations.append(("L012.attn", mean_ablation(head_mean, sub_mean, subs=[(i, forward.SUB_ATTN) for i in EARLY]), 2))
    for name, abl, i in ablations:
        keep = tuple(dict.fromkeys((f"in{i + 1}", "in3", "final")))
        f_ab, _, _ = collect(abl, keep)
        for site in keep:
            with set_mesh(mesh):
                r = probes.run(f_ab[site], ABLATED_TARGETS)
            for k, (acc, xe) in r.items():
                results[f"{name}/{site}/{k}"] = np.asarray([acc, xe])
        logger.info("probe ablation %s done (%.0fs)", name, time.time() - t0)
    results["selected"] = sel
    results["entropy"] = ent.astype(np.float16)
    return results


# ---------------------------------------------------------------------------------------------------------------
# Phase: heads


def kda_extra_stats(a, tokens, seg, valid, delim, widx):
    """Head-level extras over ``[H, S, S]`` normalized rows: delimiter mass (and the delimiter base rate over the
    previous 16 keys), the current word's earlier pieces, the word-start piece and the previous token at in-word
    positions, and the mass beyond the current word's start."""
    s = tokens.shape[0]
    t_idx = jnp.arange(s)[:, None]
    s_idx = jnp.arange(s)[None, :]
    before = s_idx < t_idx
    same_doc = seg[:, None] == seg[None, :]
    vq = valid.astype(jnp.float32)
    dmask = (before & delim[None, :]).astype(jnp.float32)
    local = (before & (t_idx - s_idx <= 16) & same_doc).astype(jnp.float32)
    base = jnp.sum(local * delim[None, :], -1) / jnp.maximum(jnp.sum(local, -1), 1.0)
    wstart = jnp.arange(s) - jnp.maximum(widx, 0)
    in_word = (s_idx >= wstart[:, None]) & before
    cont = (widx >= 1) & valid
    deep = (widx >= 2) & valid
    word_start = jnp.take_along_axis(a, jnp.broadcast_to(wstart[None, :, None], (a.shape[0], s, 1)), -1)[..., 0]
    prev = jnp.pad(jnp.diagonal(a, offset=-1, axis1=1, axis2=2), ((0, 0), (1, 0)))
    return dict(
        delim_mass=jnp.einsum("hts,ts,t->h", a, dmask, vq),
        delim_base=jnp.sum(base * vq),
        inword_mass=jnp.einsum("hts,ts,t->h", a, in_word.astype(jnp.float32), cont.astype(jnp.float32)),
        inword_prev=jnp.einsum("ht,t->h", prev, cont.astype(jnp.float32)),
        n_cont=jnp.sum(cont.astype(jnp.float32)),
        wstart_mass=jnp.einsum("ht,t->h", word_start, deep.astype(jnp.float32)),
        wstart_prev=jnp.einsum("ht,t->h", prev, deep.astype(jnp.float32)),
        n_deep=jnp.sum(deep.astype(jnp.float32)),
    )


def induction_l3_mass(lay3, tokens, seg):
    """Each L3 head's mean attention on the induction key (``tokens[s - 1] == tokens[t]``) over second-copy queries
    of one sequence."""
    a, _, _, _ = heads.mla_rows({k: lay3[k] for k in ("q", "k", "rel_bias")}, seg)
    s = tokens.shape[0]
    prev_tok = jnp.pad(tokens[:-1], (1, 0), constant_values=-1)
    same_doc = seg[:, None] == seg[None, :]
    ind = same_doc & (prev_tok[None, :] == tokens[:, None]) & (jnp.arange(s)[None, :] < jnp.arange(s)[:, None])
    has = jnp.any(ind, -1)
    return jnp.einsum("hts,ts,t->h", a, ind.astype(jnp.float32), has.astype(jnp.float32)) / jnp.maximum(jnp.sum(has), 1)


def run_heads(model, mesh, data, head_mean, sub_mean, vocab, block_means) -> dict[str, np.ndarray]:
    tokens_all, seg_all, w_all, sources = data
    pf = piece_features(vocab)
    widx_all = word_index(tokens_all, seg_all, pf)
    trace, _, _, _ = heads.make_fns()

    @jax.jit
    def head_stats(lay, tokens_b, seg_b, valid_b, delim_b, widx_b):
        mats = jnp.stack([heads.kda_matrix(lay, seg_b, h) for h in range(H)])
        v = lay["v"].astype(jnp.float32)
        mag = jnp.abs(mats)
        norm = mag / jnp.maximum(jnp.sum(mag, -1, keepdims=True), 1e-30)
        vw = mag * jnp.linalg.norm(v, axis=-1).T[:, None, :]
        vnorm = vw / jnp.maximum(jnp.sum(vw, -1, keepdims=True), 1e-30)
        st = heads.row_stats(norm, tokens_b, seg_b, valid_b)
        st.update(kda_extra_stats(norm, tokens_b, seg_b, valid_b, delim_b, widx_b))
        stv = heads.row_stats(vnorm, tokens_b, seg_b, valid_b)
        st.update({f"v_{k}": v_ for k, v_ in stv.items()})
        st.update({f"v_{k}": v_ for k, v_ in kda_extra_stats(vnorm, tokens_b, seg_b, valid_b, delim_b, widx_b).items()})
        return st, mats[:, :EXCERPT, :EXCERPT], vnorm[:, :EXCERPT, :EXCERPT]

    results: dict[str, np.ndarray] = {}
    acc: dict[str, np.ndarray] = {}
    n = tokens_all.shape[0]
    picks = np.linspace(0, n - 1, HEAD_ROWS).round().astype(int)
    picks = sorted(set((picks // BATCH).tolist()))
    t0 = time.time()
    examples = {}
    with set_mesh(mesh):
        for bi in picks:
            sl = slice(bi * BATCH, (bi + 1) * BATCH)
            batch = common.TokenBatch("eval", tokens_all[sl], seg_all[sl], w_all[sl])
            tokens, seg, w = common.device_batch(batch, mesh)
            layers = trace(model, tokens, seg)
            tokens, seg, w = forward.replicate((tokens, seg, w))
            r = 0  # one row per picked batch keeps the S x S matrices cheap
            row = bi * BATCH + r
            for i in EARLY:
                lay = {k: layers[i][k][r] for k in ("q", "k", "v", "log_decay", "beta")}
                st, ex, exv = head_stats(
                    lay,
                    tokens[r],
                    seg[r],
                    w[r] > 0,
                    jnp.asarray(pf["delim"][tokens_all[row]]),
                    jnp.asarray(widx_all[row]),
                )
                for k, v in st.items():
                    acc[f"L{i}/{k}"] = acc.get(f"L{i}/{k}", 0.0) + np.asarray(v, np.float64)
                examples.setdefault(f"L{i}", []).append(np.asarray(ex))
                examples.setdefault(f"L{i}_v", []).append(np.asarray(exv))
            examples.setdefault("tokens", []).append(tokens_all[row, :EXCERPT])
            examples.setdefault("source", []).append(sources[row])
            logger.info("heads batch %d (%.0fs)", bi, time.time() - t0)
    results.update({f"stats/{k}": v for k, v in acc.items()})
    results.update({f"example/{k}": np.stack(v) for k, v in examples.items()})

    # Induction composition.
    ind = [heads.induction_batch(BATCH, 200 + j) for j in range(INDUCTION_BATCHES)]
    period = 2 * heads.INDUCTION_LEN
    pos = np.arange(period)
    reps = common.SEQ_LEN // period
    second = np.tile((pos >= heads.INDUCTION_LEN) & (pos < period - 1), reps)
    first = np.tile(pos < heads.INDUCTION_LEN - 1, reps)
    m = functools.partial(mean_ablation, head_mean, sub_mean)
    base = forward.no_ablation()
    configs: list[tuple[str, forward.Ablation, jax.Array | None]] = [("clean", base, None)]
    for i in EARLY:
        configs.append((f"L{i}.attn", m(subs=[(i, forward.SUB_ATTN)]), None))
        configs.append((f"L{i}.routed", m(subs=[(i, forward.SUB_ROUTED)]), None))
        configs += [(f"L{i}.h{h}", m(heads_=[(i, h)]), None) for h in range(H)]
    configs.append(("L012.attn", m(subs=[(i, forward.SUB_ATTN) for i in EARLY]), None))
    configs.append(("L2.h1+h2", m(heads_=[(2, 1), (2, 2)]), None))
    configs.append(("L0.h0+h2", m(heads_=[(0, 0), (0, 2)]), None))

    def patch_mult(gates, srcs, sides):
        mult = jnp.ones((forward.NUM_GATES, L + 1, 2), jnp.float32)
        for g in gates:
            for j in srcs:
                for sd in sides:
                    mult = mult.at[g, j, sd].set(0.0)
        return mult

    for label, srcs in (("L012", (1, 2, 3)), ("L2", (3,)), ("L1", (2,)), ("L0", (1,)), ("embed", (0,))):
        configs.append((f"patch:{label}->L3.q", base, patch_mult((6,), srcs, (forward.SIDE_Q,))))
        configs.append((f"patch:{label}->L3.kv", base, patch_mult((6,), srcs, (forward.SIDE_KV,))))
    configs.append(("patch:L012->L3.qkv", base, patch_mult((6,), (1, 2, 3), (0, 1))))

    @jax.jit
    def ind_run(model, tokens, seg, abl, mult, value):
        patch = forward.GatePatch(mult, value)
        tr = forward.trace_forward(model, tokens, seg, abl, patch=patch)
        lay3 = forward.replicate({k: tr["layers"][3][k] for k in ("q", "k", "rel_bias")})
        tok, sg = forward.replicate((tokens, seg))
        mass = jnp.stack([induction_l3_mass({k: v[r] for k, v in lay3.items()}, tok[r], sg[r]) for r in range(2)])
        return tr["loss"], mass.mean(0)

    names, f_loss, s_loss, l3_mass = [], [], [], []
    with set_mesh(mesh):
        value = jnp.broadcast_to(block_means[:, None, None, :], (L + 1, BATCH, common.SEQ_LEN, forward.HIDDEN)).astype(
            jnp.bfloat16
        )
        ones = forward.no_patch(value).mult
        for name, abl, mult in configs:
            fs = ss = 0.0
            ms = np.zeros(H)
            for b in ind:
                tokens, seg, _ = common.device_batch(b, mesh)
                loss, mass = ind_run(model, tokens, seg, abl, ones if mult is None else mult, value)
                loss = np.asarray(loss)
                fs += float(loss[:, first].mean()) / len(ind)
                ss += float(loss[:, second].mean()) / len(ind)
                ms += np.asarray(mass) / len(ind)
            names.append(name)
            f_loss.append(fs)
            s_loss.append(ss)
            l3_mass.append(ms)
            logger.info("induction %s: first %.3f second %.3f L3 mass %s", name, fs, ss, np.round(ms, 3))
    results["ind_names"] = np.asarray(names)
    results["ind_first"] = np.asarray(f_loss)
    results["ind_second"] = np.asarray(s_loss)
    results["ind_l3_mass"] = np.stack(l3_mass)
    return results


# ---------------------------------------------------------------------------------------------------------------
# Phase: consume

DOWNSTREAM_GATES = tuple(range(2 * 3, forward.NUM_GATES))  # L3 attention .. final
GATE_NAMES = (*[f"{k}{i}" for i in range(L) for k in ("attn", "mlp")], "final")


def consume_configs():
    """``(name, gates, sources, sides, mode, frozen)``; sources index the AttnRes sources (1..3 = L0..L2 blocks)."""
    both = (forward.SIDE_Q, forward.SIDE_KV)
    early = (1, 2, 3)
    out = [("clean", (), (), both, "mean", False)]
    for g in DOWNSTREAM_GATES:
        gn = GATE_NAMES[g]
        out.append((f"{gn}<-L012", (g,), early, both, "mean", False))
        for j in early:
            out.append((f"{gn}<-L{j - 1}", (g,), (j,), both, "mean", False))
    for g in (6, 8, 10):
        gn = GATE_NAMES[g]
        out.append((f"{gn}.q<-L012", (g,), early, (forward.SIDE_Q,), "mean", False))
        out.append((f"{gn}.kv<-L012", (g,), early, (forward.SIDE_KV,), "mean", False))
    for g in (7, 9, 11):
        gn = GATE_NAMES[g]
        out.append((f"{gn}.router<-L012", (g,), early, (forward.SIDE_Q,), "mean", False))
        out.append((f"{gn}.experts<-L012", (g,), early, (forward.SIDE_KV,), "mean", False))
    for mode in ("mean", "resample"):
        out.append((f"all<-L012:{mode}", DOWNSTREAM_GATES, early, both, mode, False))
    for label, srcs in (("L012", early), ("L0", (1,)), ("L1", (2,)), ("L2", (3,))):
        out.append((f"all<-{label}:frozen", DOWNSTREAM_GATES, srcs, both, "mean", True))
    for j in early:
        out.append((f"all<-L{j - 1}", DOWNSTREAM_GATES, (j,), both, "mean", False))
    out.append(("all<-embed", DOWNSTREAM_GATES, (0,), both, "mean", False))
    return out


def block_means_of(model, mesh, data, num_batches: int) -> jax.Array:
    tokens_all, seg_all, w_all, _ = data

    @jax.jit
    def sums(model, tokens, seg, w):
        tr = forward.trace_forward(model, tokens, seg, forward.no_ablation())
        wf = w.astype(jnp.float32)
        return jnp.stack([jnp.einsum("bsd,bs->d", blk.astype(jnp.float32), wf) for blk in tr["blocks"]]), jnp.sum(wf)

    total, count = 0.0, 0.0
    with set_mesh(mesh):
        for bi, batch in enumerate(batches_of(tokens_all, seg_all, w_all)):
            if bi >= num_batches:
                break
            tokens, seg, w = common.device_batch(batch, mesh)
            s_, c_ = sums(model, tokens, seg, w)
            total, count = total + s_, count + c_
    return total / count  # [L + 1, D]


def run_consume(model, mesh, data, block_means) -> dict[str, np.ndarray]:
    tokens_all, seg_all, w_all, _ = data
    configs = consume_configs()
    b, s = BATCH, common.SEQ_LEN
    t = b * s
    late = (3, 4, 5)

    @jax.jit
    def clean(model, tokens, seg, w):
        tr = forward.trace_forward(model, tokens, seg, forward.no_ablation())
        wf = w.astype(jnp.float32)
        gate_w = []
        for lay in tr["layers"]:
            for key in ("w_attn", "w_mlp"):
                ww = lay[key]  # [n_src, B, S]
                gate_w.append(jnp.pad(jnp.einsum("nbs,bs->n", ww, wf), (0, L + 2 - ww.shape[0])))
        gate_w.append(jnp.pad(jnp.einsum("nbs,bs->n", tr["w_final"], wf), (0, L + 2 - tr["w_final"].shape[0])))
        sel = jnp.stack([lay["selected"] for lay in tr["layers"]])
        early_w = jnp.stack(
            [tr["layers"][i // 2][("w_attn", "w_mlp")[i % 2]][1:4] for i in range(6, 12)] + [tr["w_final"][1:4]]
        )
        return tr["loss"], jnp.stack(gate_w), sel, tr["blocks"], early_w[:, :, :, ::8]

    @jax.jit
    def patched(model, tokens, seg, mult, value, sel_clean, frozen):
        routing = forward.no_override(t)
        mask = jnp.zeros((L, t), bool).at[jnp.asarray(late)].set(frozen)
        routing = routing._replace(sel_mask=mask, sel_override=sel_clean)
        tr = forward.trace_forward(
            model, tokens, seg, forward.no_ablation(), routing=routing, patch=forward.GatePatch(mult, value)
        )
        overlap = jnp.stack(
            [
                jnp.mean(
                    jnp.sum(jnp.any(tr["layers"][i]["selected"][:, :, None] == sel_clean[i][:, None, :], -1), -1)
                    / forward.TOP_K
                )
                for i in late
            ]
        )
        units = []
        for i in late:
            lay = tr["layers"][i]
            units.append(
                (
                    jnp.zeros((forward.NUM_EXPERTS, forward.EXPERT_WIDTH))
                    .at[lay["expert_sorted"]]
                    .add(lay["expert_act"]),
                    jnp.sum(lay["shared_act"], 0),
                )
            )
        return tr["loss"], overlap, [u[0] for u in units], [u[1] for u in units]

    n_rows = CONSUME_BATCHES * b
    loss = np.zeros((len(configs), n_rows, s), np.float16)
    overlap = np.zeros((len(configs), 3))
    frozen_names = [c[0] for c in configs if c[5]]
    unit_clean = np.zeros((3, forward.NUM_EXPERTS, forward.EXPERT_WIDTH))
    unit_signed = np.zeros((len(frozen_names), 3, forward.NUM_EXPERTS, forward.EXPERT_WIDTH))
    shared_clean = np.zeros((3, forward.SHARED_WIDTH))
    shared_signed = np.zeros((len(frozen_names), 3, forward.SHARED_WIDTH))
    gate_w_sum = 0.0
    early_w = []
    weight_sum = 0.0
    t0 = time.time()
    with set_mesh(mesh):
        means_b = jnp.broadcast_to(block_means[:, None, None, :], (L + 1, b, s, forward.HIDDEN)).astype(jnp.bfloat16)
        for bi, batch in enumerate(batches_of(tokens_all, seg_all, w_all)):
            if bi >= CONSUME_BATCHES:
                break
            rows = slice(bi * b, (bi + 1) * b)
            tokens, seg, w = common.device_batch(batch, mesh)
            l0, gw, sel, blocks, ew = clean(model, tokens, seg, w)
            gate_w_sum = gate_w_sum + np.asarray(gw, np.float64)
            weight_sum += float(batch.loss_weight.sum())
            early_w.append(np.asarray(ew, np.float16))
            resampled = jnp.stack([roll_rows(blk, b) for blk in blocks]).astype(jnp.bfloat16)
            # Units under the clean routing (a frozen run with nothing patched).
            ones = forward.no_patch(means_b).mult
            # The baselines run the patched path with nothing patched (its float rounding differs from the plain
            # path's, which can flip near-tied routing choices downstream).
            l0, _, _, _ = patched(model, tokens, seg, ones, means_b, sel, False)
            _, _, u0, s0 = patched(model, tokens, seg, ones, means_b, sel, True)
            u0 = [np.asarray(x, np.float64) for x in u0]
            s0 = [np.asarray(x, np.float64) for x in s0]
            unit_clean += np.stack(u0)
            shared_clean += np.stack(s0)
            for ci, (name, gates, srcs, sides, mode, frozen) in enumerate(configs):
                if name == "clean":
                    loss[ci, rows] = np.asarray(l0)
                    overlap[ci] += 1.0 / CONSUME_BATCHES
                    continue
                mult = ones
                for g in gates:
                    for j in srcs:
                        for sd in sides:
                            mult = mult.at[g, j, sd].set(0.0)
                value = means_b if mode == "mean" else resampled
                lp, ov, u, sh = patched(model, tokens, seg, mult, value, sel, frozen)
                loss[ci, rows] = np.asarray(lp)
                overlap[ci] += np.asarray(ov) / CONSUME_BATCHES
                if frozen:
                    fi = frozen_names.index(name)
                    du = np.stack([np.asarray(x, np.float64) for x in u]) - np.stack(u0)
                    unit_signed[fi] += du
                    shared_signed[fi] += np.stack([np.asarray(x, np.float64) for x in sh]) - np.stack(s0)
            logger.info("consume batch %d (%.0fs)", bi, time.time() - t0)
    return dict(
        names=np.asarray([c[0] for c in configs]),
        loss=loss,
        overlap=overlap,
        frozen_names=np.asarray(frozen_names),
        unit_clean=unit_clean,
        unit_signed=unit_signed,
        shared_clean=shared_clean,
        shared_signed=shared_signed,
        gate_w=gate_w_sum / weight_sum,
        gate_names=np.asarray(GATE_NAMES),
        early_w=np.concatenate(early_w, axis=2),
        block_means=np.asarray(block_means),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="tn-d512-ckpt1-rno")
    parser.add_argument("--phases", default="ablate,probe,heads,consume")
    parser.add_argument("--out", default="early")
    args = parser.parse_args()
    cfg = common.model_config(args.tag)
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    head_mean, sub_mean = load_means(args.tag)
    vocab = common.vocab_strings()
    t0 = time.time()
    data = eval_rows()
    logger.info("eval rows %d (%.0fs)", data[0].shape[0], time.time() - t0)
    common.save_npz(
        args.tag,
        f"{args.out}_tokens",
        tokens=data[0].astype(np.uint16),
        segment_ids=data[1],
        weight=data[2],
        sources=data[3],
    )
    with set_mesh(mesh):
        model = forward.replicate(common.compute_model(params))
    phases = args.phases.split(",")
    block_means = None
    if "heads" in phases or "consume" in phases:
        block_means = block_means_of(model, mesh, data, 6)
    for phase in phases:
        if phase == "ablate":
            res = run_ablate(model, mesh, data, head_mean, sub_mean)
        elif phase == "probe":
            res = run_probe(model, mesh, data, head_mean, sub_mean, vocab)
        elif phase == "heads":
            res = run_heads(model, mesh, data, head_mean, sub_mean, vocab, block_means)
        elif phase == "consume":
            res = run_consume(model, mesh, data, block_means)
        else:
            raise ValueError(f"unknown phase {phase}")
        common.save_npz(args.tag, f"{args.out}_{phase}", **res)
        logger.info("phase %s done (%.0fs)", phase, time.time() - t0)


if __name__ == "__main__":
    main()
