# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Specialized experts: which routed experts own a token class, and inside one such expert, does every unit matter,
do the units split into sub-roles, and are they mutually exclusive?

Phases (``--phases``, run in order):

- ``census``: over Paloma (``--paloma-sequences`` per set) plus unseen training sequences, per layer and expert the
  routed-token count of every token class (``CLASSES``, from the token strings and the context: digits split into
  years / decimals / code literals / other numbers, punctuation, code symbols, whitespace, scripts, mid-sentence
  capitalized words) and the class totals; per routed assignment the magnitude concentration of its units
  (``c_i = act_i * ||W_down[i]||``: how many units carry 50/80/90/99% of ``sum c``, participation ratio); and a
  random sample of ``SAMPLE_ROWS`` routed rows (pre-activations) of ``SAMPLE_EXPERTS`` random experts per layer for
  the layer-wide exclusion census. Writes ``experts_census.npz``.
- ``deep`` (``--target L:E``): up to ``DEEP_ROWS`` routed rows of that expert (pre-activations, combine weight, flat
  token index) and the weights-only direct vocabulary effect of each unit (``w_down[e, i] @ w_latent_up`` times the
  MLP SConv's lag-0 tap and the final norm gain on ``W_out``, centred). Writes ``experts_deep_L{L}E{E}.npz``.
- ``ablate`` (after ``deep``): on ``WINDOWS`` 1024-token windows richest in the expert's routed tokens, the loss at
  every position with each single unit zeroed (routing unchanged), the whole expert zeroed, and the sufficiency
  curves: keep each token's top-k units by ``c`` (dynamic), the top-k units by single-unit ablation impact (static),
  and k random units. Writes ``experts_ablate_L{L}E{E}.npz``.
- ``steer`` (``--spec``, a JSON list of steering experiments): each sets a unit's activation (scale it, or set it to
  a value) wherever its expert is routed (natural routing) and records, at every position routed to the expert,
  the clean and steered log-probability of the next token, of a hypothesis token set, the KL and top-1. Writes
  ``experts_steer_<name>.npz``.
"""

import argparse
import io
import json
import logging
import time
import unicodedata

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

from experiments.grug.fast_track.interp import common, forward

logger = logging.getLogger(__name__)

BATCH = 4
L, E, I, K, V = forward.NUM_LAYERS, forward.NUM_EXPERTS, forward.EXPERT_WIDTH, forward.TOP_K, forward.VOCAB
CODE_SOURCE = "dolma_100_programing_languages"
CLASSES = (
    "all",
    "digit",
    "digit_year",
    "digit_decimal",
    "digit_code",
    "digit_other",
    "punct_prose",
    "code_sym",
    "code_any",
    "newline",
    "space",
    "cap_mid",
    "word_start",
    "continuation",
    "cyrillic",
    "cjk",
    "arabic",
    "greek",
    "latin_accent",
)
NC = len(CLASSES)
SAMPLE_EXPERTS = 32
SAMPLE_ROWS = 1024
SAMPLE_PER_BATCH = 512  # random rows of the sample experts taken per layer and batch
DEEP_ROWS = 24576
DEEP_PER_BATCH = 512
WINDOW = 1024
WINDOWS = 32
QUANTS = (0.5, 0.8, 0.9, 0.99)
PR_BINS = np.linspace(0.0, 200.0, 201)
SUFF_K = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
CODE_SYMBOLS = set("{}[]<>=|&*/\\#$@^~`;:+-%!")
SENTENCE_END = (".", "!", "?", "\n", '"', "“", ":", ")")


# ---------------------------------------------------------------------------------------------------------------
# Token classes (host)


def _script(s: str) -> str | None:
    for ch in s:
        if ch.isascii():
            continue
        name = unicodedata.name(ch, "")
        if name.startswith("CYRILLIC"):
            return "cyrillic"
        if name.startswith(("CJK", "HIRAGANA", "KATAKANA", "HANGUL")):
            return "cjk"
        if name.startswith("ARABIC"):
            return "arabic"
        if name.startswith("GREEK"):
            return "greek"
        if name.startswith("LATIN") and ch.isalpha():
            return "latin_accent"
    return None


def vocab_features(vocab: list[str]) -> dict[str, np.ndarray]:
    """Per token id: ``digit`` (all digits), ``ndig``, ``first_digit`` (-1 if none), ``punct``, ``code_sym``,
    ``newline``, ``space``, ``cap`` (a leading-space capitalized word), ``word_start``, ``continuation``,
    ``dot`` (exactly "."), ``ends_sentence`` and the script classes."""
    n = len(vocab)
    f = {k: np.zeros(n, bool) for k in ("digit", "punct", "code_sym", "newline", "space", "cap", "word_start")}
    f.update({k: np.zeros(n, bool) for k in ("continuation", "dot", "ends_sentence", "cyrillic", "cjk", "arabic")})
    f.update({k: np.zeros(n, bool) for k in ("greek", "latin_accent")})
    f["ndig"] = np.zeros(n, np.int32)
    f["first_digit"] = np.full(n, -1, np.int32)
    for i, s in enumerate(vocab):
        st = s.strip()
        if not s:
            continue
        f["newline"][i] = "\n" in s
        f["space"][i] = st == "" and "\n" not in s
        if st.isdigit() and st.isascii():
            f["digit"][i] = True
            f["ndig"][i] = len(st)
            f["first_digit"][i] = int(st[0])
        f["punct"][i] = st != "" and all(unicodedata.category(c).startswith("P") or c in CODE_SYMBOLS for c in st)
        f["code_sym"][i] = f["punct"][i] and any(c in CODE_SYMBOLS for c in st)
        f["dot"][i] = s == "."
        f["ends_sentence"][i] = s.rstrip(" ").endswith(SENTENCE_END) or s.endswith("\n")
        if st.isalpha():
            f["cap"][i] = s[0] == " " and st[0].isupper()
            f["word_start"][i] = s[0] == " "
            f["continuation"][i] = s[0] != " "
        sc = _script(s)
        if sc is not None:
            f[sc][i] = True
    return f


def digit_runs(tokens: np.ndarray, seg: np.ndarray, feats) -> dict[str, np.ndarray]:
    """``[B, S]`` per position: ``run_pos`` (index of the token in its run of consecutive digit tokens, -1 off-run),
    ``run_len`` (tokens in the run), ``run_digits`` (digits in the run), ``year`` (a 4-digit run with no decimal point
    on either side; the 15xx-20xx check needs the text, in ``class_masks``), ``decimal`` (the run follows
    "<digits>.")."""
    b, s = tokens.shape
    is_d = feats["digit"][tokens]
    out = {k: np.full((b, s), -1, np.int64) for k in ("run_pos", "run_len", "run_digits")}
    year = np.zeros((b, s), bool)
    decimal = np.zeros((b, s), bool)
    for r in range(b):
        t = 0
        while t < s:
            if not is_d[r, t]:
                t += 1
                continue
            u = t
            while u + 1 < s and is_d[r, u + 1] and seg[r, u + 1] == seg[r, t]:
                u += 1
            nd = int(feats["ndig"][tokens[r, t : u + 1]].sum())
            after_dot = t >= 2 and feats["dot"][tokens[r, t - 1]] and is_d[r, t - 2] and seg[r, t - 2] == seg[r, t]
            before_dot = u + 2 < s and feats["dot"][tokens[r, u + 1]] and is_d[r, u + 2]
            out["run_pos"][r, t : u + 1] = np.arange(u - t + 1)
            out["run_len"][r, t : u + 1] = u - t + 1
            out["run_digits"][r, t : u + 1] = nd
            decimal[r, t : u + 1] = after_dot
            year[r, t : u + 1] = nd == 4 and not after_dot and not before_dot
            t = u + 1
    out["year"] = year
    out["decimal"] = decimal
    return out


def class_masks(tokens: np.ndarray, seg: np.ndarray, valid: np.ndarray, source: str, feats, vocab) -> np.ndarray:
    """``[B, S, NC]`` float class membership of each (current) token."""
    runs = digit_runs(tokens, seg, feats)
    # Years need the run's text: a 4-digit run starting 15..20.
    run_start = runs["run_pos"] == 0
    year = runs["year"].copy()
    for r, t in zip(*np.nonzero(run_start & runs["year"]), strict=True):
        text = "".join(vocab[x] for x in tokens[r, t : t + runs["run_len"][r, t]])
        if text[:2] not in ("15", "16", "17", "18", "19", "20"):
            year[r, t : t + runs["run_len"][r, t]] = False
    is_code = source == CODE_SOURCE
    digit = feats["digit"][tokens]
    prev = np.pad(tokens[:, :-1], ((0, 0), (1, 0)))
    prev_same = np.pad(seg[:, 1:] == seg[:, :-1], ((0, 0), (1, 0)))
    cap_mid = feats["cap"][tokens] & prev_same & ~feats["ends_sentence"][prev]
    m = {
        "all": np.ones_like(digit),
        "digit": digit,
        "digit_year": digit & year & (not is_code),
        "digit_decimal": digit & runs["decimal"] & (not is_code),
        "digit_code": digit & is_code,
        "punct_prose": feats["punct"][tokens] & ~feats["code_sym"][tokens] & (not is_code),
        "code_sym": feats["code_sym"][tokens] & is_code,
        "code_any": np.full_like(digit, is_code),
        "newline": feats["newline"][tokens],
        "space": feats["space"][tokens],
        "cap_mid": cap_mid & (not is_code),
        "word_start": feats["word_start"][tokens],
        "continuation": feats["continuation"][tokens],
    }
    m["digit_other"] = digit & ~m["digit_year"] & ~m["digit_decimal"] & ~m["digit_code"]
    for sc in ("cyrillic", "cjk", "arabic", "greek", "latin_accent"):
        m[sc] = feats[sc][tokens]
    return np.stack([(m[c] & (valid > 0)).astype(np.float32) for c in CLASSES], axis=-1)


# ---------------------------------------------------------------------------------------------------------------
# Shared device helpers


def unit_norms(model) -> jax.Array:
    """``[L, E, I]`` the routed units' ``||W_down[i]||``."""
    return jnp.stack([jnp.linalg.norm(lay.mlp.expert_mlp.w_down.astype(jnp.float32), axis=-1) for lay in model.layers()])


def _assignment_combine(lay) -> jax.Array:
    """Each expert-sorted assignment's combine weight."""
    return lay["used_combine"].reshape(-1)[lay["token_sorted"] * K + lay["slot_sorted"]]


def _flat_valid(weight: jax.Array) -> jax.Array:
    return reshard(weight.reshape(-1), P(None)) > 0


def _pick_rows(mask: jax.Array, key: jax.Array, k: int) -> jax.Array:
    """``k`` random row indices where ``mask`` (-1 padded)."""
    score = jnp.where(mask, jax.random.uniform(key, mask.shape), -1.0)
    val, idx = jax.lax.top_k(score, k)
    return jnp.where(val >= 0, idx, -1)


def data_batches(paloma: int, train: int, seed: int) -> list[common.TokenBatch]:
    return [*common.paloma_batches(BATCH, max_sequences=paloma), *common.train_batches(BATCH, train, seed=seed)]


# ---------------------------------------------------------------------------------------------------------------
# census


def run_census(model, mesh, batches, feats, vocab, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    sample_experts = np.stack([np.sort(rng.choice(E, SAMPLE_EXPERTS, replace=False)) for _ in range(L)])
    norms = unit_norms(model)
    s_ex = jnp.asarray(sample_experts)

    @jax.jit
    def step(state, model, norms, tokens, seg, weight, cls, key):
        tr = forward.trace_forward(model, tokens, seg, forward.no_ablation())
        valid = _flat_valid(weight).astype(jnp.float32)
        cls = reshard(cls.reshape(-1, NC), P(None, None))
        new = dict(state)
        samples = []
        for li, lay in enumerate(tr["layers"]):
            e, tok, pre, act = lay["expert_sorted"], lay["token_sorted"], lay["expert_pre"], lay["expert_act"]
            v = valid[tok]
            w = _assignment_combine(lay)
            ct = cls[tok] * v[:, None]
            new["counts"] = new["counts"].at[li].add(jax.ops.segment_sum(ct, e, num_segments=E))
            new["wsum"] = new["wsum"].at[li].add(jax.ops.segment_sum(ct * w[:, None], e, num_segments=E))
            c = act * norms[li][e]
            srt = -jnp.sort(-c, axis=-1)
            tot = jnp.sum(srt, axis=-1, keepdims=True)
            cum = jnp.cumsum(srt, axis=-1) / jnp.maximum(tot, 1e-30)
            for qi, q in enumerate(QUANTS):
                kq = jnp.sum(cum < q, axis=-1) + 1
                new["k_hist"] = new["k_hist"].at[li, qi, jnp.clip(kq, 0, I)].add(v)
            pr = jnp.square(tot[:, 0]) / jnp.maximum(jnp.sum(c * c, axis=-1), 1e-30)
            pr_bin = jnp.clip(jnp.floor(pr).astype(jnp.int32), 0, len(PR_BINS) - 2)
            new["pr_hist"] = new["pr_hist"].at[li, pr_bin].add(v)
            new["active_hist"] = new["active_hist"].at[li, jnp.sum(pre > 0, axis=-1)].add(v)
            in_sample = jnp.any(e[:, None] == s_ex[li][None, :], axis=-1) & (v > 0)
            pick = _pick_rows(in_sample, jax.random.fold_in(key, li), SAMPLE_PER_BATCH)
            safe = jnp.maximum(pick, 0)
            samples.append(
                dict(pre=pre[safe].astype(jnp.float16), expert=jnp.where(pick >= 0, e[safe], -1), tok=tok[safe])
            )
        new["class_tokens"] = state["class_tokens"] + jnp.sum(cls * valid[:, None], axis=0)
        return new, samples

    state = dict(
        counts=jnp.zeros((L, E, NC)),
        wsum=jnp.zeros((L, E, NC)),
        k_hist=jnp.zeros((L, len(QUANTS), I + 1)),
        pr_hist=jnp.zeros((L, len(PR_BINS) - 1)),
        active_hist=jnp.zeros((L, I + 1)),
        class_tokens=jnp.zeros((NC,)),
    )
    sample_pre = np.zeros((L, SAMPLE_EXPERTS, SAMPLE_ROWS, I), np.float16)
    sample_tok = np.full((L, SAMPLE_EXPERTS, SAMPLE_ROWS), -1, np.int64)
    filled = np.zeros((L, SAMPLE_EXPERTS), np.int64)
    key = jax.random.PRNGKey(seed)
    t0 = time.time()
    toks_all, segs_all, valid_all, src_all = [], [], [], []
    offset = 0
    with set_mesh(mesh):
        for n, b in enumerate(batches):
            cls = class_masks(b.tokens, b.segment_ids, b.loss_weight, b.source, feats, vocab)
            tokens, seg, w = common.device_batch(b, mesh)
            state, samples = step(state, model, norms, tokens, seg, w, jnp.asarray(cls), jax.random.fold_in(key, n))
            if filled.min() < SAMPLE_ROWS:
                for li, smp in enumerate(samples):
                    ex, pre, tok = np.asarray(smp["expert"]), np.asarray(smp["pre"]), np.asarray(smp["tok"])
                    for j in range(SAMPLE_EXPERTS):
                        rows = np.nonzero(ex == sample_experts[li, j])[0][: SAMPLE_ROWS - filled[li, j]]
                        sl = slice(filled[li, j], filled[li, j] + len(rows))
                        sample_pre[li, j, sl] = pre[rows]
                        sample_tok[li, j, sl] = tok[rows] + offset
                        filled[li, j] += len(rows)
            offset += b.tokens.size
            toks_all.append(b.tokens.astype(np.uint16))
            segs_all.append(b.segment_ids.astype(np.int32))
            valid_all.append((b.loss_weight > 0).astype(np.uint8))
            src_all += [b.source] * b.tokens.shape[0]
            if n % 20 == 0:
                logger.info(
                    "census batch %d/%d (%.0fs) sample fill min %d", n, len(batches), time.time() - t0, filled.min()
                )
        host = {k: np.asarray(v) for k, v in state.items()}
        host["norms"] = np.asarray(norms)
    host.update(
        classes=np.asarray(CLASSES),
        quants=np.asarray(QUANTS),
        sample_experts=sample_experts,
        sample_pre=sample_pre,
        sample_tok=sample_tok,
        sample_filled=filled,
        tokens=np.concatenate(toks_all),
        segment_ids=np.concatenate(segs_all),
        valid=np.concatenate(valid_all),
        sources=np.asarray(src_all),
    )
    return host


# ---------------------------------------------------------------------------------------------------------------
# deep


def unit_vocab_effect(model, layer: int, expert: int) -> jax.Array:
    """``[I, V]`` each unit's centred direct logit effect (weights only)."""
    lay = model.layers()[layer]
    lag0 = lay.sconv_mlp.weight[0].astype(jnp.float32)
    dirs = jnp.einsum(
        "ir,rd->id", lay.mlp.expert_mlp.w_down[expert].astype(jnp.float32), lay.mlp.w_latent_up.astype(jnp.float32)
    )
    logits = (dirs * lag0) @ forward.unembed_direction(model)
    return logits - jnp.mean(logits, axis=-1, keepdims=True)


def run_deep(model, mesh, batches, layer: int, expert: int, seed: int) -> dict:
    @jax.jit
    def step(model, tokens, seg, weight, key):
        tr = forward.trace_forward(model, tokens, seg, forward.no_ablation())
        lay = tr["layers"][layer]
        valid = _flat_valid(weight)
        e, tok = lay["expert_sorted"], lay["token_sorted"]
        mask = (e == expert) & valid[tok]
        pick = _pick_rows(mask, key, DEEP_PER_BATCH)
        safe = jnp.maximum(pick, 0)
        return dict(
            pre=lay["expert_pre"][safe].astype(jnp.float16),
            tok=jnp.where(pick >= 0, tok[safe], -1),
            w=_assignment_combine(lay)[safe],
            loss=tr["loss"].reshape(-1)[tok[safe]],
            n=jnp.sum(mask),
        )

    pre, tok, wts, loss = [], [], [], []
    toks_all, segs_all, valid_all, src_all = [], [], [], []
    offset, routed, have = 0, 0, 0
    key = jax.random.PRNGKey(seed)
    t0 = time.time()
    with set_mesh(mesh):
        for n, b in enumerate(batches):
            tokens, seg, w = common.device_batch(b, mesh)
            out = jax.device_get(step(model, tokens, seg, w, jax.random.fold_in(key, n)))
            ok = out["tok"] >= 0
            pre.append(out["pre"][ok])
            tok.append(out["tok"][ok] + offset)
            wts.append(out["w"][ok])
            loss.append(out["loss"][ok])
            routed += int(out["n"])
            have += int(ok.sum())
            offset += b.tokens.size
            toks_all.append(b.tokens.astype(np.uint16))
            segs_all.append(b.segment_ids.astype(np.int32))
            valid_all.append((b.loss_weight > 0).astype(np.uint8))
            src_all += [b.source] * b.tokens.shape[0]
            if have >= DEEP_ROWS:
                break
        lens = np.asarray(unit_vocab_effect(model, layer, expert)).astype(np.float16)
        norms = np.asarray(unit_norms(model)[layer, expert])
    logger.info(
        "deep L%dE%d: %d rows of %d routed in %d batches (%.0fs)", layer, expert, have, routed, n + 1, time.time() - t0
    )
    return dict(
        pre=np.concatenate(pre)[:DEEP_ROWS],
        tok=np.concatenate(tok)[:DEEP_ROWS],
        w=np.concatenate(wts)[:DEEP_ROWS],
        loss=np.concatenate(loss)[:DEEP_ROWS],
        routed=np.asarray(routed),
        lens=lens,
        norms=norms,
        tokens=np.concatenate(toks_all),
        segment_ids=np.concatenate(segs_all),
        valid=np.concatenate(valid_all),
        sources=np.asarray(src_all),
    )


# ---------------------------------------------------------------------------------------------------------------
# ablate


def richest_windows(deep: dict) -> tuple[np.ndarray, np.ndarray]:
    """``WINDOWS`` non-overlapping ``WINDOW``-token windows (within sequences) with the most sampled routed rows:
    tokens and segment ids (each window's first token starts a segment)."""
    s = common.SEQ_LEN
    nseq = deep["tokens"].shape[0]
    per = s // WINDOW
    counts = np.bincount(deep["tok"] // WINDOW, minlength=nseq * per)
    best = np.argsort(-counts, kind="stable")[:WINDOWS]
    toks = np.stack([deep["tokens"][w // per, (w % per) * WINDOW : (w % per + 1) * WINDOW] for w in best])
    segs = np.stack([deep["segment_ids"][w // per, (w % per) * WINDOW : (w % per + 1) * WINDOW] for w in best])
    valid = np.stack([deep["valid"][w // per, (w % per) * WINDOW : (w % per + 1) * WINDOW] for w in best])
    return toks.astype(np.int32), segs - segs[:, :1], valid, best


def run_ablate(model, mesh, deep: dict, layer: int, expert: int, cls_window: np.ndarray) -> dict:
    """Single-unit and sufficiency interventions on the expert's richest windows. ``cls_window`` ``[W, S]`` marks the
    class positions (by current token)."""
    toks, segs, valid, best = richest_windows(deep)
    norms = jnp.asarray(deep["norms"])
    base = forward.no_ablation()

    def routed_mask(tr):
        lay = tr["layers"][layer]
        m = (
            jnp.zeros((toks.size,), jnp.float32)
            .at[lay["token_sorted"]]
            .max((lay["expert_sorted"] == expert).astype(jnp.float32))
        )
        return m.reshape(toks.shape)

    @jax.jit
    def run_mult(model, tokens, seg, row):
        abl = base._replace(neuron_mult=base.neuron_mult.at[layer, expert].set(row))
        tr = forward.trace_forward(model, tokens, seg, abl)
        return tr["loss"], routed_mask(tr)

    def topk_gate(k):
        def gate(li, expert_sorted, _inp, pre):
            if li != layer:
                return jnp.ones_like(pre)
            c = jnp.square(jax.nn.relu(pre)) * norms[None, :]
            srt = -jnp.sort(-c, axis=-1)
            thr = jnp.take(srt, k - 1, axis=1)[:, None]
            keep = (c >= thr) & (c > 0)
            return jnp.where((expert_sorted == expert)[:, None], keep.astype(jnp.float32), 1.0)

        return gate

    @jax.jit
    def run_topk(model, tokens, seg, k):
        tr = forward.trace_forward(model, tokens, seg, base, routed_gate=topk_gate(k))
        return tr["loss"]

    with set_mesh(mesh):
        tokens, seg = jnp.asarray(toks), jnp.asarray(segs)
        clean_loss, rmask = (np.asarray(x) for x in run_mult(model, tokens, seg, jnp.ones((I,))))
        full_loss = np.asarray(run_mult(model, tokens, seg, jnp.zeros((I,)))[0])
        score_mask = (valid > 0) & np.pad(segs[:, 1:] == segs[:, :-1], ((0, 0), (0, 1)))
        routed_pos = score_mask & (rmask > 0)
        class_pos = routed_pos & (cls_window > 0)
        t0 = time.time()
        unit_routed = np.zeros(I)
        unit_class = np.zeros(I)
        unit_all = np.zeros(I)
        for i in range(I):
            loss = np.asarray(run_mult(model, tokens, seg, jnp.ones((I,)).at[i].set(0.0))[0])
            d = loss - clean_loss
            unit_routed[i] = d[routed_pos].mean()
            unit_class[i] = d[class_pos].mean() if class_pos.any() else np.nan
            unit_all[i] = d[score_mask].mean()
            if i % 120 == 0:
                logger.info("ablate unit %d/%d (%.0fs)", i, I, time.time() - t0)

        def summary(loss):
            return np.asarray(
                [(loss - clean_loss)[m].mean() if m.any() else np.nan for m in (routed_pos, class_pos, score_mask)]
            )

        dyn = np.stack([summary(np.asarray(run_topk(model, tokens, seg, jnp.asarray(k)))) for k in SUFF_K])
        order = np.argsort(-np.nan_to_num(unit_class if class_pos.sum() > 200 else unit_routed, nan=-1e9))
        static, rand = [], []
        rng = np.random.default_rng(0)
        for k in SUFF_K:
            static.append(summary(np.asarray(run_mult(model, tokens, seg, jnp.zeros((I,)).at[order[:k]].set(1.0))[0])))
            rand.append(
                summary(
                    np.asarray(
                        run_mult(model, tokens, seg, jnp.zeros((I,)).at[rng.choice(I, k, replace=False)].set(1.0))[0]
                    )
                )
            )
    return dict(
        windows=best,
        window_tokens=toks,
        clean_loss=clean_loss,
        full_loss=full_loss,
        routed_pos=routed_pos,
        class_pos=class_pos,
        unit_routed=unit_routed,
        unit_class=unit_class,
        unit_all=unit_all,
        full=summary(full_loss),
        suff_k=np.asarray(SUFF_K),
        suff_dynamic=dyn,
        suff_static=np.stack(static),
        suff_random=np.stack(rand),
        static_order=order,
    )


# ---------------------------------------------------------------------------------------------------------------
# steer


def full_logp(model, hidden: jax.Array) -> jax.Array:
    logits = jnp.einsum("bsd,dv->bsv", hidden, model.output_proj, preferred_element_type=jnp.float32)
    if model.config.logit_soft_cap is not None:
        logits = forward.soft_cap(logits, model.config.logit_soft_cap)
    return jax.nn.log_softmax(logits, axis=-1)


def run_steer(model, mesh, deep: dict, layer: int, expert: int, spec: list[dict]) -> dict:
    """Each spec entry: ``unit``, ``mode`` ("scale" or "set"), ``value``, ``targets`` (token ids of the hypothesis).
    Steered wherever the expert is routed, on the expert's richest windows."""
    toks, segs, valid, _ = richest_windows(deep)
    base = forward.no_ablation()

    @jax.jit
    def run(model, tokens, seg, unit, mult, mean, targets):
        abl = base._replace(
            neuron_mult=base.neuron_mult.at[layer, expert, unit].set(mult),
            neuron_mean=base.neuron_mean.at[layer, expert, unit].set(mean),
        )
        tr = forward.trace_forward(model, tokens, seg, abl)
        lay = tr["layers"][layer]
        rmask = (
            jnp.zeros((toks.size,), jnp.float32)
            .at[lay["token_sorted"]]
            .max((lay["expert_sorted"] == expert).astype(jnp.float32))
        )
        lp = full_logp(model, tr["hidden"])
        lp_set = jax.scipy.special.logsumexp(jnp.where(targets[None, None, :], lp, -jnp.inf), axis=-1)
        return tr["top_ids"][..., 0], tr["loss"], rmask.reshape(toks.shape), lp_set, lp

    @jax.jit
    def kl_of(lp0, lp1):
        return jnp.sum(jnp.exp(lp0) * (lp0 - lp1), axis=-1)

    out = {}
    with set_mesh(mesh):
        tokens, seg = jnp.asarray(toks), jnp.asarray(segs)
        clean = None
        for j, s in enumerate(spec):
            targets = np.zeros(V, bool)
            targets[np.asarray(s["targets"], np.int64)] = True
            tj = jnp.asarray(targets)
            unit = jnp.asarray(s["unit"], jnp.int32)
            top0, loss0, rmask, set0, lp0 = run(model, tokens, seg, unit, 1.0, 0.0, tj)
            if clean is None:
                clean = dict(top0=top0, loss0=loss0)
            mult, mean = (float(s["value"]), 0.0) if s["mode"] == "scale" else (0.0, float(s["value"]))
            top1, loss1, _, set1, lp1 = run(model, tokens, seg, unit, mult, mean, tj)
            rec = dict(top1=top1, dloss=loss1 - loss0, set0=set0, set1=set1, kl=kl_of(lp0, lp1), routed=rmask)
            for k, v in rec.items():
                out[f"s{j}/{k}"] = np.asarray(v, np.int32 if k == "top1" else np.float32)
            logger.info("steer %d/%d %s", j + 1, len(spec), s.get("name", ""))
        out["top0"] = np.asarray(clean["top0"], np.int32)
        out["loss0"] = np.asarray(clean["loss0"], np.float32)
    out["window_tokens"] = toks
    out["window_segments"] = segs
    out["window_valid"] = valid
    out["spec"] = np.asarray(json.dumps(spec))
    return out


# ---------------------------------------------------------------------------------------------------------------


def load_npz(tag: str, name: str) -> dict:
    with fsspec.open(f"{common.OUTPUT_ROOT}/{tag}/{name}.npz", "rb") as f:
        z = np.load(io.BytesIO(f.read()))
        return {k: z[k] for k in z.files}


def window_classes(deep: dict, feats, vocab, cls_name: str) -> np.ndarray:
    toks, segs, valid, best = richest_windows(deep)
    per = common.SEQ_LEN // WINDOW
    out = np.zeros(toks.shape, np.float32)
    ci = CLASSES.index(cls_name)
    for r, w in enumerate(best):
        seq = w // per
        src = str(deep["sources"][seq])
        m = class_masks(toks[r : r + 1], segs[r : r + 1], valid[r : r + 1], src, feats, vocab)
        out[r] = m[0, :, ci]
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--phases", required=True, help="census | deep,ablate | steer")
    parser.add_argument("--target", default="", help="L:E for deep / ablate / steer")
    parser.add_argument("--cls", default="digit", help="the target's class (a CLASSES name) for the ablation")
    parser.add_argument("--spec", default="", help="steer: JSON spec path (fsspec)")
    parser.add_argument("--name", default="", help="steer: output name suffix")
    parser.add_argument("--paloma-sequences", type=int, default=16)
    parser.add_argument("--train-sequences", type=int, default=256)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    cfg = common.model_config(args.tag)
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    with set_mesh(mesh):
        model = forward.replicate(common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag]))
    vocab = common.vocab_strings()
    feats = vocab_features(vocab)
    phases = args.phases.split(",")
    layer, expert = (int(x) for x in args.target.split(":")) if args.target else (-1, -1)
    suffix = f"L{layer}E{expert}"
    t0 = time.time()
    deep = None
    for phase in phases:
        if phase == "census":
            batches = data_batches(args.paloma_sequences, args.train_sequences, args.seed)
            common.save_npz(args.tag, "experts_census", **run_census(model, mesh, batches, feats, vocab, args.seed))
        elif phase == "deep":
            batches = data_batches(args.paloma_sequences, args.train_sequences, args.seed + 1)
            deep = run_deep(model, mesh, batches, layer, expert, args.seed)
            common.save_npz(args.tag, f"experts_deep_{suffix}", **deep)
        elif phase == "ablate":
            deep = deep or load_npz(args.tag, f"experts_deep_{suffix}")
            cls_w = window_classes(deep, feats, vocab, args.cls)
            common.save_npz(args.tag, f"experts_ablate_{suffix}", **run_ablate(model, mesh, deep, layer, expert, cls_w))
        elif phase == "steer":
            deep = deep or load_npz(args.tag, f"experts_deep_{suffix}")
            with fsspec.open(args.spec, "r") as f:
                spec = json.load(f)
            res = run_steer(model, mesh, deep, layer, expert, spec)
            common.save_npz(args.tag, f"experts_steer_{suffix}{args.name}", **res)
        else:
            raise ValueError(f"unknown phase {phase}")
        logger.info("phase %s done (%.0fs)", phase, time.time() - t0)


if __name__ == "__main__":
    main()
