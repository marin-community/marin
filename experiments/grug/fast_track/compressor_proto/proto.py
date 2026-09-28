# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU toy for the stable-compressor memory idea (.agents/projects/stable-compressor-memory.md).

A small causal transformer trains on a token stream. Memory variants add a hashed n-gram table read at
the input and before the head:

- ``--learned-orders``: SGD/Adam-trained hashed n-gram embeddings (Engram-style; the fast_track bigram table).
- ``--stat-orders``: FIXED-encoder statistic tables. Row h holds S[h] = sum of v(y) over every occurrence of
  context hash h followed by next token y, and n[h] = the occurrence count, where v is a fixed
  (never-trained) next-token code. The model reads [S/n, log(1+n)] through learned linear readers. The
  table is written online after each step (write after read, so no step sees its own targets), and can
  be pre-filled from tokens the run never trains on (``--prefill``). Nothing about a row goes stale:
  the encoder v and the hash are fixed, so sums from step 0 mean the same thing at step 10k.

Both kinds can be combined. Run: JAX_PLATFORMS=cpu python proto.py --stat-orders 2 --prefill extra ...
"""

import argparse
import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

HASH_MUL = np.uint32(0x9E3779B1)
HASH_SALT = np.uint32(0x27D4EB2D)


def ngram_hash(tokens: jnp.ndarray, order: int, rows: int, salt: int) -> jnp.ndarray:
    """Hash the n-gram ending at each position (previous ``order-1`` tokens + current) into ``rows``."""
    x = jnp.zeros(tokens.shape, jnp.uint32) + jnp.uint32(salt) * HASH_SALT
    for lag in range(order - 1, -1, -1):
        prev = jnp.pad(tokens, ((0, 0), (lag, 0)), constant_values=-1)[:, : tokens.shape[1]] if lag else tokens
        x = (x + prev.astype(jnp.uint32) + jnp.uint32(1)) * HASH_MUL
        x = x ^ (x >> 15)
    return (x % jnp.uint32(rows)).astype(jnp.int32)


def rms(x):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6)


def init_params(key, cfg):
    d, v, layers = cfg.d, cfg.vocab, cfg.layers
    ks = iter(jax.random.split(key, 64))
    std = 1 / math.sqrt(d)
    p = {
        "embed": jax.random.normal(next(ks), (v, d)) * 1.0,
        "head": jax.random.normal(next(ks), (d, v)) * std * 0.5,
        "blocks": [],
    }
    for _ in range(layers):
        p["blocks"].append(
            {
                "wqkv": jax.random.normal(next(ks), (d, 3 * d)) * std,
                "wo": jnp.zeros((d, d)),
                "w1": jax.random.normal(next(ks), (d, 4 * d)) * std,
                "w2": jnp.zeros((4 * d, d)),
            }
        )
    if cfg.learned_orders:
        p["table"] = [jnp.zeros((cfg.rows, d)) for _ in cfg.learned_orders]
        p["mem_gate"] = [jnp.zeros((2, d)) for _ in cfg.learned_orders]
    if cfg.stat_orders:
        feat = cfg.code_dim + 1
        # Readers into the input stream and into the pre-head stream; the last layer is zero-init so step 0
        # is the base model. ``mlp`` readers put a GELU layer of width 2d in front.
        p["readers"] = []
        for _ in cfg.stat_orders:
            r = {"out_in": jnp.zeros((cfg.reader_width, d)), "out_head": jnp.zeros((cfg.reader_width, d))}
            if cfg.reader == "mlp":
                r["hidden"] = jax.random.normal(next(ks), (feat, cfg.reader_width)) / math.sqrt(feat)
            p["readers"].append(r)
    return p


def stat_features(table_sum, table_cnt, h):
    s = table_sum[h]
    n = table_cnt[h][..., None]
    mean = s / jnp.maximum(n, 1.0)
    return jnp.concatenate([mean, jnp.log1p(n) / 4.0], -1)


def forward(p, tokens, mem, cfg):
    """Return logits. ``mem`` is the (non-trainable) stat table tuple for memory=stat, else None."""
    b, t = tokens.shape
    d, nh = cfg.d, cfg.heads
    x = p["embed"][tokens]
    out_extra = 0.0
    for i, order in enumerate(cfg.learned_orders):
        row = p["table"][i][ngram_hash(tokens, order, cfg.rows, salt=i)]
        x = x + row * (1 + p["mem_gate"][i][0])
        out_extra = out_extra + row * p["mem_gate"][i][1]
    for i, order in enumerate(cfg.stat_orders):
        f = stat_features(mem[0][i], mem[1][i], ngram_hash(tokens, order, cfg.rows, salt=100 + i))
        r = p["readers"][i]
        if cfg.reader == "mlp":
            f = jax.nn.gelu(f @ r["hidden"])
        x = x + f @ r["out_in"]
        out_extra = out_extra + f @ r["out_head"]
    mask = jnp.tril(jnp.ones((t, t), bool))
    for blk in p["blocks"]:
        q, k, v = jnp.split(rms(x) @ blk["wqkv"], 3, -1)
        q, k, v = (a.reshape(b, t, nh, d // nh) for a in (q, k, v))
        q, k = rms(q), rms(k)
        att = jnp.einsum("bqhd,bkhd->bhqk", q, k) * (4.0 / math.sqrt(d // nh))
        att = jax.nn.softmax(jnp.where(mask, att, -1e9), -1)
        o = jnp.einsum("bhqk,bkhd->bqhd", att, v).reshape(b, t, d)
        x = x + o @ blk["wo"]
        x = x + jax.nn.gelu(rms(x) @ blk["w1"]) @ blk["w2"]
    x = rms(x) + out_extra
    return x @ p["head"]


def loss_fn(p, batch, mem, cfg):
    logits = forward(p, batch[:, :-1], mem, cfg)
    lp = jax.nn.log_softmax(logits.astype(jnp.float32), -1)
    return -jnp.mean(jnp.take_along_axis(lp, batch[:, 1:, None], -1))


def write_table(mem, tokens, code, cfg):
    """Add each (context hash, v(following tokens)) pair of ``tokens`` [B, T+1] into the stat table. With
    ``horizon`` H > 1 the value concatenates the codes of the next H tokens (code_dim / H dims each), so a row
    sketches what follows its context over H tokens, not just the next one."""
    sums, cnts = list(mem[0]), list(mem[1])
    ctx = tokens[:, :-1]
    t = ctx.shape[1]
    width = code.shape[1] // cfg.horizon
    parts = []
    for j in range(1, cfg.horizon + 1):
        future = jnp.pad(tokens[:, j:], ((0, 0), (0, j - 1)))[:, :t]
        valid = (jnp.arange(t) + j <= t)[None, :, None]
        parts.append(code[future][..., :width] * valid)
    vals = jnp.concatenate(parts, -1).reshape(-1, width * cfg.horizon)
    for i, order in enumerate(cfg.stat_orders):
        h = ngram_hash(ctx, order, cfg.rows, salt=100 + i).reshape(-1)
        sums[i] = sums[i].at[h].add(vals)
        cnts[i] = cnts[i].at[h].add(1.0)
    return tuple(sums), tuple(cnts)


def make_code(cfg, extra_tokens):
    """Fixed next-token code v: random Gaussian, or the top singular vectors of the bigram PMI of text the
    run never trains on (an 'untimed compressor' fit once)."""
    if cfg.code == "random":
        return jax.random.normal(jax.random.PRNGKey(123), (cfg.vocab, cfg.code_dim)) / math.sqrt(cfg.code_dim)
    v = cfg.vocab
    c = np.zeros((v, v), np.float64)
    np.add.at(c, (extra_tokens[:-1].astype(np.int64), extra_tokens[1:].astype(np.int64)), 1.0)
    c += 0.1
    pmi = np.log(c / c.sum()) - np.log(c.sum(1, keepdims=True) / c.sum()) - np.log(c.sum(0, keepdims=True) / c.sum())
    # Next-token side of the PMI factorisation: tokens that follow similar contexts get similar codes.
    _, s, vt = np.linalg.svd(pmi, full_matrices=False)
    code = vt[: cfg.code_dim].T * np.sqrt(s[: cfg.code_dim])
    code = code / np.sqrt(np.mean(np.sum(code**2, 1)))
    return jnp.asarray(code, jnp.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learned-orders", default="")
    ap.add_argument("--stat-orders", default="")
    ap.add_argument("--reader", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--rows", type=int, default=1 << 18)
    ap.add_argument("--code", choices=["random", "pmi"], default="random")
    ap.add_argument("--code-dim", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--prefill", choices=["none", "extra"], default="none")
    ap.add_argument("--prefill-frac", type=float, default=1.0)
    ap.add_argument("--table-lr-mult", type=float, default=10.0)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default=None)
    cfg = ap.parse_args()
    cfg.learned_orders = tuple(int(x) for x in cfg.learned_orders.split(",") if x)
    cfg.stat_orders = tuple(int(x) for x in cfg.stat_orders.split(",") if x)
    cfg.vocab = 4096
    cfg.reader_width = cfg.code_dim + 1 if cfg.reader == "linear" else 2 * cfg.d

    train = np.load(f"{cfg.data}/pg19_train.npy").astype(np.int32)
    extra = np.load(f"{cfg.data}/pg19_extra.npy").astype(np.int32)
    evals = np.load(f"{cfg.data}/pg19_eval.npy").astype(np.int32)
    w = cfg.seq + 1
    windows = train[: len(train) // w * w].reshape(-1, w)
    rng = np.random.default_rng(cfg.seed)
    windows = windows[rng.permutation(len(windows))]
    used = cfg.steps * cfg.batch
    assert used <= len(windows), (used, len(windows))
    run_windows, unused_windows = windows[:used], windows[used:]
    ev = evals[: len(evals) // w * w].reshape(-1, w)
    ev = ev[np.random.default_rng(99).permutation(len(ev))[:256]]
    # Extra text the run never trains on: the unused part of the train stream plus the held-aside books.
    extra_all = np.concatenate([unused_windows.reshape(-1), extra])

    key = jax.random.PRNGKey(cfg.seed)
    p = init_params(key, cfg)
    sched = optax.warmup_cosine_decay_schedule(0.0, cfg.lr, min(30, cfg.steps // 4), cfg.steps, cfg.lr * 0.05)

    def lr_mult(path, _):
        return "table" if any(getattr(k, "key", None) == "table" for k in path) else "main"

    labels = jax.tree_util.tree_map_with_path(lr_mult, p)
    opt = optax.multi_transform(
        {
            "main": optax.adamw(sched, b1=0.9, b2=0.95, weight_decay=0.0),
            "table": optax.adam(lambda s: sched(s) * cfg.table_lr_mult, b1=0.9, b2=0.95),
        },
        labels,
    )
    opt_state = opt.init(p)

    code = make_code(cfg, extra)
    mem = None
    if cfg.stat_orders:
        mem = (
            tuple(jnp.zeros((cfg.rows, cfg.code_dim)) for _ in cfg.stat_orders),
            tuple(jnp.zeros((cfg.rows,)) for _ in cfg.stat_orders),
        )
        if cfg.prefill == "extra":
            ex = extra_all[: len(extra_all) // w * w].reshape(-1, w)
            ex = ex[: int(len(ex) * cfg.prefill_frac)]
            writer = jax.jit(lambda m, t: write_table(m, t, code, cfg))
            for i in range(0, len(ex), 512):
                mem = writer(mem, jnp.asarray(ex[i : i + 512]))
            print(f"prefilled table from {ex.size / 1e6:.2f}M extra tokens")

    @jax.jit
    def step(p, opt_state, mem, batch):
        loss, g = jax.value_and_grad(loss_fn)(p, batch, mem, cfg)
        upd, opt_state = opt.update(g, opt_state, p)
        p = optax.apply_updates(p, upd)
        if cfg.stat_orders:
            mem = write_table(mem, batch, code, cfg)
        return p, opt_state, mem, loss

    @jax.jit
    def eval_loss(p, mem, batch):
        return loss_fn(p, batch, mem, cfg)

    def evaluate():
        return float(np.mean([eval_loss(p, mem, jnp.asarray(ev[i : i + 64])) for i in range(0, len(ev), 64)]))

    hist = []
    t0 = time.time()
    for s in range(cfg.steps):
        batch = jnp.asarray(run_windows[s * cfg.batch : (s + 1) * cfg.batch])
        p, opt_state, mem, loss = step(p, opt_state, mem, batch)
        if (s + 1) % cfg.eval_every == 0 or s == cfg.steps - 1:
            e = evaluate()
            hist.append({"step": s + 1, "tokens": (s + 1) * cfg.batch * cfg.seq, "train": float(loss), "eval": e})
            print(f"step {s + 1} train {float(loss):.4f} eval {e:.4f} ({time.time() - t0:.0f}s)", flush=True)
    if cfg.out:
        with open(cfg.out, "w") as f:
            json.dump({"cfg": {k: v for k, v in vars(cfg).items()}, "hist": hist}, f)


if __name__ == "__main__":
    main()
