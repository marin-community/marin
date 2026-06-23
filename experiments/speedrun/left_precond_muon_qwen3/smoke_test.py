# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke test: left-precond Muon (idea 4) — helper reduces to Muon at H=I + full step finite."""

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from levanter.layers.attention import AttentionMask
from levanter.models.qwen import Qwen3Config
from levanter.optim.left_precond_muon import (
    LeftPrecondMuonConfig,
    _left_precond_matrix,
    _trunc_inv_sqrt,
)
from levanter.trainer_state import take_train_step

np.random.seed(0)

# 1. _left_precond_matrix with H = c*I should match plain Muon (cos ~ 1)
out, inn = 256, 128
M = jnp.asarray(np.random.randn(out, inn).astype(np.float32))
from levanter.optim.util import zeropower_via_newtonschulz5

muon = zeropower_via_newtonschulz5(M, steps=5, eps=1e-7, coefficient_type="quintic")
muon = muon * jnp.sqrt(max(1.0, out / inn))
for c in [1.0, 7.3, 1e3]:
    H = (c * jnp.eye(out)).astype(jnp.float32)
    D = _left_precond_matrix(M, H, clamp_rel=1e-6, ns_steps=5, eps=1e-7, use_kimi_scaling=False)
    cos = float((D * muon).sum() / (jnp.linalg.norm(D) * jnp.linalg.norm(muon)))
    print(
        f"H=c*I (c={c:<6g}) cos(D,Muon)={cos:.4f}  ||D||/||Muon||={float(jnp.linalg.norm(D)/jnp.linalg.norm(muon)):.4f}"
    )

# 2. truncated pseudo-inverse: rank-deficient H zeros the null directions
Q, _ = np.linalg.qr(np.random.randn(out, out))
ev = np.zeros(out)
ev[:50] = np.logspace(0, -3, 50)  # only 50 nonzero eigenvalues
Hrank = jnp.asarray((Q * ev) @ Q.T).astype(jnp.float32)
inv = _trunc_inv_sqrt(Hrank, clamp_rel=1e-6, eps=1e-7)
print("trunc-inv finite:", bool(jnp.all(jnp.isfinite(inv))), "| rank(inv)~", int(jnp.linalg.matrix_rank(inv)))

# 3. full optimizer through take_train_step on tiny qwen3
cfg = Qwen3Config(
    max_seq_len=32,
    hidden_dim=64,
    intermediate_dim=128,
    num_layers=3,
    num_heads=4,
    num_kv_heads=2,
    scan_layers=True,
    gradient_checkpointing=False,
    tie_word_embeddings=True,
)
Vocab = hax.Axis("vocab", 128)
key = jax.random.PRNGKey(0)
model = cfg.build(Vocab, key=key)
B, P = hax.Axis("batch", 2), hax.Axis("position", 32)
tok = hax.named(jax.random.randint(key, (2, 32), 0, 128), (B, P))
am = AttentionMask.causal()


def loss_fn(m):
    return m(tok, attn_mask=am, key=None).mean().scalar()


loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
opt = LeftPrecondMuonConfig(
    lr=0.02, adam_lr=6e-4, h_beta=0.95, clamp_rel=1e-6, weight_decay=0.0, learning_rate=0.02
).build(10)
gp = eqx.filter(model, eqx.is_inexact_array)
st = opt.init(gp)
# two steps to exercise the H EMA
for i in range(2):
    new_model, st, updates = take_train_step(opt, model, st, grads, is_trainable=True)
    leaves = jax.tree_util.tree_leaves(eqx.filter(updates, eqx.is_inexact_array))
    allfin = all(bool(jnp.all(jnp.isfinite(l))) for l in leaves)
    print(f"step {i}: updates finite={allfin}  n_leaves={len(leaves)}")
    assert allfin
print("SMOKE PASS")
