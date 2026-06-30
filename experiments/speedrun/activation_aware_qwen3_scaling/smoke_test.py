"""CPU smoke test: tiny Qwen3 + activation-aware Muon end-to-end (capture + optimizer + step)."""
import jax, jax.numpy as jnp
import haliax as hax
from levanter.models.qwen import Qwen3Config
from levanter.models.lm_model import LmExample
from levanter.layers.attention import AttentionMask
from levanter.optim.activation_capture import compute_input_grams
from levanter.optim.activation_aware import ActivationAwareConfig

cfg = Qwen3Config(max_seq_len=32, hidden_dim=64, intermediate_dim=128, num_layers=3,
                  num_heads=4, num_kv_heads=2, scan_layers=True, gradient_checkpointing=False,
                  tie_word_embeddings=True)
Vocab = hax.Axis("vocab", 128)
key = jax.random.PRNGKey(0)
model = cfg.build(Vocab, key=key)

print("Embed axis:", cfg.Embed)
Batch, Pos = hax.Axis("batch", 2), hax.Axis("position", 32)
tokens = hax.named(jax.random.randint(key, (2, 32), 0, 128), (Batch, Pos))
attn_mask = AttentionMask.causal()

# 1. capture grams
grams = compute_input_grams(model, tokens, attn_mask)
print("GRAMS:", {k: v.axes for k, v in grams.items()})
for k, v in grams.items():
    assert v.array.ndim == 3, f"{k} expected [Layers,Embed,Embed2], got {v.axes}"
    assert jnp.all(jnp.isfinite(v.array)), f"{k} non-finite"
print("  attn_in[0] symmetric:", float(jnp.max(jnp.abs(grams['attn_in'].array[0] - grams['attn_in'].array[0].T))))

# 2. build optimizer + one real step through take_train_step path
opt_cfg = ActivationAwareConfig(lr=0.02, adam_lr=6e-4, damping=1e-3,
                                normalize_fro=True, weight_decay=0.0, learning_rate=0.02)
optimizer = opt_cfg.build(10)
import equinox as eqx
from levanter.trainer_state import take_train_step

def loss_fn(m):
    ex = LmExample.causal(tokens["batch", 0], loss_mask=None) if False else None
    logits = m(tokens, attn_mask=attn_mask, key=None)
    return logits.mean().scalar()

loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

new_model, new_opt_state, updates = take_train_step(
    optimizer, model, opt_state, grads, grams=grams, is_trainable=True,
)
# verify updates finite + changed
import jax.tree_util as jtu
leaves = [l for l in jtu.tree_leaves(eqx.filter(updates, eqx.is_inexact_array))]
allfin = all(bool(jnp.all(jnp.isfinite(l))) for l in leaves)
print("STEP OK. updates finite:", allfin, "| n_leaves:", len(leaves), "| loss:", float(loss))
assert allfin
print("SMOKE PASS")
