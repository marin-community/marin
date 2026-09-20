import os, jax, jax.numpy as jnp
from levanter.grug.attention import AttentionMask, attention

print("backend:", jax.default_backend(), "FUSED_BWD:", os.environ.get("FAST_TRACK_INKLING_FUSED_BWD"))
B, S, Hq, Hkv, D, L = 2, 256, 4, 1, 128, 128
ks = jax.random.split(jax.random.PRNGKey(0), 6)
q = jax.random.normal(ks[0], (B, S, Hq, D), jnp.float32) * 0.5
k = jax.random.normal(ks[1], (B, S, Hkv, D), jnp.float32) * 0.5
v = jax.random.normal(ks[2], (B, S, Hkv, D), jnp.float32) * 0.5
rel = jax.random.normal(ks[3], (B, Hq, S, L), jnp.float32) * 0.2
cot = jax.random.normal(ks[4], (B, S, Hq, D), jnp.float32)
mask = AttentionMask.causal()

def mk(impl, rel_arg):
    if rel_arg is None:
        def loss(q, k, v):
            o = attention(q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v.astype(jnp.bfloat16),
                          mask, implementation=impl, rel_bias=None)
            return jnp.sum(o.astype(jnp.float32) * cot)
        return jax.jit(jax.grad(loss, argnums=(0, 1, 2))), (q, k, v)

    def loss(q, k, v, rel):
        o = attention(q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v.astype(jnp.bfloat16),
                      mask, implementation=impl, rel_bias=rel.astype(jnp.bfloat16))
        return jnp.sum(o.astype(jnp.float32) * cot)
    return jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3))), (q, k, v, rel)

def compare(tag, rel_arg):
    gfn, gargs = mk("gpu_fa4_cute", rel_arg)
    rfn, rargs = mk("reference", rel_arg)
    gg = gfn(*gargs); gr = rfn(*rargs)
    names = ["dq", "dk", "dv"] + ([] if rel_arg is None else ["dA"])
    ok = True
    for nm, a, b in zip(names, gg, gr):
        a = a.astype(jnp.float32); b = b.astype(jnp.float32)
        d = jnp.abs(a - b); r = float(jnp.max(d) / (jnp.max(jnp.abs(b)) + 1e-6))
        p = bool(r < 0.03); ok = ok and p
        print(f"    [{tag}] {nm}: max_abs={float(jnp.max(d)):.4e} rel={r:.4e} -> {'PASS' if p else 'FAIL'}")
    print(f"=== [{tag}]", "PASS ===" if ok else "FAIL ===")

compare("NO-BIAS", None)
compare("WITH-BIAS", rel)
