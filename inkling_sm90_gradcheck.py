import os, jax, jax.numpy as jnp
from levanter.grug.attention import AttentionMask, attention

print("backend:", jax.default_backend(), "FUSED_BWD:", os.environ.get("FAST_TRACK_INKLING_FUSED_BWD"))
B, S, Hq, Hkv, D, R, L = 2, 256, 4, 1, 128, 16, 128
ks = jax.random.split(jax.random.PRNGKey(0), 7)
q = jax.random.normal(ks[0], (B, S, Hq, D), jnp.float32) * 0.5
k = jax.random.normal(ks[1], (B, S, Hkv, D), jnp.float32) * 0.5
v = jax.random.normal(ks[2], (B, S, Hkv, D), jnp.float32) * 0.5
rel_r = jax.random.normal(ks[3], (B, S, Hq, R), jnp.float32) * 0.3
rel_proj = jax.random.normal(ks[5], (R, L), jnp.float32) * 0.3
cot = jax.random.normal(ks[4], (B, S, Hq, D), jnp.float32)
mask = AttentionMask.causal()

def mk(impl, with_bias):
    if not with_bias:
        def loss(q, k, v):
            o = attention(q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v.astype(jnp.bfloat16),
                          mask, implementation=impl)
            return jnp.sum(o.astype(jnp.float32) * cot)
        return jax.jit(jax.grad(loss, argnums=(0, 1, 2))), (q, k, v)

    def loss(q, k, v, rr, rp):
        o = attention(q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v.astype(jnp.bfloat16),
                      mask, implementation=impl, rel_r=rr.astype(jnp.bfloat16), rel_proj=rp.astype(jnp.bfloat16))
        return jnp.sum(o.astype(jnp.float32) * cot)
    return jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4))), (q, k, v, rel_r, rel_proj)

def compare(tag, with_bias):
    gfn, ga = mk("gpu_fa4_cute", with_bias)
    rfn, ra = mk("reference", with_bias)
    gg = gfn(*ga); gr = rfn(*ra)
    names = ["dq", "dk", "dv"] + (["dR", "dproj"] if with_bias else [])
    ok = True
    for nm, a, b in zip(names, gg, gr):
        a = a.astype(jnp.float32); b = b.astype(jnp.float32)
        d = jnp.abs(a - b); r = float(jnp.max(d) / (jnp.max(jnp.abs(b)) + 1e-6))
        p = bool(r < 0.05); ok = ok and p
        print(f"    [{tag}] {nm}: max_abs={float(jnp.max(d)):.4e} rel={r:.4e} -> {'PASS' if p else 'FAIL'}")
    print(f"=== [{tag}]", "PASS ===" if ok else "FAIL ===")

compare("NO-BIAS", False)
compare("WITH-BIAS", True)
