"""Parity: sonic_cute full fwd+bwd (slim custom_vjp residuals) vs the scatter reference.

Run on 1 B200. Validates the residual-slimming rewrite: backward re-gathers
x_dispatch from (x, token_dispatch), recomputes h = swiglu(gu) elementwise, and
(under a mesh) stores expert weights FSDP-sharded.
"""
import traceback

import jax
import jax.numpy as jnp

from levanter.grug._moe.scatter import _moe_mlp_local_scatter
from levanter.grug._moe.sonic_cute import _moe_mlp_local_sonic_cute

T, H, I, E, K = 32768, 2560, 1280, 64, 4

k = jax.random.PRNGKey(0)
kx, ks, kw, k13, k2 = jax.random.split(k, 5)
x = (jax.random.normal(kx, (T, H)) * 0.1).astype(jnp.bfloat16)
selected = jax.random.randint(ks, (T, K), 0, E, dtype=jnp.int32)
combine = jax.nn.sigmoid(jax.random.normal(kw, (T, K))).astype(jnp.float32)
w13 = (jax.random.normal(k13, (E, H, 2 * I)) * 0.02).astype(jnp.bfloat16)
w2 = (jax.random.normal(k2, (E, I, H)) * 0.02).astype(jnp.bfloat16)


def make_loss(impl):
    def loss(x, combine, w13, w2):
        out, _ = impl(
            x, selected, combine, w13, w2, activation_fn=jax.nn.silu, num_experts=E
        )
        return jnp.sum(out.astype(jnp.float32) ** 2)

    return loss


failures = 0
try:
    ref_fn = jax.jit(jax.value_and_grad(make_loss(_moe_mlp_local_scatter), argnums=(0, 1, 2, 3)))
    got_fn = jax.jit(jax.value_and_grad(make_loss(_moe_mlp_local_sonic_cute), argnums=(0, 1, 2, 3)))
    ref_loss, ref_grads = ref_fn(x, combine, w13, w2)
    got_loss, got_grads = got_fn(x, combine, w13, w2)
    jax.block_until_ready((ref_loss, got_loss, ref_grads, got_grads))
    lrel = abs(float(got_loss) - float(ref_loss)) / (abs(float(ref_loss)) + 1e-6)
    print(f"loss ref={float(ref_loss):.4f} got={float(got_loss):.4f} rel={lrel:.5f} "
          f"{'ok' if lrel < 0.01 else 'FAIL'}", flush=True)
    if lrel >= 0.01:
        failures += 1
    for name, rg, gg in zip(("dx", "d_combine", "dw13", "dw2"), ref_grads, got_grads):
        rg32, gg32 = rg.astype(jnp.float32), gg.astype(jnp.float32)
        err = float(jnp.abs(gg32 - rg32).max())
        rel = err / (float(jnp.abs(rg32).max()) + 1e-6)
        status = "ok" if rel < 0.02 else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"{name}: max_abs {err:.6f} rel {rel:.5f} {status}", flush=True)
except Exception:
    traceback.print_exc()
    failures += 1

print("SLIM_PARITY_OK" if failures == 0 else f"SLIM_PARITY_FAIL ({failures})", flush=True)
