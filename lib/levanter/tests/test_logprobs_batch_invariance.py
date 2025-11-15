"""Minimal experiment to test if softmax_cross_entropy varies with sequence length."""

import jax.numpy as jnp
import jax

import jax
import jax.numpy as jnp
from jax import lax

def logsumexp_batch_invariant(x, axis=-1):
    # Move axis to the end
    x = jnp.moveaxis(x, axis, -1)
    *batch, n = x.shape
    x_flat = x.reshape(-1, n)  # [B, n], B = prod(batch)

    def row_logsumexp(row):
        # Standard numerically stable pattern: m + log(sum(exp(x - m)))
        m = jnp.max(row)

        def body_fun(i, acc):
            # acc is the running sum of exp(row[..] - m)
            return acc + jnp.exp(row[i] - m)

        init = jnp.array(0.0, dtype=row.dtype)
        sum_exp = lax.fori_loop(0, n, body_fun, init)
        return m + jnp.log(sum_exp)

    # vmap over rows; the inner 1D reduction is fixed-shape and sequential
    lse_flat = jax.vmap(row_logsumexp)(x_flat)  # [B]
    lse = lse_flat.reshape(*batch)              # restore batch shape
    return lse

K = 3
M = 10
key = jax.random.PRNGKey(0)
a = jax.random.uniform(key, shape=(K,M), minval=0, maxval=1.0, dtype=jnp.float32)

batched_a = jnp.array([a, a])
batched_res = logsumexp_batch_invariant(batched_a, axis=batched_a.ndim-1)[0]

res = logsumexp_batch_invariant(a, axis=a.ndim-1)

print("Individual logsumexp result:")
print(res)
print("\nBatched logsumexp result:")
print(batched_res)
assert res.shape == batched_res.shape, f"Shape mismatch: {res.shape} vs {batched_res.shape}"
print("Max absolute difference:", jnp.max(jnp.abs(batched_res - res)))

# Show one example of diverging results
max_diff_idx = jnp.argmax(jnp.abs(batched_res - res))
print(f"\nExample diverging pair at index {max_diff_idx}:")
print(f"  Individual: {res[max_diff_idx]}")
print(f"  Batched:    {batched_res[max_diff_idx]}")
print(f"  Difference: {batched_res[max_diff_idx] - res[max_diff_idx]}")
