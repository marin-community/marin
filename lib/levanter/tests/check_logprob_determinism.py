"""Minimal experiment to test if softmax_cross_entropy varies with sequence length."""

import jax.numpy as jnp
import jax

K = 3
M = 10
key = jax.random.PRNGKey(0)
a = jax.random.uniform(key, shape=(K,M), minval=0, maxval=1.0, dtype=jnp.float32)

batched_a = jnp.array([a, a])
batched_res = jax.nn.logsumexp(batched_a, axis=batched_a.ndim-1)[0]

res = jax.nn.logsumexp(a, axis=a.ndim-1)

print("Individual logsumexp result:")
print(res)
print("\nBatched logsumexp result:")
print(batched_res)
print("Max absolute difference:", jnp.max(jnp.abs(batched_res - res)))

# Show one example of diverging results
max_diff_idx = jnp.argmax(jnp.abs(batched_res - res))
print(f"\nExample diverging pair at index {max_diff_idx}:")
print(f"  Individual: {res[max_diff_idx]}")
print(f"  Batched:    {batched_res[max_diff_idx]}")
print(f"  Difference: {batched_res[max_diff_idx] - res[max_diff_idx]}")
