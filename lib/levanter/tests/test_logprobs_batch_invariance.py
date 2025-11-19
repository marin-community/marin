"""Benchmark comparing logsumexp implementations against a reference."""

import time
import jax
import jax.numpy as jnp

def logsumexp_reference(x, axis=-1):
    """Reference implementation using explicit Python loops (non-JIT)."""
    logprobs_individual = []
    for i in range(x.shape[0]):
        logprobs_individual.append(jax.nn.logsumexp(x[i:i+1], axis=axis))
    return jnp.array(logprobs_individual)

B = 16
K = 256
M = 128256
NUM_TRIALS = 10

# Generate 10 random inputs
key = jax.random.PRNGKey(0)
keys = jax.random.split(key, NUM_TRIALS)
inputs = [jax.random.uniform(k, shape=(B, K, M), minval=0, maxval=1.0, dtype=jnp.float32) for k in keys]

# Correctness test: average max absolute difference over 10 inputs
print("=" * 60)
print("CORRECTNESS TEST")
print("=" * 60)
jax_diffs = []
for i, a in enumerate(inputs):
    res_ref = logsumexp_reference(a, axis=-1)
    res_jax = jax.nn.logsumexp(a, axis=-1)
    
    jax_diff = jnp.max(jnp.abs(res_jax - res_ref))
    
    jax_diffs.append(float(jax_diff))
    print(f"Input {i+1}: jax.nn={jax_diff:.2e}")

avg_jax_diff = sum(jax_diffs) / len(jax_diffs)
print(f"\nAverage max absolute difference vs reference:")
print(f"  jax.nn.logsumexp:          {avg_jax_diff:.2e}")

# Performance test over the same 10 inputs
print("\n" + "=" * 60)
print("PERFORMANCE TEST")
print("=" * 60)

# Warmup and compile
for a in inputs[:2]:
    _ = logsumexp_reference(a, axis=-1).block_until_ready()
    _ = jax.nn.logsumexp(a, axis=-1).block_until_ready()

# Benchmark reference
start = time.time()
for a in inputs:
    _ = logsumexp_reference(a, axis=-1).block_until_ready()
ref_time = time.time() - start

# Benchmark jax.nn.logsumexp
start = time.time()
for a in inputs:
    _ = jax.nn.logsumexp(a, axis=-1).block_until_ready()
jax_time = time.time() - start

print(f"logsumexp_reference:       {ref_time*1000:.2f} ms ({ref_time*1000/NUM_TRIALS:.2f} ms/input)")
print(f"jax.nn.logsumexp:          {jax_time*1000:.2f} ms ({jax_time*1000/NUM_TRIALS:.2f} ms/input)")
print(f"\nSpeedup vs reference:")
print(f"  jax.nn.logsumexp:          {ref_time/jax_time:.2f}x")
