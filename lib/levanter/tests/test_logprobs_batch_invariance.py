"""Benchmark comparing logsumexp implementations against a reference."""

import time
import jax
import jax.numpy as jnp

def logsumexp_reference(x, axis=-1):
    """Reference implementation using explicit Python loops (non-JIT)."""
    assert axis == -1 or axis == len(x.shape) - 1, "This implementation of logsumexp only supports last axis"
    x_2d = x.reshape(-1, x.shape[-1])
    logprobs_individual = []
    for i in range(x_2d.shape[0]):
        logprobs_individual.append(jax.nn.logsumexp(x_2d[i]))
    return jnp.array(logprobs_individual).reshape(x.shape[:-1])


def logsumexp_batch_invariant(x: jax.Array, axis: int = -1) -> jax.Array:
    assert axis == -1 or axis == len(x.shape) - 1, "This implementation of logsumexp only supports last axis"
    x_2d = x.reshape(-1, x.shape[-1])
    logprobs_individual = []
    for i in range(x_2d.shape[0]):
        logprobs_individual.append(jax.jit(jax.nn.logsumexp)(x_2d[i]))
    return jnp.array(logprobs_individual).reshape(x.shape[:-1])


B = 16
K = 256
M = 128256
NUM_TRIALS = 10


def _make_input(key: jax.random.PRNGKey) -> jnp.ndarray:
    return jax.random.uniform(key, shape=(B, K, M), minval=0, maxval=1.0, dtype=jnp.float32)


# Generate fixed keys for determinism
key = jax.random.PRNGKey(0)
keys = jax.random.split(key, NUM_TRIALS)

# Correctness test: average max absolute difference over 10 inputs
print("=" * 60)
print("CORRECTNESS TEST")
print("=" * 60)
jax_diffs = []
batch_invariant_diffs = []
for i, k in enumerate(keys):
    a = _make_input(k)
    res_ref = logsumexp_reference(a, axis=-1)
    res_jax = jax.nn.logsumexp(a, axis=-1)
    res_batch_invariant = logsumexp_batch_invariant(a, axis=-1)

    jax_diff = jnp.max(jnp.abs(res_jax - res_ref))
    batch_invariant_diff = jnp.max(jnp.abs(res_batch_invariant - res_ref))

    jax_diffs.append(float(jax_diff))
    batch_invariant_diffs.append(float(batch_invariant_diff))
    print(f"Input {i+1}: jax.nn={jax_diff:.2e}")
    print(f"Input {i+1}: batch_invariant={batch_invariant_diff:.2e}")

avg_jax_diff = sum(jax_diffs) / len(jax_diffs)
avg_batch_invariant_diff = sum(batch_invariant_diffs) / len(batch_invariant_diffs)
print(f"\nAverage max absolute difference vs reference:")
print(f"  jax.nn.logsumexp:          {avg_jax_diff:.2e}")
print(f"  batch_invariant:           {avg_batch_invariant_diff:.2e}")

# Performance test over the same 10 inputs
print("\n" + "=" * 60)
print("PERFORMANCE TEST")
print("=" * 60)

# Warmup and compile (input generation not timed)
for k in keys[:2]:
    a = _make_input(k)
    _ = logsumexp_reference(a, axis=-1).block_until_ready()
    _ = jax.nn.logsumexp(a, axis=-1).block_until_ready()
    _ = logsumexp_batch_invariant(a, axis=-1).block_until_ready()

# Benchmark reference (exclude input generation time)
ref_time = 0.0
for k in keys:
    a = _make_input(k)
    start = time.time()
    _ = logsumexp_reference(a, axis=-1).block_until_ready()
    ref_time += time.time() - start

# Benchmark jax.nn.logsumexp (exclude input generation time)
jax_time = 0.0
for k in keys:
    a = _make_input(k)
    start = time.time()
    _ = jax.nn.logsumexp(a, axis=-1).block_until_ready()
    jax_time += time.time() - start

# Benchmark batch_invariant (exclude input generation time)
batch_invariant_time = 0.0
for k in keys:
    a = _make_input(k)
    start = time.time()
    _ = logsumexp_batch_invariant(a, axis=-1).block_until_ready()
    batch_invariant_time += time.time() - start

print(f"logsumexp_reference:       {ref_time*1000:.2f} ms ({ref_time*1000/NUM_TRIALS:.2f} ms/input)")
print(f"jax.nn.logsumexp:          {jax_time*1000:.2f} ms ({jax_time*1000/NUM_TRIALS:.2f} ms/input)")
print(f"logsumexp_batch_invariant: {batch_invariant_time*1000:.2f} ms ({batch_invariant_time*1000/NUM_TRIALS:.2f} ms/input)")
print(f"\nSpeedup vs reference:")
print(f"  jax.nn.logsumexp:          {ref_time/jax_time:.2f}x")
print(f"  batch_invariant:           {ref_time/batch_invariant_time:.2f}x")
