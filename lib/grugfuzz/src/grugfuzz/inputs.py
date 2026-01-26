"""Random input generators for testing model components."""

import jax
import jax.numpy as jnp


def random_tokens(
    vocab_size: int,
    batch: int = 2,
    seq: int = 32,
    seed: int = 42,
) -> jax.Array:
    """Generate random token IDs.

    Args:
        vocab_size: Size of vocabulary (tokens will be in [0, vocab_size))
        batch: Batch size
        seq: Sequence length
        seed: Random seed for reproducibility

    Returns:
        JAX array of shape (batch, seq) with dtype int32
    """
    key = jax.random.PRNGKey(seed)
    return jax.random.randint(key, (batch, seq), 0, vocab_size, dtype=jnp.int32)


def random_hidden(
    batch: int,
    seq: int,
    dim: int,
    seed: int = 42,
    dtype: jnp.dtype = jnp.float32,
    scale: float = 0.02,
) -> jax.Array:
    """Generate random hidden states for testing intermediate layers.

    Args:
        batch: Batch size
        seq: Sequence length
        dim: Hidden dimension
        seed: Random seed for reproducibility
        dtype: Output dtype
        scale: Standard deviation of the normal distribution

    Returns:
        JAX array of shape (batch, seq, dim)
    """
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, (batch, seq, dim), dtype=dtype) * scale


def random_qkv(
    batch: int,
    seq: int,
    num_heads: int,
    head_dim: int,
    seed: int = 42,
    dtype: jnp.dtype = jnp.float32,
    scale: float = 0.02,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Generate random Q, K, V tensors for attention testing.

    Args:
        batch: Batch size
        seq: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        seed: Random seed for reproducibility
        dtype: Output dtype
        scale: Standard deviation of the normal distribution

    Returns:
        Tuple of (Q, K, V) JAX arrays, each of shape (batch, num_heads, seq, head_dim)
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 3)
    shape = (batch, num_heads, seq, head_dim)
    q = jax.random.normal(keys[0], shape, dtype=dtype) * scale
    k = jax.random.normal(keys[1], shape, dtype=dtype) * scale
    v = jax.random.normal(keys[2], shape, dtype=dtype) * scale
    return q, k, v


def random_attention_mask(
    batch: int,
    seq: int,
    seed: int = 42,
    pad_ratio: float = 0.1,
) -> jax.Array:
    """Generate random attention mask with some padding.

    Args:
        batch: Batch size
        seq: Sequence length
        seed: Random seed for reproducibility
        pad_ratio: Approximate ratio of positions to mask out

    Returns:
        JAX array of shape (batch, seq) with dtype bool,
        True = attend, False = mask out
    """
    key = jax.random.PRNGKey(seed)
    # Generate random mask
    rand = jax.random.uniform(key, (batch, seq))
    return rand > pad_ratio


def random_kv_cache(
    batch: int,
    cache_len: int,
    num_kv_heads: int,
    head_dim: int,
    seed: int = 42,
    dtype: jnp.dtype = jnp.float32,
    scale: float = 0.02,
) -> tuple[jax.Array, jax.Array]:
    """Generate random KV cache for testing cached attention.

    Args:
        batch: Batch size
        cache_len: Length of cached sequence
        num_kv_heads: Number of KV heads
        head_dim: Dimension per head
        seed: Random seed for reproducibility
        dtype: Output dtype
        scale: Standard deviation of the normal distribution

    Returns:
        Tuple of (K_cache, V_cache) JAX arrays,
        each of shape (batch, num_kv_heads, cache_len, head_dim)
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 2)
    shape = (batch, num_kv_heads, cache_len, head_dim)
    k_cache = jax.random.normal(keys[0], shape, dtype=dtype) * scale
    v_cache = jax.random.normal(keys[1], shape, dtype=dtype) * scale
    return k_cache, v_cache
