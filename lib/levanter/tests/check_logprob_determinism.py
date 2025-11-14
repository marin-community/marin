"""Minimal experiment to test if softmax_cross_entropy varies with sequence length."""

import jax.numpy as jnp
import optax


def test_logprob_sequence_independence():
    """Test if logprobs differ when computed all at once vs one at a time."""
    
    # Create dummy data: 5 tokens, vocab size 100
    seq_len = 5
    vocab_size = 100
    
    logits = jnp.array([
        [1.0, 2.0, 0.5] + [0.0] * (vocab_size - 3),  # token 0
        [0.3, 1.5, 2.1] + [0.0] * (vocab_size - 3),  # token 1
        [2.0, 0.1, 1.0] + [0.0] * (vocab_size - 3),  # token 2
        [0.5, 3.0, 0.2] + [0.0] * (vocab_size - 3),  # token 3
        [1.2, 0.8, 2.5] + [0.0] * (vocab_size - 3),  # token 4
    ], dtype=jnp.float32)
    
    labels = jnp.array([1, 2, 0, 1, 2], dtype=jnp.int32)
    
    # Compute all at once
    logprobs_all = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    
    # Compute one at a time
    logprobs_individual = []
    for i in range(seq_len):
        logprob = optax.softmax_cross_entropy_with_integer_labels(
            logits[i:i+1], labels[i:i+1]
        )
        logprobs_individual.append(logprob[0])
    
    logprobs_individual = jnp.array(logprobs_individual)
    
    # Compare
    print("All at once:", logprobs_all)
    print("One at a time:", logprobs_individual)
    print("Difference:", logprobs_all - logprobs_individual)
    print("Max absolute difference:", jnp.max(jnp.abs(logprobs_all - logprobs_individual)))
    
    # Assert they're identical (or very close)
    assert jnp.allclose(logprobs_all, logprobs_individual, atol=1e-6), \
        f"Logprobs differ! Max diff: {jnp.max(jnp.abs(logprobs_all - logprobs_individual))}"


if __name__ == "__main__":
    test_logprob_sequence_independence()
