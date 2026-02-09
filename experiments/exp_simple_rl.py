# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Simple RL training system in Levanter.

Simplified version based on https://github.com/bwasti/spirl/blob/64b126fb68fac404e67ff1677469f6cb647a4e4a/simple_rl.py

Key simplifications:
- Single example
- Batch size 1 with group_size=4 samples for GRPO
- Local execution on v4-8
- Simple mesh with no data/tensor parallelism
- Exact token ID matching for rewards
- GRPO advantages computed exactly like spirl
"""

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh

import haliax as hax
from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.llama import LlamaConfig


def compute_grpo_advantages(rewards: jnp.ndarray, group_size: int = 4) -> jnp.ndarray:
    """
    Compute advantages using GRPO-style group ranking.
    Direct port from spirl simple_rl.py

    GRPO groups samples from the same prompt and ranks them by reward.
    This is a simplified version that just uses mean-centering within groups.

    Args:
        rewards: [batch] where batch = num_groups * group_size
        group_size: Number of samples per prompt

    Returns:
        advantages: [batch]
    """
    batch_size = rewards.shape[0]
    assert batch_size % group_size == 0, f"Batch size {batch_size} must be divisible by group_size {group_size}"

    num_groups = batch_size // group_size
    rewards_grouped = rewards.reshape(num_groups, group_size)

    # Compute advantages: reward - group_mean
    group_means = rewards_grouped.mean(axis=1, keepdims=True)
    advantages_grouped = rewards_grouped - group_means

    # Flatten back
    advantages = advantages_grouped.reshape(-1)

    return advantages


def compute_reward(generated_tokens: np.ndarray, expected_tokens: np.ndarray) -> float:
    """
    Compute reward based on exact token ID match.

    Args:
        generated_tokens: Array of generated token IDs
        expected_tokens: Array of expected token IDs (e.g., tokenized "Paris")

    Returns:
        1.0 if expected_tokens appear consecutively in generated_tokens, else 0.0
    """
    if np.array_equal(generated_tokens, expected_tokens):
        return 1.0
    else:
        return 0.0


def sample_completion(
    model,
    prompt_tokens: jax.Array,  # Named array
    max_tokens: int,
    key: jax.random.PRNGKey,
    temperature: float = 1.0,
) -> tuple[np.ndarray, list[float]]:
    """
    Sample tokens autoregressively and track logprobs.

    Args:
        model: The language model
        prompt_tokens: Named array of prompt token IDs [position]
        max_tokens: Maximum number of tokens to generate
        key: JAX random key
        temperature: Sampling temperature

    Returns:
        tokens: Generated token IDs (numpy array, not including prompt)
        log_probs: Per-token log probabilities (list of floats)
    """
    # Convert to regular array for easier manipulation
    current_tokens = prompt_tokens.array
    generated_tokens = []
    log_probs = []

    for _ in range(max_tokens):
        # Create named array for current sequence
        current_named = hax.named(current_tokens, ["position"])
        current_named = hax.shard(current_named)

        # Forward pass
        logits = model(input_ids=current_named, attn_mask=AttentionMask.causal(), key=key)

        # Get last position logits
        last_logits = logits.array[-1, :].astype(jnp.float32) / temperature

        # Sample token
        key, subkey = jax.random.split(key)
        sampled_token = jax.random.categorical(subkey, last_logits)

        # Compute log prob
        log_probs_dist = jax.nn.log_softmax(last_logits)
        log_prob = log_probs_dist[sampled_token]

        generated_tokens.append(int(sampled_token))
        log_probs.append(float(log_prob))

        # Append to sequence
        current_tokens = jnp.concatenate([current_tokens, jnp.array([sampled_token])])

    return np.array(generated_tokens), log_probs


def compute_policy_gradient_loss(
    model,
    prompt_tokens: jax.Array,
    completions_tokens: list[np.ndarray],
    advantages: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> jax.Array:
    """
    Compute policy gradient loss by re-evaluating completions under current policy.

    Args:
        model: Current policy model
        prompt_tokens: Named array of prompt tokens
        completions_tokens: List of generated token arrays (one per sample)
        advantages: GRPO advantages [group_size]
        key: JAX random key

    Returns:
        loss: Scalar loss value
    """
    # Re-evaluate log probs under current policy
    new_total_log_probs = []

    for completion_tokens in completions_tokens:
        # Concatenate prompt + completion
        full_sequence = jnp.concatenate([prompt_tokens.array, jnp.array(completion_tokens)])
        full_named = hax.named(full_sequence, ["position"])
        full_named = hax.shard(full_named)

        # Forward pass
        logits = model(input_ids=full_named, attn_mask=AttentionMask.causal(), key=key)
        logits_array = logits.array.astype(jnp.float32)

        # Get logits for predicting completion tokens
        # logits[i] predicts token[i+1]
        # We want logits for positions [len(prompt)-1 : len(prompt)+len(completion)-1]
        prompt_len = len(prompt_tokens.array)
        completion_len = len(completion_tokens)

        # Extract relevant logits
        relevant_logits = logits_array[prompt_len - 1 : prompt_len + completion_len - 1, :]

        # Compute log probs
        log_probs_dist = jax.nn.log_softmax(relevant_logits, axis=-1)

        # Extract log probs for actual tokens
        token_log_probs = log_probs_dist[:, completion_tokens]

        # Sum over sequence
        total_log_prob = jnp.sum(token_log_probs)
        new_total_log_probs.append(total_log_prob)

    new_total_log_probs = jnp.array(new_total_log_probs)

    # Policy gradient loss: -E[log_prob * advantage]
    pg_loss = -(new_total_log_probs * advantages).mean()

    return pg_loss


def main():
    parser = argparse.ArgumentParser(description="Simple RL training in Levanter")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace checkpoint to load",
    )
    parser.add_argument("--num-steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--group-size", type=int, default=16, help="Number of samples per prompt for GRPO")
    parser.add_argument("--max-gen-tokens", type=int, default=1, help="Maximum tokens to generate")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")

    # Load tokenizer
    tokenizer = load_tokenizer(args.checkpoint)

    # Create simple mesh with all devices as replicas (no data parallelism requirement)
    # This allows batch_size=1 to work without sharding constraints
    num_devices = jax.device_count()
    devices = jax.devices()
    mesh = Mesh(np.array(devices).reshape(num_devices, 1, 1), axis_names=("replica", "data", "model"))
    axis_mapping = {"batch": "data", "embed": "model"}

    print(f"Using {num_devices} devices in simple mesh configuration")

    with hax.axis_mapping(axis_mapping), mesh:
        print("Loading HF checkpoint...")
        converter = HFCheckpointConverter(
            LlamaConfig,
            reference_checkpoint=args.checkpoint,
            tokenizer=tokenizer,
        )

        # Load config and override attention backend
        hf_config = converter.hf_config_from_hf_checkpoint(args.checkpoint)
        model_config = converter.config_from_hf_config(hf_config, overrides={"attn_backend": AttentionBackend.VANILLA})

        model = converter.load_pretrained(
            model_config.model_type,
            ref=args.checkpoint,
            config=model_config,
            dtype=jnp.float32,
            axis_mapping=axis_mapping,
        )

        print("Model loaded successfully!")

        # Setup training
        prompt = "The number of Rs in the word strawberry is"
        expected_answer = " three"

        # Tokenize
        prompt_tokens_list = tokenizer.encode(prompt, add_special_tokens=False)
        expected_tokens = np.array(tokenizer.encode(expected_answer, add_special_tokens=False))

        print(f"\nPrompt: '{prompt}'")
        print(f"Expected answer: '{expected_answer}'")
        print(f"Expected tokens: {expected_tokens}")

        # Create named array for prompt
        prompt_array = np.array(prompt_tokens_list, dtype=np.int32)
        prompt_named = hax.named(prompt_array, ["position"])
        prompt_named = hax.shard(prompt_named)

        # Setup optimizer
        optimizer = optax.adam(args.learning_rate)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        print(f"\nStarting training for {args.num_steps} steps...")
        print(f"Group size: {args.group_size}, Max gen tokens: {args.max_gen_tokens}")
        print("=" * 80)

        # Training loop
        for step in range(args.num_steps):
            key = jax.random.PRNGKey(step)

            # 1. Rollout: Generate group_size samples
            completions = []

            for _ in range(args.group_size):
                key, subkey = jax.random.split(key)
                tokens, _ = sample_completion(model, prompt_named, args.max_gen_tokens, subkey, args.temperature)
                completions.append(tokens)

            # 2. Compute rewards (exact token match)
            rewards = jnp.array([compute_reward(comp, expected_tokens) for comp in completions])

            # 3. Compute GRPO advantages
            advantages = compute_grpo_advantages(rewards, args.group_size)

            # 4. Compute loss and gradients
            def loss_fn(model, completions=completions, advantages=advantages, key=key):
                return compute_policy_gradient_loss(model, prompt_named, completions, advantages, key)

            loss, grads = jax.value_and_grad(loss_fn)(model)

            # 5. Update model
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)

            # 6. Logging
            mean_reward = float(rewards.mean())

            print(f"Step {step:3d}: loss={float(loss):7.4f}, mean_reward={mean_reward:.2f}")

            # Decode all completions for inspection
            decoded_completions = [tokenizer.decode(comp, skip_special_tokens=True) for comp in completions]
            completions_str = ", ".join(f"'{comp}'" for comp in decoded_completions)
            print(f"  Samples: {completions_str}")
            print(f"  Rewards: {[float(r) for r in rewards]}")
            print(f"  Advantages: {[float(a) for a in advantages]}")

        print("\n" + "=" * 80)
        print("Training complete!")

        # Final evaluation: generate a few samples
        print("\nFinal samples:")
        key = jax.random.PRNGKey(999)
        for i in range(5):
            key, subkey = jax.random.split(key)
            tokens, _ = sample_completion(model, prompt_named, args.max_gen_tokens, subkey, temperature=0.7)
            completion = tokenizer.decode(tokens, skip_special_tokens=True)
            reward = compute_reward(tokens, expected_tokens)
            print(f"  {i+1}. '{completion}' (reward={reward})")


if __name__ == "__main__":
    main()
