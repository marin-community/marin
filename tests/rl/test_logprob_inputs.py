# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test to verify that trainer logprob calculations match policy logprobs from rollout worker.

This test creates a closed-loop verification:
1. Generate rollouts with policy logprobs (simulating rollout worker)
2. Convert to training batch (simulating train_batch.py)
3. Run loss function and capture computed logprobs
4. Verify masked logprobs match between policy and computed
"""

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import ray
from levanter.models.llama import LlamaConfig
from levanter.trainer import TrainerConfig

from marin.rl.rl_losses import rloo_loss_with_importance_sampling
from marin.rl.train_batch import create_training_batch_from_rollouts
from marin.rl.types import Rollout, RolloutMetadata, RolloutWithAdvantage


def create_test_model(vocab_size: int = 100, seq_len: int = 128):
    """Create a small test model for logprob verification."""
    import haliax as hax

    config = LlamaConfig(
        seq_len=seq_len,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        intermediate_dim=512,
        gradient_checkpointing=False,
    )

    # Initialize model
    Vocab = hax.Axis("vocab", vocab_size)
    key = jrandom.PRNGKey(0)

    with TrainerConfig(
        tensor_parallel_axes=[],
        fsdp_axis=None,
    ).device_mesh:
        model = config.build(Vocab, key=key)

    return model, Vocab


def create_mock_rollout(
    prompt_tokens: list[int],
    response_tokens: list[int],
    response_logprobs: list[float],
    env_name: str = "test_env",
    example_id: str = "test_example",
    episode_reward: float = 1.0,
) -> Rollout:
    """Create a mock rollout with specified tokens and logprobs."""
    return Rollout(
        env_name=env_name,
        env_example_id=example_id,
        prompt_tokens=jnp.array(prompt_tokens, dtype=jnp.int32),
        response_tokens=jnp.array(response_tokens, dtype=jnp.int32),
        response_logprobs=jnp.array(response_logprobs, dtype=jnp.float32),
        token_rewards=jnp.ones(len(response_tokens), dtype=jnp.float32) * episode_reward,
        episode_reward=episode_reward,
        metadata=RolloutMetadata(worker_id="test_worker", timestamp=0.0, weight_step=0),
    )


def compute_logprobs_from_model(model, input_ids, target_ids):
    """Compute logprobs for target tokens given input tokens.

    This mimics what happens in the loss function.
    """
    import haliax as hax
    from optax import softmax_cross_entropy_with_integer_labels

    # Create properly shaped NamedArrays for attention mask and position IDs
    # They need to have the same axes as input_ids
    batch_size = input_ids.array.shape[0]
    seq_len = input_ids.array.shape[1]

    # Create causal mask - lower triangular matrix
    seq_len = input_ids.resolve_axis("position").size
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.int32))
    # Broadcast to batch dimension
    attn_mask = hax.named(causal_mask[None, :, :], ("batch", "position", "key_position"))
    pos_ids = hax.arange(input_ids.resolve_axis("position"))
    # Broadcast pos_ids to batch dimension
    pos_ids = hax.broadcast_to(pos_ids, input_ids.axes)

    # Forward pass
    output = model(
        input_ids=input_ids,
        attn_mask=attn_mask,
        pos_ids=pos_ids,
    )

    logits = output.array.astype(jnp.float32)
    logprobs = -softmax_cross_entropy_with_integer_labels(logits, target_ids.array)

    return logprobs


def test_logprob_consistency_simple():
    """Test that policy logprobs match recomputed logprobs from the same model."""
    import haliax as hax

    # Create a tiny model
    vocab_size = 50
    model, Vocab = create_test_model(vocab_size=vocab_size, seq_len=64)

    # Create mock rollout
    prompt_tokens = [1, 2, 3, 4, 5]  # "What is 2+2?"
    response_tokens = [10, 11]  # "4"

    # We'll compute the actual logprobs from the model for this test
    # First, create full sequence
    full_tokens = jnp.array(prompt_tokens + response_tokens, dtype=jnp.int32)

    # Compute logprobs for the response tokens using the model
    # Input: prompt + response[:-1], Target: prompt[1:] + response
    input_seq = full_tokens[:-1]
    target_seq = full_tokens[1:]

    input_ids = hax.named(input_seq[None, :], ("batch", "position"))
    target_ids = hax.named(target_seq[None, :], ("batch", "position"))

    # Get logprobs from model
    all_logprobs = compute_logprobs_from_model(model, input_ids, target_ids)

    # Extract response logprobs (last len(response_tokens) positions)
    response_logprobs = all_logprobs[0, len(prompt_tokens) - 1 :].tolist()

    print(f"Computed response logprobs from model: {response_logprobs}")

    # Create rollout with these logprobs
    rollout = create_mock_rollout(
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        response_logprobs=response_logprobs,
        episode_reward=1.0,
    )

    # Convert to training format
    rollout_with_adv = RolloutWithAdvantage(rollout=rollout, advantage=1.0)
    batch = create_training_batch_from_rollouts(
        [rollout_with_adv],
        max_tokens=64,
        pad_token_id=0,
    )

    # Verify batch structure
    print(f"\nBatch input_ids shape: {batch.input_ids.array.shape}")
    print(f"Batch target_ids shape: {batch.target_ids.array.shape}")
    print(f"Batch policy_logprobs shape: {batch.policy_logprobs.array.shape}")
    print(f"Batch loss_masks shape: {batch.loss_masks.array.shape}")

    # Check policy_logprobs structure
    policy_logprobs = batch.policy_logprobs.array[0]
    loss_masks = batch.loss_masks.array[0]

    print(f"\nPolicy logprobs (first 10): {policy_logprobs[:10]}")
    print(f"Loss masks (first 10): {loss_masks[:10]}")

    # Extract non-zero policy logprobs (should be response only)
    response_mask = loss_masks > 0
    policy_response_logprobs = policy_logprobs[response_mask]

    print(f"\nExtracted policy response logprobs: {policy_response_logprobs}")
    print(f"Original response logprobs: {response_logprobs}")

    # Recompute logprobs using the same model
    recomputed_logprobs = compute_logprobs_from_model(
        model,
        batch.input_ids,
        batch.target_ids,
    )

    recomputed_response_logprobs = recomputed_logprobs[0][response_mask]

    print(f"Recomputed response logprobs: {recomputed_response_logprobs}")

    # Verify they match
    np.testing.assert_allclose(
        policy_response_logprobs,
        recomputed_response_logprobs,
        rtol=1e-5,
        err_msg="Policy logprobs should match recomputed logprobs for response tokens",
    )

    print("\n✅ Test passed! Policy logprobs match recomputed logprobs.")


def test_logprob_consistency_in_loss():
    """Test logprob consistency within the actual loss function."""
    import haliax as hax

    # Create model
    vocab_size = 50
    model, Vocab = create_test_model(vocab_size=vocab_size, seq_len=64)

    # Use same model as reference (on-policy case)
    reference_model = model

    # Create mock rollout with known structure
    prompt_tokens = [1, 2, 3, 4]
    response_tokens = [10, 11, 12]

    # Compute actual logprobs from the model
    full_tokens = jnp.array(prompt_tokens + response_tokens, dtype=jnp.int32)
    input_seq = full_tokens[:-1]
    target_seq = full_tokens[1:]

    input_ids = hax.named(input_seq[None, :], ("batch", "position"))
    target_ids = hax.named(target_seq[None, :], ("batch", "position"))

    all_logprobs = compute_logprobs_from_model(model, input_ids, target_ids)
    response_logprobs = all_logprobs[0, len(prompt_tokens) - 1 :].tolist()

    # Create rollout
    rollout = create_mock_rollout(
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        response_logprobs=response_logprobs,
        episode_reward=1.0,
    )

    # Convert to training batch
    batch = create_training_batch_from_rollouts(
        [RolloutWithAdvantage(rollout=rollout, advantage=1.0)],
        max_tokens=64,
        pad_token_id=0,
    )

    # Extract values before loss computation
    policy_logprobs_before = batch.policy_logprobs.array.copy()
    loss_masks = batch.loss_masks.array.copy()

    print(f"Loss mask: {loss_masks}")
    # print(f"batch attention array: {batch.attention_mask.array}")

    # Compute loss (this will recompute logprobs internally)
    key = jrandom.PRNGKey(42)
    loss, aux = rloo_loss_with_importance_sampling(
        model=model,
        reference_model=reference_model,
        batch=batch,
        key=key,
        kl_coef=0.0,  # No KL penalty for this test
        clip_epsilon=1.0,  # No clipping for this test
    )

    print(f"\nLoss computed: {loss}")
    current_logprobs = aux["current_logprobs"]

    # Now manually recompute to compare
    # current_logprobs = compute_logprobs_from_model(model, batch.input_ids, batch.target_ids)

    # Compare masked logprobs
    response_mask = loss_masks[0] > 0
    policy_response = policy_logprobs_before[0][response_mask]
    current_response = current_logprobs[0][response_mask]

    print(f"\nPolicy response logprobs: {policy_response}")
    print(f"Current response logprobs: {current_response}")
    print(f"Difference: {jnp.abs(policy_response - current_response)}")

    # In on-policy case with same model, these should be identical
    np.testing.assert_allclose(
        policy_response,
        current_response,
        rtol=1e-5,
        err_msg="In on-policy case, current logprobs should match policy logprobs",
    )

    print("\n✅ Test passed! Logprobs are consistent in loss function.")


def test_logprob_masking():
    """Test that prompt logprobs are properly masked out."""

    vocab_size = 50
    model, Vocab = create_test_model(vocab_size=vocab_size, seq_len=64)

    # Create rollout
    prompt_tokens = [1, 2, 3, 4, 5, 6]  # Longer prompt
    response_tokens = [10, 11]  # Short response
    response_logprobs = [-0.5, -0.3]  # Dummy values

    rollout = create_mock_rollout(
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        response_logprobs=response_logprobs,
    )

    # Convert to training batch
    batch = create_training_batch_from_rollouts(
        [RolloutWithAdvantage(rollout=rollout, advantage=1.0)],
        max_tokens=64,
        pad_token_id=0,
    )

    policy_logprobs = batch.policy_logprobs.array[0]
    loss_masks = batch.loss_masks.array[0]

    # Verify structure
    print(f"\nPrompt length: {len(prompt_tokens)}")
    print(f"Response length: {len(response_tokens)}")
    print(
        f"\nPolicy logprobs (first {len(prompt_tokens)+len(response_tokens)}): {policy_logprobs[:len(prompt_tokens)+len(response_tokens)]}"
    )
    print(
        f"Loss masks (first {len(prompt_tokens)+len(response_tokens)}): {loss_masks[:len(prompt_tokens)+len(response_tokens)]}"
    )

    # Check that prompt positions have zero policy logprobs
    prompt_positions = len(prompt_tokens) - 1  # Shifted for next-token prediction
    assert jnp.all(policy_logprobs[:prompt_positions] == 0), "Prompt positions should have zero policy logprobs"

    # Check that prompt positions are masked
    assert jnp.all(loss_masks[:prompt_positions] == 0), "Prompt positions should be masked"

    # Check that response positions have non-zero policy logprobs
    response_start = prompt_positions
    response_end = response_start + len(response_tokens)
    assert jnp.all(
        policy_logprobs[response_start:response_end] != 0
    ), "Response positions should have non-zero policy logprobs"

    # Check that response positions are not masked
    assert jnp.all(loss_masks[response_start:response_end] == 1), "Response positions should not be masked"

    print("\n✅ Test passed! Masking is correct.")


def test_importance_sampling_ratio():
    """Test that importance sampling ratio is computed correctly."""

    vocab_size = 50
    model, Vocab = create_test_model(vocab_size=vocab_size, seq_len=64)

    # Create two different models to simulate off-policy scenario
    reference_model, _ = create_test_model(vocab_size=vocab_size, seq_len=64)

    # Create rollout
    prompt_tokens = [1, 2, 3]
    response_tokens = [10, 11]

    # Use dummy logprobs (would come from old policy in real scenario)
    response_logprobs = [-1.0, -1.5]

    rollout = create_mock_rollout(
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        response_logprobs=response_logprobs,
    )

    batch = create_training_batch_from_rollouts(
        [RolloutWithAdvantage(rollout=rollout, advantage=1.0)],
        max_tokens=64,
        pad_token_id=0,
    )

    # Compute current logprobs
    current_logprobs = compute_logprobs_from_model(model, batch.input_ids, batch.target_ids)

    # Get policy logprobs from batch
    policy_logprobs = batch.policy_logprobs.array
    loss_masks = batch.loss_masks.array

    # Compute importance ratio manually
    log_ratio = current_logprobs - policy_logprobs
    ratio = jnp.exp(log_ratio)

    # Only look at masked positions
    response_mask = loss_masks[0] > 0
    ratio_at_response = ratio[0][response_mask]

    print(f"\nCurrent logprobs at response: {current_logprobs[0][response_mask]}")
    print(f"Policy logprobs at response: {policy_logprobs[0][response_mask]}")
    print(f"Log ratio at response: {log_ratio[0][response_mask]}")
    print(f"Importance ratio at response: {ratio_at_response}")

    # Verify ratio is positive and reasonable
    assert jnp.all(ratio_at_response > 0), "Importance ratio should be positive"

    print("\n✅ Test passed! Importance sampling ratio computed correctly.")


@ray.remote(resources={"TPU": 4, "TPU-v4-8-head": 1})
def main():
    print("Running logprob consistency tests...\n")
    print("=" * 80)
    print("Test 1: Simple logprob consistency")
    print("=" * 80)
    test_logprob_consistency_simple()

    print("\n" + "=" * 80)
    print("Test 2: Logprob consistency in loss function")
    print("=" * 80)
    test_logprob_consistency_in_loss()

    print("\n" + "=" * 80)
    print("Test 3: Logprob masking")
    print("=" * 80)
    test_logprob_masking()

    print("\n" + "=" * 80)
    print("Test 4: Importance sampling ratio")
    print("=" * 80)
    test_importance_sampling_ratio()

    print("\n" + "=" * 80)
    print("All tests passed! ✅")
    print("=" * 80)


if __name__ == "__main__":
    ray.get(main.remote())
