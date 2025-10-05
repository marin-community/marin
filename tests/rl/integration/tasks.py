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

"""Task-specific helpers for cats and sequential digits integration tests."""

import time

import numpy as np

from marin.rl.types import Rollout, RolloutBatch, RolloutGroup, RolloutMetadata
from tests.rl.integration.config import (
    DummyTokenizer,
    compute_model_logprobs,
    encode_prompt_and_response,
    run_inference_with_engine,
)

# ============================================================================
# Cats Task
# ============================================================================


def compute_cats_reward(response: str) -> float:
    """Compute reward for cat-themed responses using MoarCatsTask logic."""
    num_cats = response.lower().count("cat")
    love_cats = response.lower().count("love cats")
    return (num_cats + (10 * love_cats)) / (1 + len(response))


def create_cats_rollout_batch(
    policy_model,
    batch_size: int,
    tokenizer=None,
    max_input_length: int = 16,
    max_output_length: int = 16,
    pad_token_id: int = 0,
    worker_id: str = "test_worker",
) -> RolloutBatch:
    """Create a rollout batch with cat-themed examples using real model logprob computation."""
    if tokenizer is None:
        tokenizer = DummyTokenizer()

    # Generate synthetic prompt/response examples
    prompts = [
        "i like cats, give me moar cats",
        "do you like cats?",
        "cats",
        "moar cats",
    ]
    positive_words = ["cats", "love"]
    negative_words = ["like", "feel", "for", "give", "me", "moar"]

    examples = []
    rng = np.random.default_rng(42)

    for _ in range(batch_size):
        prompt = rng.choice(prompts)
        if rng.random() < 0.5:
            response = " ".join(rng.choice(positive_words, size=rng.integers(1, 8)))
        else:
            response = " ".join(rng.choice(negative_words, size=rng.integers(1, 8)))
        examples.append((prompt, response))

    # Encode examples
    encoded_examples = []
    for prompt_text, response_text in examples:
        encoded = encode_prompt_and_response(
            prompt_text,
            response_text,
            tokenizer,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            pad_token_id=pad_token_id,
        )
        encoded_examples.append(encoded)

    # Stack arrays for logprob computation
    prompt_tokens = np.stack([ex["prompt_tokens"] for ex in encoded_examples])
    prompt_masks = np.stack([ex["prompt_attention_mask"] for ex in encoded_examples])
    response_tokens = np.stack([ex["response_tokens"] for ex in encoded_examples])
    response_masks = np.stack([ex["response_attention_mask"] for ex in encoded_examples])

    policy_logprobs = compute_model_logprobs(
        policy_model,
        prompt_tokens,
        prompt_masks,
        response_tokens,
        response_masks,
    )

    # Create individual rollouts
    rollouts = []
    for i in range(len(examples)):
        prompt_text, response_text = examples[i]
        encoded_ex = encoded_examples[i]
        episode_reward = compute_cats_reward(response_text)

        # Extract individual arrays (remove batch dimension)
        individual_prompt = encoded_ex["prompt_tokens"][prompt_masks[i] == 1]
        individual_response = encoded_ex["response_tokens"][response_masks[i] == 1]
        individual_logprobs = policy_logprobs[i][response_masks[i] == 1]

        # Token rewards (simple: use episode reward for all response tokens)
        token_rewards = np.full(len(individual_response), episode_reward, dtype=np.float32)

        rollout = Rollout(
            env_name="mock:cats",
            env_example_id=f"cats_example_{i}",
            prompt_tokens=individual_prompt,
            response_tokens=individual_response,
            response_logprobs=individual_logprobs,
            token_rewards=token_rewards,
            episode_reward=episode_reward,
        )
        rollouts.append(rollout)

    # Group rollouts
    group = RolloutGroup(rollouts=rollouts)

    # Use a fixed large step number to ensure these batches are never discarded
    metadata = RolloutMetadata(worker_id=worker_id, timestamp=time.time(), weight_step=10000)
    return RolloutBatch(groups=[group], metadata=metadata)


def validate_cats_model(model, tokenizer) -> dict[str, str]:
    """Test trained model on cat-themed prompts."""
    print("\n" + "=" * 60)
    print("Testing trained model responses:")
    print("=" * 60)

    test_prompts = [
        # prompts from our training data (for train only test)
        "i like cats, give me moar cats",
        "do you like cats?",
        "cats",
        "moar cats",
        # novel prompts
        "moar moar",
        "i love i love",
        "  ",
    ]

    tokenizer = DummyTokenizer()

    _, texts = run_inference_with_engine(
        model=model,
        prompts=test_prompts,
        tokenizer=tokenizer,
        max_tokens=64,
        temperature=0.8,
    )

    for i, (prompt, response) in enumerate(zip(test_prompts, texts, strict=True)):
        print(f"\nPrompt {i + 1}: {prompt}")
        print(f"Response: {response}")

        # Check if response contains cats
        cat_count = response.lower().count("cat")
        if cat_count > 0:
            print(f"  ✓ Contains {cat_count} cat references!")
        else:
            print("  - No cat references found")

    # at least responses should have cats, we should have at least 10 total
    cat_count = 0
    cat_response_count = 0
    for response in texts:
        cat_count += response.lower().count("cat")
        if response.lower().count("cat") > 0:
            cat_response_count += 1

    assert cat_response_count >= 3, f"Expected at least 3 cat responses, got {cat_response_count}"
    assert cat_count >= 10, f"Expected at least 10 cat references, got {cat_count}"

    return {prompt: response for prompt, response in zip(test_prompts, texts, strict=True)}


# ============================================================================
# Sequential Digits Task
# ============================================================================


def compute_sequential_digits_reward(response: str) -> float:
    """Compute reward for sequential digit responses.

    Mirrors the logic in SequentialDigitsTask.compute_reward()
    """
    if not response:
        return -1.0

    # Extract digits from response
    digits = [c for c in response if c.isdigit()]

    if not digits:
        return -2.0

    # Count sequential increasing pairs
    sequential_count = 0
    for i in range(len(digits) - 1):
        curr = int(digits[i])
        next_digit = int(digits[i + 1])

        if next_digit == (curr + 1) % 10 or (curr == 9 and next_digit == 0):
            sequential_count += 1

    non_sequential_count = len(digits) - 1 - sequential_count
    base_reward = sequential_count * 1.0 - non_sequential_count * 0.5
    length_bonus = min(len(digits) / 10.0, 1.0) * 0.5
    non_digit_chars = len(response) - len(digits)
    non_digit_penalty = non_digit_chars * 0.1

    total_reward = base_reward + length_bonus - non_digit_penalty
    return max(-2.0, min(2.0, total_reward))


def create_sequential_digits_rollout_batch(
    policy_model,
    batch_size: int,
    tokenizer=None,
    max_input_length: int = 16,
    max_output_length: int = 16,
    pad_token_id: int = 0,
    worker_id: str = "test_worker",
) -> RolloutBatch:
    """Create a rollout batch with sequential digit examples."""
    if tokenizer is None:
        tokenizer = DummyTokenizer()

    # Generate synthetic prompt/response examples
    prompts = [
        "Count from 0:",
        "Sequence:",
        "0 1 2 3",
        "digits:",
    ]

    # Positive responses: sequential digits
    positive_responses = [
        "0123456789",
        "12345678",
        "234567",
        "01234",
        "56789",
    ]

    # Negative responses: random/backwards digits, or text
    negative_responses = [
        "97531",
        "42",
        "9876543210",
        "5231",
        "cats",
        "cats cats",
    ]

    examples = []
    rng = np.random.default_rng(42)

    for _ in range(batch_size):
        prompt = rng.choice(prompts)
        # 60% chance of positive (sequential) response
        if rng.random() < 0.6:
            response = rng.choice(positive_responses)
        else:
            response = rng.choice(negative_responses)
        examples.append((prompt, response))

    # Encode examples
    encoded_examples = []
    for prompt_text, response_text in examples:
        encoded = encode_prompt_and_response(
            prompt_text,
            response_text,
            tokenizer,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            pad_token_id=pad_token_id,
        )
        encoded_examples.append(encoded)

    # Stack arrays for logprob computation
    prompt_tokens = np.stack([ex["prompt_tokens"] for ex in encoded_examples])
    prompt_masks = np.stack([ex["prompt_attention_mask"] for ex in encoded_examples])
    response_tokens = np.stack([ex["response_tokens"] for ex in encoded_examples])
    response_masks = np.stack([ex["response_attention_mask"] for ex in encoded_examples])

    policy_logprobs = compute_model_logprobs(
        policy_model,
        prompt_tokens,
        prompt_masks,
        response_tokens,
        response_masks,
    )

    # Create individual rollouts
    rollouts = []
    for i in range(len(examples)):
        prompt_text, response_text = examples[i]
        encoded_ex = encoded_examples[i]
        episode_reward = compute_sequential_digits_reward(response_text)

        # Extract individual arrays (remove batch dimension)
        individual_prompt = encoded_ex["prompt_tokens"][prompt_masks[i] == 1]
        individual_response = encoded_ex["response_tokens"][response_masks[i] == 1]
        individual_logprobs = policy_logprobs[i][response_masks[i] == 1]

        # Token rewards (use episode reward for all response tokens)
        token_rewards = np.full(len(individual_response), episode_reward, dtype=np.float32)

        rollout = Rollout(
            env_name="mock:sequential_digits",
            env_example_id=f"seq_digits_example_{i}",
            prompt_tokens=individual_prompt,
            response_tokens=individual_response,
            response_logprobs=individual_logprobs,
            token_rewards=token_rewards,
            episode_reward=episode_reward,
        )
        rollouts.append(rollout)

    # Group rollouts
    group = RolloutGroup(rollouts=rollouts)

    # Use fixed large step number to ensure batches are never discarded
    metadata = RolloutMetadata(worker_id=worker_id, timestamp=time.time(), weight_step=10000)
    return RolloutBatch(groups=[group], metadata=metadata)


def validate_sequential_digits_model(model, tokenizer) -> dict[str, str]:
    """Test trained model on sequential digit generation."""
    print("\n" + "=" * 60)
    print("Testing trained model for sequential digit generation:")
    print("=" * 60)

    test_prompts = [
        # Training-like prompts
        "Count from 0:",
        "Sequence:",
        "0 1 2 3",
        # Novel prompts
        "digits:",
        "Numbers:",
        "",  # Empty prompt test
    ]

    _, texts = run_inference_with_engine(
        model=model,
        prompts=test_prompts,
        tokenizer=tokenizer,
        max_tokens=16,
        temperature=0.8,
    )

    results = {}
    total_sequential_score = 0

    for i, (prompt, response) in enumerate(zip(test_prompts, texts, strict=True)):
        print(f"\nPrompt {i + 1}: '{prompt}'")
        print(f"Response: '{response}'")

        # Analyze response
        digits = [c for c in response if c.isdigit()]
        sequential_pairs = 0

        if len(digits) >= 2:
            for j in range(len(digits) - 1):
                curr = int(digits[j])
                next_d = int(digits[j + 1])
                if next_d == (curr + 1) % 10 or (curr == 9 and next_d == 0):
                    sequential_pairs += 1

        digit_ratio = len(digits) / max(len(response), 1)
        sequence_score = sequential_pairs / max(len(digits) - 1, 1) if len(digits) > 1 else 0

        print(f"  Digits: {digits}")
        print(f"  Sequential pairs: {sequential_pairs}/{max(len(digits) - 1, 0)}")
        print(f"  Digit ratio: {digit_ratio:.2f}")
        print(f"  Sequence score: {sequence_score:.2f}")

        if sequence_score >= 0.5:
            print("  ✓ Good sequential structure!")
        elif len(digits) > 0:
            print("  ~ Some digits present")
        else:
            print("  ✗ No digits found")

        results[prompt] = response
        total_sequential_score += sequence_score

    # Validation assertions
    avg_sequence_score = total_sequential_score / len(test_prompts)
    print(f"\n{'=' * 60}")
    print(f"Average sequence score: {avg_sequence_score:.2f}")
    print(f"{'=' * 60}")

    # Count responses with good sequential structure
    good_responses = 0
    for response in texts:
        digits = [c for c in response if c.isdigit()]
        if len(digits) > 1:
            seq_pairs = 0
            for j in range(len(digits) - 1):
                curr = int(digits[j])
                next_d = int(digits[j + 1])
                if next_d == (curr + 1) % 10 or (curr == 9 and next_d == 0):
                    seq_pairs += 1
            score = seq_pairs / (len(digits) - 1)
            if score >= 0.5:
                good_responses += 1

    # Relaxed criteria for sequential digits (harder task than cats)
    assert good_responses >= 1, f"Expected at least 1 response with good sequential structure, got {good_responses}"
    assert avg_sequence_score >= 0.20, f"Expected average sequence score >= 0.20, got {avg_sequence_score:.2f}"

    return results
