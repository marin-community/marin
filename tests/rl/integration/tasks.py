# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task-specific helpers for cats and sequential digits integration tests."""

import time

import numpy as np
from marin.rl.decoding import DecodingConfig
from marin.rl.types import Rollout, RolloutBatch, RolloutGroup, RolloutGroupMetadata, RolloutMetadata

from tests.rl.integration.config import (
    DummyTokenizer,
    compute_model_logprobs,
    encode_prompt_and_response,
    run_inference_with_engine,
)

SYNTHETIC_ROLLOUT_TEMPERATURE = 1.0
SYNTHETIC_ROLLOUT_TOP_K = None
SYNTHETIC_ROLLOUTS_PER_GROUP = 4
SYNTHETIC_TASK_VERSION = "integration-v1"


def _is_response_truncated(response_text: str, tokenizer: DummyTokenizer, max_output_length: int) -> bool:
    response_token_count = len(tokenizer.encode(response_text, add_special_tokens=False))
    return response_token_count > max_output_length


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
    step: int,
    tokenizer=None,
    max_input_length: int = 16,
    max_output_length: int = 16,
    pad_token_id: int = 0,
    worker_id: str = "test_worker",
    batch_index: int = 0,
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
    positive_words = ["cats", "love cats"]
    negative_words = ["like", "feel", "for", "give", "me", "moar"]

    examples = []
    rng = np.random.default_rng(42)

    for group_idx, group_start in enumerate(range(0, batch_size, SYNTHETIC_ROLLOUTS_PER_GROUP)):
        prompt = prompts[group_idx % len(prompts)]
        group_size = min(SYNTHETIC_ROLLOUTS_PER_GROUP, batch_size - group_start)
        for sample_idx in range(group_size):
            if sample_idx % 2 == 0:
                response = " ".join(rng.choice(positive_words, size=rng.integers(1, 8)))
            else:
                response = " ".join(rng.choice(negative_words, size=rng.integers(1, 8)))
            examples.append((group_idx, prompt, response))

    # Encode examples
    encoded_examples = []
    for _, prompt_text, response_text in examples:
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

    policy_logprobs = np.asarray(
        compute_model_logprobs(
            policy_model,
            prompt_tokens,
            prompt_masks,
            response_tokens,
            response_masks,
        )
    )

    num_groups = (batch_size + SYNTHETIC_ROLLOUTS_PER_GROUP - 1) // SYNTHETIC_ROLLOUTS_PER_GROUP
    grouped_rollouts: list[list[Rollout]] = [[] for _ in range(num_groups)]
    for i in range(len(examples)):
        group_idx, prompt_text, response_text = examples[i]
        encoded_ex = encoded_examples[i]
        episode_reward = compute_cats_reward(response_text)

        # Extract individual arrays (remove batch dimension)
        response_mask = response_masks[i].astype(bool)
        individual_prompt = encoded_ex["prompt_tokens"][prompt_masks[i] == 1]
        individual_response = encoded_ex["response_tokens"][response_mask]
        individual_logprobs = policy_logprobs[i][response_mask]

        # Token rewards (simple: use episode reward for all response tokens)
        token_rewards = np.full(len(individual_response), episode_reward, dtype=np.float32)

        group_id = f"{worker_id}:cats:{step}:batch-{batch_index}:group-{group_idx}"
        trace_id = f"{worker_id}:cats:{step}:batch-{batch_index}:trace-{group_idx}"
        rollout = Rollout(
            env_name="mock:cats",
            env_example_id=f"cats_example_{group_idx}_{len(grouped_rollouts[group_idx])}",
            prompt_tokens=individual_prompt,
            response_tokens=individual_response,
            response_logprobs=individual_logprobs,
            token_rewards=token_rewards,
            episode_reward=episode_reward,
            decoding=DecodingConfig(
                temperature=SYNTHETIC_ROLLOUT_TEMPERATURE,
                top_k=SYNTHETIC_ROLLOUT_TOP_K,
                max_output_tokens=max_output_length,
            ).as_trace(),
            is_truncated=_is_response_truncated(response_text, tokenizer, max_output_length),
            metadata=RolloutMetadata(
                worker_id=worker_id,
                timestamp=time.time(),
                weight_step=step,
                group_id=group_id,
                trace_id=trace_id,
                task_name="mock:cats",
                task_version=SYNTHETIC_TASK_VERSION,
            ),
        )
        grouped_rollouts[group_idx].append(rollout)

    groups = [
        RolloutGroup(
            rollouts=rollouts,
            metadata=RolloutGroupMetadata(
                group_id=rollouts[0].metadata.group_id,
                trace_id=rollouts[0].metadata.trace_id,
                task_name="mock:cats",
                task_version=SYNTHETIC_TASK_VERSION,
            ),
        )
        for rollouts in grouped_rollouts
        if rollouts
    ]

    metadata = RolloutMetadata(worker_id=worker_id, timestamp=time.time(), weight_step=step)
    return RolloutBatch(groups=groups, metadata=metadata)


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
    step: int,
    tokenizer=None,
    max_input_length: int = 16,
    max_output_length: int = 16,
    pad_token_id: int = 0,
    worker_id: str = "test_worker",
    batch_index: int = 0,
) -> RolloutBatch:
    """Create a rollout batch with sequential digit examples."""
    if tokenizer is None:
        tokenizer = DummyTokenizer()

    examples = []
    rng = np.random.default_rng(42)
    bad_responses = [
        "cats cats",
        "love cats",
        "i like cats",
    ]

    for group_idx, group_start in enumerate(range(0, batch_size, SYNTHETIC_ROLLOUTS_PER_GROUP)):
        start_idx = rng.integers(0, 5)
        end_idx = start_idx + rng.integers(start_idx, 9)
        prompt = f"{start_idx} to {end_idx}:"
        group_size = min(SYNTHETIC_ROLLOUTS_PER_GROUP, batch_size - group_start)
        for sample_idx in range(group_size):
            if sample_idx % 2 == 0:
                response = "".join(str(i) for i in range(start_idx, end_idx))
            else:
                response = rng.choice(bad_responses)
            examples.append((group_idx, prompt, response))

    # Encode examples
    encoded_examples = []
    for _, prompt_text, response_text in examples:
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

    policy_logprobs = np.asarray(
        compute_model_logprobs(
            policy_model,
            prompt_tokens,
            prompt_masks,
            response_tokens,
            response_masks,
        )
    )

    # Create individual rollouts
    num_groups = (batch_size + SYNTHETIC_ROLLOUTS_PER_GROUP - 1) // SYNTHETIC_ROLLOUTS_PER_GROUP
    grouped_rollouts: list[list[Rollout]] = [[] for _ in range(num_groups)]
    for i in range(len(examples)):
        group_idx, prompt_text, response_text = examples[i]
        encoded_ex = encoded_examples[i]
        episode_reward = compute_sequential_digits_reward(response_text)

        # Extract individual arrays (remove batch dimension)
        response_mask = response_masks[i].astype(bool)
        individual_prompt = encoded_ex["prompt_tokens"][prompt_masks[i] == 1]
        individual_response = encoded_ex["response_tokens"][response_mask]
        individual_logprobs = policy_logprobs[i][response_mask]

        # Token rewards (use episode reward for all response tokens)
        token_rewards = np.full(len(individual_response), episode_reward, dtype=np.float32)

        group_id = f"{worker_id}:sequential_digits:{step}:batch-{batch_index}:group-{group_idx}"
        trace_id = f"{worker_id}:sequential_digits:{step}:batch-{batch_index}:trace-{group_idx}"
        rollout = Rollout(
            env_name="mock:sequential_digits",
            env_example_id=f"seq_digits_example_{group_idx}_{len(grouped_rollouts[group_idx])}",
            prompt_tokens=individual_prompt,
            response_tokens=individual_response,
            response_logprobs=individual_logprobs,
            token_rewards=token_rewards,
            episode_reward=episode_reward,
            decoding=DecodingConfig(
                temperature=SYNTHETIC_ROLLOUT_TEMPERATURE,
                top_k=SYNTHETIC_ROLLOUT_TOP_K,
                max_output_tokens=max_output_length,
            ).as_trace(),
            is_truncated=_is_response_truncated(response_text, tokenizer, max_output_length),
            metadata=RolloutMetadata(
                worker_id=worker_id,
                timestamp=time.time(),
                weight_step=step,
                group_id=group_id,
                trace_id=trace_id,
                task_name="mock:sequential_digits",
                task_version=SYNTHETIC_TASK_VERSION,
            ),
        )
        grouped_rollouts[group_idx].append(rollout)

    groups = [
        RolloutGroup(
            rollouts=rollouts,
            metadata=RolloutGroupMetadata(
                group_id=rollouts[0].metadata.group_id,
                trace_id=rollouts[0].metadata.trace_id,
                task_name="mock:sequential_digits",
                task_version=SYNTHETIC_TASK_VERSION,
            ),
        )
        for rollouts in grouped_rollouts
        if rollouts
    ]

    metadata = RolloutMetadata(worker_id=worker_id, timestamp=time.time(), weight_step=step)
    return RolloutBatch(groups=groups, metadata=metadata)


def validate_sequential_digits_model(model, tokenizer) -> dict[str, str]:
    """Test trained model on sequential digit generation."""
    print("\n" + "=" * 60)
    print("Testing trained model for sequential digit generation:")
    print("=" * 60)

    test_prompts = [
        "0 to 9:",
        "1 to 5:",
        "2 to 7:",
        "3 to 8:",
        "4 to 9:",
        "3 to 6:",
        "5 to 9:",
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

        digits = [c for c in response if c.isdigit()]
        sequence_score = 0
        last_digit = -1
        for d in digits:
            curr_digit = int(d)
            if last_digit != -1:
                if curr_digit > last_digit:
                    sequence_score += 1
            last_digit = curr_digit

        digit_ratio = len(digits) / max(len(response), 1)
        sequence_score = sequence_score / max(len(digits), 1)
        print(f"  Digits: {digits}")
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

    return results
