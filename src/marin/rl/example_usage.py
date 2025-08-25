#!/usr/bin/env python3
"""Example usage of the new in-memory ReplayBuffer + BatchMaker system.

This script demonstrates how to:
1. Create a ReplayBuffer (in-memory) and extend it with rollouts
2. Create a BatchMaker and add rollouts to it
3. Sample from the ReplayBuffer and create a training batch from the BatchMaker
"""

import time
import uuid

import jax
import numpy as np

from .datatypes import InferenceMetadata, RolloutRecord, Turn
from .batch_maker import RlooBatchMaker
from .replay_buffer import ReplayBuffer


def create_mock_rollout(
    environment: str, example_id: str, policy_version: str, message: str, reward: float
) -> RolloutRecord:
    """Create a mock rollout for demonstration purposes."""
    # Assistant turn with dummy logprobs and minimal metadata
    turn = Turn(
        message=message,
        role="assistant",
        logprobs=np.array([-0.1, -0.2, -0.3], dtype=np.float32),
        reward=reward,
        inference_metadata=InferenceMetadata(model_version="test_model", usage={}),
    )

    # Create rollout record
    rollout = RolloutRecord(
        environment=environment,
        example_id=example_id,
        policy_version=policy_version,
        rollout_uid=str(uuid.uuid4()),
        turns=[turn],
        created_ts=time.time(),
        metadata={"test": True},
        replica_id="test_replica",
    )

    return rollout


def main():
    print("Starting RL Dataset System Demo (in-memory)")

    # 1. Create RlooBatchMaker (pure in-memory)
    print("\nCreating RlooBatchMaker...")
    batch_maker = RlooBatchMaker(rng_seed=42)

    # 2. Create in-memory ReplayBuffer
    print("Creating ReplayBuffer (in-memory)...")
    rb = ReplayBuffer(prng_key=jax.random.PRNGKey(0), min_group_size=2)

    # 3. Add mock rollouts
    print("\nAdding rollouts to ReplayBuffer and BatchMaker...")
    rollouts = [
        create_mock_rollout("math_problems", "problem_1", "v1", "The answer is 4", 0.5),
        create_mock_rollout("math_problems", "problem_1", "v2", "The answer is 4", 1.0),  # Better version
        create_mock_rollout("math_problems", "problem_2", "v1", "The answer is 9", 0.8),
    ]

    # Extend replay buffer and feed batch maker
    rb.extend(rollouts)
    for r in rollouts:
        batch_maker.add_rollout(r)

    # 4. Sample from replay buffer
    print("\nSampling from ReplayBuffer...")
    sampled = rb.sample(bsize=2, step=0)
    print(f"  Sampled {len(sampled)} rollouts with non-zero advantage candidates")

    # 5. Create a batch using BatchMaker
    print("\nCreating a training batch...")
    batch = batch_maker.create_batch(batch_size=2)
    if batch:
        metadata = batch_maker.get_batch_metadata(batch)
        print(f"  Batch size: {len(batch)}")
        print(f"  Batch metadata: {metadata}")
    else:
        print("  No batch created - insufficient data")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
