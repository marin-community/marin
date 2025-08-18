#!/usr/bin/env python3
"""Example usage of the new ReplayBuffer + BatchMaker system.

This script demonstrates how to:
1. Create a ReplayBuffer with a BatchMaker
2. Add rollouts to the buffer
3. Use the ReplayBuffer to create and store training batches
"""

import numpy as np
from pathlib import Path
import tempfile
import time
import uuid

from .datatypes import RolloutRecord, Turn, InferenceMetadata
from .batch_maker import GrpoBatchMaker
from .replay_buffer import ReplayBuffer


def create_mock_rollout(
    environment: str, example_id: str, policy_version: str, message: str, reward: float
) -> RolloutRecord:
    """Create a mock rollout for demonstration purposes."""
    # Create a simple turn with the message
    turn = Turn(
        message=message,
        logprobs=np.array([-0.1, -0.2, -0.3], dtype=np.float32),
        role="assistant",
        reward=reward,
        inference_metadata=InferenceMetadata(
            model_version="test_model", sampling_params={"temperature": 0.7}, inference_time=0.1
        ),
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
    """Main demonstration function."""
    print("Starting RL Dataset System Demo")

    # Create temporary directory for storage
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Using temporary directory: {temp_path}")

        # 1. Create GrpoBatchMaker (no I/O responsibilities)
        print("\nCreating GrpoBatchMaker...")
        batch_maker = GrpoBatchMaker(rng_seed=42)  # Optional seed for reproducible sampling

        # 2. Create ReplayBuffer with BatchMaker
        print("Creating ReplayBuffer with BatchMaker...")
        replay_buffer = ReplayBuffer.remote(
            root_path=str(temp_path), compression="zstd", target_group_size=2, batch_maker=batch_maker  # Small for demo
        )

        # 3. Add mock rollouts
        print("\nAdding rollouts to ReplayBuffer...")

        # Create rollouts for the same problem with different policy versions
        rollouts = [
            create_mock_rollout("math_problems", "problem_1", "v1", "The answer is 4", 0.5),
            create_mock_rollout("math_problems", "problem_1", "v2", "The answer is 4", 1.0),  # Better version
            create_mock_rollout("math_problems", "problem_2", "v1", "The answer is 9", 0.8),
        ]

        for i, rollout in enumerate(rollouts):
            print(
                f"  Adding rollout {i+1}: {rollout.environment}/{rollout.example_id} (policy {rollout.policy_version})"
            )
            replay_buffer.add_rollout.remote(rollout)
            time.sleep(0.1)  # Small delay

        # 4. Flush the replay buffer to seal groups
        print("\nFlushing replay buffer...")
        replay_buffer.flush.remote()

        # 5. Create and store training batches using ReplayBuffer
        print("\nCreating and storing training batches...")
        batch_size = 2

        # Create and store a batch
        batch_id = replay_buffer.create_and_store_batch.remote(batch_size)

        if batch_id:
            print(f"  Created and stored batch with ID: {batch_id}")

            # Get batch metadata from batch maker
            batch = replay_buffer.create_batch.remote(batch_size)
            if batch:
                metadata = batch_maker.get_batch_metadata(batch)
                print(f"  Batch metadata: {metadata}")
        else:
            print("  No batch created - insufficient data")

        # 6. Show what was created
        print("\nFiles created:")
        print(f"  Experiences: {list((temp_path).glob('*.parquet'))}")
        print(f"  Batches: {list((temp_path / 'batches').glob('*.parquet'))}")

        print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
